from fastapi import FastAPI, File, UploadFile, HTTPException, BackgroundTasks, Query, Request, Form, Body
from fastapi.responses import FileResponse, JSONResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
import shutil
import os
import io
import json
import uvicorn
import zlib
import pickle
import tempfile
import asyncio
import uuid
import time
import hashlib
import logging
import warnings
from pathlib import Path
from typing import Optional
import logging 

# Medical imaging imports
from oct_converter.dicom import create_dicom_from_oct
from oct_converter.readers import E2E, FDS, FDA
from oct_converter.dicom.fda_meta import fda_dicom_metadata
import zipfile
import pydicom
from PIL import Image
import numpy as np
from scipy.io import savemat
import scipy.io as sio

# Enhanced OCT flattening imports
from scipy import ndimage
from scipy.signal import find_peaks

import logging 
logger = logging.getLogger("kodiac_v1")


from riv_desktop.s3_api import bucket_name, s3

# Optional imports with fallbacks
try:
    import pylibjpeg
    PYLIBJPEG_AVAILABLE = True
    logger.info("pylibjpeg is available for JPEG decompression")
except ImportError:
    PYLIBJPEG_AVAILABLE = False
    logger.warning("pylibjpeg not available - some compressed DICOM files may fail")

try:
    import gdcm
    GDCM_AVAILABLE = True
    logger.info("GDCM is available for advanced DICOM decompression")
except ImportError:
    GDCM_AVAILABLE = False
    logger.warning("GDCM not available - some compressed DICOM files may fail")
except Exception as e:
    GDCM_AVAILABLE = False
    logger.warning(f"GDCM failed to load (DLL issue): {str(e)} - will use pylibjpeg fallback")

try:
    import cv2
    CV2_AVAILABLE = True
    logger.info("OpenCV available for image processing")
except ImportError:
    CV2_AVAILABLE = False
    logger.warning("OpenCV not available - OCT flattening may not work properly")
    cv2 = None

logger = logging.getLogger("kodiac_v1")
logger.setLevel(logging.DEBUG)

app = FastAPI()

app.mount("/static", StaticFiles(directory="static"), name="static")

@app.on_event("shutdown")
async def shutdown_event():
    """Handle application shutdown."""
    cleanup_on_shutdown()

# Memory cache for storing processed images - MUST be defined before importing s3_api
stored_images = {}

# S3 API Router - import after defining stored_images

from riv_desktop.s3_api import router as s3_router

app.include_router(s3_router)



# S3 credentials are managed through the /api/set-s3-credentials endpoint in s3_api.py

# Performance optimization: Add connection pooling and caching
import asyncio
from concurrent.futures import ThreadPoolExecutor, as_completed

# Parallel processing configuration for FDA files
FDA_PARALLEL_CONFIG = {
    "max_frame_workers": 12,  # Maximum workers for frame processing
    "max_png_workers": 8,     # Maximum workers for PNG conversion
    "min_workers": 2,         # Minimum workers to use
    "progress_log_interval": 10  # Log progress every N frames
}

# Global thread pool for CPU-intensive operations
thread_pool = ThreadPoolExecutor(max_workers=4)

# Cache for expensive operations
operation_cache = {}
CACHE_TTL = 300  # 5 minutes

# Get the current file's directory
current_dir = os.path.dirname(os.path.abspath(__file__))

# Construct the path to the static files directory
static_dir = os.path.join(current_dir)

# Ensure the directory exists
if not os.path.exists(static_dir):
    raise RuntimeError(f"Static files directory does not exist: {static_dir}")

# Mount UI assets under /ui to serve frontend scripts and styles
app.mount("/ui", StaticFiles(directory=os.path.join(current_dir, "ui")), name="ui")


@app.get("/")
async def read_index():
    return FileResponse(os.path.join(static_dir, "index.html"))


@app.get("/health")
def health_check():
    return {"status": "healthy"}


@app.get("/api/test")
async def test_backend():
    """Test endpoint for frontend connection verification."""
    return {"message": "Backend connection successful!"}


# Add these constants near the top of your file
CACHE_DIR = Path("cache")
CACHE_DIR.mkdir(exist_ok=True)
MAX_FRAMES_PER_EYE = 97

# CRC-based caching system
CRC_CACHE = {}  # In-memory CRC to file path mapping


def calculate_crc32(file_path: str) -> str:
    """Calculate CRC32 checksum of a file."""
    with open(file_path, 'rb') as f:
        buffer = f.read()
        return format(zlib.crc32(buffer) & 0xFFFFFFFF, '08x')


def calculate_content_crc32(content: bytes) -> str:
    """Calculate CRC32 checksum of content bytes."""
    return format(zlib.crc32(content) & 0xFFFFFFFF, '08x')


def get_cache_path(crc: str) -> Path:
    """Get the cache directory path for a given CRC."""
    return CACHE_DIR / f"{crc}"


def get_file_crc_from_metadata(file_path: str, metadata: dict = None) -> str:
    """Generate CRC from file path and metadata for consistent caching. Fallback to path-only if metadata is missing or unstable."""
    if metadata and all(k in metadata for k in ("size", "last_modified")):
        crc_data = {
            'path': file_path,
            'size': metadata.get('size', 0),
            'last_modified': metadata.get('last_modified', ''),
            'frame': metadata.get('frame', 0)
        }
        content = json.dumps(crc_data, sort_keys=True).encode('utf-8')
        logger.info(f"[CRC] Hashing with metadata: {crc_data}")
        return calculate_content_crc32(content)
    else:
        logger.info(f"[CRC] Hashing with path only: {file_path}")
        return calculate_content_crc32(file_path.encode('utf-8'))


def save_to_cache(crc: str, data: dict, file_type: str, file_info: dict):
    """Enhanced save to cache with validation."""
    try:
        # Determine cache directory based on file type
        cache_dir = CACHE_DIR / file_type
        cache_dir.mkdir(parents=True,
                        exist_ok=True)  # Hierarchical directory structure
        cache_path = cache_dir / crc  # CRC as the final directory name
        cache_path.mkdir(exist_ok=True)

        # Validate data before saving
        if not data or len(data) == 0:
            logger.warning(f"No data to cache for CRC: {crc}")
            return False

        # Save metadata with timestamp
        metadata = {
            "number_of_frames": len(data),
            "cached_at": time.time(),
            "cache_version": "1.0",
            "crc": crc,
            "file_info": file_info
        }

        with open(cache_path / "metadata.pkl", "wb") as f:
            pickle.dump(metadata, f)

        # Save individual frames
        frames_saved = 0
        for frame_num, img_data in data.items():
            try:
                frame_path = cache_path / f"frame_{frame_num}.jpg"
                with open(frame_path, "wb") as f:
                    img_data.seek(0)
                    f.write(img_data.getvalue())
                frames_saved += 1
            except Exception as e:
                logger.error(f"Failed to save frame {frame_num}: {str(e)}")

        logger.info(
            f"Successfully cached {frames_saved} frames for CRC: {crc}")
        logger.info(f"Saving to cache directory: {cache_path}")
        return frames_saved > 0

    except Exception as e:
        logger.error(f"Failed to save cache for CRC {crc}: {str(e)}")
        return False


def load_from_cache(crc: str) -> tuple[dict, dict]:
    """Enhanced load from cache with validation."""
    try:
        # Search all file type subdirectories
        for file_type in ["dicom", "e2e", "fda"]:
            cache_dir = CACHE_DIR / file_type
            cache_path = cache_dir / crc

            if cache_path.exists():
                break  # Found the cache, so exit the loop
        else:
            # No cache found in any directory
            return {}, {}

        if not cache_path.exists():
            return {}, {}

        # Load and validate metadata
        metadata_file = cache_path / "metadata.pkl"
        if not metadata_file.exists():
            logger.warning(f"Cache metadata missing for CRC: {crc}")
            return {}, {}

        with open(metadata_file, "rb") as f:
            metadata = pickle.load(f)

        # Check cache age (optional - expire after 7 days)
        cache_age = time.time() - metadata.get("cached_at", 0)
        if cache_age > (7 * 24 * 3600):  # 7 days
            logger.info(
                f"Cache expired for CRC: {crc} (age: {cache_age/3600:.1f} hours)"
            )
            cleanup_cache_entry(crc)
            return {}, {}

        # Load frames
        cached_images = {}
        expected_frames = metadata.get("number_of_frames", 0)

        for frame_file in cache_path.glob("frame_*.jpg"):
            try:
                frame_num = int(frame_file.stem.split('_')[1])
                img_data = io.BytesIO()
                with open(frame_file, "rb") as f:
                    img_data.write(f.read())
                img_data.seek(0)
                cached_images[frame_num] = img_data
            except Exception as e:
                logger.error(f"Failed to load frame {frame_file}: {str(e)}")

        # For E2E files, restore the complete data structure
        file_info = metadata.get("file_info", {})
        if file_info.get("file_type") == "e2e":
            logger.info(f"E2E file_info from cache: {file_info}")
            
            # Restore the complete E2E structure
            restored_data = {}
            
            # Add the frame images with their original keys
            frame_mapping = file_info.get("frame_mapping", {})
            logger.info(f"Frame mapping from cache: {frame_mapping}")
            logger.info(f"Cached images keys: {list(cached_images.keys())}")
            
            for frame_num, original_key in frame_mapping.items():
                if frame_num in cached_images:
                    restored_data[original_key] = cached_images[frame_num]
                    logger.info(f"Restored frame {frame_num} -> {original_key}")
                else:
                    logger.warning(f"Frame {frame_num} not found in cached_images")
            
            # Add E2E metadata
            restored_data["file_type"] = "e2e"
            restored_data["left_eye_data"] = file_info.get("left_eye_data", {"dicom": [], "oct": []})
            restored_data["right_eye_data"] = file_info.get("right_eye_data", {"dicom": [], "oct": []})
            restored_data["timestamp"] = time.time()
            restored_data["crc"] = crc
            
            logger.info(f"Restored E2E data keys: {list(restored_data.keys())}")
            logger.info(
                f"Successfully restored E2E data structure from cache for CRC: {crc}"
            )
            return restored_data, metadata
        else:
            # For regular files, validate frame count
            if len(cached_images) != expected_frames:
                logger.warning(
                    f"Cache incomplete for CRC: {crc}. Expected {expected_frames}, got {len(cached_images)}"
                )
                cleanup_cache_entry(crc)
                return {}, {}

            logger.info(
                f"Successfully loaded {len(cached_images)} frames from cache for CRC: {crc}"
            )
            return cached_images, metadata
    except Exception as e:
        logger.error(f"Error loading from cache for CRC {crc}: {str(e)}")
        return {}, {}

def cleanup_cache_entry(crc: str):
    """Clean up a specific cache entry."""
    try:
        # Search all file type subdirectories
        for file_type in ["dicom", "e2e", "fda"]:
            cache_dir = CACHE_DIR / file_type
            cache_path = cache_dir / crc
            if cache_path.exists():
                shutil.rmtree(cache_path)
                logger.info(f"Cleaned up cache entry: {crc}")
                return

    except Exception as e:
        logger.error(f"Failed to cleanup cache entry {crc}: {str(e)}")

def cleanup_old_cache_entries(max_age_days: int = 7):
    """Clean up old cache entries."""
    try:
        current_time = time.time()
        max_age_seconds = max_age_days * 24 * 3600

        for file_type in ["dicom", "e2e", "fda"]:
            cache_dir = CACHE_DIR / file_type
            if not cache_dir.exists():
                continue
            for cache_dir_crc in cache_dir.iterdir():
                if cache_dir_crc.is_dir():
                    metadata_file = cache_dir_crc / "metadata.pkl"
                    if metadata_file.exists():
                        try:
                            with open(metadata_file, "rb") as f:
                                metadata = pickle.load(f)

                            cache_age = current_time - metadata.get(
                                "cached_at", 0)
                            if cache_age > max_age_seconds:
                                shutil.rmtree(cache_dir_crc)
                                logger.info(
                                    f"Cleaned up old cache entry: {cache_dir_crc.name}"
                                )
                        except Exception as e:
                            logger.error(
                                f"Error checking cache age for {cache_dir_crc}: {str(e)}"
                            )

    except Exception as e:
        logger.error(f"Failed to cleanup old cache entries: {str(e)}")


# CRC-based endpoint for getting file CRC
@app.get("/api/get-file-crc")
async def get_file_crc(path: str = Query(...)):
    """Get CRC checksum for a file path."""
    try:
        # Check if we have this file's CRC cached
        if path in CRC_CACHE:
            logger.info(f"Returning cached CRC for {path}: {CRC_CACHE[path]}")
            return {"crc": CRC_CACHE[path], "source": "cache"}


        # For S3 paths, use the S3 CRC calculation method to ensure consistency
        if not os.path.exists(path):
            # This is likely an S3 path, use S3 CRC calculation
            try:
                from riv_desktop.s3_api import s3, bucket_name
                if s3 and bucket_name:
                    # Use the SAME method as download_dicom_from_s3
                    try:
                        head_response = s3.head_object(Bucket=bucket_name, Key=path)
                        file_size = head_response.get('ContentLength', 0)
                        last_modified = head_response.get('LastModified', '').isoformat() if head_response.get('LastModified') else ''
                        etag = head_response.get('ETag', '').strip('"')
                        
                        # Calculate CRC based on file metadata (SAME as S3 download)
                        metadata_str = f"{path}:{etag}:{last_modified}:{file_size}"
                        file_crc = format(zlib.crc32(metadata_str.encode('utf-8')) & 0xFFFFFFFF, '08x')
                        
                        CRC_CACHE[path] = file_crc
                        logger.info(f"Generated S3 metadata CRC for {path}: {file_crc}")
                        return {"crc": file_crc, "source": "s3_metadata"}
                    except Exception as head_error:
                        logger.warning(f"S3 head_object failed for {path}: {str(head_error)}")
                        # Fallback to path-only CRC
                        file_crc = format(zlib.crc32(path.encode('utf-8')) & 0xFFFFFFFF, '08x')
                        CRC_CACHE[path] = file_crc
                        logger.info(f"Generated fallback CRC for {path}: {file_crc}")
                        return {"crc": file_crc, "source": "fallback"}
                else:
                    logger.warning(f"S3 client not available for CRC calculation")
                    # Fallback to path-only CRC
                    file_crc = format(zlib.crc32(path.encode('utf-8')) & 0xFFFFFFFF, '08x')
                    CRC_CACHE[path] = file_crc
                    logger.info(f"Generated fallback CRC for {path}: {file_crc}")
                    return {"crc": file_crc, "source": "fallback"}
            except Exception as s3_error:
                logger.warning(f"S3 CRC calculation failed for {path}: {str(s3_error)}")
                # Fallback to path-only CRC
                file_crc = format(zlib.crc32(path.encode('utf-8')) & 0xFFFFFFFF, '08x')
                CRC_CACHE[path] = file_crc
                logger.info(f"Generated fallback CRC for {path}: {file_crc}")
                return {"crc": file_crc, "source": "fallback"}

        # For local files, use file metadata
        stat = os.stat(path)
        metadata = {
            'size': stat.st_size,
            'last_modified': str(stat.st_mtime)
        }

        # Try to get file metadata if local
        metadata = None
        if os.path.exists(path):
            stat = os.stat(path)
            metadata = {
                'size': stat.st_size,
                'last_modified': str(stat.st_mtime)
            }

        file_crc = get_file_crc_from_metadata(path, metadata)
        CRC_CACHE[path] = file_crc
        logger.info(f"Generated CRC for {path}: {file_crc}")
        return {"crc": file_crc, "source": "generated"}
    except Exception as e:
        logger.error(f"Error getting CRC for {path}: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Error getting file CRC: {str(e)}")


# Utility functions
def apply_windowing(pixels, dicom):
    """Apply DICOM windowing (contrast adjustment) with enhanced error handling."""
    try:
        window_center = dicom.get('WindowCenter', None)
        window_width = dicom.get('WindowWidth', None)

        if window_center is not None and window_width is not None:
            # Handle cases where WC/WW are lists or sequences
            if hasattr(window_center,
                       '__iter__') and not isinstance(window_center, str):
                window_center = float(window_center[0])
            else:
                window_center = float(window_center)

            if hasattr(window_width,
                       '__iter__') and not isinstance(window_width, str):
                window_width = float(window_width[0])
            else:
                window_width = float(window_width)

            logger.debug(
                f"Applying windowing: WC={window_center}, WW={window_width}")

            # Apply windowing
            min_value = window_center - (window_width / 2)
            max_value = window_center + (window_width / 2)

            # Clip and normalize
            pixels = np.clip(pixels, min_value, max_value)
            pixels = (pixels - min_value) / (max_value - min_value)
            pixels = (pixels * 255.0).astype(np.uint8)

        else:
            logger.debug(
                "No windowing parameters found, using auto-normalization")
            # Auto-normalize based on pixel intensity range
            pixels_min = pixels.min()
            pixels_max = pixels.max()

            if pixels_max > pixels_min:
                pixels = (pixels - pixels_min) / (pixels_max - pixels_min)
                pixels = (pixels * 255.0).astype(np.uint8)
            else:
                # Handle edge case where all pixels have same value
                pixels = np.full_like(pixels, 128, dtype=np.uint8)

    except Exception as e:
        logger.warning(
            f"Windowing failed, using fallback normalization: {str(e)}")
        # Fallback normalization
        try:
            pixels = pixels.astype(np.float64)
            pixels_min = pixels.min()
            pixels_max = pixels.max()

            if pixels_max > pixels_min:
                pixels = (pixels - pixels_min) / (pixels_max -
                                                  pixels_min) * 255.0
            else:
                pixels = np.full_like(pixels, 128.0)

            pixels = pixels.astype(np.uint8)
        except Exception as fallback_error:
            logger.error(
                f"Even fallback normalization failed: {str(fallback_error)}")
            raise

    return pixels


def convert_dicom_to_image(pixels: np.ndarray, frame_number: int = 0) -> Image:
    """Convert a DICOM file to a PIL Image, optionally selecting a frame."""
    # Handle multi-frame DICOMs
    if len(pixels.shape) == 3:
        pixels = pixels[frame_number]
    return Image.fromarray(pixels)


def check_dicom_compression(dicom_dataset) -> tuple[bool, str]:
    """
    Check if DICOM file is compressed and identify the compression type.

    Returns:
        tuple: (is_compressed, compression_type)
    """
    try:
        transfer_syntax = dicom_dataset.file_meta.TransferSyntaxUID
        #//logger.info(dicom_dataset, "dicom_dataset details")

        # Common compressed transfer syntaxes
        compressed_syntaxes = {
            '1.2.840.10008.1.2.4.50': 'JPEG Baseline',
            '1.2.840.10008.1.2.4.51': 'JPEG Extended',
            '1.2.840.10008.1.2.4.57': 'JPEG Lossless',
            '1.2.840.10008.1.2.4.70': 'JPEG Lossless SV1',
            '1.2.840.10008.1.2.4.80': 'JPEG-LS Lossless',
            '1.2.840.10008.1.2.4.81': 'JPEG-LS Near Lossless',
            '1.2.840.10008.1.2.4.90': 'JPEG 2000 Lossless',
            '1.2.840.10008.1.2.4.91': 'JPEG 2000',
            '1.2.840.10008.1.2.5': 'RLE Lossless',
        }

        compression_type = compressed_syntaxes.get(str(transfer_syntax),
                                                   'Unknown')
        is_compressed = str(transfer_syntax) in compressed_syntaxes

        return is_compressed, compression_type

    except AttributeError:
        # No transfer syntax info available
        return False, 'Unknown'


def decompress_dicom_with_fallbacks(dicom_dataset,
                                    file_path: str) -> Optional[np.ndarray]:
    """
    Attempt to decompress DICOM pixel data using multiple fallback methods.

    Args:
        dicom_dataset: pydicom dataset
        file_path: path to the DICOM file

    Returns:
        numpy array of pixel data or None if all methods fail
    """
    is_compressed, compression_type = check_dicom_compression(dicom_dataset)

    if not is_compressed:
        logger.info("DICOM file is not compressed, using standard pixel_array")
        try:
            return dicom_dataset.pixel_array
        except Exception as e:
            logger.warning(f"Standard pixel_array access failed: {str(e)}")

    logger.info(
        f"Detected compressed DICOM with {compression_type} compression")

    # Method 1: Try pydicom with pylibjpeg
    if PYLIBJPEG_AVAILABLE:
        try:
            logger.info("Attempting decompression with pylibjpeg...")
            # Force pydicom to use pylibjpeg for decompression
            with warnings.catch_warnings():
                warnings.simplefilter("ignore")
                pixel_array = dicom_dataset.pixel_array
            logger.info("Successfully decompressed with pylibjpeg")
            return pixel_array
        except Exception as e:
            logger.warning(f"pylibjpeg decompression failed: {str(e)}")
            
        # Additional pylibjpeg method: try with explicit handler
        try:
            logger.info("Attempting pylibjpeg with explicit handler...")
            import pylibjpeg
            pixel_array = dicom_dataset.pixel_array
            logger.info("Successfully decompressed with pylibjpeg explicit handler")
            return pixel_array
        except Exception as e:
            logger.warning(f"pylibjpeg explicit handler failed: {str(e)}")
    else:
        logger.warning("pylibjpeg not available - this is the primary decompression method")

    # Method 2: Try with GDCM by re-reading the file
    if GDCM_AVAILABLE:
        try:
            logger.info("Attempting decompression with GDCM...")
            # Re-read with fresh pydicom instance
            dicom_gdcm = pydicom.dcmread(file_path, force=True)
            dicom_gdcm.decompress(handler_name='gdcm')
            pixel_array = dicom_gdcm.pixel_array
            logger.info("Successfully decompressed with GDCM")
            return pixel_array
        except Exception as e:
            logger.warning(f"GDCM decompression failed: {str(e)}")
    else:
        logger.info("GDCM not available, skipping GDCM decompression method")

    # Method 3: Try forcing decompression to explicit VR
    try:
        logger.info("Attempting forced decompression...")
        # Create a copy to avoid modifying the original
        dicom_copy = pydicom.dcmread(file_path, force=True)
        dicom_copy.decompress()
        pixel_array = dicom_copy.pixel_array
        logger.info("Successfully decompressed with forced decompression")
        return pixel_array
    except Exception as e:
        logger.warning(f"Forced decompression failed: {str(e)}")

    # Method 4: Try basic pixel array access with error handling
    try:
        logger.info("Attempting basic pixel array access...")
        # Sometimes the original dataset works despite initial failure
        pixel_array = dicom_dataset.pixel_array
        logger.info("Successfully accessed pixel array on retry")
        return pixel_array
    except Exception as e:
        logger.warning(f"Basic pixel array access failed: {str(e)}")

    return None


def enhance_contrast(img, clip_limit=2.0, grid_size=(8, 8)):
    """Enhanced contrast using CLAHE"""
    if CV2_AVAILABLE:
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=grid_size)
        return clahe.apply(img)
    else:
        # Fallback: simple histogram stretching
        img_min, img_max = np.percentile(img, [2, 98])
        if img_max > img_min:
            enhanced = np.clip((img - img_min) / (img_max - img_min) * 255, 0,
                               255)
        else:
            enhanced = img
        return enhanced.astype(np.uint8)


def detect_rpe_curve(img,
                     roi_start,
                     roi_end,
                     kernel_size=151,
                     poly_degree=2,
                     force_straight=True):
    """
    Detect RPE curve using gradient-based method with improved logic from ztest file.

    1) Compute the vertical gradient using Sobel and normalized [0, 255] for EDGE DETECTION
    2) For each column in the ROI, find the maximum gradient value
    3) If force_straight=True, apply polynomial fitting of degree 2 for smoothened curve
    4) If polynomial fitting fails, use median filtering with kernel_size
    """
    height, width = img.shape

    # Compute vertical gradient using Sobel
    if CV2_AVAILABLE:
        grad_y = cv2.Sobel(img, cv2.CV_64F, 0, 1, ksize=3)
        grad_y = np.abs(grad_y).astype(np.float32)
        grad_y = cv2.normalize(grad_y, None, 0, 255,
                               cv2.NORM_MINMAX).astype(np.uint8)
    else:
        # Fallback gradient computation
        grad_y = np.gradient(img.astype(np.float32), axis=0)
        grad_y = np.abs(grad_y)
        grad_y = ((grad_y - grad_y.min()) / (grad_y.max() - grad_y.min()) *
                  255).astype(np.uint8)

    # Find maximum gradient in each column within ROI
    raw_curve = np.zeros(width, dtype=np.int32)
    for j in range(width):
        col = grad_y[roi_start:roi_end, j]
        if len(col) > 0:
            raw_curve[j] = roi_start + np.argmax(col)
        else:
            raw_curve[j] = roi_start

    # Apply smoothing
    valid_x = np.arange(width)
    valid_y = raw_curve

    if force_straight:
        try:
            # Polynomial fitting for smoother curve
            coeffs = np.polyfit(valid_x, valid_y, poly_degree)
            smooth_curve = np.polyval(coeffs, valid_x)
        except Exception:
            # Fallback to median filtering
            from scipy.signal import medfilt
            smooth_curve = medfilt(raw_curve, kernel_size=kernel_size)
    else:
        from scipy.signal import medfilt
        smooth_curve = medfilt(raw_curve, kernel_size=kernel_size)

    return smooth_curve.astype(np.float32)


def flatten_oct_image(image: np.ndarray) -> np.ndarray:
    """
    Enhanced OCT image flattening using the logic from ztest file.

    1) Convert RGB to grayscale if needed
    2) Find ROI automatically based on vertical histogram and cusum
    3) Apply blur for better curvature detection
    4) Call RPE detection function
    5) Flatten the image using detected curve
    """
    try:
        logger.info(
            "Applying enhanced OCT flattening algorithm with new logic")

        # Ensure image is 2D grayscale
        if len(image.shape) > 2:
            if CV2_AVAILABLE:
                image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                image = np.mean(image, axis=2)

        # Convert to uint8 if needed
        if image.dtype != np.uint8:
            image = ((image - image.min()) / (image.max() - image.min()) *
                     255).astype(np.uint8)

        height, width = image.shape
        logger.info(f"Processing image with shape: {image.shape}")

        # Automatic ROI detection based on vertical histogram
        # Sum each row to create vertical histogram
        vertical_hist = np.sum(image, axis=1)

        # Find the region with significant tissue (non-zero intensities)
        # Use cumulative sum to find tissue boundaries
        cumsum = np.cumsum(vertical_hist)
        total_sum = cumsum[-1]

        # Find ROI boundaries (5% to 70% of cumulative intensity)
        roi_start = np.where(cumsum >= 0.05 * total_sum)[0]
        roi_end = np.where(cumsum >= 0.70 * total_sum)[0]

        if len(roi_start) > 0 and len(roi_end) > 0:
            roi_start = max(0, roi_start[0] - 10)  # Add small margin
            roi_end = min(height - 1, roi_end[0] + 10)
        else:
            # Fallback ROI
            roi_start = height // 6
            roi_end = min(height - 1, height * 2 // 3)

        logger.info(f"Detected ROI: {roi_start} to {roi_end}")

        # Apply slight blur for better curvature detection
        if CV2_AVAILABLE:
            blurred_img = cv2.GaussianBlur(image, (3, 3), 0.5)
        else:
            from scipy import ndimage
            blurred_img = ndimage.gaussian_filter(image, sigma=0.5)

        # Detect RPE curve
        rpe_curve = detect_rpe_curve(blurred_img,
                                     roi_start,
                                     roi_end,
                                     kernel_size=151,
                                     poly_degree=2,
                                     force_straight=True)

        # Create flattened image
        flattened = np.zeros_like(image)

        # Calculate reference level (median of curve)
        reference_level = int(np.median(rpe_curve))

        # Flatten each column
        for col in range(width):
            curve_point = int(rpe_curve[col])
            shift = reference_level - curve_point

            # Apply vertical shift with bounds checking
            if shift > 0:
                # Shift down
                end_idx = min(height, height - shift)
                flattened[shift:, col] = image[:end_idx, col]
            elif shift < 0:
                # Shift up
                start_idx = abs(shift)
                if start_idx < height:
                    flattened[:height + shift, col] = image[start_idx:, col]
            else:
                # No shift needed
                flattened[:, col] = image[:, col]

        # Apply contrast enhancement
        flattened = enhance_contrast(flattened,
                                     clip_limit=2.0,
                                     grid_size=(8, 8))

        logger.info("Enhanced OCT flattening completed successfully")
        return flattened

    except Exception as e:
        logger.error(f"Error in enhanced OCT flattening: {e}")
        raise

def compute_quality_score(image: np.ndarray) -> float:
    """Compute OCT frame quality score [0,1] using intensity range and edge content."""
    try:
        if len(image.shape) > 2:
            image = np.mean(image, axis=2)
        img = image.astype(np.float32)
        p2, p98 = np.percentile(img, [2, 98])
        dyn = max(p98 - p2, 1e-6)
        dyn_norm = np.clip(dyn / 255.0, 0.0, 1.0)
        grad = np.gradient(img, axis=0)
        edge_energy = np.mean(np.abs(grad)) / 255.0
        edge_norm = np.clip(edge_energy * 4.0, 0.0, 1.0)
        return float(np.clip(0.6 * dyn_norm + 0.4 * edge_norm, 0.0, 1.0))
    except Exception:
        return 0.0

def detect_ilm_curve(img: np.ndarray, roi_top_frac: float = 0.05, roi_bottom_frac: float = 0.35) -> np.ndarray:
    """Detect ILM curve near the top using gradient maxima within a top ROI."""
    try:
        if len(img.shape) > 2:
            img = np.mean(img, axis=2)
        h, w = img.shape
        roi_start = int(h * roi_top_frac)
        roi_end = int(h * roi_bottom_frac)
        if roi_end <= roi_start:
            roi_start, roi_end = 0, max(1, h // 3)
        grad_y = np.gradient(img.astype(np.float32), axis=0)
        curve = np.zeros(w, dtype=np.int32)
        for j in range(w):
            col = grad_y[roi_start:roi_end, j]
            if len(col) > 0:
                curve[j] = roi_start + int(np.argmax(col))
            else:
                curve[j] = roi_start
        from scipy import ndimage as _nd
        return _nd.median_filter(curve, size=5).astype(np.float32)
    except Exception:
        return np.zeros(img.shape[1], dtype=np.float32)

def compute_oct_thickness_metrics(img: np.ndarray) -> dict:
    """Compute simple retinal thickness metrics using ILM and RPE curves."""
    try:
        if len(img.shape) > 2:
            img = np.mean(img, axis=2)
        h = img.shape[0]
        norm = (img.astype(np.float32) - img.min()) / max(img.ptp(), 1e-6)
        rpe = detect_retinal_surface_enhanced(norm)
        ilm = detect_ilm_curve(img)
        thickness = rpe - ilm
        thickness = np.clip(thickness, 0, h)
        center_span = slice(max(0, img.shape[1] // 2 - 20), min(img.shape[1], img.shape[1] // 2 + 20))
        return {
            "median_thickness": float(np.median(thickness)),
            "std_thickness": float(np.std(thickness)),
            "center_thickness": float(np.median(thickness[center_span]))
        }
    except Exception:
        return {"median_thickness": 0.0, "std_thickness": 0.0, "center_thickness": 0.0}


def flatten_oct_image_enhanced(image: np.ndarray) -> np.ndarray:
    """
    Enhanced OCT image flattening algorithm with better surface detection.

    Args:
        image: Input OCT image as numpy array

    Returns:
        Flattened OCT image
    """
    try:
        logger.info("Applying enhanced OCT flattening algorithm")

        # Ensure image is 2D
        if len(image.shape) > 2:
            image = np.mean(image, axis=2)

        # Convert to float for processing
        img_float = image.astype(np.float32)

        # Normalize image
        img_min, img_max = np.min(img_float), np.max(img_float)
        if img_max > img_min:
            img_normalized = (img_float - img_min) / (img_max - img_min)
        else:
            img_normalized = img_float

        # Enhanced surface detection
        surface_points = detect_retinal_surface_enhanced(img_normalized)

        # Apply polynomial fitting for smoother surface
        x_coords = np.arange(len(surface_points))
        try:
            # Fit polynomial to surface points
            poly_coeffs = np.polyfit(x_coords,
                                     surface_points,
                                     deg=min(3,
                                             len(surface_points) - 1))
            surface_smooth = np.polyval(poly_coeffs, x_coords)
        except:
            # Fallback to Gaussian smoothing
            surface_smooth = ndimage.gaussian_filter1d(surface_points, sigma=3)

        # Create flattened image with interpolation
        flattened = create_flattened_image(img_normalized, surface_smooth)

        # Post-processing: enhance contrast
        flattened = enhance_contrast(flattened)

        # Convert back to uint8
        flattened_scaled = (flattened * 255).astype(np.uint8)

        logger.info("Enhanced OCT flattening completed successfully")
        return flattened_scaled

    except Exception as e:
        logger.error(f"Error in enhanced OCT flattening: {e}")
        # Fallback to basic algorithm
        logger.info("Falling back to basic OCT flattening")
        return flatten_oct_image(image)


def detect_retinal_surface_enhanced(image: np.ndarray) -> np.ndarray:
    """
    Enhanced retinal surface detection using multiple methods.

    Args:
        image: Normalized OCT image

    Returns:
        Array of surface points for each column
    """
    try:
        height, width = image.shape
        surface_points = np.zeros(width)

        # Method 1: Gradient-based detection
        gradient_y = np.gradient(image, axis=0)

        # Method 2: Intensity-based detection
        for col in range(width):
            column = image[:, col]
            gradient_col = gradient_y[:, col]

            # Find strong positive gradients (dark to bright transitions)
            strong_gradients = np.where(gradient_col > 0.1)[0]

            if len(strong_gradients) > 0:
                # Take the first strong gradient as surface
                surface_points[col] = strong_gradients[0]
            else:
                # Fallback: find maximum intensity in upper portion
                upper_portion = column[:height // 3]
                if len(upper_portion) > 0:
                    surface_points[col] = np.argmax(upper_portion)
                else:
                    surface_points[col] = 0

        # Remove outliers using median filtering
        surface_filtered = ndimage.median_filter(surface_points, size=5)

        return surface_filtered

    except Exception as e:
        logger.error(f"Error in surface detection: {e}")
        # Fallback to simple surface detection
        return np.zeros(image.shape[1])


def create_flattened_image(image: np.ndarray,
                           surface_points: np.ndarray) -> np.ndarray:
    """
    Create flattened image by aligning surface points.

    Args:
        image: Input OCT image
        surface_points: Detected surface points for each column

    Returns:
        Flattened image
    """
    try:
        height, width = image.shape
        flattened = np.zeros_like(image)

        # Calculate reference surface level (median)
        reference_level = int(np.median(surface_points))

        for col in range(width):
            surface_row = int(surface_points[col])
            shift = reference_level - surface_row

            # Apply shift with bounds checking
            if shift > 0:
                # Shift down
                end_idx = min(height, height - shift)
                flattened[shift:, col] = image[:end_idx, col]
            elif shift < 0:
                # Shift up
                start_idx = abs(shift)
                flattened[:height + shift, col] = image[start_idx:, col]
            else:
                # No shift needed
                flattened[:, col] = image[:, col]

        return flattened

    except Exception as e:
        logger.error(f"Error creating flattened image: {e}")
        return image


def enhance_contrast(image: np.ndarray,
                     clip_limit=2.0,
                     grid_size=(8, 8)) -> np.ndarray:
    """
    Enhance contrast of the flattened OCT image.

    Args:
        image: Input image
        clip_limit: CLAHE clip limit
        grid_size: CLAHE grid size

    Returns:
        Contrast-enhanced image
    """
    try:
        # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
        if CV2_AVAILABLE:
            # Convert to uint8 for CLAHE if needed
            if image.dtype != np.uint8:
                if image.max() <= 1.0:
                    # Image is normalized to [0,1], scale to [0,255]
                    img_uint8 = (image * 255).astype(np.uint8)
                    should_normalize_back = True
                else:
                    # Image is already in [0,255] range
                    img_uint8 = image.astype(np.uint8)
                    should_normalize_back = False
            else:
                img_uint8 = image
                should_normalize_back = False

            # Apply CLAHE
            clahe = cv2.createCLAHE(clipLimit=clip_limit,
                                    tileGridSize=grid_size)
            enhanced = clahe.apply(img_uint8)

            # Convert back to original format if needed
            if should_normalize_back:
                return enhanced.astype(np.float32) / 255.0
            else:
                return enhanced
        else:
            # Fallback: simple histogram stretching
            img_min, img_max = np.percentile(image, [2, 98])
            if img_max > img_min:
                enhanced = np.clip((image - img_min) / (img_max - img_min), 0,
                                   1)
            else:
                enhanced = image

            return enhanced

    except Exception as e:
        logger.error(f"Error enhancing contrast: {e}")
        return image


def validate_oct_image(image: np.ndarray) -> bool:
    """
    Validate if the image is suitable for OCT flattening.

    Args:
        image: Input image

    Returns:
        True if image is valid for OCT processing
    """
    try:
        # Check if image has reasonable dimensions
        if len(image.shape) < 2:
            return False

        height, width = image.shape[:2]

        # OCT images should have reasonable aspect ratio
        if height < 50 or width < 50:
            return False

        # Check if image has sufficient dynamic range
        img_range = np.max(image) - np.min(image)
        if img_range < 10:  # Very low contrast
            return False

        return True

    except Exception as e:
        logger.error(f"Error validating OCT image: {e}")
        return False


def preprocess_oct_image(image: np.ndarray) -> np.ndarray:
    """
    Preprocess OCT image before flattening.

    Args:
        image: Raw OCT image

    Returns:
        Preprocessed image
    """
    try:
        # Remove noise using Gaussian filter
        if len(image.shape) == 2:
            denoised = ndimage.gaussian_filter(image, sigma=0.5)
        else:
            denoised = image

        # Normalize intensity
        img_min, img_max = np.percentile(denoised, [1, 99])
        if img_max > img_min:
            normalized = np.clip((denoised - img_min) / (img_max - img_min), 0,
                                 1)
        else:
            normalized = denoised

        return normalized

    except Exception as e:
        logger.error(f"Error preprocessing OCT image: {e}")
        return image


def postprocess_flattened_image(image: np.ndarray) -> np.ndarray:
    """
    Post-process flattened OCT image.

    Args:
        image: Flattened OCT image

    Returns:
        Post-processed image
    """
    try:
        # Apply slight sharpening
        if CV2_AVAILABLE:
            kernel = np.array([[-1, -1, -1], [-1, 9, -1], [-1, -1, -1]])
            sharpened = cv2.filter2D(image, -1, kernel)
            # Blend with original
            result = 0.8 * image + 0.2 * sharpened
        else:
            result = image

        # Ensure values are in valid range
        result = np.clip(result, 0, 255)

        return result.astype(np.uint8)

    except Exception as e:
        logger.error(f"Error post-processing flattened image: {e}")
        return image.astype(np.uint8)


# Replace the old apply_oct_flattening function with this enhanced version
def apply_oct_flattening(pixels: np.ndarray,
                         is_middle_frame: bool = False) -> np.ndarray:
    """
    Apply enhanced OCT flattening algorithm using the new logic from ztest file.

    Args:
        pixels: Input pixel data (2D array for single frame or 3D for volume)
        is_middle_frame: If True, pixels is already a 2D middle frame
    """
    try:
        # Handle input data
        if not is_middle_frame and len(pixels.shape) == 3:
            # Use middle frame from 3D volume
            middle_index = pixels.shape[0] // 2
            pixels = pixels[middle_index]
            logger.info(
                f"Using middle frame {middle_index} from 3D volume for flattening"
            )

        # Ensure we have 2D data
        if len(pixels.shape) != 2:
            raise ValueError(
                f"Expected 2D pixel data for flattening, got shape: {pixels.shape}"
            )

        logger.info(f"Flattening 2D image with shape: {pixels.shape}")

        # Apply the new enhanced flattening algorithm
        flattened = flatten_oct_image(pixels)

        logger.info("Enhanced OCT flattening completed successfully")
        return flattened

    except Exception as e:
        logger.error(f"OCT flattening failed: {str(e)}", exc_info=True)
        # Return original image if flattening fails
        try:
            if len(pixels.shape) == 3:
                result_pixels = pixels[pixels.shape[0] // 2]
            else:
                result_pixels = pixels

            if result_pixels.dtype != np.uint8:
                pixels_min = result_pixels.min()
                pixels_max = result_pixels.max()
                if pixels_max > pixels_min:
                    return ((result_pixels - pixels_min) /
                            (pixels_max - pixels_min) * 255).astype(np.uint8)
                else:
                    return np.full_like(result_pixels, 128, dtype=np.uint8)
            return result_pixels.astype(np.uint8)
        except Exception as fallback_error:
            logger.error(f"Even fallback failed: {str(fallback_error)}")
            # Last resort - return a blank image
            return np.zeros((512, 512), dtype=np.uint8)


def process_frame_parallel(frame_data, frame_idx):
    """
    Enhanced parallel frame processing for FDA files with better normalization and error handling.
    
    Args:
        frame_data: numpy array of frame pixel data
        frame_idx: index of the frame being processed
        
    Returns:
        tuple: (frame_idx, processed_frame_data)
    """
    try:
        # Enhanced normalization using percentiles for better contrast
        vmin, vmax = np.percentile(frame_data, (2, 98))  # Slightly more conservative percentiles
        
        # Handle edge case where min and max are the same
        if vmax <= vmin:
            vmax = vmin + 1e-5
            
        # Normalize to 8-bit with better precision
        frame_8bit = ((frame_data - vmin) / (vmax - vmin) * 255).astype(np.uint8)
        
        # Apply slight contrast enhancement
        frame_8bit = np.clip(frame_8bit, 0, 255)
        
        return frame_idx, frame_8bit
    except Exception as e:
        logger.error(f"Error processing frame {frame_idx}: {str(e)}")
        # Return original frame data as fallback
        return frame_idx, frame_data.astype(np.uint8)


def populate_stored_images_from_cache(key, cached_images, metadata, compression_type, is_compressed, number_of_frames):
    """Populate stored_images[key] with all frames and metadata from cache."""
    stored_images[key] = {}
    frame_count = 0
    for frame_num, img_data in cached_images.items():
        if isinstance(frame_num, int):
            stored_images[key][frame_num] = img_data
            frame_count += 1
    stored_images[key]["timestamp"] = time.time()
    stored_images[key]["crc"] = metadata.get("crc", key)
    stored_images[key]["compression_type"] = compression_type
    stored_images[key]["is_compressed"] = is_compressed
    if number_of_frames > 1:
        stored_images[key]["is_oct"] = True
        stored_images[key]["middle_frame_index"] = number_of_frames // 2
    else:
        stored_images[key]["is_oct"] = False
    logger.info(f"[populate_stored_images_from_cache] Loaded {frame_count} frames into memory for key: {key}")


def process_dicom_file(file_path: str, key: str, crc: str):
    """
    Enhanced DICOM processing function with compressed DICOM support and CRC-based caching.
    Now uses parallel frame processing for multi-frame DICOMs.
    Optimized for RGB images to display directly without unnecessary processing.
    """
    # Safety check for .EX.DCM files
    if file_path.lower().endswith('.ex.dcm'):
        logger.warning(f"EX.DCM file detected in process_dicom_file: {file_path} - this should not happen")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "EX.DCM files are not supported",
                "message": "WORK IN PROGRESS - This file type is not yet supported"
            }
        )
    
    logger.info(f"Processing DICOM file: {file_path}")
    logger.info(f"[process_dicom_file] Using CRC key: {crc} for file_path: {file_path}")
    try:
        # Check if this file has an S3 key that needs immediate mapping
        s3_key = None
        if key in stored_images and "s3_key" in stored_images[key]:
            s3_key = stored_images[key]["s3_key"]
            logger.info(f"Found S3 key for mapping: {s3_key}")

        # Check CRC-based cache first
        cache_path = get_cache_path(crc)
        if cache_path.exists():
            logger.info(f"Loading from CRC cache: {crc}")
            cached_images, metadata = load_from_cache(crc)
            if cached_images:
                logger.info(f"Restoring {len(cached_images)} items from cache for CRC: {crc}")
                logger.info(f"Cached items keys: {list(cached_images.keys())}")
                file_info = metadata.get("file_info", {})
                compression_type = file_info.get("compression_type")
                is_compressed = file_info.get("is_compressed")
                number_of_frames = len([k for k in cached_images.keys() if isinstance(k, int)])
                populate_stored_images_from_cache(crc, cached_images, metadata, compression_type, is_compressed, number_of_frames)
                logger.info(f"Successfully loaded {number_of_frames} frames from CRC cache into memory for key: {crc}")
                # IMMEDIATELY map the file_path to the CRC key for UI access
                stored_images[file_path] = stored_images[crc]
                logger.info(f"[IMMEDIATE] Mapped original file path '{file_path}' to CRC key '{crc}' in stored_images.")
                # IMMEDIATELY map the S3 key if available
                if s3_key:
                    stored_images[s3_key] = stored_images[crc]
                    logger.info(f"[IMMEDIATE] Mapped S3 key '{s3_key}' to CRC key '{crc}' in stored_images.")
                if os.path.exists(file_path):
                    os.remove(file_path)
                return JSONResponse(
                    content={
                        "message": "File loaded from CRC cache.",
                        "number_of_frames": number_of_frames,
                        "dicom_file_path": crc,
                        "cache_source": "disk"
                    })

        # Continue with normal processing if no cache or cache failed...
        # Read DICOM file
        dicom = pydicom.dcmread(file_path, force=True)
        logger.info(f"Successfully read DICOM file: {file_path}")

        # Store the raw DICOM bytes for flattening operations
        with open(file_path, 'rb') as f:
            dicom_bytes = f.read()

        # Check compression status
        is_compressed, compression_type = check_dicom_compression(dicom)
        if is_compressed:
            logger.info(f"Detected compressed DICOM: {compression_type}")

        # Get number of frames - ensure it's at least 1
        number_of_frames = max(1, dicom.get("NumberOfFrames", 1))
        logger.info(f"Number of frames: {number_of_frames}")

        # Attempt to get pixel data with fallbacks
        pixels = decompress_dicom_with_fallbacks(dicom, file_path)

        if pixels is None:
            error_msg = f"Failed to decompress DICOM file with {compression_type} compression. "
            if not PYLIBJPEG_AVAILABLE and not GDCM_AVAILABLE:
                error_msg += "Install pylibjpeg or gdcm for compressed DICOM support: "
                error_msg += "pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg pylibjpeg-rle"

            logger.error(error_msg)
            raise HTTPException(
                status_code=422,
                detail={
                    "error":
                    "Compressed DICOM decompression failed",
                    "compression_type":
                    compression_type,
                    "suggestions": [
                        "Install pylibjpeg: pip install pylibjpeg pylibjpeg-libjpeg pylibjpeg-openjpeg",
                        "Install GDCM: pip install gdcm",
                        "Convert DICOM to uncompressed format"
                    ]
                })

        logger.info(
            f"Successfully extracted pixel data. Shape: {pixels.shape}")

        # Check if this is an RGB image (fundus/retinal photography)
        is_rgb_image = False
        if len(pixels.shape) == 3 and pixels.shape[2] == 3:
            # Check photometric interpretation and samples per pixel
            photometric = getattr(dicom, 'PhotometricInterpretation', '')
            samples_per_pixel = getattr(dicom, 'SamplesPerPixel', 1)
            
            if photometric == 'RGB' or samples_per_pixel == 3:
                is_rgb_image = True
                logger.info("RGB image detected - optimizing for direct display")
            elif photometric == 'MONOCHROME2' and samples_per_pixel == 3:
                # Some RGB images are incorrectly tagged as monochrome
                is_rgb_image = True
                logger.info("RGB image detected (monochrome tag) - optimizing for direct display")
        
        # For RGB images, use minimal processing for better performance
        if is_rgb_image:
            logger.info("Processing RGB image with minimal processing for optimal display")
            
            # Initialize storage for this key
            if key not in stored_images:
                stored_images[key] = {}

            # Store the raw DICOM bytes for flattening
            stored_images[key]["dicom_bytes"] = dicom_bytes
            stored_images[key]["timestamp"] = time.time()
            stored_images[key]["crc"] = crc
            stored_images[key]["is_rgb"] = True  # Mark as RGB image
            stored_images[key]["is_oct"] = False  # Not OCT
            stored_images[key]["is_fda"] = False
            stored_images[key]["is_fds"] = False

            # For RGB images, process only the first frame (usually single frame)
            if number_of_frames == 1:
                logger.info("Single frame RGB image - processing for direct display")
                
                # Convert RGB array directly to PIL Image
                try:
                    # Ensure pixel values are in uint8 range
                    if pixels.dtype != np.uint8:
                        if pixels.max() > 255:
                            # Scale down if values exceed 255
                            pixels = (pixels / pixels.max() * 255).astype(np.uint8)
                        else:
                            pixels = pixels.astype(np.uint8)
                    
                    # Create PIL Image directly from RGB array
                    img = Image.fromarray(pixels, mode='RGB')
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG', optimize=True)
                    img_byte_arr.seek(0)
                    
                    stored_images[key][0] = img_byte_arr
                    logger.info(f"RGB image processed and cached successfully. Size: {img_byte_arr.getbuffer().nbytes} bytes")
                    
                except Exception as e:
                    logger.error(f"Error processing RGB image: {str(e)}")
                    # Fall back to standard processing
                    is_rgb_image = False
                    logger.info("Falling back to standard DICOM processing")
            
            # If multi-frame RGB (rare), process normally
            if number_of_frames > 1 or not is_rgb_image:
                logger.info("Multi-frame RGB or fallback - using standard processing")
                is_rgb_image = False

        # Apply windowing (skip for RGB images as they're already properly formatted)
        if not is_rgb_image:
            try:
                pixels = apply_windowing(pixels, dicom)
                logger.info("Applied DICOM windowing successfully")
            except Exception as e:
                logger.warning(f"Windowing failed, using raw pixels: {str(e)}")
                # Normalize to 0-255 range as fallback
                pixels = ((pixels - pixels.min()) / (pixels.max() - pixels.min()) *
                          255).astype(np.uint8)
        else:
            logger.info("Skipping windowing for RGB image - already properly formatted")

        # After: pixels = apply_windowing(pixels, dicom)
        # Add OCT detection and middle frame extraction (skip for RGB images)

        # Initialize storage for this key first (skip if already done for RGB)
        if not is_rgb_image:
            if key not in stored_images:
                stored_images[key] = {}

            # Store the raw DICOM bytes for flattening
            stored_images[key]["dicom_bytes"] = dicom_bytes
            stored_images[key]["timestamp"] = time.time()
            stored_images[key]["crc"] = crc

        # Detect if this is likely an OCT image (multi-frame with depth)
        # Skip OCT detection for RGB images as they're fundus photography
        is_oct_image = False
        middle_frame_pixels = None

        if not is_rgb_image and number_of_frames > 1:
            # Assume multi-frame images are OCT scans (but not RGB)
            is_oct_image = True
            middle_frame_index = number_of_frames // 2

            logger.info(
                f"Detected OCT image with {number_of_frames} frames (indexed 0-{number_of_frames-1})"
            )
            logger.info(
                f"Using middle frame INDEX {middle_frame_index} (frame #{middle_frame_index + 1} of {number_of_frames}) for flattening"
            )

            # Extract middle frame for flattening
            if len(pixels.shape) == 3:
                middle_frame_pixels = pixels[middle_frame_index].copy()
            else:
                middle_frame_pixels = pixels.copy()

            # Store OCT-specific data
            stored_images[key]["is_oct"] = True
            stored_images[key]["middle_frame_index"] = middle_frame_index
            stored_images[key]["middle_frame_pixels"] = middle_frame_pixels
            logger.info(
                f"Stored middle frame pixels with shape: {middle_frame_pixels.shape}"
            )
        else:
            if is_rgb_image:
                logger.info("RGB image detected - skipping OCT processing")
            else:
                logger.info(f"Single frame image detected - not treated as OCT")
            stored_images[key]["is_oct"] = False

        # --- Parallel frame processing for multi-frame DICOMs ---
        # Skip complex processing for RGB images as they're already processed
        if not is_rgb_image:
            logger.info(f"Processing {number_of_frames} frame(s) in parallel (except median)")
            frame_indices = list(range(number_of_frames))
            # For multi-frame, skip the median frame in parallel processing (as in your code)
            if number_of_frames > 1:
                median_idx = number_of_frames // 2
                frames_to_process = [i for i in frame_indices if i != median_idx]
            else:
                frames_to_process = frame_indices

            processed_frames = {}
            if number_of_frames > 1:
                logger.info(f"Starting parallel processing of {len(frames_to_process)} frames (excluding median index {median_idx})")
                with ThreadPoolExecutor(max_workers=8) as executor:
                    future_to_idx = {
                        executor.submit(process_frame_parallel, pixels[idx], idx): idx for idx in frames_to_process
                    }
                    for future in as_completed(future_to_idx):
                        idx, processed_frame = future.result()
                        processed_frames[idx] = processed_frame
                logger.info(f"Parallel processing complete. Processed {len(processed_frames)} frames. Adding median frame (index {median_idx}) to results.")
                # Add the median frame (not processed in parallel) for consistency
                processed_frames[median_idx] = pixels[median_idx] if len(pixels.shape) == 3 else pixels
            else:
                logger.info("Single frame DICOM detected. No parallel processing needed.")
                processed_frames[0] = pixels if len(pixels.shape) == 2 else pixels[0]
        else:
            logger.info("RGB image detected - skipping parallel processing (already optimized)")
            # RGB images are already processed above, so we can skip this section

        # Store processed frames as PNGs in stored_images for better quality and consistency
        # Skip for RGB images as they're already stored above
        if not is_rgb_image:
            for frame in range(number_of_frames):
                try:
                    img_data = processed_frames[frame]
                    img = convert_dicom_to_image(img_data, 0)  # Already 2D
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')  # Use PNG for better quality and consistency
                    img_byte_arr.seek(0)
                    stored_images[key][frame] = img_byte_arr
                    logger.debug(f"Processed frame {frame + 1}/{number_of_frames}")
                except Exception as e:
                    logger.error(f"Failed to process frame {frame}: {str(e)}")
                    raise
        else:
            logger.info("RGB image detected - frames already stored with optimized processing")

        # Save to hierarchical CRC-based cache
        try:
            cache_data = {
                k: v
                for k, v in stored_images[key].items() if isinstance(k, int)
            }
            file_info = {
                "name": os.path.basename(file_path),
                "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                "compression_type": compression_type,
                "is_compressed": is_compressed
            }
            save_to_cache(crc, cache_data, "dicom", file_info)
            logger.info(f"Saved processed DICOM images to CRC cache: {crc}")
        except Exception as e:
            logger.warning(f"Failed to save to CRC cache: {str(e)}")
        # --- Ensure all frames are loaded into memory after first processing ---
        try:
            cached_images, metadata = load_from_cache(crc)
            if cached_images:
                populate_stored_images_from_cache(crc, cached_images, metadata, compression_type, is_compressed, number_of_frames)
                logger.info(f"Ensured all frames are loaded into memory after processing for key: {crc}")
                # IMMEDIATELY map the file_path to the CRC key for UI access
                stored_images[file_path] = stored_images[crc]
                logger.info(f"[IMMEDIATE] Mapped original file path '{file_path}' to CRC key '{crc}' in stored_images.")
                # IMMEDIATELY map the S3 key if available
                if s3_key:
                    stored_images[s3_key] = stored_images[crc]
                    logger.info(f"[IMMEDIATE] Mapped S3 key '{s3_key}' to CRC key '{crc}' in stored_images.")
        except Exception as e:
            logger.warning(f"Failed to reload frames from cache after processing: {str(e)}")

        # Clean up temporary file
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {str(e)}")

        logger.info(
            f"Successfully processed DICOM file with {number_of_frames} frames"
        )

        return JSONResponse(
            content={
                "message": "File uploaded successfully.",
                "number_of_frames": number_of_frames,
                "dicom_file_path": crc,
                "cache_source": "fresh_download",
                "compression_info": {
                    "is_compressed": is_compressed,
                    "compression_type": compression_type
                }
            })

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing DICOM file: {str(e)}",
                     exc_info=True)

        # Clean up file on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass

        raise HTTPException(status_code=500,
                            detail=f"Error processing DICOM file: {str(e)}")


def process_e2e_file(file_path: str, key: str, crc: str):
    """
    Processes an .e2e file, extracts OCT and fundus data for both eyes,
    creates DICOM files, flattened OCT images, and caches them.
    Now also extracts and flattens all original OCT frames for each eye.
    """
    # Safety check for .EX.DCM files
    if file_path.lower().endswith('.ex.dcm'):
        logger.warning(f"EX.DCM file detected in process_e2e_file: {file_path} - this should not happen")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "EX.DCM files are not supported",
                "message": "WORK IN PROGRESS - This file type is not yet supported"
            }
        )
    
    # Start timing for overall processing
    overall_start_time = time.time()
    logger.info(f"[E2E TIMING] Starting E2E file processing: {os.path.basename(file_path)}")
    
    try:
        oct_file = E2E(file_path)
        logger.info("E2E file detected")

        if key not in stored_images:
            stored_images[key] = {}
        stored_images[key]["file_type"] = "e2e"
        stored_images[key]["left_eye_data"] = {"dicom": [], "oct": [], "original_oct": []}
        stored_images[key]["right_eye_data"] = {"dicom": [], "oct": [], "original_oct": []}
        stored_images[key]["timestamp"] = time.time()
        stored_images[key]["crc"] = crc

        original_filename = os.path.basename(file_path).lower()
        logger.info(f"Processing E2E file: {original_filename}")

        def get_laterality_from_filename(filename):
            filename_lower = filename.lower()
            if filename_lower.endswith('l.e2e') or '_l.' in filename_lower:
                return 'L'
            elif filename_lower.endswith('r.e2e') or '_r.' in filename_lower:
                return 'R'
            else:
                if 'l' in filename_lower:
                    return 'L'
                elif 'r' in filename_lower:
                    return 'R'
                return 'L'

        file_laterality = get_laterality_from_filename(original_filename)
        logger.info(f"Detected eye laterality from filename: {file_laterality}")
        per_eye_metrics = {
            'L': {"processed_frames": 0, "quality_scores": [], "thickness_values": [], "volumes": []},
            'R': {"processed_frames": 0, "quality_scores": [], "thickness_values": [], "volumes": []}
        }
        def choose_eye() -> str:
            nonlocal per_eye_original_frame_counts
            left_rem = MAX_FRAMES_PER_EYE - per_eye_original_frame_counts['L']
            right_rem = MAX_FRAMES_PER_EYE - per_eye_original_frame_counts['R']
            return 'R' if right_rem > left_rem else 'L'
        per_eye_original_frame_counts = {'L': 0, 'R': 0}

        # 1. Process Fundus images (unchanged)
        fundus_start_time = time.time()
        try:
            fundus_images = oct_file.read_fundus_image()
            logger.info(f"Processing {len(fundus_images)} fundus images for E2E file")
            
            right_eye_fundus_count = 0
            left_eye_fundus_count = 0
            
            for i, fundus_image in enumerate(fundus_images):
                laterality = 'L' if (i % 2 == 0) else 'R'
                eye_key = "left_eye_data" if laterality == 'L' else "right_eye_data"
                fundus_key = f"{key}_{laterality}_fundus_{i}"
                
                # Log right eye processing specifically
                if laterality == 'R':
                    logger.info(f"[RIGHT EYE] Processing fundus image {i+1}/{len(fundus_images)} - Key: {fundus_key}")
                    right_eye_fundus_count += 1
                else:
                    logger.info(f"[LEFT EYE] Processing fundus image {i+1}/{len(fundus_images)} - Key: {fundus_key}")
                    left_eye_fundus_count += 1
                
                img_byte_arr = io.BytesIO()
                if hasattr(fundus_image, 'image') and fundus_image.image is not None:
                    fundus_data = fundus_image.image
                    if isinstance(fundus_data, np.ndarray):
                        if fundus_data.dtype != np.uint8:
                            fundus_normalized = ((fundus_data - fundus_data.min()) / (fundus_data.max() - fundus_data.min()) * 255).astype(np.uint8)
                        else:
                            fundus_normalized = fundus_data
                        if len(fundus_normalized.shape) == 3:
                            pil_image = Image.fromarray(fundus_normalized)
                        else:
                            pil_image = Image.fromarray(fundus_normalized, mode='L')
                    else:
                        pil_image = fundus_data
                    pil_image.save(img_byte_arr, format='PNG')  # Use PNG for better quality and consistency
                    img_byte_arr.seek(0)
                    stored_images[key][fundus_key] = img_byte_arr
                    stored_images[key][eye_key]["dicom"].append(fundus_key)
                    
                    # Log successful processing for right eye
                    if laterality == 'R':
                        logger.info(f"[RIGHT EYE] Successfully processed fundus image {i+1} - Size: {img_byte_arr.getbuffer().nbytes} bytes")
                else:
                    logger.warning(f"[{laterality} EYE] Fundus image {i+1} has no image data")
            
            logger.info(f"[RIGHT EYE] Fundus processing complete - {right_eye_fundus_count} images processed")
            logger.info(f"[LEFT EYE] Fundus processing complete - {left_eye_fundus_count} images processed")
            
        except Exception as e:
            logger.warning(f"Error processing fundus images: {str(e)}")

        # 2. Process OCT volumes, flatten, and cache all frames
        oct_start_time = time.time()
        try:
            oct_volumes = oct_file.read_oct_volume()
            logger.info(f"Processing {len(oct_volumes)} OCT volumes for E2E file")
            
            right_eye_oct_count = 0
            left_eye_oct_count = 0
            
            for i, oct_volume in enumerate(oct_volumes):
                laterality = choose_eye()
                eye_key = "left_eye_data" if laterality == 'L' else "right_eye_data"
                
                # Log OCT volume processing for right eye specifically
                if laterality == 'R':
                    logger.info(f"[RIGHT EYE] Processing OCT volume {i+1}/{len(oct_volumes)}")
                    right_eye_oct_count += 1
                else:
                    logger.info(f"[LEFT EYE] Processing OCT volume {i+1}/{len(oct_volumes)}")
                    left_eye_oct_count += 1
                
                # Extract all frames from the OCT volume for both flattening and original storage
                if hasattr(oct_volume, 'volume') and len(oct_volume.volume) > 0:
                    volume_frame_count = len(oct_volume.volume)
                    logger.info(f"[{laterality} EYE] OCT volume {i+1} contains {volume_frame_count} frames - Processing both flattened and original frames")
                    if volume_frame_count != MAX_FRAMES_PER_EYE:
                        logger.info(f"[{laterality} EYE] Enforcing {MAX_FRAMES_PER_EYE}-frame cap per eye as industrial standard")
                    try:
                        frames_list = list(oct_volume.volume)
                        dims = [f.shape[:2] if isinstance(f, np.ndarray) else None for f in frames_list]
                        dim_consistent = len(set(dims)) <= 1
                        per_eye_metrics[laterality]["volumes"].append({
                            "frame_count": volume_frame_count,
                            "expected": MAX_FRAMES_PER_EYE,
                            "dimension_consistent": dim_consistent
                        })
                    except Exception as ve:
                        per_eye_metrics[laterality]["volumes"].append({"error": str(ve)})
                    
                    # Process middle frame for flattening first (to maintain current functionality)
                    middle_frame_index = len(oct_volume.volume) // 2
                    oct_slice = oct_volume.volume[middle_frame_index]
                    logger.info(f"[{laterality} EYE] Using middle frame {middle_frame_index+1} for flattening")
                    
                    # Flattened OCT processing (unchanged - maintains current functionality)
                    try:
                        flattened_oct_key = f"{key}_{laterality}_oct_flattened_{i}"
                        logger.info(f"[{laterality} EYE] Starting OCT flattening process - Target key: {flattened_oct_key}")
                        
                        flattened_img_byte_arr = io.BytesIO()
                        flattened_oct_array = apply_oct_flattening(oct_slice, is_middle_frame=True)
                        
                        if isinstance(flattened_oct_array, np.ndarray):
                            flattened_pil_image = Image.fromarray(flattened_oct_array)
                        else:
                            flattened_pil_image = flattened_oct_array
                        
                        flattened_pil_image.save(flattened_img_byte_arr, format='PNG')  # Use PNG for better quality and consistency
                        flattened_img_byte_arr.seek(0)
                        stored_images[key][flattened_oct_key] = flattened_img_byte_arr
                        stored_images[key][eye_key]["oct"].append(flattened_oct_key)  # Append the flattened key
                        
                        logger.info(f"[{laterality} EYE] Successfully flattened OCT volume {i+1} - Size: {flattened_img_byte_arr.getbuffer().nbytes} bytes")
                        
                    except Exception as flatten_error:
                        logger.warning(f"[{laterality} EYE] Error flattening OCT volume {i+1}: {str(flatten_error)}")
                    
                    # Now process all original OCT frames and FLATTEN them
                    logger.info(f"[{laterality} EYE] Starting original OCT frame processing and flattening for volume {i+1}")
                    original_frames_processed = 0
                    
                    remaining_allowance = MAX_FRAMES_PER_EYE - per_eye_original_frame_counts[laterality]
                    if remaining_allowance <= 0:
                        logger.info(f"[{laterality} EYE] Already reached {MAX_FRAMES_PER_EYE} frames. Skipping additional frames in volume {i+1}.")
                    frames_to_process = min(remaining_allowance, volume_frame_count)
                    for local_idx in range(frames_to_process):
                        original_oct_slice = oct_volume.volume[local_idx]
                        try:
                            global_idx = per_eye_original_frame_counts[laterality]
                            original_frame_key = f"{key}_{laterality}_original_oct_frame_{global_idx:04d}"
                            original_img_byte_arr = io.BytesIO()
                            
                            if isinstance(original_oct_slice, np.ndarray):
                                # FLATTEN the original frame instead of storing it as-is
                                logger.info(f"[{laterality} EYE] Flattening original OCT frame {local_idx+1}/{frames_to_process}")
                                flattened_oct_array = apply_oct_flattening(original_oct_slice, is_middle_frame=True)
                                
                                if isinstance(flattened_oct_array, np.ndarray):
                                    # The frame is now flattened, save it
                                    original_pil_image = Image.fromarray(flattened_oct_array)
                                else:
                                    original_pil_image = flattened_oct_array
                            else:
                                # If not numpy array, try to flatten it directly
                                flattened_oct_array = apply_oct_flattening(original_oct_slice, is_middle_frame=True)
                                if isinstance(flattened_oct_array, np.ndarray):
                                    original_pil_image = Image.fromarray(flattened_oct_array)
                                else:
                                    original_pil_image = flattened_oct_array
                            
                            original_pil_image.save(original_img_byte_arr, format='PNG')
                            original_img_byte_arr.seek(0)
                            stored_images[key][original_frame_key] = original_img_byte_arr
                            stored_images[key][eye_key]["original_oct"].append(original_frame_key)
                            
                            original_frames_processed += 1
                            per_eye_original_frame_counts[laterality] += 1
                            try:
                                per_eye_metrics[laterality]["processed_frames"] += 1
                                per_eye_metrics[laterality]["quality_scores"].append(compute_quality_score(flattened_oct_array))
                                tm = compute_oct_thickness_metrics(flattened_oct_array)
                                per_eye_metrics[laterality]["thickness_values"].append(tm.get("median_thickness", 0.0))
                            except Exception as me:
                                logger.warning(f"[{laterality} EYE] Metrics computation failed for frame {local_idx+1}: {str(me)}")
                            
                            # Log every 10th frame to avoid log spam
                            if local_idx % 10 == 0 or local_idx == frames_to_process - 1:
                                logger.info(f"[{laterality} EYE] Processed and FLATTENED original OCT frame {local_idx+1}/{frames_to_process} - Key: {original_frame_key}")
                                
                        except Exception as frame_error:
                            logger.warning(f"[{laterality} EYE] Error processing and flattening original OCT frame {local_idx+1}: {str(frame_error)}")
                    
                    logger.info(f"[{laterality} EYE] Completed original OCT frame processing and flattening for volume {i+1} - {original_frames_processed}/{frames_to_process} frames flattened")
                    
                else:
                    logger.warning(f"[{laterality} EYE] OCT volume {i+1} has no volume data or is empty")
            
            logger.info(f"[RIGHT EYE] OCT processing complete - {right_eye_oct_count} volumes processed")
            logger.info(f"[LEFT EYE] OCT processing complete - {left_eye_oct_count} volumes processed")
            
        except Exception as e:
            logger.warning(f"Error processing OCT volumes: {str(e)}")

        # Save to CRC-based cache (all frames, including original_oct)
        cache_start_time = time.time()
        try:
            cache_data = {k: v for k, v in stored_images[key].items() if isinstance(v, io.BytesIO)}
            file_info = {"name": os.path.basename(file_path), "size": os.path.getsize(file_path) if os.path.exists(file_path) else 0}
            save_to_cache(crc, cache_data, "e2e", file_info)
            logger.info(f"Saved E2E processed images to CRC cache: {crc}")
        except Exception as e:
            logger.warning(f"Failed to save E2E to CRC cache: {str(e)}")

        if os.path.exists(file_path):
            os.remove(file_path)

        # Generate comprehensive summary with right eye details
        total_images = len([k for k in stored_images[key].keys() if isinstance(stored_images[key][k], io.BytesIO)])
        right_eye_data = stored_images[key]["right_eye_data"]
        left_eye_data = stored_images[key]["left_eye_data"]
        
        logger.info(f"[RIGHT EYE] Final summary:")
        logger.info(f"[RIGHT EYE]   - DICOM/Fundus images: {len(right_eye_data['dicom'])}")
        logger.info(f"[RIGHT EYE]   - Flattened OCT images: {len(right_eye_data['oct'])}")
        logger.info(f"[RIGHT EYE]   - Original OCT frames: {len(right_eye_data['original_oct'])} (now flattened)")
        logger.info(f"[RIGHT EYE]   - Total right eye images: {len(right_eye_data['dicom']) + len(right_eye_data['oct']) + len(right_eye_data['original_oct'])}")
        
        logger.info(f"[LEFT EYE] Final summary:")
        logger.info(f"[LEFT EYE]   - DICOM/Fundus images: {len(left_eye_data['dicom'])}")
        logger.info(f"[LEFT EYE]   - Flattened OCT images: {len(left_eye_data['oct'])}")
        logger.info(f"[LEFT EYE]   - Original OCT frames: {len(left_eye_data['original_oct'])} (now flattened)")
        logger.info(f"[LEFT EYE]   - Total left eye images: {len(left_eye_data['dicom']) + len(left_eye_data['oct']) + len(left_eye_data['original_oct'])}")
        
        # Calculate timing for each processing step
        overall_end_time = time.time()
        overall_duration = overall_end_time - overall_start_time
        
        fundus_end_time = time.time()
        fundus_duration = fundus_end_time - fundus_start_time
        
        oct_end_time = time.time()
        oct_duration = oct_end_time - oct_start_time
        
        cache_end_time = time.time()
        cache_duration = cache_end_time - cache_start_time
        
        # Log comprehensive timing information
        logger.info(f"[E2E TIMING] ===== PROCESSING TIME BREAKDOWN =====")
        logger.info(f"[E2E TIMING] File: {os.path.basename(file_path)}")
        logger.info(f"[E2E TIMING] Fundus processing: {fundus_duration:.2f} seconds")
        logger.info(f"[E2E TIMING] OCT processing: {oct_duration:.2f} seconds")
        logger.info(f"[E2E TIMING] Cache saving: {cache_duration:.2f} seconds")
        logger.info(f"[E2E TIMING] Total processing time: {overall_duration:.2f} seconds")
        logger.info(f"[E2E TIMING] ======================================")
        
        logger.info(f"[E2E PROCESSING] Complete - Total images processed: {total_images} (including flattened original OCT frames)")

        # Build diagnostics summary
        left_diag = {
            "expected_frames": MAX_FRAMES_PER_EYE,
            "processed_frames": per_eye_metrics['L']["processed_frames"],
            "complete": per_eye_metrics['L']["processed_frames"] == MAX_FRAMES_PER_EYE,
            "quality_mean": float(np.mean(per_eye_metrics['L']["quality_scores"])) if per_eye_metrics['L']["quality_scores"] else 0.0,
            "quality_median": float(np.median(per_eye_metrics['L']["quality_scores"])) if per_eye_metrics['L']["quality_scores"] else 0.0,
            "thickness_median": float(np.median(per_eye_metrics['L']["thickness_values"])) if per_eye_metrics['L']["thickness_values"] else 0.0,
            "thickness_std": float(np.std(per_eye_metrics['L']["thickness_values"])) if per_eye_metrics['L']["thickness_values"] else 0.0,
            "volumes": per_eye_metrics['L']["volumes"]
        }
        right_diag = {
            "expected_frames": MAX_FRAMES_PER_EYE,
            "processed_frames": per_eye_metrics['R']["processed_frames"],
            "complete": per_eye_metrics['R']["processed_frames"] == MAX_FRAMES_PER_EYE,
            "quality_mean": float(np.mean(per_eye_metrics['R']["quality_scores"])) if per_eye_metrics['R']["quality_scores"] else 0.0,
            "quality_median": float(np.median(per_eye_metrics['R']["quality_scores"])) if per_eye_metrics['R']["quality_scores"] else 0.0,
            "thickness_median": float(np.median(per_eye_metrics['R']["thickness_values"])) if per_eye_metrics['R']["thickness_values"] else 0.0,
            "thickness_std": float(np.std(per_eye_metrics['R']["thickness_values"])) if per_eye_metrics['R']["thickness_values"] else 0.0,
            "volumes": per_eye_metrics['R']["volumes"]
        }
        try:
            diag_dir = CACHE_DIR / 'e2e' / crc
            diag_dir.mkdir(parents=True, exist_ok=True)
            import json as _json
            with open(diag_dir / f"{crc}_L_diagnostics.json", 'w') as f:
                _json.dump(left_diag, f, indent=2)
            with open(diag_dir / f"{crc}_R_diagnostics.json", 'w') as f:
                _json.dump(right_diag, f, indent=2)
        except Exception as de:
            logger.warning(f"Failed to write diagnostics files: {str(de)}")
        
        return JSONResponse(
            content={
                "message": "E2E file processed and cached successfully.",
                "number_of_frames": total_images,
                "dicom_file_path": key,
                "cache_source": "fresh_download",
                "file_type": "e2e",
                "left_eye_data": stored_images[key]["left_eye_data"],
                "right_eye_data": stored_images[key]["right_eye_data"],
                "processing_time_seconds": round(overall_duration, 2),
                "left_eye_diagnostics": left_diag,
                "right_eye_diagnostics": right_diag
            })
    except Exception as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        logger.error(f"Error processing E2E file: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing E2E file: {str(e)}")


def process_fds_file(file_path: str, key: str, crc: str):
    """
    FDS format processing function - WORK IN PROGRESS.
    FDS files are currently not supported and will return a work-in-progress message.
    """
    # Safety check for .EX.DCM files
    if file_path.lower().endswith('.ex.dcm'):
        logger.warning(f"EX.DCM file detected in process_fds_file: {file_path} - this should not happen")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "EX.DCM files are not supported",
                "message": "WORK IN PROGRESS - This file type is not yet supported"
            }
        )
    
    # Check for .FDS files and return "WORK IN PROGRESS" message
    if file_path.lower().endswith('.fds'):
        logger.info(f"FDS file detected: {file_path} - returning WORK IN PROGRESS message")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "FDS files are not supported",
                "message": "WORK IN PROGRESS - This file type is not yet supported"
            }
        )
    
    try:
        # Check if this file has an S3 key that needs immediate mapping
        s3_key = None
        if key in stored_images and "s3_key" in stored_images[key]:
            s3_key = stored_images[key]["s3_key"]
            logger.info(f"Found S3 key for FDS mapping: {s3_key}")

        # Check CRC-based cache first
        cache_path = get_cache_path(crc)
        if cache_path.exists():
            logger.info(f"Loading from CRC cache: {crc}")
            cached_images, metadata = load_from_cache(crc)

            if cached_images:  # Only proceed if we have cached images
                # Initialize storage for this key
                stored_images[key] = {}

                # Restore all cached frames
                for frame_num, img_data in cached_images.items():
                    if isinstance(frame_num, int):  # Only frame numbers, not metadata
                        stored_images[key][frame_num] = img_data

                # Store essential metadata for proper functioning
                stored_images[key]["timestamp"] = time.time()
                stored_images[key]["crc"] = crc
                stored_images[key]["is_fds"] = True  # Mark as FDS file

                # Restore file info from cache metadata
                file_info = metadata.get("file_info", {})
                if file_info:
                    # Restore compression info
                    stored_images[key]["compression_type"] = file_info.get("compression_type")
                    stored_images[key]["is_compressed"] = file_info.get("is_compressed")
                    
                    # Determine if this was an OCT image based on frame count
                    number_of_frames = len([k for k in cached_images.keys() if isinstance(k, int)])
                    if number_of_frames > 1:
                        stored_images[key]["is_oct"] = True
                        stored_images[key]["middle_frame_index"] = number_of_frames // 2
                        logger.info(f"Restored FDS OCT metadata: {number_of_frames} frames, middle frame index: {stored_images[key]['middle_frame_index']}")
                    else:
                        stored_images[key]["is_oct"] = False
                        logger.info(f"Restored FDS single frame metadata: {number_of_frames} frames")

                # IMMEDIATELY map the S3 key if available
                if s3_key:
                    stored_images[s3_key] = stored_images[key]
                    logger.info(f"[IMMEDIATE] Mapped S3 key '{s3_key}' to key '{key}' in stored_images for FDS.")

                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)

                logger.info(
                    f"Successfully loaded {len([k for k in cached_images.keys() if isinstance(k, int)])} frames from CRC cache"
                )

                return JSONResponse(
                    content={
                                    "message": "File loaded from CRC cache.",
                                    "number_of_frames": len([k for k in cached_images.keys() if isinstance(k, int)]),
                                    "dicom_file_path": key,
                                    "cache_source": "disk"
                                })

        # Try to process as FDA file first (FDS and FDA are often compatible)
        try:
            fda = FDA(file_path)
            logger.info(f"Successfully read FDS file as FDA: {file_path}")
            
            oct = fda.read_oct_volume()  # use oct.volume for frames
            meta = fda_dicom_metadata(oct)  # use as is

            number_of_frames = len(oct.volume)
            
            # Normalize
            frames = normalize_volume(oct.volume)
            pixel_data = np.array(frames).astype(np.uint16)

            if pixel_data is None:
                raise Exception("Failed to extract pixel data from FDS file")

            logger.info(
                f"Successfully extracted pixel data. Shape: {pixel_data.shape}")

            # Apply windowing to fds file
            try:
                pixel_data = pixel_data.astype(np.float64)
                pixels_min = pixel_data.min()
                pixels_max = pixel_data.max()

                if pixels_max > pixels_min:
                    pixel_data = (pixel_data - pixels_min) / (pixels_max -
                                                              pixels_min) * 255.0
                else:
                    pixel_data = np.full_like(pixel_data, 128.0)

                pixel_data = pixel_data.astype(np.uint8)
                logger.info(f"Windowing applied successfully")
            except Exception as fallback_error:
                logger.error(
                    f"Even fallback normalization failed: {str(fallback_error)}")
                raise

            # Initialize storage for this key
            if key not in stored_images:
                stored_images[key] = {}

            dicom_bytes = None  # TODO: research what are the dicom bytes for fds images

            # Store the raw DICOM bytes for flattening
            stored_images[key]["dicom_bytes"] = dicom_bytes
            stored_images[key]["timestamp"] = time.time()
            stored_images[key]["crc"] = crc
            stored_images[key]["is_fds"] = True  # Mark as FDS file

            # Store middle frame pixels for flattening (FDS files are typically multi-frame)
            if number_of_frames > 1:
                middle_frame_index = number_of_frames // 2
                stored_images[key]["middle_frame_index"] = middle_frame_index
                stored_images[key]["middle_frame_pixels"] = pixel_data[middle_frame_index].copy()
                logger.info(f"Stored middle frame pixels for FDS flattening with shape: {pixel_data[middle_frame_index].shape}")

            # Process each frame
            logger.info(f"Processing {number_of_frames} frame(s)")
            for frame in range(number_of_frames):
                try:
                    img = convert_dicom_to_image(pixel_data, frame)
                    img_byte_arr = io.BytesIO()
                    img.save(img_byte_arr, format='PNG')  # Use PNG for better quality and consistency
                    img_byte_arr.seek(0)
                    stored_images[key][frame] = img_byte_arr
                    logger.debug(f"Processed frame {frame + 1}/{number_of_frames}")
                except Exception as e:
                    logger.error(f"Failed to process frame {frame}: {str(e)}")
                    raise

            # Save to hierarchical CRC-based cache
            try:
                cache_data = {
                    k: v
                    for k, v in stored_images[key].items() if isinstance(k, int)
                }
                file_info = {
                    "name":
                    os.path.basename(file_path),
                    "size":
                    os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                    "compression_type": None,
                    "is_compressed": False
                }
                save_to_cache(crc, cache_data, "fds", file_info)
                logger.info(f"Saved processed FDS images to CRC cache: {crc}")
            except Exception as e:
                logger.warning(f"Failed to save to CRC cache: {str(e)}")

            # Clean up temporary file
            try:
                os.remove(file_path)
                logger.info(f"Cleaned up temporary file: {file_path}")
            except Exception as e:
                logger.warning(f"Failed to clean up file {file_path}: {str(e)}")

            logger.info(
                f"Successfully processed FDS file with {number_of_frames} frames")

            return JSONResponse(
                content={
                    "message": "File uploaded successfully.",
                    "number_of_frames": number_of_frames,
                    "dicom_file_path": key,
                    "cache_source": "fresh_download",
                    "compression_info": {
                        "is_compressed": False,
                        "compression_type": None
                    }
                })
                
        except Exception as fda_error:
            logger.warning(f"Failed to process FDS as FDA: {str(fda_error)}")
            # Fall back to not supported
            if os.path.exists(file_path):
                os.remove(file_path)
            
            return JSONResponse(
                content={
                    "error": f"FDS file format not supported: {str(fda_error)}",
            "number_of_frames": 0,
                    "dicom_file_path": key,
            "cache_source": "not_supported"
        })
    except Exception as e:
        logger.error(f"Error processing FDS file: {str(e)}", exc_info=True)
            


# Add OCT flattening functionality
@app.get("/api/flatten_dicom_image")
async def flatten_dicom_image(dicom_file_path: str = Query(...)):
    """
    Enhanced flattening with better error handling and fallback methods.
    """
    logger.info(f"Flattening request for file key: {dicom_file_path}")

    try:
        if dicom_file_path not in stored_images:
            raise HTTPException(status_code=404,
                                detail="DICOM file not found in memory.")

        data = stored_images[dicom_file_path]

        # Check if already flattened and cached
        if "flattened_0" in data:
            logger.info(
                f"Serving cached flattened image for {dicom_file_path}")
            image_buffer = data["flattened_0"]
            image_buffer.seek(0)
            return StreamingResponse(image_buffer, media_type="image/png")

        # Method 1: Check if this is an OCT/FDA/FDS image with stored middle frame
        if (data.get("is_oct", False) or data.get("is_fda", False) or data.get("is_fds", False)) and "middle_frame_pixels" in data:
            logger.info(
                f"Processing OCT flattening using stored middle frame for {dicom_file_path}"
            )

            # Use the pre-stored middle frame pixels
            middle_frame_pixels = data["middle_frame_pixels"]
            middle_frame_index = data.get("middle_frame_index", 0)

            logger.info(
                f"Using middle frame INDEX {middle_frame_index} (frame #{middle_frame_index + 1}) for enhanced OCT flattening"
            )
            logger.info(
                f"Middle frame pixels shape: {middle_frame_pixels.shape}")
            logger.info(
                f"Pixel value range: {middle_frame_pixels.min()} - {middle_frame_pixels.max()}"
            )

            # Apply enhanced OCT flattening algorithm to the middle frame
            flattened_pixels = apply_oct_flattening(middle_frame_pixels,
                                                    is_middle_frame=True)
            logger.info(
                f"Flattening completed. Output shape: {flattened_pixels.shape}"
            )
        # Method 2: Try to use DICOM bytes if available
        elif "dicom_bytes" in data and data["dicom_bytes"] is not None:
            logger.info(
                f"Processing flattening using stored DICOM bytes for {dicom_file_path}"
            )

            # Create a temporary file from the stored bytes
            import tempfile
            with tempfile.NamedTemporaryFile(delete=False,
                                             suffix='.dcm') as tmp:
                tmp.write(data["dicom_bytes"])
                temp_dicom_path = tmp.name

            try:
                # Read the DICOM file for flattening
                dicom = pydicom.dcmread(temp_dicom_path, force=True)
                pixels = decompress_dicom_with_fallbacks(
                    dicom, temp_dicom_path)

                if pixels is None:
                    raise HTTPException(
                        status_code=500,
                        detail="Failed to extract pixel data for flattening")

                # Apply OCT flattening algorithm
                flattened_pixels = apply_oct_flattening(pixels,
                                                        is_middle_frame=False)

            finally:
                # Clean up temporary file
                if os.path.exists(temp_dicom_path):
                    os.remove(temp_dicom_path)

        # Method 3: Fallback - try to reconstruct from stored frame data
        else:
            logger.info(
                f"Using fallback method: reconstructing from frame data for {dicom_file_path}"
            )

            # Get available frames
            frame_keys = [k for k in data.keys() if isinstance(k, int)]
            if not frame_keys:
                raise HTTPException(
                    status_code=400,
                    detail="No frame data available for flattening")

            # Use the first available frame (or middle frame if multiple)
            if len(frame_keys) > 1:
                middle_frame_key = sorted(frame_keys)[len(frame_keys) // 2]
            else:
                middle_frame_key = frame_keys[0]

            logger.info(
                f"Using frame {middle_frame_key} for fallback flattening")

            # Get the frame image data
            frame_buffer = data[middle_frame_key]
            frame_buffer.seek(0)

            # Convert JPEG buffer back to numpy array
            from PIL import Image
            import numpy as np

            pil_image = Image.open(frame_buffer)
            if pil_image.mode != 'L':  # Convert to grayscale if needed
                pil_image = pil_image.convert('L')

            # Convert to numpy array
            frame_pixels = np.array(pil_image)

            # Apply flattening to this frame
            flattened_pixels = apply_oct_flattening(frame_pixels,
                                                    is_middle_frame=True)

        # Convert to image and return (use PIL.Image explicitly)
        from PIL import Image as PILImage
        flattened_img = PILImage.fromarray(flattened_pixels)

        # Save to buffer
        flattened_buffer = io.BytesIO()
        flattened_img.save(flattened_buffer, format='PNG')
        flattened_buffer.seek(0)

        # Cache the result
        data["flattened_0"] = flattened_buffer

        # Return the flattened image
        flattened_buffer.seek(0)
        return StreamingResponse(flattened_buffer, media_type="image/png")

    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Error flattening image: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"Error flattening image: {str(e)}")


# NEW ENDPOINT: Get frame information for multi-frame DICOM - FIXED for single-frame support
@app.get("/api/view_frames/{file_key}")
async def view_frames(file_key: str):
    """Get information about all frames in a DICOM file - supports both single and multi-frame."""
    logger.info(f"Getting frame info for file key: {file_key}")

    try:
        if file_key not in stored_images:
            raise HTTPException(status_code=404,
                                detail="DICOM file not found in memory.")

        frames_data = stored_images[file_key]
        # Count only integer keys (frames), exclude metadata keys
        frame_keys = [k for k in frames_data.keys() if isinstance(k, int)]
        number_of_frames = len(frame_keys)

        # Ensure we have at least 1 frame
        if number_of_frames == 0:
            logger.warning(f"No frames found for file key: {file_key}")
            raise HTTPException(status_code=404,
                                detail="No frames found in DICOM file.")

        # Return frame URLs/info - works for both single and multi-frame
        frame_urls = []
        for frame_num in sorted(frame_keys):
            frame_urls.append(
                f"/api/view_dicom_png?frame={frame_num}&dicom_file_path={file_key}"
            )

        logger.info(
            f"Returning {number_of_frames} frame(s) for file key: {file_key}")

        return {
            "number_of_frames": number_of_frames,
            "frame_urls": frame_urls,
            "file_key": file_key
        }

    except Exception as e:
        logger.error(f"Error getting frame info: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Error getting frame info: {str(e)}")


@app.get("/api/view_e2e_eye")
async def view_e2e_eye(frame: int = Query(...),
                       dicom_file_path: str = Query(...),
                       eye: str = Query(...)):
    logger.info(
        f"Received request to view E2E frame {frame} from file {dicom_file_path} for {eye} eye"
    )

    try:
        if dicom_file_path not in stored_images:
            raise HTTPException(status_code=404,
                                detail="E2E file not found in memory.")

        data = stored_images[dicom_file_path]

        if data.get("file_type") != "e2e":
            raise HTTPException(status_code=400,
                                detail="File is not an E2E file.")

        # Get the appropriate eye data
        eye_key = "left_eye_data" if eye.lower(
        ) == "left" else "right_eye_data"

        if eye_key not in data:
            logger.error(f"Eye key '{eye_key}' not found in data. Available keys: {list(data.keys())}")
            raise HTTPException(status_code=404,
                                detail=f"No data found for {eye} eye.")

        eye_data = data[eye_key]
        
        # Enhanced logging for right eye requests
        if eye.lower() == "right":
            logger.info(f"[RIGHT EYE REQUEST] Processing frame {frame} request")
            logger.info(f"[RIGHT EYE REQUEST] Eye data structure: {eye_data}")
            logger.info(f"[RIGHT EYE REQUEST] DICOM images available: {len(eye_data.get('dicom', []))}")
            logger.info(f"[RIGHT EYE REQUEST] OCT images available: {len(eye_data.get('oct', []))}")
            logger.info(f"[RIGHT EYE REQUEST] Flattened original OCT frames available: {len(eye_data.get('original_oct', []))} (now flattened)")
        else:
            logger.info(f"Eye data for {eye} eye: {eye_data}")

        # Get available images for this eye (DICOM, OCT, and flattened original OCT frames)
        all_images = eye_data.get("dicom", []) + eye_data.get("oct", []) + eye_data.get("original_oct", [])
        
        if eye.lower() == "right":
            logger.info(f"[RIGHT EYE REQUEST] Combined image list ({len(all_images)} total): {all_images}")
        else:
            logger.info(f"All images for {eye} eye: {all_images}")

        if frame >= len(all_images):
            logger.error(f"Frame {frame} out of range. Available frames: {len(all_images)}")
            raise HTTPException(
                status_code=404,
                detail=f"Frame {frame} not found for {eye} eye.")

        frame_key = all_images[frame]
        
        if eye.lower() == "right":
            logger.info(f"[RIGHT EYE REQUEST] Selected frame key: {frame_key}")
        else:
            logger.info(f"Frame key for frame {frame}: {frame_key}")

        if frame_key not in data:
            logger.error(f"Frame key '{frame_key}' not found in data. Available keys: {list(data.keys())}")
            raise HTTPException(status_code=404,
                                detail=f"Frame data '{frame_key}' not found.")

        buf = data[frame_key]
        buf.seek(0)
        
        if eye.lower() == "right":
            logger.info(f"[RIGHT EYE REQUEST] Successfully retrieved frame data - Size: {buf.getbuffer().nbytes} bytes")

        return StreamingResponse(buf, media_type="image/jpeg")

    except HTTPException as e:
        logger.error(
            f"Error retrieving E2E frame {frame} for {eye} eye: {str(e)}")
        raise e
    except Exception as e:
        logger.error(
            f"Error retrieving E2E frame {frame} for {eye} eye: {str(e)}",
            exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"Error processing E2E file: {str(e)}")


@app.get("/api/get_e2e_tree_data")
async def get_e2e_tree_data(dicom_file_path: str = Query(...)):
    """Get tree structure data for E2E file, now including flattened original_oct branch."""
    logger.info(f"Getting E2E tree data for {dicom_file_path}")
    try:
        if dicom_file_path not in stored_images:
            logger.error(f"File {dicom_file_path} not found in stored_images. Available keys: {list(stored_images.keys())}")
            raise HTTPException(status_code=404, detail="E2E file not found in memory.")
        data = stored_images[dicom_file_path]
        logger.info(f"File data keys: {list(data.keys())}")
        logger.info(f"File type: {data.get('file_type')}")
        if data.get("file_type") != "e2e":
            logger.error(f"File {dicom_file_path} is not an E2E file. File type: {data.get('file_type')}")
            raise HTTPException(status_code=400, detail="File is not an E2E file.")
        left_eye_data = data.get("left_eye_data", {"dicom": [], "oct": [], "original_oct": []})
        right_eye_data = data.get("right_eye_data", {"dicom": [], "oct": [], "original_oct": []})
        
        # Enhanced logging for right eye tree data
        logger.info(f"[RIGHT EYE TREE] Right eye data summary:")
        logger.info(f"[RIGHT EYE TREE]   - DICOM/Fundus images: {len(right_eye_data.get('dicom', []))}")
        logger.info(f"[RIGHT EYE TREE]   - Flattened OCT images: {len(right_eye_data.get('oct', []))}")
        logger.info(f"[RIGHT EYE TREE]   - Original OCT frames: {len(right_eye_data.get('original_oct', []))} (now flattened)")
        logger.info(f"[RIGHT EYE TREE]   - Total right eye items: {len(right_eye_data.get('dicom', [])) + len(right_eye_data.get('oct', [])) + len(right_eye_data.get('original_oct', []))}")
        
        logger.info(f"[LEFT EYE TREE] Left eye data summary:")
        logger.info(f"[LEFT EYE TREE]   - DICOM/Fundus images: {len(left_eye_data.get('dicom', []))}")
        logger.info(f"[LEFT EYE TREE]   - Flattened OCT images: {len(left_eye_data.get('oct', []))}")
        logger.info(f"[LEFT EYE TREE]   - Original OCT frames: {len(left_eye_data.get('original_oct', []))} (now flattened)")
        logger.info(f"[LEFT EYE TREE]   - Total left eye items: {len(left_eye_data.get('dicom', [])) + len(left_eye_data.get('oct', [])) + len(left_eye_data.get('original_oct', []))}")
        
        return JSONResponse(
            content={
                "left_eye": {
                    "dicom": left_eye_data.get("dicom", []),
                    "oct": left_eye_data.get("oct", []),
                    "original_oct": left_eye_data.get("original_oct", [])
                },
                "right_eye": {
                    "dicom": right_eye_data.get("dicom", []),
                    "oct": right_eye_data.get("oct", []),
                    "original_oct": right_eye_data.get("original_oct", [])
                },
                "file_type": "e2e"
            })
    except Exception as e:
        logger.error(f"Error getting E2E tree data: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error getting tree data: {str(e)}")


@app.get("/api/view_dicom_png")
async def view_dicom_png(frame: int = Query(...),
                         dicom_file_path: str = Query(...),
                         v: str = Query(None)):
    """Serve a specific frame from the preprocessed DICOM file stored in memory with CRC-based caching. For OCT files, serve the flattened image for the middle frame."""
    logger.info(
        f"Received request to view DICOM PNG for frame {frame} from file {dicom_file_path}"
    )
    logger.info(f"CRC version parameter: {v}")
    logger.info(f"Stored images in memory: {len(stored_images)} entries")

    try:
        if dicom_file_path not in stored_images:
            raise HTTPException(status_code=404,
                                detail="DICOM file not found in memory.")
        logger.info(
            f"Retrieving frame {frame} from DICOM file {dicom_file_path}")

        data = stored_images[dicom_file_path]

        # Check if this is an OCT/FDA/FDS and if the requested frame is the middle frame
        is_oct = data.get("is_oct", False)
        is_fda = data.get("is_fda", False)
        is_fds = data.get("is_fds", False)
        frame_keys = [k for k in data.keys() if isinstance(k, int)]
        
        logger.info(f"File metadata - is_oct: {is_oct}, is_fda: {is_fda}, is_fds: {is_fds}")
        logger.info(f"Available frame keys: {frame_keys}")
        logger.info(f"Requested frame: {frame}")
        
        if (is_oct or is_fda or is_fds) and frame_keys:
            middle_frame_key = sorted(frame_keys)[len(frame_keys) // 2]
            if frame == middle_frame_key:
                # Serve the flattened image if available, otherwise generate and cache it
                if "flattened_0" in data:
                    logger.info(f"Serving cached flattened image for {dicom_file_path}")
                    buf = data["flattened_0"]
                    buf.seek(0)
                    headers = {
                        "Cache-Control": "public, max-age=31536000, immutable",
                        "ETag": f'"{v}"' if v else None
                    }
                    headers = {k: v for k, v in headers.items() if v is not None}
                    return StreamingResponse(buf, media_type="image/png", headers=headers)
                # Generate the flattened image
                if "middle_frame_pixels" in data:
                    middle_frame_pixels = data["middle_frame_pixels"]
                    flattened_pixels = apply_oct_flattening(middle_frame_pixels, is_middle_frame=True)
                else:
                    # Fallback: reconstruct from stored frame data
                    frame_buffer = data[middle_frame_key]
                    frame_buffer.seek(0)
                    from PIL import Image
                    import numpy as np
                    pil_image = Image.open(frame_buffer)
                    if pil_image.mode != 'L':
                        pil_image = pil_image.convert('L')
                    frame_pixels = np.array(pil_image)
                    flattened_pixels = apply_oct_flattening(frame_pixels, is_middle_frame=True)
                from PIL import Image as PILImage
                flattened_img = PILImage.fromarray(flattened_pixels)
                flattened_buffer = io.BytesIO()
                flattened_img.save(flattened_buffer, format='PNG')
                flattened_buffer.seek(0)
                data["flattened_0"] = flattened_buffer
                flattened_buffer.seek(0)
                headers = {
                    "Cache-Control": "public, max-age=31536000, immutable",
                    "ETag": f'"{v}"' if v else None
                }
                headers = {k: v for k, v in headers.items() if v is not None}
                return StreamingResponse(flattened_buffer, media_type="image/png", headers=headers)

        # Default: serve the original frame as before
        if frame not in data:
            raise HTTPException(status_code=404, detail="Frame not found.")
        logger.info(
            f"Frame {frame} found in stored images for {dicom_file_path}")

        buf = data[frame]
        logger.info(
            f"Buffer size for frame {frame}: {buf.getbuffer().nbytes} bytes")
        buf.seek(0)
        logger.info(f"Returning frame {frame} as PNG response")

        # Set CRC-based cache headers for browser caching
        headers = {
            "Cache-Control": "public, max-age=31536000, immutable",
            "ETag": f'"{v}"' if v else None
        }
        headers = {k: v for k, v in headers.items() if v is not None}

        return StreamingResponse(buf, media_type="image/png", headers=headers)

    except Exception as e:
        logger.error(f"Error retrieving DICOM frame {frame}: {str(e)}",
                     exc_info=True)
        raise HTTPException(status_code=500,
                            detail=f"Error processing DICOM file: {str(e)}")


def extract_all_dicom_metadata(dicom_files):
    """Extract all available metadata from DICOM files, excluding pixel data."""
    all_metadata = []

    for dicom_file in dicom_files:
        dicom = pydicom.dcmread(dicom_file)

        dicom_metadata = {}

        # Loop over all DICOM tags and extract their values, excluding pixel data
        for elem in dicom:
            # Exclude the Pixel Data tag (0x7FE0, 0x0010)
            if elem.tag != (0x7FE0,
                            0x0010):  # (0x7FE0, 0x0010) is the Pixel Data tag
                dicom_metadata[elem.name] = str(
                    elem.value
                )  # Convert to string to avoid serialization issues

        all_metadata.append(dicom_metadata)

    return all_metadata


# Utility Function
def normalize_volume(vol: list[np.ndarray]) -> list[np.ndarray]:
    """Normalizes pixel intensities within a range of 0-100.

    Args:
        vol: List of frames
    Returns:
        Normalized list of frames
    """
    arr = np.array(vol)
    norm_vol = []
    diff_arr = arr.max() - arr.min()
    for i in arr:
        temp = ((i - arr.min()) / diff_arr) * 100
        norm_vol.append(temp)
    return norm_vol


# main function for fda processing
def process_fda_file(file_path: str, key: str, crc: str):
    """
    Enhanced FDA format processing function with compressed DICOM support and CRC-based caching.

    Args:
        - file_path: .fda file path
        - key: key
        - crc: cache

    Returns:
        - JSON response
    """
    # Safety check for .EX.DCM files
    if file_path.lower().endswith('.ex.dcm'):
        logger.warning(f"EX.DCM file detected in process_fda_file: {file_path} - this should not happen")
        raise HTTPException(
            status_code=422,
            detail={
                "error": "EX.DCM files are not supported",
                "message": "WORK IN PROGRESS - This file type is not yet supported"
            }
        )
    
    try:
        # Check if this file has an S3 key that needs immediate mapping
        s3_key = None
        if key in stored_images and "s3_key" in stored_images[key]:
            s3_key = stored_images[key]["s3_key"]
            logger.info(f"Found S3 key for FDA mapping: {s3_key}")

        # Check CRC-based cache first
        cache_path = get_cache_path(crc)
        if cache_path.exists():
            logger.info(f"Loading from CRC cache: {crc}")
            cached_images, metadata = load_from_cache(crc)

            if cached_images:  # Only proceed if we have cached images
                # Initialize storage for this key
                stored_images[key] = {}

                # Restore all cached frames
                for frame_num, img_data in cached_images.items():
                    if isinstance(frame_num, int):  # Only frame numbers, not metadata
                        stored_images[key][frame_num] = img_data

                # Store essential metadata for proper functioning
                stored_images[key]["timestamp"] = time.time()
                stored_images[key]["crc"] = crc
                stored_images[key]["is_fda"] = True  # Mark as FDA file

                # Restore file info from cache metadata
                file_info = metadata.get("file_info", {})
                if file_info:
                    # Restore compression info
                    stored_images[key]["compression_type"] = file_info.get("compression_type")
                    stored_images[key]["is_compressed"] = file_info.get("is_compressed")
                    
                    # Determine if this was an OCT image based on frame count
                    number_of_frames = len([k for k in cached_images.keys() if isinstance(k, int)])
                    if number_of_frames > 1:
                        stored_images[key]["is_oct"] = True
                        stored_images[key]["middle_frame_index"] = number_of_frames // 2
                        logger.info(f"Restored FDA OCT metadata: {number_of_frames} frames, middle frame index: {stored_images[key]['middle_frame_index']}")
                    else:
                        stored_images[key]["is_oct"] = False
                        logger.info(f"Restored FDA single frame metadata: {number_of_frames} frames")

                # IMMEDIATELY map the S3 key if available
                if s3_key:
                    stored_images[s3_key] = stored_images[key]
                    logger.info(f"[IMMEDIATE] Mapped S3 key '{s3_key}' to key '{key}' in stored_images for FDA.")

                # Clean up temporary file
                if os.path.exists(file_path):
                    os.remove(file_path)

                logger.info(
                    f"Successfully loaded {len([k for k in cached_images.keys() if isinstance(k, int)])} frames from CRC cache"
                )

                return JSONResponse(
                    content={
                        "message": "File loaded from CRC cache.",
                        "number_of_frames": len([k for k in cached_images.keys() if isinstance(k, int)]),
                        "dicom_file_path": key,
                        "cache_source": "disk"
                    })

        # continue with first time processing
        fda = FDA(file_path)
        logger.info(f"Successfully read fda file: {file_path}")

        compression_type = None
        is_compressed = None

        oct = fda.read_oct_volume()  # use oct.volume for frames
        meta = fda_dicom_metadata(oct)  # use as is

        number_of_frames = len(oct.volume)
        per_frame = []
        pixel_data_bytes = list()  # TODO

        # Normalize
        frames = normalize_volume(oct.volume)

        pixel_data = np.array(frames).astype(np.uint16)

        if pixel_data is None:
            error_msg = f"Failed to convert fda to dicom."
            compression_type = None
            logger.error(error_msg)
            raise HTTPException(status_code=422,
                                detail={
                                    "error":
                                    "Compressed DICOM decompression failed",
                                    "compression_type":
                                    compression_type,
                                    "suggestions":
                                    ["Convert FDA to DICOM separately."]
                                })

        logger.info(
            f"Successfully extracted pixel data. Shape: {pixel_data.shape}")

        # Apply windowing to fda file
        try:
            pixel_data = pixel_data.astype(np.float64)
            pixels_min = pixel_data.min()
            pixels_max = pixel_data.max()

            if pixels_max > pixels_min:
                pixel_data = (pixel_data - pixels_min) / (pixels_max -
                                                          pixels_min) * 255.0
            else:
                pixel_data = np.full_like(pixel_data, 128.0)

            pixel_data = pixel_data.astype(np.uint8)
            logger.info(f"Windowing applied successfully")
        except Exception as fallback_error:
            logger.error(
                f"Even fallback normalization failed: {str(fallback_error)}")
            raise

        # Initialize storage for this key
        if key not in stored_images:
            stored_images[key] = {}

        dicom_bytes = None  #TODO: research what are the dicom bytes for fda and e2e images

        # Store the raw DICOM bytes for flattening
        stored_images[key]["dicom_bytes"] = dicom_bytes
        stored_images[key]["timestamp"] = time.time()
        stored_images[key]["crc"] = crc
        stored_images[key]["is_fda"] = True  # Mark as FDA file

        # Store middle frame pixels for flattening (FDA files are typically multi-frame)
        if number_of_frames > 1:
            middle_frame_index = number_of_frames // 2
            stored_images[key]["middle_frame_index"] = middle_frame_index
            stored_images[key]["middle_frame_pixels"] = pixel_data[middle_frame_index].copy()
            logger.info(f"Stored middle frame pixels for FDA flattening with shape: {pixel_data[middle_frame_index].shape}")

        # --- Enhanced Parallel frame processing for FDA files ---
        logger.info(f"Processing {number_of_frames} frame(s) with enhanced parallel processing")
        
        # Performance monitoring
        frame_processing_start = time.time()
        
        # Determine optimal number of workers based on frame count and system capabilities
        import os
        cpu_count = os.cpu_count() or 4
        optimal_workers = min(
            cpu_count, 
            number_of_frames, 
            FDA_PARALLEL_CONFIG["max_frame_workers"]
        )
        optimal_workers = max(optimal_workers, FDA_PARALLEL_CONFIG["min_workers"])
        
        logger.info(f"Using {optimal_workers} parallel workers for frame processing (CPU cores: {cpu_count}, frames: {number_of_frames})")
        
        processed_frames = {}
        
        if number_of_frames > 1:
            # Process all frames in parallel for better performance
            frame_indices = list(range(number_of_frames))
            
            logger.info(f"Starting enhanced parallel processing of {number_of_frames} frames")
            
            with ThreadPoolExecutor(max_workers=optimal_workers) as executor:
                # Submit all frame processing tasks
                future_to_idx = {
                    executor.submit(process_frame_parallel, pixel_data[idx], idx): idx 
                    for idx in frame_indices
                }
                
                # Collect results as they complete
                completed_count = 0
                for future in as_completed(future_to_idx):
                    try:
                        idx, processed_frame = future.result()
                        processed_frames[idx] = processed_frame
                        completed_count += 1
                        
                        # Log progress based on configuration
                        if (completed_count % FDA_PARALLEL_CONFIG["progress_log_interval"] == 0 or 
                            completed_count == number_of_frames):
                            logger.info(f"Parallel processing progress: {completed_count}/{number_of_frames} frames completed")
                            
                    except Exception as e:
                        frame_idx = future_to_idx[future]
                        logger.error(f"Failed to process frame {frame_idx}: {str(e)}")
                        # Use original frame data as fallback
                        processed_frames[frame_idx] = pixel_data[frame_idx].astype(np.uint8)
                        completed_count += 1
                        
            frame_processing_time = time.time() - frame_processing_start
            logger.info(f"Enhanced parallel processing complete. Successfully processed {len(processed_frames)} frames in {frame_processing_time:.2f} seconds.")
            
        else:
            logger.info("Single frame FDA detected. Processing single frame.")
            processed_frames[0] = process_frame_parallel(pixel_data[0] if len(pixel_data.shape) == 3 else pixel_data, 0)[1]
            frame_processing_time = time.time() - frame_processing_start
            logger.info(f"Single frame processing complete in {frame_processing_time:.2f} seconds.")

        # --- Parallel PNG conversion and storage for FDA frames ---
        logger.info(f"Converting {number_of_frames} processed frames to PNG format in parallel")
        
        # Performance monitoring for PNG conversion
        png_conversion_start = time.time()
        
        def convert_frame_to_png(frame_data, frame_idx):
            """Convert processed frame data to PNG format."""
            try:
                img = convert_dicom_to_image(frame_data, 0)  # Already 2D
                img_byte_arr = io.BytesIO()
                img.save(img_byte_arr, format='PNG', optimize=True)  # Use PNG with optimization
                img_byte_arr.seek(0)
                return frame_idx, img_byte_arr
            except Exception as e:
                logger.error(f"Failed to convert frame {frame_idx} to PNG: {str(e)}")
                raise
        
        # Use parallel processing for PNG conversion
        png_workers = min(optimal_workers, FDA_PARALLEL_CONFIG["max_png_workers"])
        logger.info(f"Using {png_workers} parallel workers for PNG conversion")
        
        with ThreadPoolExecutor(max_workers=png_workers) as executor:
            # Submit all PNG conversion tasks
            png_future_to_idx = {
                executor.submit(convert_frame_to_png, processed_frames[idx], idx): idx 
                for idx in range(number_of_frames)
            }
            
            # Collect PNG conversion results
            png_completed_count = 0
            for future in as_completed(png_future_to_idx):
                try:
                    frame_idx, png_data = future.result()
                    stored_images[key][frame_idx] = png_data
                    png_completed_count += 1
                    
                    # Log progress based on configuration
                    if (png_completed_count % FDA_PARALLEL_CONFIG["progress_log_interval"] == 0 or 
                        png_completed_count == number_of_frames):
                        logger.info(f"PNG conversion progress: {png_completed_count}/{number_of_frames} frames completed")
                        
                except Exception as e:
                    frame_idx = png_future_to_idx[future]
                    logger.error(f"Failed to convert frame {frame_idx} to PNG: {str(e)}")
                    raise
        
        png_conversion_time = time.time() - png_conversion_start
        logger.info(f"PNG conversion complete. Successfully converted {png_completed_count} frames in {png_conversion_time:.2f} seconds.")
        
        # Log total processing performance
        total_processing_time = frame_processing_time + png_conversion_time
        logger.info(f"Total FDA processing time: {total_processing_time:.2f} seconds (Frame processing: {frame_processing_time:.2f}s, PNG conversion: {png_conversion_time:.2f}s)")

        # Save to hierarchical CRC-based cache
        try:
            cache_data = {
                k: v
                for k, v in stored_images[key].items() if isinstance(k, int)
            }
            file_info = {
                "name":
                os.path.basename(file_path),
                "size":
                os.path.getsize(file_path) if os.path.exists(file_path) else 0,
                "compression_type":
                compression_type,
                "is_compressed":
                is_compressed
            }
            save_to_cache(crc, cache_data, "fda", file_info)
            logger.info(f"Saved processed FDA images to CRC cache: {crc}")
        except Exception as e:
            logger.warning(f"Failed to save to CRC cache: {str(e)}")

        # Clean up temporary file
        try:
            os.remove(file_path)
            logger.info(f"Cleaned up temporary file: {file_path}")
        except Exception as e:
            logger.warning(f"Failed to clean up file {file_path}: {str(e)}")

        logger.info(
            f"Successfully processed e2e file with {number_of_frames} frames")

        return JSONResponse(
            content={
                "message": "File uploaded successfully.",
                "number_of_frames": number_of_frames,
                "dicom_file_path": key,
                "cache_source": "fresh_download",
                "compression_info": {
                    "is_compressed": is_compressed,
                    "compression_type": compression_type
                }
            })
    except HTTPException:
        # Re-raise HTTP exceptions
        raise
    except Exception as e:
        logger.error(f"Unexpected error processing e2e file: {str(e)}",
                     exc_info=True)

        # Clean up file on error
        try:
            if os.path.exists(file_path):
                os.remove(file_path)
        except:
            pass

        raise HTTPException(status_code=500,
                            detail=f"Error processing e2e file: {str(e)}")


@app.get("/api/fda-parallel-config")
async def get_fda_parallel_config():
    """Get current FDA parallel processing configuration."""
    return JSONResponse(content={
        "config": FDA_PARALLEL_CONFIG,
        "system_info": {
            "cpu_count": os.cpu_count(),
            "current_workers": thread_pool._max_workers
        }
    })


@app.post("/api/fda-parallel-config")
async def update_fda_parallel_config(config: dict = Body(..., embed=True)):
    """Update FDA parallel processing configuration."""
    global FDA_PARALLEL_CONFIG
    
    # Validate configuration
    valid_keys = ["max_frame_workers", "max_png_workers", "min_workers", "progress_log_interval"]
    for key in config:
        if key not in valid_keys:
            raise HTTPException(status_code=400, detail=f"Invalid config key: {key}")
    
    # Update configuration
    FDA_PARALLEL_CONFIG.update(config)
    
    logger.info(f"Updated FDA parallel processing config: {FDA_PARALLEL_CONFIG}")
    
    return JSONResponse(content={
        "message": "Configuration updated successfully",
        "config": FDA_PARALLEL_CONFIG
    })


@app.post("/api/inspect_all_metadata")
async def inspect_all_metadata(files: list[UploadFile] = File(...)):
    """
    Endpoint to inspect all available DICOM metadata for debugging, excluding pixel data.
    """
    try:
        # Check for .EX.DCM files
        for file in files:
            if file.filename.lower().endswith('.ex.dcm'):
                raise HTTPException(
                    status_code=422,
                    detail={
                        "error": "EX.DCM files are not supported",
                        "message": "WORK IN PROGRESS - This file type is not yet supported"
                    }
                )
        
        # Read the uploaded DICOM files
        dicom_files = [file.file for file in files]

        # Extract all metadata from the DICOM files
        all_metadata = extract_all_dicom_metadata(dicom_files)

        # Return all metadata as a JSON response
        return JSONResponse(content={"all_metadata": all_metadata})

    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Error processing DICOM files: {str(e)}")


@app.get("/api/check_dicom_ready")
async def check_dicom_ready(dicom_file_path: str):
    """
    Check if the DICOM image for the given file path has been fully processed and is ready to be served.
    """
    try:
        # Check if the dicom_file_path exists in stored_images
        if dicom_file_path in stored_images:
            # Get the number of frames to confirm that all images are stored
            number_of_frames = len([
                k for k in stored_images[dicom_file_path].keys()
                if isinstance(k, int)
            ])

            # If we have frames in memory, the DICOM is ready
            if number_of_frames > 0:
                return {"ready": True, "number_of_frames": number_of_frames}
        # If the DICOM is not fully processed, return not ready
        return {"ready": False}
    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error checking DICOM readiness: {str(e)}")


@app.post("/api/extract_3d_pixel_array/")
async def extract_3d_pixel_array(file: UploadFile = File(...)):
    """
    Extracts the 3D pixel array (X, Y, Z) and the number of bits from a DICOM file.
    """
    try:
        # Check for .EX.DCM files
        if file.filename.lower().endswith('.ex.dcm'):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "EX.DCM files are not supported",
                    "message": "WORK IN PROGRESS - This file type is not yet supported"
                }
            )
        
        # Save the uploaded DICOM file temporarily
        dicom_file_path = file.filename
        with open(dicom_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read the DICOM file
        dicom = pydicom.dcmread(dicom_file_path)

        # Extract pixel data
        pixel_array = dicom.pixel_array  # 3D pixel array (X, Y, Z)
        X, Y, Z = pixel_array.shape if len(
            pixel_array.shape) == 3 else (pixel_array.shape[0],
                                          pixel_array.shape[1], 1)

        # Extract number of bits per pixel
        bits_allocated = dicom.BitsAllocated

        # Clean up the file
        os.remove(dicom_file_path)

        # Return the dimensions and bits
        return JSONResponse(content={
            "X": X,
            "Y": Y,
            "Z": Z,
            "bits_per_pixel": bits_allocated
        })
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting 3D pixel array: {str(e)}")


@app.post("/api/extract_2d_pixel_array/")
async def extract_2d_pixel_array(
        file: UploadFile = File(...), frame_index: int = 0):
    """
    Extracts a 2D pixel array (slice) from the 3D DICOM array.
    You can specify the frame (Z) index to extract.
    """
    try:
        # Check for .EX.DCM files
        if file.filename.lower().endswith('.ex.dcm'):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "EX.DCM files are not supported",
                    "message": "WORK IN PROGRESS - This file type is not yet supported"
                }
            )
        
        # Save the uploaded DICOM file temporarily
        dicom_file_path = file.filename
        with open(dicom_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read the DICOM file
        dicom = pydicom.dcmread(dicom_file_path)

        # Extract 3D pixel data and get 2D slice
        pixel_array = dicom.pixel_array
        if len(pixel_array.shape) == 3:
            if frame_index >= pixel_array.shape[0]:
                raise HTTPException(status_code=400,
                                    detail="Frame index out of range.")
            pixel_slice = pixel_array[frame_index]  # Extract 2D slice
        else:
            pixel_slice = pixel_array  # The file is 2D, so return as is

        # Clean up the file
        os.remove(dicom_file_path)

        # Return the 2D pixel array as a list (for JSON serialization)
        return JSONResponse(content={"2d_pixel_array": pixel_slice.tolist()})
    except Exception as e:
        raise HTTPException(
            status_code=500,
            detail=f"Error extracting 2D pixel array: {str(e)}")


@app.post("/api/extract_lossless_pixel_data_npy/")
async def extract_lossless_pixel_data_npy(file: UploadFile = File(...)):
    """
    Extract the raw, lossless pixel data from a DICOM file and save it to a .npy (NumPy) file.
    """
    try:
        # Check for .EX.DCM files
        if file.filename.lower().endswith('.ex.dcm'):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "EX.DCM files are not supported",
                    "message": "WORK IN PROGRESS - This file type is not yet supported"
                }
            )
        
        # Save the uploaded DICOM file temporarily
        dicom_file_path = file.filename
        with open(dicom_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read the DICOM file
        dicom = pydicom.dcmread(dicom_file_path)

        # Extract raw pixel data (lossless)
        raw_pixel_data = dicom.pixel_array  # This is the raw, unprocessed pixel data

        # Save pixel data to an .npy file
        npy_file_path = f"{dicom_file_path}_pixel_data.npy"
        np.save(npy_file_path, raw_pixel_data)

        # Clean up the temporary DICOM file
        os.remove(dicom_file_path)

        # Return the path to the saved .npy file for download
        return FileResponse(npy_file_path,
                            media_type='application/octet-stream',
                            filename=os.path.basename(npy_file_path))

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error processing DICOM file: {str(e)}")


@app.post("/api/extract_lossless_pixel_data_mat/")
async def extract_lossless_pixel_data_mat(file: UploadFile = File(...)):
    """
    Extract the raw, lossless pixel data from a DICOM file and save it to a .mat (MATLAB) file.
    """
    try:
        # Check for .EX.DCM files
        if file.filename.lower().endswith('.ex.dcm'):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "EX.DCM files are not supported",
                    "message": "WORK IN PROGRESS - This file type is not yet supported"
                }
            )
        
        # Save the uploaded DICOM file temporarily
        dicom_file_path = file.filename
        with open(dicom_file_path, "wb") as buffer:
            shutil.copyfileobj(file.file, buffer)

        # Read the DICOM file
        dicom = pydicom.dcmread(dicom_file_path)

        # Extract raw pixel data (lossless)
        raw_pixel_data = dicom.pixel_array  # This is the raw, unprocessed pixel data

        # Save pixel data to a .mat file
        mat_file_path = f"{dicom_file_path}_pixel_data.mat"
        savemat(mat_file_path, {"pixel_data": raw_pixel_data})

        # Clean up the temporary DICOM file
        os.remove(dicom_file_path)

        # Return the path to the saved .mat file for download
        return FileResponse(mat_file_path,
                            media_type='application/octet-stream',
                            filename=os.path.basename(mat_file_path))

    except Exception as e:
        raise HTTPException(status_code=500,
                            detail=f"Error processing DICOM file: {str(e)}")


@app.post("/api/dicom_to_mat_npy_zip")
async def dicom_to_mat_npy_zip(file: UploadFile = File(...)):
    """
    This endpoint processes one or multiple DICOM file(s) to extract the following:

    - 3D pixel array dimensions (X, Y, Z) and bits per pixel
    - Raw, lossless pixel data saved in both .npy and .mat formats
    - Slice thickness (if available in the DICOM metadata)
    - All DICOM metadata (excluding pixel data)

    The endpoint returns a zipped folder with:

    - A JSON file containing X, Y, Z, bits_per_pixel, and slice_thickness
    - A JSON file containing all DICOM metadata
    - A .npy file with raw pixel data
    - A .mat file with raw pixel data

    Input:
    - One DICOM file

    Output:
    - Zipped folder containing the files
    """
    try:
        # Check for .EX.DCM files
        if file.filename.lower().endswith('.ex.dcm'):
            raise HTTPException(
                status_code=422,
                detail={
                    "error": "EX.DCM files are not supported",
                    "message": "WORK IN PROGRESS - This file type is not yet supported"
                }
            )
        
        try:
            # Extract the file name without the extension
            original_file_name = os.path.splitext(file.filename)[0]

            # Generate a unique ID for the folder
            unique_id = str(uuid.uuid4())
            dicom_file_path = f"{unique_id}_{file.filename}"

            # Save the uploaded DICOM file temporarily
            with open(dicom_file_path, "wb") as buffer:
                shutil.copyfileobj(file.file, buffer)

            # Read the DICOM file
            dicom = pydicom.dcmread(dicom_file_path)
            pixel_array = dicom.pixel_array
            bits_per_pixel = dicom.BitsAllocated

            # Extract slice thickness
            slice_thickness = dicom.get('SliceThickness', None)
            if slice_thickness is None and hasattr(
                    dicom, 'SharedFunctionalGroupsSequence'):
                try:
                    shared_group = dicom.SharedFunctionalGroupsSequence[0]
                    slice_thickness = float(
                        shared_group.PixelMeasuresSequence[0].SliceThickness)
                except Exception:
                    slice_thickness = "N/A"

            slice_thickness_str = f"{slice_thickness}_mm" if slice_thickness != "N/A" else "N/A"

            # Shape of the 3D pixel array
            Z, Y, X = pixel_array.shape if len(
                pixel_array.shape) == 3 else (1, *pixel_array.shape)

            # Prepare metadata as JSON
            metadata = {
                "X": X,
                "Y": Y,
                "Z": Z,
                "bits_per_pixel": bits_per_pixel,
                "slice_thickness": slice_thickness_str
            }

            # Prepare output file names
            json_file_name = f"pixel_dim_{original_file_name}.json"
            npy_file_name = f"raw_pixels_numpy_{original_file_name}.npy"
            mat_file_name = f"raw_pixels_matlab_{original_file_name}.mat"
            full_metadata_json = f"full_nonpixel_metadata_{original_file_name}.json"
            zip_file_name = f"{original_file_name}_dicom_data.zip"

            # Write the pixel dimension JSON metadata
            with open(json_file_name, "w") as json_file:
                json.dump(metadata, json_file)

            # Save the pixel data in .npy
            np.save(npy_file_name, pixel_array)

            # Save the pixel data in .mat
            sio.savemat(mat_file_name, {"pixel_data": pixel_array})

            # Extract full DICOM metadata (excluding pixel data)
            full_metadata = extract_all_dicom_metadata([dicom_file_path])

            # Write the full metadata JSON
            with open(full_metadata_json, "w") as json_file:
                json.dump(full_metadata, json_file)

            # Create a zip file containing both JSON files, .npy, and .mat
            with zipfile.ZipFile(zip_file_name, 'w') as zipf:
                zipf.write(json_file_name, os.path.basename(json_file_name))
                zipf.write(npy_file_name, os.path.basename(npy_file_name))
                zipf.write(mat_file_name, os.path.basename(mat_file_name))
                zipf.write(full_metadata_json,
                           os.path.basename(full_metadata_json))

            # Clean up individual files after zipping them
            os.remove(json_file_name)
            os.remove(npy_file_name)
            os.remove(mat_file_name)
            os.remove(full_metadata_json)
            os.remove(dicom_file_path)

            # Return the zip file for download
            return FileResponse(zip_file_name,
                                media_type='application/zip',
                                filename=os.path.basename(zip_file_name))

        except Exception as e:
            print(f"Error: {str(e)}")
            raise HTTPException(status_code=500,
                                detail=f"Error processing DICOM file: {str(e)}")
    except Exception as e:
        print(f"Error: {str(e)}")
        raise HTTPException(status_code=500,
                            detail=f"Error processing DICOM file: {str(e)}")


def cleanup_temp_files(temp_files: list):
    """Background task to clean up temporary files."""
    for file_path in temp_files:
        if os.path.exists(file_path):
            os.remove(file_path)


def cleanup_on_shutdown():
    """Cleanup function to be called on application shutdown."""
    logger.info("Cleaning up resources on shutdown...")
    
    # Clean up thread pool
    if 'thread_pool' in globals():
        thread_pool.shutdown(wait=True)
        logger.info("Thread pool shutdown complete")
    
    # Clean up memory cache
    if 'stored_images' in globals():
        stored_images.clear()
        logger.info("Memory cache cleared")
    
    # Clean up operation cache
    if 'operation_cache' in globals():
        operation_cache.clear()
        logger.info("Operation cache cleared")
    
    # Clean up temp files
    cleanup_temp_files([])
    logger.info("Cleanup complete")


# Enhanced cache status endpoint with CRC information
@app.get("/api/cache-status")
async def get_cache_status():
    """Get cache statistics including CRC cache information."""
    try:
        cache_stats = {
            "memory_entries": len(stored_images),
            "disk_cache_size": 0,
            "disk_entries": 0,
            "total_size_mb": 0,
            "crc_mappings": len(CRC_CACHE)
        }

        if CACHE_DIR.exists():
            for cache_dir in CACHE_DIR.iterdir():
                if cache_dir.is_dir():
                    cache_stats["disk_entries"] += 1
                    for file in cache_dir.rglob("*"):
                        if file.is_file():
                            cache_stats["disk_cache_size"] += file.stat(
                            ).st_size

        cache_stats["total_size_mb"] = cache_stats["disk_cache_size"] / (1024 *
                                                                         1024)

        return cache_stats

    except Exception as e:
        logger.error(f"Error getting cache status: {str(e)}")
        return {"error": str(e)}


@app.post("/api/clear-cache")
async def clear_cache():
    """Clear all caches (memory and disk)."""
    try:
        # Clear memory cache
        stored_images.clear()
        logger.info("Memory cache cleared")
        
        # Clear disk cache
        cleared_files = 0
        for file_type in ["dicom", "e2e", "fda"]:
            cache_dir = CACHE_DIR / file_type
            if cache_dir.exists():
                for cache_entry in cache_dir.iterdir():
                    if cache_entry.is_dir():
                        try:
                            import shutil
                            shutil.rmtree(cache_entry)
                            cleared_files += 1
                        except Exception as e:
                            logger.warning(f"Failed to clear cache entry {cache_entry}: {str(e)}")
        
        logger.info(f"Disk cache cleared: {cleared_files} entries removed")
        
        return {
            "message": "Cache cleared successfully",
            "memory_cleared": True,
            "disk_entries_cleared": cleared_files
        }
    except Exception as e:
        logger.error(f"Error clearing cache: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error clearing cache: {str(e)}")


@app.get("/api/file_info/{dicom_file_path}")
async def get_file_info(dicom_file_path: str):
    """Get file information"""
    if dicom_file_path not in stored_images:
        raise HTTPException(status_code=404, detail="File not found")

    file_data = stored_images[dicom_file_path]

    if file_data.get("type") == "e2e":
        eye_data = file_data.get("eye_data", {})
        return JSONResponse(
            content={
                "type": "e2e",
                "left_eye_frames": len(eye_data.get("left_eye", {})),
                "right_eye_frames": len(eye_data.get("right_eye", {})),
                "has_eye_data": True,
                "metadata": eye_data.get("metadata", {})
            })
    else:
        num_frames = len([k for k in file_data.keys() if isinstance(k, int)])
        return JSONResponse(content={
            "type": "dicom",
            "number_of_frames": num_frames,
            "has_eye_data": False
        })


@app.get("/api/file_tree_structure")
async def get_file_tree_structure():
    """Get hierarchical tree structure for file organization"""
    try:
        # This will be the tree structure that appears under each eye viewport
        tree_structure = {
            "folders": [{
                "name": "Recent Files",
                "type": "folder",
                "children": [],
                "icon": "folder"
            }, {
                "name": "E2E Files",
                "type": "folder",
                "children": [],
                "icon": "eye"
            }, {
                "name": "DICOM Files",
                "type": "folder",
                "children": [],
                "icon": "medical"
            }],
            "recent_files": []
        }

        # Populate with currently loaded files
        for key, file_data in stored_images.items():
            file_info = {
                "id": key,
                "name": f"File_{key[:8]}",
                "type": file_data.get("type", "dicom"),
                "timestamp": file_data.get("timestamp", time.time()),
                "frames": 0
            }

            if file_data.get("type") == "e2e":
                eye_data = file_data.get("eye_data", {})
                file_info["frames"] = max(len(eye_data.get("left_eye", {})),
                                          len(eye_data.get("right_eye", {})))
                # Add to E2E folder
                tree_structure["folders"][1]["children"].append(file_info)
            else:
                file_info["frames"] = len(
                    [k for k in file_data.keys() if isinstance(k, int)])
                # Add to DICOM folder
                tree_structure["folders"][2]["children"].append(file_info)

            # Add to recent files (limit to 5 most recent)
            tree_structure["recent_files"].append(file_info)

        # Sort recent files by timestamp
        tree_structure["recent_files"].sort(key=lambda x: x["timestamp"],
                                            reverse=True)
        tree_structure["recent_files"] = tree_structure["recent_files"][:5]

        return JSONResponse(content=tree_structure)

    except Exception as e:
        logger.error(f"Error generating tree structure: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))


@app.post("/api/pre-cache-files")
async def pre_cache_files(file_paths: list[str] = Body(..., embed=True, description="List of S3 file paths to pre-cache")):
    """
    Pre-cache multiple files by downloading them from S3, processing them,
    and saving them to the cache for faster access later.
    """
    try:
        logger.info(f"Starting pre-cache operation for {len(file_paths)} files")
        
        # Import S3 functionality
        try:
            from riv_desktop.s3_api import s3, bucket_name
        except ImportError:
            logger.error("Could not import S3 functionality")
            raise HTTPException(status_code=500, detail="S3 functionality not available")
        
        if not s3 or not bucket_name:
            raise HTTPException(status_code=503, detail="S3 not configured. Please set credentials first.")
        
        results = {
            "total_files": len(file_paths),
            "successful": 0,
            "failed": 0,
            "skipped": 0,
            "errors": [],
            "processed_files": []
        }
        
        for i, file_path in enumerate(file_paths):
            try:
                logger.info(f"Processing file {i+1}/{len(file_paths)}: {file_path}")
                
                # Check for .EX.DCM files and skip them
                if file_path.lower().endswith('.ex.dcm'):
                    logger.info(f"EX.DCM file detected: {file_path} - skipping pre-cache")
                    results["skipped"] += 1
                    results["processed_files"].append({
                        "path": file_path,
                        "status": "skipped_ex_dcm",
                        "message": "WORK IN PROGRESS - File type not yet supported"
                    })
                    continue
                
                # Check for .FDS files and skip them
                if file_path.lower().endswith('.fds'):
                    logger.info(f"FDS file detected: {file_path} - skipping pre-cache")
                    results["skipped"] += 1
                    results["processed_files"].append({
                        "path": file_path,
                        "status": "skipped_fds",
                        "message": "WORK IN PROGRESS - File type not yet supported"
                    })
                    continue
                
                # Check if file is already cached
                file_extension = os.path.splitext(file_path)[1].lower()
                # Process all file types - let the processing functions handle unsupported types
                logger.info(f"Processing file: {file_path} (extension: {file_extension})")
                
                # Generate CRC for this file
                try:
                    head_response = s3.head_object(Bucket=bucket_name, Key=file_path)
                    file_size = head_response.get('ContentLength', 0)
                    last_modified = head_response.get('LastModified', '').isoformat() if head_response.get('LastModified') else ''
                    etag = head_response.get('ETag', '').strip('"')
                    
                    metadata_str = f"{file_path}:{etag}:{last_modified}:{file_size}"
                    crc = format(zlib.crc32(metadata_str.encode('utf-8')) & 0xFFFFFFFF, '08x')
                    
                except Exception as e:
                    logger.warning(f"Could not get S3 metadata for {file_path}: {str(e)}")
                    crc = format(zlib.crc32(file_path.encode('utf-8')) & 0xFFFFFFFF, '08x')
                
                # Check if already cached
                cache_path = get_cache_path(crc)
                if cache_path.exists():
                    logger.info(f"File already cached: {file_path}")
                    results["skipped"] += 1
                    results["processed_files"].append({
                        "path": file_path,
                        "status": "already_cached",
                        "crc": crc
                    })
                    continue
                
                # Download file from S3
                logger.info(f"Downloading {file_path} from S3")
                obj = s3.get_object(Bucket=bucket_name, Key=file_path)
                
                # Create temp file
                with tempfile.NamedTemporaryFile(delete=False, suffix=file_extension) as tmp:
                    temp_path = tmp.name
                    
                    # Stream download with progress tracking
                    chunk_size = 1024 * 1024  # 1MB chunks
                    total_size = obj.get('ContentLength', 0)
                    downloaded_size = 0
                    
                    for chunk in obj['Body'].iter_chunks(chunk_size=chunk_size):
                        tmp.write(chunk)
                        downloaded_size += len(chunk)
                        if total_size > 0:
                            progress = (downloaded_size / total_size) * 100
                            if downloaded_size % (5 * 1024 * 1024) == 0:  # Log every 5MB
                                logger.info(f"Pre-cache download progress for {file_path}: {progress:.1f}% ({downloaded_size}/{total_size} bytes)")
                
                logger.info(f"Downloaded to temp file: {temp_path}")
                
                # Process the file based on its type
                key = str(uuid.uuid4())
                result = None
                
                try:
                    # Try to process based on file extension
                    if file_extension == '.dcm':
                        result = process_dicom_file(temp_path, key, crc)
                    elif file_extension == '.e2e':
                        result = process_e2e_file(temp_path, key, crc)
                    elif file_extension == '.fda':
                        result = process_fda_file(temp_path, key, crc)
                    else:
                        # For unsupported file types, just cache the raw file
                        logger.info(f"Unsupported file type {file_extension}, caching raw file: {file_path}")
                        
                        # Store the raw file in memory
                        with open(temp_path, 'rb') as f:
                            file_content = f.read()
                        
                        stored_images[key] = {
                            "raw_content": file_content,
                            "file_type": "raw",
                            "original_path": file_path,
                            "timestamp": time.time(),
                            "crc": crc
                        }
                        
                        # Save to cache
                        cache_data = {"raw_content": file_content}
                        file_info = {
                            "name": os.path.basename(file_path),
                            "size": len(file_content),
                            "file_type": "raw"
                        }
                        save_to_cache(crc, cache_data, "raw", file_info)
                        
                        result = JSONResponse(content={
                            "message": "Raw file cached successfully.",
                            "number_of_frames": 1,
                            "dicom_file_path": key,
                            "cache_source": "raw_file",
                            "file_type": "raw"
                        })
                    
                    # Clean up temp file
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                    
                    if result:
                        results["successful"] += 1
                        results["processed_files"].append({
                            "path": file_path,
                            "status": "success",
                            "crc": crc,
                            "frames": result.body.get("number_of_frames", 0) if hasattr(result, 'body') else 0
                        })
                        logger.info(f"Successfully processed and cached: {file_path}")
                    else:
                        results["failed"] += 1
                        results["errors"].append(f"Processing failed for: {file_path}")
                        
                except Exception as process_error:
                    logger.error(f"Error processing {file_path}: {str(process_error)}")
                    results["failed"] += 1
                    results["errors"].append(f"Processing error for {file_path}: {str(process_error)}")
                    
                    # Clean up temp file on error
                    if os.path.exists(temp_path):
                        os.remove(temp_path)
                
                # Add small delay to prevent overwhelming the system
                await asyncio.sleep(0.1)
                
            except Exception as e:
                logger.error(f"Error processing file {file_path}: {str(e)}")
                results["failed"] += 1
                results["errors"].append(f"Error for {file_path}: {str(e)}")
        
        logger.info(f"Pre-cache operation completed. Success: {results['successful']}, Failed: {results['failed']}, Skipped: {results['skipped']}")
        
        return JSONResponse(content={
            "message": "Pre-cache operation completed",
            "results": results
        })
        
    except Exception as e:
        logger.error(f"Error in pre-cache operation: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error in pre-cache operation: {str(e)}")


def main():
    # Run the Uvicorn server
    uvicorn.run("main:app", host="localhost", port=8000, reload=True)


if __name__ == "__main__":
    main()

@app.get("/api/view_e2e_oct_frame")
async def view_e2e_oct_frame(dicom_file_path: str = Query(...), eye: str = Query(...), frame_idx: int = Query(...)):
    """Serve a specific flattened original OCT frame image for a given eye and frame index."""
    logger.info(f"Request for flattened original OCT frame {frame_idx} from {eye} eye in file {dicom_file_path}")
    try:
        if dicom_file_path not in stored_images:
            raise HTTPException(status_code=404, detail="E2E file not found in memory.")
        data = stored_images[dicom_file_path]
        if data.get("file_type") != "e2e":
            raise HTTPException(status_code=400, detail="File is not an E2E file.")
        eye_key = "left_eye_data" if eye.lower() == "left" else "right_eye_data"
        if eye_key not in data:
            raise HTTPException(status_code=404, detail=f"No data found for {eye} eye.")
        frame_list = data[eye_key].get("original_oct", [])
        if frame_idx < 0 or frame_idx >= len(frame_list):
            raise HTTPException(status_code=404, detail=f"Flattened original OCT frame {frame_idx} not found for {eye} eye.")
        frame_key = frame_list[frame_idx]
        if frame_key not in data:
            raise HTTPException(status_code=404, detail=f"Flattened original OCT frame data '{frame_key}' not found.")
        buf = data[frame_key]
        buf.seek(0)
        return StreamingResponse(buf, media_type="image/jpeg")
    except HTTPException as e:
        logger.error(f"Error retrieving flattened original OCT frame {frame_idx} for {eye} eye: {str(e)}")
        raise e
    except Exception as e:
        logger.error(f"Error retrieving flattened original OCT frame {frame_idx} for {eye} eye: {str(e)}", exc_info=True)
        raise HTTPException(status_code=500, detail=f"Error processing E2E file: {str(e)}")


@app.get("/api/debug_stored_images")
async def debug_stored_images():
    """Debug endpoint to check what's in stored_images."""
    try:
        debug_info = {}
        for key, data in stored_images.items():
            debug_info[key] = {
                "keys": list(data.keys()),
                "frame_count": len([k for k in data.keys() if isinstance(k, int)]),
                "is_oct": data.get("is_oct", False),
                "is_fda": data.get("is_fda", False),
                "is_fds": data.get("is_fds", False),
                "file_type": data.get("file_type", "unknown"),
                "timestamp": data.get("timestamp", 0),
                "crc": data.get("crc", "unknown")
            }
        return JSONResponse(content=debug_info)
    except Exception as e:
        logger.error(f"Error in debug endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")


@app.get("/api/debug_crc_calculation")
async def debug_crc_calculation(path: str = Query(...)):
    """Debug endpoint to test CRC calculation for an S3 path."""
    try:
        from riv_desktop.s3_api import s3, bucket_name
        
        if not s3 or not bucket_name:
            return {"error": "S3 not configured"}
        
        # Test the same calculation used in download_dicom_from_s3
        try:
            head_response = s3.head_object(Bucket=bucket_name, Key=path)
            file_size = head_response.get('ContentLength', 0)
            last_modified = head_response.get('LastModified', '').isoformat() if head_response.get('LastModified') else ''
            etag = head_response.get('ETag', '').strip('"')
            
            metadata_str = f"{path}:{etag}:{last_modified}:{file_size}"
            s3_metadata_crc = format(zlib.crc32(metadata_str.encode('utf-8')) & 0xFFFFFFFF, '08x')
            
            # Also calculate path-only CRC for comparison
            path_only_crc = format(zlib.crc32(path.encode('utf-8')) & 0xFFFFFFFF, '08x')
            
            return {
                "path": path,
                "s3_metadata_crc": s3_metadata_crc,
                "path_only_crc": path_only_crc,
                "metadata_string": metadata_str,
                "file_size": file_size,
                "etag": etag,
                "last_modified": last_modified,
                "match": s3_metadata_crc == path_only_crc
            }
        except Exception as e:
            return {"error": f"S3 head_object failed: {str(e)}"}
            
    except Exception as e:
        logger.error(f"Error in CRC debug endpoint: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error: {str(e)}")

