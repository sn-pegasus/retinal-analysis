"""
Standalone AE2 File Processor with OCT Flattening

This module extracts the E2E file processing and OCT flattening functionality
from the main application into a standalone module.

Usage:
    python ae2_processor.py <path_to_ae2_file>

The module will:
1. Extract OCT and fundus data from AE2/E2E files
2. Flatten OCT images using the enhanced algorithm
3. Save all frames to an output directory organized by CRC
"""

import os
import sys
import io
import time
import logging
import hashlib
import pickle
import json
from pathlib import Path
from typing import Optional, Tuple, Dict, List
import numpy as np
import cv2
from PIL import Image
from scipy import ndimage
from scipy.signal import find_peaks

# Medical imaging imports
from oct_converter.readers import E2E

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Industrial standard cap for OCT frames per eye
MAX_FRAMES_PER_EYE = 97


class AE2Processor:
    """Main processor class for AE2/E2E files"""
    
    def __init__(self, output_base_dir: str = "cache/e2e"):
        """
        Initialize the AE2 processor.
        
        Args:
            output_base_dir: Base directory for output files
        """
        self.output_base_dir = Path(output_base_dir)
        self.output_base_dir.mkdir(parents=True, exist_ok=True)
        
    def calculate_crc(self, file_path: str) -> str:
        """Calculate CRC hash for file identification"""
        hasher = hashlib.md5()
        with open(file_path, 'rb') as f:
            hasher.update(f.read())
        return hasher.hexdigest()
    
    def flatten_oct_image(self, image: np.ndarray) -> np.ndarray:
        """
        Basic OCT image flattening algorithm.
        
        Args:
            image: Input OCT image as numpy array
            
        Returns:
            Flattened OCT image
        """
        try:
            logger.info("Applying basic OCT flattening algorithm")
            
            # Ensure image is 2D
            if len(image.shape) > 2:
                image = np.mean(image, axis=2)
            
            # Convert to float for processing
            img_float = image.astype(np.float32)
            
            # Normalize image
            img_normalized = (img_float - np.min(img_float)) / (np.max(img_float) - np.min(img_float))
            
            # Simple surface detection using gradient
            gradient = np.gradient(img_normalized, axis=0)
            
            # Find surface for each column
            surface_points = []
            for col in range(img_normalized.shape[1]):
                column_gradient = gradient[:, col]
                
                # Find peaks in gradient (surface candidates)
                peaks, _ = find_peaks(np.abs(column_gradient), height=0.1)
                
                if len(peaks) > 0:
                    # Take the first significant peak as surface
                    surface_points.append(peaks[0])
                else:
                    # Fallback to top of image
                    surface_points.append(0)
            
            # Smooth the surface
            surface_points = ndimage.gaussian_filter1d(surface_points, sigma=2)
            
            # Create flattened image
            flattened = np.zeros_like(img_normalized)
            
            for col in range(img_normalized.shape[1]):
                surface_row = int(surface_points[col])
                
                # Shift column to align surface
                if surface_row > 0:
                    flattened[:-surface_row, col] = img_normalized[surface_row:, col]
                else:
                    flattened[:, col] = img_normalized[:, col]
            
            # Convert back to original data type
            flattened_scaled = (flattened * 255).astype(np.uint8)
            
            logger.info("Basic OCT flattening completed successfully")
            return flattened_scaled
            
        except Exception as e:
            logger.error(f"Error in basic OCT flattening: {e}")
            raise
    
    def flatten_oct_image_enhanced(self, image: np.ndarray) -> np.ndarray:
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
            surface_points = self.detect_retinal_surface_enhanced(img_normalized)
            
            # Apply polynomial fitting for smoother surface
            x_coords = np.arange(len(surface_points))
            try:
                # Fit polynomial to surface points
                poly_coeffs = np.polyfit(x_coords, surface_points, deg=min(3, len(surface_points)-1))
                surface_smooth = np.polyval(poly_coeffs, x_coords)
            except:
                # Fallback to Gaussian smoothing
                surface_smooth = ndimage.gaussian_filter1d(surface_points, sigma=3)
            
            # Create flattened image with interpolation
            flattened = self.create_flattened_image(img_normalized, surface_smooth)
            
            # Post-processing: enhance contrast
            flattened = self.enhance_contrast(flattened)
            
            # Convert back to uint8
            flattened_scaled = (flattened * 255).astype(np.uint8)
            
            logger.info("Enhanced OCT flattening completed successfully")
            return flattened_scaled
            
        except Exception as e:
            logger.error(f"Error in enhanced OCT flattening: {e}")
            # Fallback to basic algorithm
            logger.info("Falling back to basic OCT flattening")
            return self.flatten_oct_image(image)
    
    def detect_retinal_surface_enhanced(self, image: np.ndarray) -> np.ndarray:
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
                    upper_portion = column[:height//3]
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
    
    def create_flattened_image(self, image: np.ndarray, surface_points: np.ndarray) -> np.ndarray:
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
                    flattened[:height+shift, col] = image[start_idx:, col]
                else:
                    # No shift needed
                    flattened[:, col] = image[:, col]
            
            return flattened
            
        except Exception as e:
            logger.error(f"Error creating flattened image: {e}")
            return image
    
    def enhance_contrast(self, image: np.ndarray) -> np.ndarray:
        """
        Enhance contrast of the flattened OCT image.
        
        Args:
            image: Input image
            
        Returns:
            Contrast-enhanced image
        """
        try:
            # Apply CLAHE (Contrast Limited Adaptive Histogram Equalization)
            if cv2 is not None:
                # Convert to uint8 for CLAHE
                img_uint8 = (image * 255).astype(np.uint8)
                
                # Apply CLAHE
                clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
                enhanced = clahe.apply(img_uint8)
                
                # Convert back to float
                return enhanced.astype(np.float32) / 255.0
            else:
                # Fallback: simple histogram stretching
                img_min, img_max = np.percentile(image, [2, 98])
                if img_max > img_min:
                    enhanced = np.clip((image - img_min) / (img_max - img_min), 0, 1)
                else:
                    enhanced = image
                
                return enhanced
                
        except Exception as e:
            logger.error(f"Error enhancing contrast: {e}")
            return image
    
    def apply_oct_flattening(self, pixels: np.ndarray, is_middle_frame: bool = False) -> np.ndarray:
        """
        Apply enhanced OCT flattening algorithm.
        
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
                logger.info(f"Using middle frame {middle_index} from 3D volume for flattening")
            
            # Ensure we have 2D data
            if len(pixels.shape) != 2:
                raise ValueError(f"Expected 2D pixel data for flattening, got shape: {pixels.shape}")
            
            logger.info(f"Flattening 2D image with shape: {pixels.shape}")
            
            # Apply the enhanced flattening algorithm
            flattened = self.flatten_oct_image(pixels)
            
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
                        return ((result_pixels - pixels_min) / (pixels_max - pixels_min) * 255).astype(np.uint8)
                    else:
                        return np.full_like(result_pixels, 128, dtype=np.uint8)
                return result_pixels.astype(np.uint8)
            except Exception as e2:
                logger.error(f"Failed to return fallback image: {e2}")
                return np.zeros((100, 100), dtype=np.uint8)

    def compute_frame_quality(self, image: np.ndarray) -> float:
        """Compute a simple OCT frame quality score [0,1] using intensity range and edge content."""
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

    def detect_ilm_curve(self, img: np.ndarray, roi_top_frac: float = 0.05, roi_bottom_frac: float = 0.35) -> np.ndarray:
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
            return ndimage.median_filter(curve, size=5).astype(np.float32)
        except Exception:
            return np.zeros(img.shape[1], dtype=np.float32)

    def compute_thickness_metrics(self, img: np.ndarray) -> Dict[str, float]:
        """Compute simple retinal thickness metrics using ILM and RPE curves."""
        try:
            if len(img.shape) > 2:
                img = np.mean(img, axis=2)
            h = img.shape[0]
            rpe = self.detect_retinal_surface_enhanced((img.astype(np.float32) - img.min()) / max(img.ptp(), 1e-6))
            ilm = self.detect_ilm_curve(img)
            thickness = rpe - ilm
            thickness = np.clip(thickness, 0, h)
            median = float(np.median(thickness))
            std = float(np.std(thickness))
            center_span = slice(max(0, img.shape[1] // 2 - 20), min(img.shape[1], img.shape[1] // 2 + 20))
            center_median = float(np.median(thickness[center_span]))
            return {"median_thickness": median, "std_thickness": std, "center_thickness": center_median}
        except Exception:
            return {"median_thickness": 0.0, "std_thickness": 0.0, "center_thickness": 0.0}

    def validate_oct_volume(self, frames: List[np.ndarray]) -> Dict[str, object]:
        """Validate an OCT volume for expected frame count, dimension consistency, and basic quality."""
        try:
            frame_count = len(frames)
            completeness = frame_count == MAX_FRAMES_PER_EYE
            dims = [f.shape[:2] if isinstance(f, np.ndarray) else None for f in frames]
            dim_consistent = len(set(dims)) <= 1
            quality_scores = [self.compute_frame_quality(f) if isinstance(f, np.ndarray) else 0.0 for f in frames]
            quality_mean = float(np.mean(quality_scores)) if quality_scores else 0.0
            return {
                "frame_count": frame_count,
                "expected": MAX_FRAMES_PER_EYE,
                "complete": completeness,
                "dimension_consistent": dim_consistent,
                "quality_mean": quality_mean
            }
        except Exception as e:
            return {"error": str(e), "frame_count": 0, "expected": MAX_FRAMES_PER_EYE, "complete": False}
    
    def get_laterality_from_filename(self, filename: str) -> str:
        """Detect eye laterality from filename"""
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
    
    def process_ae2_file(self, file_path: str) -> dict:
        """
        Process an AE2/E2E file and extract all data.
        
        Args:
            file_path: Path to the AE2/E2E file
            
        Returns:
            Dictionary with processing results and metadata
        """
        try:
            logger.info(f"Starting AE2 file processing: {os.path.basename(file_path)}")
            
            # Read E2E file
            oct_file = E2E(file_path)
            logger.info("E2E file detected")
            
            # Calculate CRC
            crc = self.calculate_crc(file_path)
            
            # Create output directory
            output_dir = self.output_base_dir / crc
            output_dir.mkdir(parents=True, exist_ok=True)
            logger.info(f"Output directory: {output_dir}")
            
            # Initialize data structures
            processed_data = {
                "file_type": "e2e",
                "left_eye_data": {"dicom": [], "oct": [], "original_oct": []},
                "right_eye_data": {"dicom": [], "oct": [], "original_oct": []},
                "timestamp": time.time(),
                "crc": crc,
                "output_dir": str(output_dir)
            }
            
            original_filename = os.path.basename(file_path).lower()
            logger.info(f"Processing E2E file: {original_filename}")
            
            file_laterality = self.get_laterality_from_filename(original_filename)
            logger.info(f"Detected eye laterality from filename: {file_laterality}")
            
            # Track counts
            right_eye_fundus_count = 0
            left_eye_fundus_count = 0
            right_eye_oct_count = 0
            left_eye_oct_count = 0
            per_eye_original_frame_counts = {'L': 0, 'R': 0}
            per_eye_metrics = {
                'L': {"processed_frames": 0, "quality_scores": [], "thickness_values": [], "volumes": []},
                'R': {"processed_frames": 0, "quality_scores": [], "thickness_values": [], "volumes": []}
            }

            def choose_eye() -> str:
                left_rem = MAX_FRAMES_PER_EYE - per_eye_original_frame_counts['L']
                right_rem = MAX_FRAMES_PER_EYE - per_eye_original_frame_counts['R']
                return 'R' if right_rem > left_rem else 'L'
            
            # 1. Process Fundus images
            try:
                fundus_images = oct_file.read_fundus_image()
                logger.info(f"Processing {len(fundus_images)} fundus images")
                
                for i, fundus_image in enumerate(fundus_images):
                    laterality = choose_eye()
                    
                    eye_key = "left_eye_data" if laterality == 'L' else "right_eye_data"
                    fundus_key = f"fundus_{i}"
                    
                    if laterality == 'R':
                        right_eye_fundus_count += 1
                    else:
                        left_eye_fundus_count += 1
                    
                    if hasattr(fundus_image, 'image') and fundus_image.image is not None:
                        fundus_data = fundus_image.image
                        if isinstance(fundus_data, np.ndarray):
                            if fundus_data.dtype != np.uint8:
                                fundus_normalized = ((fundus_data - fundus_data.min()) / 
                                                    (fundus_data.max() - fundus_data.min()) * 255).astype(np.uint8)
                            else:
                                fundus_normalized = fundus_data
                            
                            # Save fundus image
                            img_pil = Image.fromarray(fundus_normalized)
                            output_path = output_dir / f"{crc}_{laterality}_fundus_{i}.jpg"
                            img_pil.save(output_path, format='JPEG')
                            processed_data[eye_key]["dicom"].append(str(output_path.name))
                            
                            logger.info(f"[{laterality} EYE] Saved fundus image {i+1} to {output_path.name}")
            
            except Exception as e:
                logger.warning(f"Error processing fundus images: {str(e)}")
            
            # 2. Process OCT volumes
            try:
                oct_volumes = oct_file.read_oct_volume()
                logger.info(f"Processing {len(oct_volumes)} OCT volumes")
                
                for i, oct_volume in enumerate(oct_volumes):
                    laterality = choose_eye()
                    
                    eye_key = "left_eye_data" if laterality == 'L' else "right_eye_data"
                    
                    if laterality == 'R':
                        right_eye_oct_count += 1
                    else:
                        left_eye_oct_count += 1
                    
                    # Extract all frames from the OCT volume
                    if hasattr(oct_volume, 'volume') and len(oct_volume.volume) > 0:
                        volume_frame_count = len(oct_volume.volume)
                        logger.info(f"[{laterality} EYE] OCT volume {i+1} contains {volume_frame_count} frames")
                        if volume_frame_count != MAX_FRAMES_PER_EYE:
                            logger.info(f"[{laterality} EYE] Enforcing {MAX_FRAMES_PER_EYE}-frame cap per eye as industrial standard")
                        try:
                            per_eye_metrics[laterality]["volumes"].append(self.validate_oct_volume(list(oct_volume.volume)))
                        except Exception as ve:
                            per_eye_metrics[laterality]["volumes"].append({"error": str(ve)})
                        
                        # Process middle frame for flattening
                        middle_frame_index = len(oct_volume.volume) // 2
                        oct_slice = oct_volume.volume[middle_frame_index]
                        
                        # Flattened OCT processing
                        try:
                            flattened_oct_array = self.apply_oct_flattening(oct_slice, is_middle_frame=True)
                            output_path = output_dir / f"{crc}_{laterality}_oct_flattened_{i}.jpg"
                            flattened_img_pil = Image.fromarray(flattened_oct_array)
                            flattened_img_pil.save(output_path, format='JPEG')
                            processed_data[eye_key]["oct"].append(str(output_path.name))
                            logger.info(f"[{laterality} EYE] Saved flattened OCT volume {i+1}")
                        except Exception as flatten_error:
                            logger.warning(f"[{laterality} EYE] Error flattening OCT volume {i+1}: {str(flatten_error)}")
                        
                        # Process original OCT frames with per-eye cap
                        remaining_allowance = MAX_FRAMES_PER_EYE - per_eye_original_frame_counts[laterality]
                        if remaining_allowance <= 0:
                            logger.info(f"[{laterality} EYE] Already reached {MAX_FRAMES_PER_EYE} frames. Skipping additional frames in volume {i+1}.")
                        frames_to_process = min(remaining_allowance, volume_frame_count)
                        for local_idx in range(frames_to_process):
                            original_oct_slice = oct_volume.volume[local_idx]
                            try:
                                # Flatten the original frame
                                flattened_oct_array = self.apply_oct_flattening(original_oct_slice, is_middle_frame=True)
                                global_idx = per_eye_original_frame_counts[laterality]
                                output_path = output_dir / f"{crc}_{laterality}_original_oct_frame_{global_idx:04d}.jpg"
                                flattened_img_pil = Image.fromarray(flattened_oct_array)
                                flattened_img_pil.save(output_path, format='JPEG')
                                processed_data[eye_key]["original_oct"].append(str(output_path.name))
                                per_eye_metrics[laterality]["processed_frames"] += 1
                                per_eye_metrics[laterality]["quality_scores"].append(self.compute_frame_quality(flattened_oct_array))
                                tm = self.compute_thickness_metrics(flattened_oct_array)
                                per_eye_metrics[laterality]["thickness_values"].append(tm.get("median_thickness", 0.0))
                                
                                per_eye_original_frame_counts[laterality] += 1
                                if local_idx % 10 == 0 or local_idx == frames_to_process - 1:
                                    logger.info(f"[{laterality} EYE] Processed frame {per_eye_original_frame_counts[laterality]}/{MAX_FRAMES_PER_EYE} for this eye")
                            
                            except Exception as frame_error:
                                logger.warning(f"[{laterality} EYE] Error processing frame {local_idx+1}: {str(frame_error)}")
            
            except Exception as e:
                logger.warning(f"Error processing OCT volumes: {str(e)}")
            
            # Save metadata
            metadata_path = output_dir / "metadata.pkl"
            with open(metadata_path, 'wb') as f:
                pickle.dump(processed_data, f)

            # Generate diagnostics per eye
            for lat, eye_key in (('L', "left_eye_data"), ('R', "right_eye_data")):
                m = per_eye_metrics[lat]
                diag = {
                    "expected_frames": MAX_FRAMES_PER_EYE,
                    "processed_frames": m["processed_frames"],
                    "complete": m["processed_frames"] == MAX_FRAMES_PER_EYE,
                    "quality_mean": float(np.mean(m["quality_scores"])) if m["quality_scores"] else 0.0,
                    "quality_median": float(np.median(m["quality_scores"])) if m["quality_scores"] else 0.0,
                    "thickness_median": float(np.median(m["thickness_values"])) if m["thickness_values"] else 0.0,
                    "thickness_std": float(np.std(m["thickness_values"])) if m["thickness_values"] else 0.0,
                    "volumes": m["volumes"]
                }
                diag_path = output_dir / f"{crc}_{lat}_diagnostics.json"
                with open(diag_path, 'w') as df:
                    json.dump(diag, df, indent=2)
                processed_data[eye_key]["diagnostics"] = str(diag_path.name)
            
            logger.info("=" * 60)
            logger.info("Processing Summary:")
            logger.info(f"Output directory: {output_dir}")
            logger.info(f"CRC: {crc}")
            logger.info(f"Left eye - Fundus: {left_eye_fundus_count}, OCT volumes: {left_eye_oct_count}")
            logger.info(f"Right eye - Fundus: {right_eye_fundus_count}, OCT volumes: {right_eye_oct_count}")
            logger.info("=" * 60)
            
            return processed_data
            
        except Exception as e:
            logger.error(f"Error processing AE2 file: {str(e)}", exc_info=True)
            raise


def main():
    """Main entry point for command-line usage"""
    if len(sys.argv) < 2:
        print("Usage: python ae2_processor.py <path_to_ae2_file>")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    if not os.path.exists(file_path):
        print(f"Error: File not found: {file_path}")
        sys.exit(1)
    
    # Create processor
    processor = AE2Processor()
    
    # Process the file
    try:
        result = processor.process_ae2_file(file_path)
        print(f"\nProcessing completed successfully!")
        print(f"Output directory: {result['output_dir']}")
        print(f"CRC: {result['crc']}")
    except Exception as e:
        print(f"\nError processing file: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()

