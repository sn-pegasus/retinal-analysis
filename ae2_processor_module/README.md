# AE2/E2E File Processor Module

A standalone module for processing AE2/E2E files with OCT flattening functionality.

## Overview

This module extracts the E2E file processing and OCT flattening functionality from the main application into a standalone processor. It can process AE2/E2E files and extract all OCT frames (both original and flattened), fundus images, and save them to an organized directory structure.

## Features

- **E2E File Processing**: Extracts OCT and fundus data from AE2/E2E files
- **OCT Flattening**: Applies enhanced OCT flattening algorithms to align retinal surfaces
- **Organized Output**: Saves all extracted frames to a structured directory (similar to `cache/e2e/076ec4d2/`)
- **Dual Eye Support**: Handles both left and right eye data
- **Comprehensive Extraction**: Extracts:
  - Fundus images
  - Flattened OCT volumes
  - All original OCT frames (also flattened for consistency)

## Installation

```bash
# Install dependencies
pip install -r requirements.txt
```

## Usage

### Command Line Usage

```bash
python ae2_processor.py <path_to_ae2_file>
```

Example:
```bash
python ae2_processor.py data/patient1_L.e2e
```

### Programmatic Usage

```python
from ae2_processor import AE2Processor

# Create processor instance
processor = AE2Processor(output_base_dir="cache/e2e")

# Process an AE2 file
result = processor.process_ae2_file("path/to/file.e2e")

# Access results
print(f"Output directory: {result['output_dir']}")
print(f"CRC: {result['crc']}")
print(f"Left eye - Fundus: {len(result['left_eye_data']['dicom'])}")
print(f"Left eye - OCT volumes: {len(result['left_eye_data']['oct'])}")
print(f"Left eye - Original frames: {len(result['left_eye_data']['original_oct'])}")
```

## Output Structure

The processor creates a directory structure similar to:

```
cache/e2e/
└── 076ec4d2/
    ├── metadata.pkl
    ├── 076ec4d2_L_fundus_0.jpg
    ├── 076ec4d2_L_fundus_1.jpg
    ├── 076ec4d2_L_oct_flattened_0.jpg
    ├── 076ec4d2_L_oct_flattened_1.jpg
    ├── 076ec4d2_L_original_oct_frame_0000.jpg
    ├── 076ec4d2_L_original_oct_frame_0001.jpg
    ├── ... (all original OCT frames)
    ├── 076ec4d2_R_fundus_0.jpg
    └── ... (right eye images)
```

## Key Components

### AE2Processor Class

Main class that handles:
- File CRC calculation for unique identification
- E2E file reading and parsing
- OCT flattening using multiple algorithms
- Image saving and organization

### OCT Flattening Algorithms

1. **Basic Algorithm**: Uses gradient-based surface detection
2. **Enhanced Algorithm**: Combines gradient detection, polynomial fitting, and contrast enhancement

### Helper Functions

- `flatten_oct_image()`: Basic OCT flattening
- `flatten_oct_image_enhanced()`: Enhanced flattening with better surface detection
- `detect_retinal_surface_enhanced()`: Detects retinal surface using multiple methods
- `create_flattened_image()`: Aligns retinal surface to create flattened image
- `enhance_contrast()`: Improves image contrast using CLAHE

## Using in Another Project

To use this module in a different project:

1. Copy the entire `ae2_processor_module` folder to your project
2. Install the dependencies: `pip install -r requirements.txt`
3. Import and use:

```python
from ae2_processor_module import AE2Processor

processor = AE2Processor(output_base_dir="my_output_folder")
result = processor.process_ae2_file("path/to/myfile.e2e")
```

## Logging

The module provides comprehensive logging at INFO level:
- File processing progress
- Eye laterality detection
- OCT volume processing
- Frame extraction progress
- Error handling and warnings

## Example Output

```
2024-01-15 10:30:00 - INFO - Starting AE2 file processing: patient1_L.e2e
2024-01-15 10:30:01 - INFO - E2E file detected
2024-01-15 10:30:01 - INFO - Output directory: cache/e2e/6e0e40ed-e4f6-4e0e-a503-04b382414c75
2024-01-15 10:30:01 - INFO - Processing E2E file: patient1_l.e2e
2024-01-15 10:30:01 - INFO - Detected eye laterality from filename: L
2024-01-15 10:30:02 - INFO - Processing 2 fundus images
2024-01-15 10:30:02 - INFO - [L EYE] Saved fundus image 1 to 6e0e40ed-e4f6-4e0e-a503-04b382414c75_L_fundus_0.jpg
2024-01-15 10:30:05 - INFO - Processing 1 OCT volumes
2024-01-15 10:30:05 - INFO - [L EYE] OCT volume 1 contains 97 frames
...
2024-01-15 10:32:00 - INFO - ============================================================
2024-01-15 10:32:00 - INFO - Processing Summary:
2024-01-15 10:32:00 - INFO - Output directory: cache/e2e/6e0e40ed-e4f6-4e0e-a503-04b382414c75
2024-01-15 10:32:00 - INFO - CRC: 6e0e40ed-e4f6-4e0e-a503-04b382414c75
2024-01-15 10:32:00 - INFO - Left eye - Fundus: 2, OCT volumes: 1
2024-01-15 10:32:00 - INFO - Right eye - Fundus: 0, OCT volumes: 0
2024-01-15 10:32:00 - INFO - ============================================================
```

## Notes

- The module preserves all original functionality from the main application
- OCT frames are flattened using the enhanced algorithm for consistency
- The output format mirrors the existing cache structure (e.g., `cache/e2e/076ec4d2/`)
- Each processed file gets its own directory based on CRC hash


