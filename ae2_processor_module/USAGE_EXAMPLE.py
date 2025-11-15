"""
Quick Usage Examples for AE2 Processor Module

This file demonstrates various ways to use the AE2 processor in your project.
"""

from ae2_processor import AE2Processor
import sys

# Example 1: Basic usage
def example_basic():
    """Basic file processing"""
    processor = AE2Processor(output_base_dir="cache/e2e")
    result = processor.process_ae2_file("path/to/your/file.e2e")
    
    print(f"Processed successfully!")
    print(f"CRC: {result['crc']}")
    print(f"Output: {result['output_dir']}")


# Example 2: Custom output directory
def example_custom_output():
    """Use custom output directory"""
    processor = AE2Processor(output_base_dir="my_projects_processed_files")
    result = processor.process_ae2_file("data/patient_L.e2e")
    
    print(f"Files saved to: {result['output_dir']}")


# Example 3: Process multiple files
def example_multiple_files():
    """Process multiple E2E files"""
    files = ["patient1_L.e2e", "patient1_R.e2e", "patient2_L.e2e"]
    processor = AE2Processor(output_base_dir="batch_processing")
    
    for file_path in files:
        try:
            result = processor.process_ae2_file(file_path)
            print(f"✓ Processed {file_path} -> CRC: {result['crc']}")
        except Exception as e:
            print(f"✗ Failed to process {file_path}: {e}")


# Example 4: Access detailed data
def example_detailed_access():
    """Access detailed processing results"""
    processor = AE2Processor()
    result = processor.process_ae2_file("data/patient.e2e")
    
    left_eye = result['left_eye_data']
    right_eye = result['right_eye_data']
    
    print("\nLeft Eye Data:")
    print(f"  Fundus images: {len(left_eye['dicom'])}")
    print(f"  Flattened OCT: {len(left_eye['oct'])}")
    print(f"  Original frames: {len(left_eye['original_oct'])}")
    
    print("\nRight Eye Data:")
    print(f"  Fundus images: {len(right_eye['dicom'])}")
    print(f"  Flattened OCT: {len(right_eye['oct'])}")
    print(f"  Original frames: {len(right_eye['original_oct'])}")


# Example 5: Access processed images
def example_access_images():
    """Access the saved image files"""
    import os
    from pathlib import Path
    
    processor = AE2Processor()
    result = processor.process_ae2_file("data/patient.e2e")
    
    output_dir = Path(result['output_dir'])
    
    # List all generated images
    image_files = list(output_dir.glob("*.jpg"))
    print(f"Generated {len(image_files)} image files:")
    
    for img_file in image_files:
        print(f"  - {img_file.name}")

if __name__ == "__main__":
    print("AE2 Processor Usage Examples")
    print("=" * 50)
    print("\nTo run these examples:")
    print("1. Update the file paths to point to your E2E files")
    print("2. Uncomment the example you want to run")
    print("3. Run: python USAGE_EXAMPLE.py")
    print("\nOr import and use in your own code:")
    print("  from ae2_processor import AE2Processor")
    print("  processor = AE2Processor()")
    print("  result = processor.process_ae2_file('file.e2e')")


