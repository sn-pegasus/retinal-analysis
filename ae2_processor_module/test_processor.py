"""
Test script for AE2 Processor Module

This script tests the AE2 processor by processing an E2E file.
Usage: python test_processor.py <path_to_e2e_file>
"""

import sys
import os
from pathlib import Path

# Add parent directory to path to import the processor
sys.path.insert(0, str(Path(__file__).parent.parent))

from ae2_processor_module.ae2_processor import AE2Processor


def test_processor(file_path: str):
    """Test the AE2 processor with a given file"""
    
    # Check if file exists
    if not os.path.exists(file_path):
        print(f"❌ Error: File not found: {file_path}")
        return False
    
    print("=" * 70)
    print("AE2 Processor Test")
    print("=" * 70)
    print(f"Input file: {file_path}")
    print(f"File size: {os.path.getsize(file_path) / (1024*1024):.2f} MB")
    print()
    
    try:
        # Create processor with test output directory
        processor = AE2Processor(output_base_dir="test_output")
        
        print("Processing file...")
        print("-" * 70)
        
        # Process the file
        result = processor.process_ae2_file(file_path)
        
        # Display results
        print()
        print("=" * 70)
        print("✅ Processing Completed Successfully!")
        print("=" * 70)
        print(f"CRC: {result['crc']}")
        print(f"Output directory: {result['output_dir']}")
        print()
        
        left_eye = result['left_eye_data']
        right_eye = result['right_eye_data']
        
        print("📊 Processing Summary:")
        print(f"  Left Eye:")
        print(f"    - Fundus images: {len(left_eye['dicom'])}")
        print(f"    - Flattened OCT volumes: {len(left_eye['oct'])}")
        print(f"    - Original OCT frames: {len(left_eye['original_oct'])}")
        print()
        print(f"  Right Eye:")
        print(f"    - Fundus images: {len(right_eye['dicom'])}")
        print(f"    - Flattened OCT volumes: {len(right_eye['oct'])}")
        print(f"    - Original OCT frames: {len(right_eye['original_oct'])}")
        print()
        
        # Count total output files
        output_dir = Path(result['output_dir'])
        total_files = len(list(output_dir.glob("*.jpg")))
        print(f"  Total output files: {total_files}")
        print()
        print(f"📁 Files saved to: {output_dir.absolute()}")
        print("=" * 70)
        
        return True
        
    except Exception as e:
        print()
        print("=" * 70)
        print(f"❌ Processing Failed: {str(e)}")
        print("=" * 70)
        import traceback
        traceback.print_exc()
        return False


def main():
    """Main entry point"""
    if len(sys.argv) < 2:
        print("Usage: python test_processor.py <path_to_e2e_file>")
        print()
        print("Example:")
        print("  python test_processor.py data/patient_L.e2e")
        print()
        print("If you need to test with a file, provide the path to an .e2e file.")
        sys.exit(1)
    
    file_path = sys.argv[1]
    
    # Run the test
    success = test_processor(file_path)
    
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()


