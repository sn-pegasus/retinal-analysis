"""
Demo script showing how to use the AE2 Processor

This demonstrates the usage without requiring an actual E2E file.
"""

import sys
from pathlib import Path

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from ae2_processor_module.ae2_processor import AE2Processor


def print_usage():
    """Print usage instructions"""
    print("=" * 70)
    print("AE2 Processor Module - Demo & Usage Instructions")
    print("=" * 70)
    print()
    print("To use this module with an actual E2E file:")
    print()
    print("1. Ensure you have an .e2e file")
    print("2. Install dependencies:")
    print("   pip install -r requirements.txt")
    print()
    print("3. Run the processor:")
    print("   python ae2_processor.py path/to/your/file.e2e")
    print()
    print("   OR")
    print()
    print("   python test_processor.py path/to/your/file.e2e")
    print()
    print("4. Use in your code:")
    print("""
from ae2_processor_module import AE2Processor

# Create processor
processor = AE2Processor(output_base_dir="cache/e2e")

# Process file
result = processor.process_ae2_file("path/to/file.e2e")

# Access results
print(f"CRC: {result['crc']}")
print(f"Output: {result['output_dir']}")
print(f"Left eye - Fundus: {len(result['left_eye_data']['dicom'])}")
print(f"Left eye - OCT volumes: {len(result['left_eye_data']['oct'])}")
print(f"Left eye - Frames: {len(result['left_eye_data']['original_oct'])}")
""")
    print()
    print("=" * 70)
    print("Module Structure:")
    print("=" * 70)
    print()
    print("ae2_processor_module/")
    print("├── ae2_processor.py       # Main processor")
    print("├── __init__.py            # Package initialization")
    print("├── requirements.txt        # Dependencies")
    print("├── README.md               # Full documentation")
    print("├── MODULE_SUMMARY.md       # Quick summary")
    print("├── USAGE_EXAMPLE.py        # Code examples")
    print("├── test_processor.py       # Test script")
    print("└── demo.py                 # This file")
    print()
    print("=" * 70)
    print("Key Features:")
    print("=" * 70)
    print("✓ Extracts fundus images from E2E files")
    print("✓ Extracts OCT volumes and all frames")
    print("✓ Flattens OCT images using enhanced algorithms")
    print("✓ Organizes output by CRC hash")
    print("✓ Supports both left and right eyes")
    print("✓ Standalone module (no dependencies on main.py)")
    print()
    print("=" * 70)


if __name__ == "__main__":
    print_usage()
    
    # Ask if user has a file to test
    print("\nDo you have an E2E file to test? (y/n)")
    response = input().strip().lower()
    
    if response == 'y':
        print("\nEnter the path to your E2E file:")
        file_path = input().strip()
        
        try:
            processor = AE2Processor(output_base_dir="test_output")
            result = processor.process_ae2_file(file_path)
            
            print("\n✅ Success!")
            print(f"Output directory: {result['output_dir']}")
            print(f"CRC: {result['crc']}")
            
        except Exception as e:
            print(f"\n❌ Error: {e}")
    else:
        print("\nTo use this module:")
        print("1. Get an E2E file")
        print("2. Run: python test_processor.py <file_path>")
        print("\nSee README.md for more details.")


