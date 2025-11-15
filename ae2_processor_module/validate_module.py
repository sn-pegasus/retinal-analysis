"""
Module Validation Script

This script validates that the AE2 processor module is properly structured
and all dependencies are available.
"""

import sys
from pathlib import Path


def validate_imports():
    """Check if all required imports work"""
    print("Validating imports...")
    print("-" * 60)
    
    issues = []
    
    # Check basic Python imports
    try:
        import os
        import sys
        import io
        import time
        import logging
        import hashlib
        import pickle
        from pathlib import Path
        from typing import Optional, Tuple, Dict, List
        print("[OK] Standard library imports")
    except Exception as e:
        issues.append(f"Standard library: {e}")
        print(f"[FAIL] Standard library imports failed: {e}")
    
    # Check numpy
    try:
        import numpy as np
        print(f"[OK] numpy {np.__version__}")
    except ImportError as e:
        issues.append(f"numpy: {e}")
        print(f"[FAIL] numpy not installed: {e}")
    
    # Check cv2 (OpenCV)
    try:
        import cv2
        print(f"[OK] opencv-python {cv2.__version__}")
    except ImportError as e:
        issues.append(f"opencv: {e}")
        print(f"[FAIL] opencv-python not installed: {e}")
    
    # Check PIL
    try:
        from PIL import Image
        print(f"[OK] pillow (PIL)")
    except ImportError as e:
        issues.append(f"pillow: {e}")
        print(f"[FAIL] pillow not installed: {e}")
    
    # Check scipy
    try:
        from scipy import ndimage
        from scipy.signal import find_peaks
        print("[OK] scipy")
    except ImportError as e:
        issues.append(f"scipy: {e}")
        print(f"[FAIL] scipy not installed: {e}")
    
    # Check oct_converter
    try:
        from oct_converter.readers import E2E
        print("[OK] oct-converter")
    except ImportError as e:
        issues.append(f"oct_converter: {e}")
        print(f"[FAIL] oct-converter not installed: {e}")
    
    return issues


def validate_module_structure():
    """Check if module files exist"""
    print("\nValidating module structure...")
    print("-" * 60)
    
    required_files = [
        "ae2_processor.py",
        "__init__.py",
        "requirements.txt",
        "README.md"
    ]
    
    module_dir = Path(__file__).parent
    missing_files = []
    
    for file in required_files:
        file_path = module_dir / file
        if file_path.exists():
            print(f"[OK] {file}")
        else:
            print(f"[FAIL] {file} - MISSING")
            missing_files.append(file)
    
    return missing_files


def validate_class_structure():
    """Check if AE2Processor class is properly defined"""
    print("\nValidating class structure...")
    print("-" * 60)
    
    try:
        sys.path.insert(0, str(Path(__file__).parent.parent))
        from ae2_processor_module.ae2_processor import AE2Processor
        
        # Check if class has required methods
        required_methods = [
            'calculate_crc',
            'flatten_oct_image',
            'flatten_oct_image_enhanced',
            'detect_retinal_surface_enhanced',
            'create_flattened_image',
            'enhance_contrast',
            'apply_oct_flattening',
            'get_laterality_from_filename',
            'process_ae2_file'
        ]
        
        missing_methods = []
        for method in required_methods:
            if hasattr(AE2Processor, method):
                print(f"[OK] {method}()")
            else:
                print(f"[FAIL] {method}() - MISSING")
                missing_methods.append(method)
        
        return missing_methods
        
    except Exception as e:
        print(f"[FAIL] Failed to import AE2Processor: {e}")
        return [str(e)]


def main():
    """Run all validations"""
    print("=" * 60)
    print("AE2 Processor Module Validation")
    print("=" * 60)
    print()
    
    # Validate imports
    import_issues = validate_imports()
    print()
    
    # Validate structure
    structure_issues = validate_module_structure()
    print()
    
    # Validate class
    class_issues = validate_class_structure()
    print()
    
    # Summary
    print("=" * 60)
    print("Validation Summary")
    print("=" * 60)
    
    total_issues = len(import_issues) + len(structure_issues) + len(class_issues)
    
    if total_issues == 0:
        print("[SUCCESS] All validations passed! Module is ready to use.")
        print()
        print("To test with an E2E file:")
        print("  python test_processor.py path/to/file.e2e")
        print()
        print("Or run the demo:")
        print("  python demo.py")
    else:
        print(f"[WARNING] Found {total_issues} issue(s):")
        print()
        if import_issues:
            print("Missing imports:")
            for issue in import_issues:
                print(f"  - {issue}")
            print()
            print("Install missing dependencies:")
            print("  pip install -r requirements.txt")
            print()
        if structure_issues:
            print("Missing files:")
            for issue in structure_issues:
                print(f"  - {issue}")
            print()
        if class_issues:
            print("Missing methods:")
            for issue in class_issues:
                print(f"  - {issue}")
            print()
    
    print("=" * 60)
    
    return total_issues == 0


if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)

