# Testing the AE2 Processor Module

## Quick Test Instructions

### Option 1: Using the Test Script

```bash
python test_processor.py <path_to_your_e2e_file>
```

Example:
```bash
python test_processor.py C:\data\patient_L.e2e
```

### Option 2: Using the Demo Script

```bash
python demo.py
```

This will show usage instructions and prompt you to enter a file path if available.

### Option 3: Direct Usage

```python
from ae2_processor_module import AE2Processor

processor = AE2Processor(output_base_dir="test_output")
result = processor.process_ae2_file("path/to/your/file.e2e")

print(f"CRC: {result['crc']}")
print(f"Output: {result['output_dir']}")
```

## What to Expect

When you run the test with an E2E file, you should see:

1. **Processing logs** showing:
   - E2E file detection
   - Fundus image processing
   - OCT volume processing
   - OCT frame extraction and flattening

2. **Output directory** created at:
   - `test_output/[CRC]/` - for test output
   - `cache/e2e/[CRC]/` - for default output

3. **Files saved**:
   - Fundus images: `[CRC]_L_fundus_0.jpg`, etc.
   - Flattened OCT: `[CRC]_L_oct_flattened_0.jpg`, etc.
   - Original frames: `[CRC]_L_original_oct_frame_0000.jpg`, etc.
   - Metadata: `metadata.pkl`

## Example Output

```
======================================================================
AE2 Processor Test
======================================================================
Input file: data/patient_L.e2e
File size: 15.32 MB

Processing file...
----------------------------------------------------------------------
2024-10-27 20:45:00 - INFO - Starting AE2 file processing: patient_L.e2e
2024-10-27 20:45:01 - INFO - E2E file detected
2024-10-27 20:45:01 - INFO - Output directory: test_output/abc123...
2024-10-27 20:45:01 - INFO - Processing E2E file: patient_l.e2e
2024-10-27 20:45:02 - INFO - Processing 2 fundus images
...
2024-10-27 20:47:00 - INFO - ============================================================
2024-10-27 20:47:00 - INFO - Processing Summary:
2024-10-27 20:47:00 - INFO - Output directory: test_output/abc123...
2024-10-27 20:47:00 - INFO - CRC: abc123...
2024-10-27 20:47:00 - INFO - Left eye - Fundus: 2, OCT volumes: 1
2024-10-27 20:47:00 - INFO - Right eye - Fundus: 0, OCT volumes: 0
2024-10-27 20:47:00 - INFO - ============================================================

======================================================================
✅ Processing Completed Successfully!
======================================================================
CRC: abc123def456...
Output directory: test_output/abc123def456...

📊 Processing Summary:
  Left Eye:
    - Fundus images: 2
    - Flattened OCT volumes: 1
    - Original OCT frames: 97

  Right Eye:
    - Fundus images: 0
    - Flattened OCT volumes: 0
    - Original OCT frames: 0

  Total output files: 100
======================================================================
```

## Troubleshooting

### Error: Module not found
Make sure you're running from the correct directory:
```bash
cd kodiak-poc-main
python ae2_processor_module/test_processor.py file.e2e
```

### Error: File not found
Make sure the E2E file path is correct and the file exists.

### Error: Import errors
Install dependencies:
```bash
pip install -r ae2_processor_module/requirements.txt
```

## Testing Without a File

If you don't have an E2E file yet, run:
```bash
python ae2_processor_module/demo.py
```

This will show you instructions and usage examples.


