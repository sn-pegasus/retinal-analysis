# OCT Processing Pipeline

## Overview
This pipeline validates, processes, and analyzes OCT data from E2E/AE2 files, producing clinically relevant metrics and diagnostic outputs. It integrates with existing modules (`ae2_processor_module/ae2_processor.py`, `main.py`) and adheres to an industrial standard of 97 frames per eye.

## Validation
- Input completeness: Enforces per-eye cap of `97` frames; flags volumes with unexpected frame counts.
- Dimension consistency: Confirms all frames in a volume share identical dimensions.
- Basic quality assessment: Computes a per-frame quality score using intensity dynamic range and edge energy.

## Image Processing
- Flattening: Enhanced OCT flattening aligns retinal surfaces for consistent analysis.
- Contrast enhancement: CLAHE improves layer visibility and robustness of curve detection.
- Surface detection: Detects RPE and ILM curves using gradient-based methods with smoothing.

## Dataset Handling
- Per-eye counters guarantee the total processed frames do not exceed `97` across volumes.
- Middle-frame flattening is preserved per volume for preview and consistency.

## Clinically Relevant Metrics
- Thickness estimation: Computes per-frame retinal thickness (median, standard deviation, center region) from RPE–ILM separation.
- Quality summary: Aggregates quality scores (mean, median) per eye.
- Completeness flags: Indicates whether each eye meets the expected `97` frames.

## Error Handling
- Robust try/except blocks around volume reads, frame processing, flattening, and metric computation.
- Skips invalid frames; records errors in diagnostics for traceability.

## Diagnostic Output
- AE2 standalone writes per-eye diagnostics to `cache/e2e/<crc>/<crc>_<L|R>_diagnostics.json`.
- FastAPI `process_e2e_file` returns `left_eye_diagnostics` and `right_eye_diagnostics` in the JSON response and writes the same files to the CRC cache directory.

## Integration
- AE2 standalone: Use `AE2Processor.process_ae2_file(path)` to process and inspect output and diagnostics in `cache/e2e/<crc>/`.
- FastAPI service: Upload E2E via the existing API and consume the response fields and cached diagnostics.

## Validation Procedures
- Frame count conformity: Each eye must have `97` processed frames; discrepancies are flagged.
- Quality acceptance: Monitor quality score distributions; very low averages may indicate acquisition issues.
- Thickness stability: Review thickness variability (`std`) and center-region values to detect artifacts or pathology.

## File/Function References
- AE2 processing and diagnostics: `ae2_processor_module/ae2_processor.py`
- FastAPI E2E pipeline and diagnostics: `main.py`