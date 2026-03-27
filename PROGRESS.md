# Cohere CoreML Progress (Handoff Snapshot)

## Latest Update (Pure CoreML CLI Milestone)
- Added canonical Swift CLI target:
  - `swift_runner/Sources/pure_coreml_asr_cli/main.swift`
  - Contract: `swift run pure_coreml_asr_cli --audio <path.wav> [--trace-json <path>]`
  - Output line: `decoded_text=...`
- Added pure pipeline exporter:
  - `scripts/export_coreml_pure_pipeline.py`
  - Exports:
    - `artifacts/cohere_frontend.mlpackage`
    - `artifacts/cohere_encoder.mlpackage`
    - `artifacts/cohere_decoder_fullseq_masked.mlpackage`
    - `artifacts/coreml_manifest.json`
- Added parity runner for pure CLI:
  - `scripts/validate_pure_coreml_cli.py`
- Updated PyTorch trace for deterministic comparison:
  - `scripts/trace_pytorch_decode.py` disables dither and supports manifest prompt ids.
- ANE root-cause update:
  - Symptom: first ANE decode after fresh `.mlmodelc` compilation could diverge on the demo clip at token index `23`.
  - Wrong cold token path: `5380` (`▁Council`)
  - Correct warm/PyTorch token path: `2231, 4512` (`▁European`, `▁Parliament`)
  - Mitigation now implemented in the Swift CLI:
    - if ANE models are freshly compiled in `artifacts/.compiled`, the CLI runs one full dry decode pass and discards it
    - the second pass is used as the real result
  - Verified after mitigation:
    - demo clip cold ANE: exact IDs match PyTorch
    - demo clip warm ANE: exact IDs match PyTorch
    - short clean clip `needle.`: exact IDs match PyTorch on ANE

## Space Incident + Recovery
- CoreML conversion left very large temp compile artifacts under:
  - `/private/var/folders/6d/2vs25c0n37b8_z73k_xrvv900000gn/T`
- Cleaned stale `cohere_*` temp model directories and failed temp `.mlpackage` files.
- Recovered significant space; root/data free space increased back to ~`19 GiB` at cleanup time.
- Added persistent compile cache for Swift CLI:
  - `artifacts/.compiled/*.mlmodelc`
  - Prevents repeated temp recompiles and reduces cold `load_ms` after first run.
- Added temp cleanup utility:
  - `scripts/cleanup_coreml_temp.sh`
  - Supports `DRY_RUN=1`.

## Goal
Build a pure CoreML ASR pipeline (`audio wav -> text`) for `CohereLabs/cohere-transcribe-03-2026` before any FluidVoice integration.

## Environment + Versions
- Workspace: `/Users/barathwajanandan/Documents/random_apps/cohere_coreml`
- Python venv: `.venv` created with `/usr/bin/python3` (3.9.6)
- Key package pins (`requirements.txt`):
  - `torch==2.2.2`
  - `transformers==4.57.6`
  - `huggingface_hub==0.36.2`
  - `coremltools==8.3.0`
  - `librosa==0.10.2.post1`
- Note:
  - Homebrew Python 3.14 was previously broken on this machine (module load/signing issue), so this flow uses system python for venv bootstrap.

## What Was Implemented
- Baseline transcription script:
  - `scripts/run_baseline.py`
- CoreML export scripts:
  - First-step decoder parity model: `scripts/export_coreml.py`
  - Full-sequence decoder model: `scripts/export_coreml_fullseq.py`
  - Cached decoder model (experimental): `scripts/export_coreml_cached.py`
- Python parity script:
  - `scripts/test_coreml_python.py`
- Swift runners:
  - `swift_runner` (first-step sanity)
  - `swift_fullseq_runner` (exact path)
  - `swift_cached_runner` (experimental)
- Trace + diff tooling:
  - `scripts/trace_pytorch_decode.py`
  - `scripts/compare_decode_traces.py`
- Multi-file validators:
  - `scripts/validate_fullseq_parity_set.py`
  - `scripts/validate_cached_parity_set.py`
- Orchestration:
  - `scripts/run_all.sh` now defaults to exact fullseq path.
  - Cached path is opt-in via `RUN_CACHED_EXPERIMENTAL=1`.

## Key Artifacts Produced
- Baseline:
  - `artifacts/baseline_result.json`
- First-step decoder:
  - `artifacts/cohere_first_step.mlpackage`
  - `artifacts/cohere_first_step_int8.mlpackage`
  - `reports/python_parity_report.json`
- Full-sequence decoder:
  - `artifacts/cohere_decoder_fullseq.mlpackage`
  - `reports/export_fullseq_report.json`
- Cached decoder:
  - `artifacts/cohere_decoder_cached.mlpackage`
  - `reports/export_cached_report.json`
- Validation summaries:
  - `reports/fullseq_parity_set_summary.json`
  - `reports/cached_parity_set_summary.json`

## Validation Outcomes (Important)
- Baseline sample text from Cohere demo audio works:
  - `"If not, there will be a big crisis between you and the European Parliament."`
- First-step parity:
  - top token match = true
  - high logits cosine similarity (~0.99978)
- Fullseq parity set (3 wavs):
  - `reports/fullseq_parity_set_summary.json` => `all_exact_match: true`
- Cached parity set (3 wavs):
  - `reports/cached_parity_set_summary.json` => `all_exact_match: false`
  - Conclusion: cached is faster but not accuracy-safe yet.

## Critical Decision Taken
- **Accuracy-first default:** Use fullseq decoder path for production-grade behavior now.
- **Cached path remains experimental** until exact token parity is achieved across eval set.

## Remaining Gap to Pure CoreML End-to-End
- Pure CoreML CLI path now exists and runs end-to-end (`wav -> text`) in Swift + CoreML.
- Remaining hardening item:
  1. Re-run the ANE exactness sweep on a broader local validation set beyond the demo + short clean clip.

## Useful Commands
- Standard exact run:
  - `HF_TOKEN=... ./scripts/run_all.sh`
- Fullseq no-drop validation set:
  - `HF_TOKEN=... python scripts/validate_fullseq_parity_set.py --audio-files <wav1> <wav2> <wav3>`
- Cached experimental validation:
  - `HF_TOKEN=... python scripts/validate_cached_parity_set.py --audio-files <wav1> <wav2> <wav3>`

## Known Non-Blocking Warnings
- `urllib3` LibreSSL warning appears in this env; pipeline still runs.
- CoreML conversion warnings about int64->int32 and output renaming are present; outputs were handled explicitly.
