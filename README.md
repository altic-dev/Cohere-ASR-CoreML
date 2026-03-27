# Cohere ASR -> CoreML (macOS local)

This workspace validates `CohereLabs/cohere-transcribe-03-2026`, exports CoreML decoder variants, and runs parity in Python + Swift.

## What is implemented

- `scripts/setup_env.sh`: creates `.venv` with conservative versions for macOS compatibility.
- `scripts/run_baseline.py`: runs baseline transcript on Cohere model and saves output/timing.
- `scripts/export_coreml.py`: exports first-step decoder logits model for precise PyTorch/CoreML parity.
- `scripts/export_coreml_fullseq.py`: exports fixed-length full-sequence decoder (`input_ids + decoder_attention_mask`).
- `scripts/export_coreml_cached.py`: exports a CoreML decoder-step model with explicit KV-cache I/O for Swift autoregressive generation.
- `scripts/test_coreml_python.py`: compares first-step CoreML logits vs PyTorch reference.
- `scripts/trace_pytorch_decode.py`: emits PyTorch token-by-token decode trace.
- `scripts/compare_decode_traces.py`: compares token traces exactly.
- `scripts/validate_fullseq_parity_set.py`: multi-file exactness validator for fullseq path.
- `scripts/validate_cached_parity_set.py`: multi-file validator for cached path (experimental).
- `swift_runner`: first-step parity runner (sanity check).
- `swift_fullseq_runner`: full-sequence Swift decode runner.
- `swift_cached_runner`: KV-cache Swift decode runner (faster).
- `scripts/run_all.sh`: baseline -> exports -> Python parity -> all Swift runners.

## Decode modes

- First-step model: deterministic numeric parity check.
- Full-sequence model: simple Swift loop with mask updates.
- Cached model: optimized Swift token loop using KV cache tensors (experimental).

## Prerequisites

- macOS with Xcode command line tools
- `/usr/bin/python3` available
- Hugging Face token with access to the gated model:
  - `export HF_TOKEN='...'`

## Run

```bash
cd /Users/barathwajanandan/Documents/random_apps/cohere_coreml
chmod +x scripts/setup_env.sh scripts/run_all.sh
./scripts/setup_env.sh
./scripts/run_all.sh
```

## Recommended run

```bash
cd /Users/barathwajanandan/Documents/random_apps/cohere_coreml
source .venv/bin/activate
HF_TOKEN=... python scripts/run_baseline.py
HF_TOKEN=... python scripts/export_coreml_fullseq.py
HF_TOKEN=... python scripts/export_coreml_cached.py
cd swift_runner
swift run swift_fullseq_runner \
  /Users/barathwajanandan/Documents/random_apps/cohere_coreml/artifacts/cohere_decoder_fullseq.mlpackage \
  /Users/barathwajanandan/Documents/random_apps/cohere_coreml/artifacts/swift_fullseq_input.json \
  96
swift run swift_cached_runner \
  /Users/barathwajanandan/Documents/random_apps/cohere_coreml/artifacts/cohere_decoder_cached.mlpackage \
  /Users/barathwajanandan/Documents/random_apps/cohere_coreml/artifacts/swift_cached_input.json \
  96
```

## Single-command E2E (audio in -> text out)

```bash
cd /Users/barathwajanandan/Documents/random_apps/cohere_coreml
source .venv/bin/activate
HF_TOKEN=... python scripts/run_e2e_audio_to_text.py \
  --audio /Users/barathwajanandan/Documents/random_apps/voxtral/results/downloads/clean_speech_wav.wav
```

## Exactness validation (no-drop gate)

```bash
cd /Users/barathwajanandan/Documents/random_apps/cohere_coreml
source .venv/bin/activate
HF_TOKEN=... python scripts/validate_fullseq_parity_set.py \
  --audio-files \
  /Users/barathwajanandan/Documents/random_apps/voxtral/results/downloads/clean_speech_wav.wav \
  /Users/barathwajanandan/Documents/random_apps/voxtral/results/downloads/speech_noisyish_wav.wav \
  /Users/barathwajanandan/Documents/random_apps/voxtral/results/downloads/speech_variant_wav.wav
```

## Key outputs

- Baseline transcript: `artifacts/baseline_result.json`
- First-step export report: `reports/export_report.json`
- Full-seq export report: `reports/export_fullseq_report.json`
- Cached export report: `reports/export_cached_report.json`
- Python parity: `reports/python_parity_report.json`
- First-step packages: `artifacts/cohere_first_step*.mlpackage`
- Full-seq package: `artifacts/cohere_decoder_fullseq.mlpackage`
- Cached package: `artifacts/cohere_decoder_cached.mlpackage`
- Full-seq parity summary: `reports/fullseq_parity_set_summary.json`
- Cached parity summary: `reports/cached_parity_set_summary.json`

## Current status

- Baseline transcript matches expected sample sentence.
- Python first-step parity: top-token match true, cosine similarity ~0.99978.
- Swift full-seq decode produces:
  - `If not, there will be a big crisis between you and the European Parliament.`
- Full-seq strict token parity across 3 local speech files: exact match true.
- Cached decode is fast and can match on some clips but is not yet exact across all tested clips.
