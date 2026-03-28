# Cohere ASR -> CoreML (macOS local)

This workspace validates `CohereLabs/cohere-transcribe-03-2026`, exports CoreML artifacts, and runs Swift CoreML ASR locally.

## What is implemented

- `scripts/setup_env.sh`: creates `.venv` with conservative versions for macOS compatibility.
- `scripts/run_baseline.py`: runs baseline transcript on Cohere model and saves output/timing.
- `scripts/export_coreml.py`: exports first-step decoder logits model for precise PyTorch/CoreML parity.
- `scripts/export_coreml_fullseq.py`: exports fixed-length full-sequence decoder (`input_ids + decoder_attention_mask`).
- `scripts/export_coreml_cached.py`: exports a CoreML decoder-step model with explicit KV-cache I/O for Swift autoregressive generation.
- `scripts/export_coreml_pure_pipeline.py`: exports pure pipeline stages (`frontend`, `encoder`, `decoder`) + `artifacts/coreml_manifest.json`.
- `scripts/test_coreml_python.py`: compares first-step CoreML logits vs PyTorch reference.
- `scripts/trace_pytorch_decode.py`: emits PyTorch token-by-token decode trace.
- `scripts/compare_decode_traces.py`: compares token traces exactly.
- `scripts/validate_fullseq_parity_set.py`: multi-file exactness validator for fullseq path.
- `scripts/validate_cached_parity_set.py`: multi-file validator for cached path (experimental).
- `scripts/validate_pure_coreml_cli.py`: parity validator for pure Swift CoreML CLI path.
- `swift_runner`: first-step parity runner (sanity check).
- `swift_fullseq_runner`: full-sequence Swift decode runner.
- `swift_cached_runner`: KV-cache Swift decode runner (faster).
- `pure_coreml_asr_cli`: canonical Swift CLI for `--audio <wav> -> decoded_text=...`.
- `scripts/run_all.sh`: baseline -> exports -> Python parity -> all Swift runners.

## Decode modes

- First-step model: deterministic numeric parity check.
- Full-sequence model: simple Swift loop with mask updates.
- Cached model: optimized Swift token loop using KV cache tensors (experimental).
- Pure pipeline model: frontend + encoder + fullseq decoder chained in Swift.

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

## Canonical Pure CoreML CLI (Swift)

```bash
cd /Users/barathwajanandan/Documents/random_apps/cohere_coreml
source .venv/bin/activate
HF_TOKEN=... python scripts/export_coreml_pure_pipeline.py --max-new-tokens 96
cd swift_runner
swift run pure_coreml_asr_cli \
  --audio /Users/barathwajanandan/Documents/random_apps/voxtral/results/downloads/clean_speech_wav.wav \
  --compute ane \
  --trace-json /Users/barathwajanandan/Documents/random_apps/cohere_coreml/reports/swift_pure_trace.json
```

Notes:
- CLI caches compiled CoreML models in:
  - `/Users/barathwajanandan/Documents/random_apps/cohere_coreml/artifacts/.compiled`
- This avoids repeated huge temp compiles and keeps `load_ms` low on repeat runs.
- Compute mode:
  - `--compute ane` (default, faster)
  - `--compute gpu`
  - `--compute all`
  - `--compute cpu` (use for strict parity checks)
- ANE cold-start note:
  - The first decode after fresh ANE `.mlmodelc` compilation was observed to drift on the demo clip (`Council` vs `European Parliament`).
  - The CLI now performs one full dry warmup decode when it compiles fresh ANE artifacts, then uses the second pass as the real result.

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

For pure Swift CLI parity:

```bash
cd /Users/barathwajanandan/Documents/random_apps/cohere_coreml
source .venv/bin/activate
HF_TOKEN=... python scripts/validate_pure_coreml_cli.py \
  --audio-files \
  /Users/barathwajanandan/Documents/random_apps/voxtral/results/downloads/clean_speech_wav.wav \
  /Users/barathwajanandan/Documents/random_apps/voxtral/results/downloads/speech_noisyish_wav.wav \
  /Users/barathwajanandan/Documents/random_apps/voxtral/results/downloads/speech_variant_wav.wav
```

By default this validator reuses existing exported CoreML artifacts and does not re-export each run.
Use `--reexport` only when model/export code changes.

```bash
HF_TOKEN=... python scripts/validate_pure_coreml_cli.py --reexport ...
```

## LibriSpeech WER (HF dataset)

The [openslr/librispeech_asr](https://huggingface.co/datasets/openslr/librispeech_asr) corpus is 16 kHz English read speech (CC BY 4.0), aligned with this project’s frontend sample rate.

`scripts/eval_librispeech_wer.py` downloads a slice of the split (default: first 50 `test` utterances from config `clean`), runs `pure_coreml_asr_cli` per clip, and reports **corpus WER** with simple normalization (lower case, strip punctuation). Compare **Core ML weight settings** by pointing at different export trees:

- `artifacts_fp16` — FP16 weights, no palettization (`--quantize none`)
- `artifacts_palettize6` / `artifacts_palettize4` — 6-bit / 4-bit palettized encoder+decoder (see `export_coreml_pure_pipeline.py --quantize`)

Use **`--compute gpu`** (or `ane` / `cpu`) consistently when comparing. **PyTorch** baseline (model loaded once per run, faster for large N): `--backend pytorch` and optional `--pytorch-dtype float16` on CUDA.

```bash
source .venv/bin/activate
pip install -r requirements.txt   # includes datasets, jiwer
cd swift_runner && swift build -c release && cd ..
# Core ML: three conditions (slow: one Swift process per clip per condition)
python scripts/eval_librispeech_wer.py --max-samples 50 --compute gpu --output-json reports/librispeech_wer.json
# Full test-clean (thousands of clips): --max-samples -1  (long run)
# PyTorch reference WER on same clips
HF_TOKEN=... python scripts/eval_librispeech_wer.py --backend pytorch --max-samples 200
```

## Disk Cleanup (CoreML temp)

CoreML conversion/compile may leave large temporary files under `/private/var/folders/.../T`.

Dry run:

```bash
cd /Users/barathwajanandan/Documents/random_apps/cohere_coreml
DRY_RUN=1 ./scripts/cleanup_coreml_temp.sh
```

Cleanup:

```bash
cd /Users/barathwajanandan/Documents/random_apps/cohere_coreml
./scripts/cleanup_coreml_temp.sh
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
- Pure pipeline packages:
  - `artifacts/cohere_frontend.mlpackage`
  - `artifacts/cohere_encoder.mlpackage`
  - `artifacts/cohere_decoder_fullseq_masked.mlpackage`
  - `artifacts/coreml_manifest.json`
- LibriSpeech WER summary: `reports/librispeech_wer.json`

## Current status

- Baseline transcript matches expected sample sentence.
- Python first-step parity: top-token match true, cosine similarity ~0.99978.
- Swift full-seq decode produces:
  - `If not, there will be a big crisis between you and the European Parliament.`
- Full-seq strict token parity across 3 local speech files: exact match true.
- Cached decode is fast and can match on some clips but is not yet exact across all tested clips.
- Pure Swift CLI path is working end-to-end (`audio -> text`) with deterministic CPU decode.
- One strict token-parity edge case remains on very short 8 kHz clips (capitalization/punctuation first-token drift).
