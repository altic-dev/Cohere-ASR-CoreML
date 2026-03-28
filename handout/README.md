# Handout: Cohere ASR → CoreML (macOS)

Self-contained copy of **source** for sharing (talks, blog, job packet). **Neural network weights are not included** (multi‑GB); generate them with the Python export script or copy `*.mlpackage` from your own build.

## Layout

| Path | What |
|------|------|
| **`scripts/`** | Python: export to CoreML, benchmark, chunk-merge tests, `setup_env.sh` |
| **`swift_runner/`** | Swift package: `pure_coreml_asr_cli`, `ChunkMergeCore`, tests |
| **`sample_manifest/`** | Example `coreml_manifest.json` (metadata + tokenizer; **not** the `.mlpackage` weights) |

## Pipeline (reminder)

1. **Python** (`scripts/export_coreml_pure_pipeline.py`) — loads PyTorch model from Hugging Face, writes **`*.mlpackage`** + **`coreml_manifest.json`**.
2. **CoreML** — drop the four packages + manifest into a folder (e.g. `artifacts_fp16/`).
3. **Swift** (`swift_runner/`) — loads packages + manifest, runs on-device ASR (chunked long audio + overlap merge).

## Generate CoreML (not stored here)

From **this `handout/` directory** (after installing deps — see `scripts/setup_env.sh`):

```bash
export HF_TOKEN="hf_..."   # access to CohereLabs/cohere-transcribe-03-2026
python3 -m venv .venv && source .venv/bin/activate

python scripts/export_coreml_pure_pipeline.py \
  --artifacts-dir ./my_artifacts \
  --max-audio-seconds 30 \
  --overlap-seconds 5
```

Then point Swift at `./my_artifacts` (must contain `coreml_manifest.json` and all `cohere_*.mlpackage` dirs).

## Run Swift CLI

```bash
cd swift_runner   # under handout/
swift build -c release
./.build/release/pure_coreml_asr_cli \
  --audio /path/to/audio.wav \
  --artifacts-dir ../my_artifacts \
  --compiled-cache-dir ../my_artifacts/.compiled \
  --compute ane \
  --decoder-mode cached \
  --overlap-seconds 5
```

## License / redistribution

- **Your code**: add your own `LICENSE` if you publish a fork.
- **Cohere model**: follow [Hugging Face model card](https://huggingface.co/CohereLabs/cohere-transcribe-03-2026) and license for weights and any redistributed CoreML conversion.
