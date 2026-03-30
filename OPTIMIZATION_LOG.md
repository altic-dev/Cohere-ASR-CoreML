# CoreML ASR Optimization Log

**Model**: CohereLabs/cohere-transcribe-03-2026
**Platform**: macOS 15 / Apple Silicon (M4 Pro)
**Framework**: CoreML via coremltools 8.3.0
**Date range**: March 27-30, 2026

---

## Pipeline Architecture

```
Audio -> Frontend (STFT/log-mel, fp32)
      -> Encoder (conformer, 1024-dim, 8 layers)
      -> CrossKV Projector (cross-attn K/V for all decoder layers)
      -> Cached Decoder (transformer, 8 layers, autoregressive with KV cache)
      -> Token IDs -> Text
```

Long audio is chunked into 30s windows with 5s overlap. Chunks are merged via suffix-prefix matching.

---

## Optimization History

### Phase 1: Initial CoreML Export (palettize6 baseline)

- Exported frontend, encoder, fullseq decoder, cached decoder as `.mlpackage`
- Float16 precision for encoder/decoder, float32 for frontend
- Palettize6 (6-bit, 64 clusters) weight compression
- Double-buffered KV cache with `outputBackings` for zero-copy decode steps
- Swift CLI (`pure_coreml_asr_cli`) with batch `--audio-list` support

**Result**: ~35x realtime on 10-min audio (GPU), 2.43% WER on LibriSpeech test-clean

### Phase 2: Cross-Attention K/V Pre-computation

**Problem**: The cached decoder recomputes cross-attention K and V projections from `encoder_hidden_states` at every decode step. With 8 layers x 2 projections x ~96 steps = 1,536 redundant matmuls per chunk.

**Solution**: Created `CrossKVProjector` module (12MB) that runs once per chunk after the encoder, producing pre-computed `cross_k` and `cross_v` tensors. Modified `DecoderStepCachedV2` to accept these instead of raw encoder output.

**Files changed**:
- `scripts/export_coreml_pure_pipeline.py` -- Added `CrossKVProjector` and `DecoderStepCachedV2` classes
- `swift_runner/Sources/pure_coreml_asr_cli/main.swift` -- Added `CrossKVMeta`, cross-KV model loading, updated `runCachedPipeline`

**Result**: ~48x realtime on 10-min audio (GPU), 2.43% WER (unchanged)

### Phase 3: Stateful KV Cache (attempted, reverted)

**Problem**: Hypothesized that CoreML's `StateType` API (macOS 15+) with `MLState` could provide faster in-place KV cache updates vs double-buffering.

**Solution**: Created `DecoderStepStateful` using `register_buffer` for `cache_k`/`cache_v` and `torch.scatter` for position-wise updates. Updated Swift runner to use `decoderModel.makeState()` and `prediction(from:using:options:)`.

**Result**: 2.3x SLOWER decoder (1,146ms vs 500ms). The `scatter` + state read/write overhead exceeded the double-buffering benefit. `outputBackings` already provides effective zero-copy. **Reverted.**

### Phase 4: Async Chunk Pipelining (attempted, cancelled)

**Problem**: For multi-chunk audio, encoder (~236ms) and decoder (~661ms) run sequentially per chunk. Overlapping encoder(N+1) with decoder(N) could save ~5.9s for 25 chunks.

**Solution**: Implemented GCD-based prefetch: while decoder processes chunk N, a background thread pre-computes frontend+encoder+crossKV for chunk N+1.

**Result**: No improvement (12.8s async vs 12.4s sequential). Both encoder and decoder compete for the same GPU, causing contention that negates the overlap benefit. Would only help with heterogeneous compute (e.g., encoder on ANE + decoder on GPU). **Cancelled.**

### Phase 5: Mixed Quantization (palettize4 encoder)

**Problem**: Encoder is 26% of per-chunk time. Palettize4 could be faster and 31% smaller.

**Solution**: Exported full pipeline with palettize4 quantization (4-bit, 16 clusters).

**Result**: Same speed as palettize6 on GPU (dequantization cost is identical for both bit widths). WER increased from 2.43% to 2.50% (negligible). Encoder size reduced from 1.3GB to 893MB. **Viable for size-sensitive deployment.**

### Export Issue: ANE Daemon Hang

During development, discovered that the ANE (Apple Neural Engine) daemon on this machine is unresponsive. Any CoreML model compilation/validation that touches ANE hangs indefinitely. Workaround: changed `compute_units` from `CPU_AND_NE` to `CPU_AND_GPU` in the export script. Also discovered that `minimum_deployment_target=ct.target.macOS15` causes a 10x encoder slowdown on GPU vs `ct.target.macOS13` -- only use macOS15 target for models that require it (e.g., stateful).

---

## Current State (as of March 30, 2026)

### Recommended Production Model: `artifacts_p4_full`

- Palettize4 + cross-KV optimization
- Total model size: ~1.1GB
- WER: 2.50% on LibriSpeech test-clean (500 samples)
- Speed: 17.3x realtime (batch, 500 samples) / ~48x realtime (single 10-min file, warm)

### Alternative: `artifacts_crosskv`

- Palettize6 + cross-KV optimization
- Total model size: ~1.5GB
- WER: 2.43% on LibriSpeech test-clean
- Speed: 17.3x realtime (batch) / ~48x realtime (single file)

### Artifacts Directory Map

| Directory | Quantization | Cross-KV | Notes |
|-----------|-------------|----------|-------|
| `artifacts_palettize6` | palettize6 | No | Original baseline |
| `artifacts_crosskv` | palettize6 | Yes | Best accuracy |
| `artifacts_p4_full` | palettize4 | Yes | Best size/accuracy tradeoff |
| `artifacts_fp16` | none (fp16) | No | Uncompressed reference |
| `artifacts_mixed_p4enc_p6dec` | mixed | Yes | Broken (cross-run incompatibility) |
| `artifacts_stateful` | palettize6 | No | Stateful KV (slower, experimental) |

---

## Benchmark Results (LibriSpeech test-clean, 500 samples, 3,809s audio)

| Variant | WER% | RTF | Speed | Wall(s) | Encoder Size |
|---------|------|-----|-------|---------|-------------|
| palettize6 (baseline) | 2.43 | 0.0670 | 14.9x | 255.3 | 1.3GB |
| crosskv (p6 + cross-KV) | 2.43 | 0.0577 | 17.3x | 219.6 | 1.3GB |
| p4_full (p4 + cross-KV) | 2.50 | 0.0579 | 17.3x | 220.5 | 893MB |

### Per-Component Profiling (single chunk, GPU, warm)

| Component | Time | % of Total |
|-----------|------|-----------|
| Frontend | 3.7ms | 0.4% |
| Encoder | 235.9ms | 26.0% |
| CrossKV Projector | 4.8ms | 0.5% |
| Decoder (96 steps) | 661.3ms | 73.0% |
| **Total per chunk** | **905.6ms** | |
| **Per decode step** | **6.89ms** | |

---

## Future Optimization Ideas (not yet attempted)

### High potential
1. **ANE+GPU heterogeneous compute** -- Run encoder on ANE, decoder on GPU in parallel. Requires fixing ANE daemon (likely just a macOS restart). Could nearly halve wall time.
2. **Reduce overlap from 5s to 2-3s** -- Fewer chunks = fewer encoder passes. Needs accuracy validation on long-form audio.
3. **Float16 frontend** -- Currently fp32 for precision. May not actually matter. Would speed up the 3.7ms frontend.

### Medium potential
4. **Speculative decoding** -- Small draft decoder guesses tokens, main decoder verifies in batches.
5. **Batch encoder inference** -- Feed multiple chunks' features at once if GPU memory allows.
6. **Reduce decoder_max_len** -- Currently 108. Shorter max reduces cache update cost per step.

### Research-level
7. **Custom Metal kernels** for attention bottlenecks
8. **Pruning** encoder attention heads (needs retraining)
9. **Dynamic early exit** -- Stop decoding when confidence exceeds threshold
10. **Knowledge distillation** -- Train a smaller student model

---



### Phase 6: ANE Ablation Study + Heterogeneous Async Pipeline

**Problem**: Previous async pipelining (Phase 4) failed because encoder and decoder both ran on GPU, causing contention. Goal: determine which models can run on ANE and exploit heterogeneous compute.

**Approach**: Added `--compute-split fine:F,E,C,D` to Swift CLI for per-model compute unit control. Ran ablation matrix testing each model individually on ANE.

**Ablation results** (single LibriSpeech clip, palettize6 `artifacts_crosskv`):

| Test | Frontend | Encoder | Decoder | Pipeline | Status |
|------|----------|---------|---------|----------|--------|
| Baseline (all GPU) | 98ms | 745ms | 133ms | 1041ms | OK |
| frontend=ANE | 2.8ms | 237ms | 74ms | 361ms | OK |
| encoder=ANE | - | - | - | - | TIMEOUT (120s) |
| crossKV=ANE | 3.9ms | 237ms | 74ms | 361ms | OK |
| decoder=ANE | 6.1ms | 250ms | 73ms | 392ms | OK |
| small3=ANE (front+ckv+dec) | 2.7ms | 247ms | 71ms | 363ms | OK |

**Key finding**: The encoder is the ONLY model that hangs on ANE. All smaller models (frontend, cross-KV, decoder) run fine and faster on ANE.

**Root cause**: Palettized weights break ANE compilation (dequant ops). FP16 (uncompressed) encoder runs on ANE fine at 533ms, but palettize6 encoder on GPU is faster at 249ms. So encoder stays on GPU.

**Solution**: `--compute-split ane_small` preset: frontend+crossKV+decoder on ANE, encoder on GPU. Combined with async overlap (encoder(N+1) on GPU while decoder(N) runs on ANE), since they use different hardware.

**Result**:

| Config | Pipeline (10-min, avg 3 runs) | RTF | Speed |
|--------|------------------------------|-----|-------|
| All GPU (sequential) | 15,078ms | 0.0251 | 39.8x |
| **ane_small + async overlap** | **8,346ms** | **0.0139** | **71.9x** |

**1.81x speedup** with identical accuracy. The async pipeline now works because encoder (GPU) and decoder (ANE) use different hardware with no contention.

**Files changed**:
- `swift_runner/Sources/pure_coreml_asr_cli/main.swift` -- Added `--compute-split fine:F,E,C,D` syntax, `ane_small` preset, `ComputeModelRole` enum, `computeForRole()` dispatch, async chunk overlap with `DispatchQueue`/`DispatchSemaphore`

---

## Key Technical Lessons

1. **`outputBackings` is already near-optimal** for KV cache management on GPU. CoreML `StateType` adds overhead rather than removing it.
2. **GPU cannot pipeline, but GPU+ANE can** -- Running two models on the same GPU causes contention. But encoder on GPU + decoder on ANE uses different hardware, enabling effective async overlap (1.81x speedup).
3. **Palettize4 vs palettize6 same speed on GPU** -- Metal dequantization cost is identical for both bit widths. Only disk size differs.
4. **`minimum_deployment_target` matters** -- macOS15 target generates different MIL ops that are 10x slower on GPU for the encoder. Use macOS13 unless model requires macOS15 features.
5. **Palettized weights break ANE compilation** -- Palettize4/6/8 encoders hang on ANE; uncompressed FP16 works fine. The issue is dequantization ops in the ANE compiler, not model size. Small palettized models (decoder, cross-KV) still work on ANE.

---

## Eval Setup

- **Dataset**: LibriSpeech test-clean (2,620 total samples, using first 500 sorted)
- **Total audio**: 3,809 seconds (~63 minutes)
- **WER normalization**: lowercase, strip punctuation (keep apostrophes), collapse whitespace
- **Tool**: `jiwer` library
- **Scripts**: `scripts/benchmark_eval.py`, `scripts/eval_librispeech_wer.py`
- **Swift CLI**: `swift_runner/.build/release/pure_coreml_asr_cli` with `--audio-list` for batch processing
