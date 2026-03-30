#!/usr/bin/env python3
"""Benchmark + WER eval for CoreML ASR variants on LibriSpeech test-clean.

Usage:
    python scripts/benchmark_eval.py --num-samples 500
    python scripts/benchmark_eval.py --num-samples 500 --variants palettize6,crosskv
"""
import argparse, json, os, re, subprocess, sys, tempfile, time
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import wer as compute_wer

ROOT = Path(__file__).resolve().parents[1]
SWIFT_BIN = ROOT / "swift_runner" / ".build" / "release" / "pure_coreml_asr_cli"
CACHE_DIR = ROOT / ".compiled_cache"
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data" / "librispeech_test_clean"

ALL_VARIANTS = {
    "palettize6":     ROOT / "artifacts_palettize6",
    "crosskv":        ROOT / "artifacts_crosskv",
    "mixed_p4enc":    ROOT / "artifacts_mixed_p4enc_p6dec",
    "p4_full":        ROOT / "artifacts_p4_full",
}


def norm(s):
    s = s.lower()
    s = re.sub(r"[^\w\s']", "", s)
    return re.sub(r"\s+", " ", s).strip()


def load_samples(n):
    base = DATA_DIR / "LibriSpeech" / "test-clean"
    if not base.exists():
        for candidate in DATA_DIR.rglob("*.trans.txt"):
            base = candidate.parent.parent.parent
            break
    samples = []
    for trans_file in sorted(base.rglob("*.trans.txt")):
        folder = trans_file.parent
        with open(trans_file) as f:
            for line in f:
                parts = line.strip().split(maxsplit=1)
                if len(parts) < 2:
                    continue
                utt_id, text = parts
                flac_path = folder / f"{utt_id}.flac"
                if flac_path.exists():
                    wav_path = str(flac_path).replace(".flac", ".wav")
                    if not os.path.exists(wav_path):
                        audio, sr = sf.read(str(flac_path), dtype="float32")
                        sf.write(wav_path, audio, sr)
                    else:
                        audio, sr = sf.read(wav_path, dtype="float32")
                    samples.append((wav_path, text, len(audio) / sr))
                if len(samples) >= n:
                    return samples
    return samples


def batch_transcribe(wav_paths, artifacts_dir):
    list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    for p in wav_paths:
        list_file.write(p + "\n")
    list_file.close()

    try:
        t0 = time.time()
        r = subprocess.run(
            [str(SWIFT_BIN), "--audio-list", list_file.name,
             "--artifacts-dir", str(artifacts_dir),
             "--compute", "gpu", "--decoder-mode", "cached",
             "--compiled-cache-dir", str(CACHE_DIR)],
            capture_output=True, text=True, timeout=7200)
        wall_sec = time.time() - t0
    except subprocess.TimeoutExpired:
        return {}, 0.0
    finally:
        os.unlink(list_file.name)

    if r.returncode != 0:
        print(f"  ERROR (rc={r.returncode}): {r.stderr[:300]}", flush=True)
        return {}, wall_sec

    results = {}
    current_file = None
    for line in r.stdout.splitlines():
        if line.startswith("audio_file="):
            current_file = line[len("audio_file="):]
        elif line.startswith("decoded_text=") and current_file:
            results[current_file] = line[len("decoded_text="):].strip()
    return results, wall_sec


def run_eval(variant_name, artifacts_dir, wavs, refs, durations):
    total_audio_sec = sum(durations)
    print(f"\n{'='*60}", flush=True)
    print(f"  {variant_name}: {len(wavs)} samples, {total_audio_sec:.1f}s audio", flush=True)
    print(f"  artifacts: {artifacts_dir}", flush=True)
    print(f"{'='*60}", flush=True)

    hyp_dict, wall_sec = batch_transcribe(wavs, artifacts_dir)
    hyps = [hyp_dict.get(w, "") for w in wavs]
    valid_count = sum(1 for h in hyps if h)

    valid = [(norm(r), norm(h)) for r, h in zip(refs, hyps) if h]
    wer_pct = None
    if valid:
        valid_refs, valid_hyps = zip(*valid)
        wer_pct = round(compute_wer(list(valid_refs), list(valid_hyps)) * 100, 2)

    rtf = wall_sec / total_audio_sec if total_audio_sec > 0 else 0
    faster_x = 1.0 / rtf if rtf > 0 else 0

    result = {
        "variant": variant_name,
        "num_samples": len(wavs),
        "num_valid": valid_count,
        "total_audio_sec": round(total_audio_sec, 1),
        "wall_sec": round(wall_sec, 1),
        "rtf": round(rtf, 5),
        "faster_than_realtime_x": round(faster_x, 1),
        "wer_pct": wer_pct,
    }

    print(f"  Wall time:  {wall_sec:.1f}s", flush=True)
    print(f"  RTF:        {rtf:.5f}", flush=True)
    print(f"  Speed:      {faster_x:.1f}x realtime", flush=True)
    print(f"  WER:        {wer_pct}%  ({valid_count}/{len(wavs)} valid)", flush=True)

    return result, hyps


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num-samples", type=int, default=500)
    parser.add_argument("--variants", type=str, default=None,
                        help="Comma-separated variant names (default: all available)")
    args = parser.parse_args()

    variants = ALL_VARIANTS
    if args.variants:
        names = [v.strip() for v in args.variants.split(",")]
        variants = {k: v for k, v in ALL_VARIANTS.items() if k in names}
    variants = {k: v for k, v in variants.items() if v.exists()}

    print(f"Variants: {list(variants.keys())}", flush=True)
    print(f"Samples:  {args.num_samples}", flush=True)

    print(f"\nLoading {args.num_samples} LibriSpeech samples...", flush=True)
    samples = load_samples(args.num_samples)
    wavs = [s[0] for s in samples]
    refs = [s[1] for s in samples]
    durations = [s[2] for s in samples]
    print(f"Loaded {len(wavs)} samples, total audio: {sum(durations):.1f}s\n", flush=True)

    all_results = []
    for vname, vpath in variants.items():
        result, hyps = run_eval(vname, vpath, wavs, refs, durations)
        all_results.append(result)

    print(f"\n{'='*60}", flush=True)
    print("  SUMMARY", flush=True)
    print(f"{'='*60}", flush=True)
    print(f"{'Variant':<20} {'WER%':>8} {'RTF':>10} {'Speed':>10} {'Wall(s)':>10}", flush=True)
    print("-" * 60, flush=True)
    for r in all_results:
        wer_str = f"{r['wer_pct']:>7.2f}%" if r['wer_pct'] is not None else "  FAIL "
        print(f"{r['variant']:<20} {wer_str} {r['rtf']:>10.5f} {r['faster_than_realtime_x']:>9.1f}x {r['wall_sec']:>9.1f}s", flush=True)

    out = ROOT / "reports" / "benchmark_eval.json"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(all_results, indent=2))
    print(f"\nSaved to {out}", flush=True)


if __name__ == "__main__":
    main()
