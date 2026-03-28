#!/usr/bin/env python3
"""Evaluate CoreML ASR on LibriSpeech test-clean: WER for FP16, palettize6, palettize8.
Uses batch mode (--audio-list) to load model once per variant."""
import json, os, re, subprocess, sys, tarfile, tempfile, time
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import soundfile as sf
from jiwer import wer as compute_wer

ROOT = Path(__file__).resolve().parents[1]
SWIFT_BIN = ROOT / "swift_runner" / ".build" / "release" / "pure_coreml_asr_cli"
NUM_SAMPLES = int(sys.argv[1]) if len(sys.argv) > 1 else 2620
CACHE_DIR = ROOT / ".compiled_cache"
CACHE_DIR.mkdir(exist_ok=True)
DATA_DIR = ROOT / "data" / "librispeech_test_clean"

VARIANTS = {
    "fp16":       ROOT / "artifacts_fp16",
    "palettize6": ROOT / "artifacts_palettize6",
    "palettize8": ROOT / "artifacts_palettize8",
}
VARIANTS = {k: v for k, v in VARIANTS.items() if v.exists()}
print(f"Variants found: {list(VARIANTS.keys())}", flush=True)


def norm(s):
    s = s.lower()
    s = re.sub(r"[^\w\s']", "", s)
    return re.sub(r"\s+", " ", s).strip()


def download_librispeech():
    if DATA_DIR.exists() and any(DATA_DIR.rglob("*.flac")):
        print(f"LibriSpeech test-clean already at {DATA_DIR}", flush=True)
        return
    DATA_DIR.mkdir(parents=True, exist_ok=True)
    url = "https://www.openslr.org/resources/12/test-clean.tar.gz"
    tar_path = DATA_DIR / "test-clean.tar.gz"
    print(f"Downloading test-clean.tar.gz...", flush=True)
    subprocess.run(["curl", "-L", "-o", str(tar_path), url], check=True)
    print("Extracting...", flush=True)
    with tarfile.open(tar_path, "r:gz") as tar:
        tar.extractall(path=DATA_DIR)
    tar_path.unlink()
    print("Done extracting.", flush=True)


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
    """Transcribe all files in one process using --audio-list. Returns dict {path: text}."""
    list_file = tempfile.NamedTemporaryFile(mode="w", suffix=".txt", delete=False)
    for p in wav_paths:
        list_file.write(p + "\n")
    list_file.close()

    try:
        r = subprocess.run(
            [str(SWIFT_BIN), "--audio-list", list_file.name,
             "--artifacts-dir", str(artifacts_dir),
             "--compute", "gpu", "--decoder-mode", "cached",
             "--compiled-cache-dir", str(CACHE_DIR)],
            capture_output=True, text=True, timeout=7200)
    except subprocess.TimeoutExpired:
        return {}
    finally:
        os.unlink(list_file.name)

    if r.returncode != 0:
        print(f"  ERROR: {r.stderr[:200]}", flush=True)
        return {}

    results = {}
    current_file = None
    for line in r.stdout.splitlines():
        if line.startswith("audio_file="):
            current_file = line[len("audio_file="):]
        elif line.startswith("decoded_text=") and current_file:
            results[current_file] = line[len("decoded_text="):].strip()
    return results


download_librispeech()
print(f"\nLoading {NUM_SAMPLES} samples...", flush=True)
samples = load_samples(NUM_SAMPLES)
wavs = [s[0] for s in samples]
refs = [s[1] for s in samples]
durations = [s[2] for s in samples]
total_audio_sec = sum(durations)
print(f"Loaded {len(wavs)} samples, total audio: {total_audio_sec:.1f}s ({total_audio_sec/3600:.1f}h)\n", flush=True)

all_hyps = {}
results = {}


def run_variant(vname, vpath):
    print(f"=== {vname}: starting batch transcription ({len(wavs)} files) ===", flush=True)
    t0 = time.time()
    hyp_dict = batch_transcribe(wavs, vpath)
    elapsed = time.time() - t0
    hyps = [hyp_dict.get(w, "") for w in wavs]
    valid_count = sum(1 for h in hyps if h)
    print(f"  {vname}: done in {elapsed:.0f}s ({valid_count}/{len(wavs)} valid, "
          f"{elapsed/len(wavs):.2f}s/sample)", flush=True)
    return vname, hyps


# Run all variants in parallel
print(f"Running {len(VARIANTS)} variants in parallel (batch mode)...", flush=True)
t_start = time.time()

with ThreadPoolExecutor(max_workers=len(VARIANTS)) as pool:
    futures = {pool.submit(run_variant, vn, vp): vn for vn, vp in VARIANTS.items()}
    for fut in as_completed(futures):
        vname, hyps = fut.result()
        all_hyps[vname] = hyps

total_time = time.time() - t_start
print(f"\nAll variants done in {total_time:.0f}s\n", flush=True)

# Compute WER
for vname, hyps in all_hyps.items():
    valid = [(norm(r), norm(h)) for r, h in zip(refs, hyps) if h]
    if not valid:
        print(f"  {vname}: no valid transcriptions!", flush=True)
        continue
    valid_refs, valid_hyps = zip(*valid)
    w = compute_wer(list(valid_refs), list(valid_hyps))
    results[vname] = {
        "wer_vs_ground_truth": round(w * 100, 2),
        "num_valid": len(valid_refs),
        "num_failed": len(refs) - len(valid_refs),
        "total_audio_sec": round(total_audio_sec, 1),
        "dataset": "librispeech_test_clean",
        "num_samples": len(refs),
    }
    print(f"  {vname}: WER = {w*100:.2f}% ({len(valid_refs)} valid)", flush=True)

if "fp16" in all_hyps:
    for vname in [k for k in all_hyps if k != "fp16"]:
        pairs = [(norm(f), norm(v)) for f, v in zip(all_hyps["fp16"], all_hyps[vname]) if f and v]
        if pairs:
            f_list, v_list = zip(*pairs)
            cross = compute_wer(list(f_list), list(v_list))
            if vname in results:
                results[vname]["wer_vs_fp16_hyp"] = round(cross * 100, 2)
            print(f"  {vname} vs fp16: {cross*100:.2f}%", flush=True)

# Sample comparisons
if "fp16" in all_hyps:
    print("\n--- Sample comparisons ---", flush=True)
    for i in range(min(5, len(refs))):
        print(f"\n[{i}] REF: {refs[i]}", flush=True)
        for vname in all_hyps:
            marker = " " if norm(all_hyps[vname][i]) == norm(refs[i]) else "*"
            print(f"  {marker} {vname}: {all_hyps[vname][i]}", flush=True)

print("\n=== Final Summary ===", flush=True)
print(json.dumps(results, indent=2), flush=True)

out = ROOT / "reports" / "librispeech_wer.json"
out.parent.mkdir(parents=True, exist_ok=True)
out.write_text(json.dumps(results, indent=2))
print(f"\nSaved to {out}", flush=True)
