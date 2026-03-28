#!/usr/bin/env python3
"""Benchmark CoreML CLI: optional warmup runs, then N timed runs; parses long-audio *_ms_total lines."""
import json, os, re, subprocess, sys, time
from glob import glob
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]


def ffprobe(p):
    r = subprocess.run(
        ["ffprobe", "-v", "error", "-show_entries", "format=duration", "-of", "default=noprint_wrappers=1:nokey=1", str(p)],
        capture_output=True,
        text=True,
    )
    return float(r.stdout.strip())


def parse_swift_stdout(stdout: str) -> dict:
    m = {}
    for line in stdout.splitlines():
        if "=" not in line:
            continue
        k, _, v = line.partition("=")
        k, v = k.strip(), v.strip()
        if k.endswith("_ms") or k.endswith("_ms_total"):
            try:
                m[k] = float(v)
            except ValueError:
                m[k] = v
        else:
            m[k] = v
    return m


def infer_ms_from_parsed(m: dict) -> float:
    if "frontend_ms_total" in m and "encoder_ms_total" in m and "decoder_ms_total" in m:
        return m["frontend_ms_total"] + m["encoder_ms_total"] + m["decoder_ms_total"]
    return m["frontend_ms"] + m["encoder_ms"] + m["decoder_ms"]


def total_reported_ms(m: dict) -> float:
    if "load_ms" not in m:
        raise KeyError("load_ms missing in CLI output")
    return m.get("load_ms", 0) + m.get("audio_ms", 0) + infer_ms_from_parsed(m)


def run_swift(audio, swift, ad, compute, dec, compiled_cache=None, overlap_seconds=None):
    cmd = [str(swift), "--audio", str(audio), "--artifacts-dir", str(ad), "--compute", compute, "--decoder-mode", dec]
    if compiled_cache:
        cmd += ["--compiled-cache-dir", str(compiled_cache)]
    if overlap_seconds is not None:
        cmd += ["--overlap-seconds", str(overlap_seconds)]
    r = subprocess.run(cmd, capture_output=True, text=True)
    if r.returncode:
        raise RuntimeError(r.stdout + r.stderr)
    return parse_swift_stdout(r.stdout)


def norm(s):
    return " ".join(s.lower().split())


def pt_once(model, proc, path, dev):
    import torch

    t0 = time.perf_counter()
    with torch.inference_mode():
        if hasattr(model, "transcribe"):
            texts = model.transcribe(processor=proc, audio_files=[path], language="en")
        else:
            from transformers import pipeline

            asr = pipeline(
                "automatic-speech-recognition",
                model=model,
                tokenizer=proc.tokenizer,
                feature_extractor=proc.feature_extractor,
                device="mps" if dev == "mps" else -1,
            )
            o = asr(path, generate_kwargs={"language": "en"})
            texts = [o["text"] if isinstance(o, dict) else str(o)]
    return (texts[0] if texts else "").strip(), time.perf_counter() - t0


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser()
    ap.add_argument("--audio-glob", default=str(ROOT / "artifacts" / "*.wav"))
    ap.add_argument("--artifacts-dir", default=str(ROOT / "artifacts"))
    ap.add_argument("--compiled-cache-dir", default=None, help="Stabilizes load_ms after first compile (e.g. artifacts_fp16/.compiled)")
    ap.add_argument("--overlap-seconds", type=float, default=5.0)
    ap.add_argument("--swift-bin", default=str(ROOT / "swift_runner" / ".build" / "release" / "pure_coreml_asr_cli"))
    ap.add_argument(
        "--compute",
        default="ane",
        help="One mode or comma-separated list, e.g. cpu,ane,gpu",
    )
    ap.add_argument("--decoder-mode", default="cached")
    ap.add_argument("--warmup-runs", type=int, default=1)
    ap.add_argument("--timed-runs", type=int, default=5)
    ap.add_argument("--pytorch", action="store_true")
    ap.add_argument("--output-json", default=str(ROOT / "reports" / "benchmark_rtf_accuracy.json"))
    a = ap.parse_args()
    swift = Path(a.swift_bin)
    if not swift.is_file():
        sys.exit("missing " + str(swift) + " (run: cd swift_runner && swift build -c release)")
    ad = Path(a.artifacts_dir)
    cache = Path(a.compiled_cache_dir) if a.compiled_cache_dir else None
    computes = [x.strip() for x in str(a.compute).split(",") if x.strip()]
    wavs = sorted(Path(x) for x in glob(a.audio_glob))
    if not wavs:
        sys.exit("no audio matched --audio-glob")

    pm = pp = None
    pdev = "cpu"
    if a.pytorch and os.environ.get("HF_TOKEN"):
        import torch
        from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

        pdev = "mps" if torch.backends.mps.is_available() else "cpu"
        t = os.environ["HF_TOKEN"]
        pp = AutoProcessor.from_pretrained("CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True, token=t)
        pm = AutoModelForSpeechSeq2Seq.from_pretrained(
            "CohereLabs/cohere-transcribe-03-2026", trust_remote_code=True, token=t
        ).to(pdev)
        pm.eval()
    elif a.pytorch:
        print("no HF_TOKEN", file=sys.stderr)
        a.pytorch = False

    by_compute = {}
    for compute in computes:
        clips = []
        for wav in wavs:
            dur = ffprobe(wav)
            if dur <= 0:
                continue
            for _ in range(a.warmup_runs):
                run_swift(wav, swift, ad, compute, a.decoder_mode, cache, a.overlap_seconds)
            inf, tot, last = [], [], {}
            for _ in range(a.timed_runs):
                last = run_swift(wav, swift, ad, compute, a.decoder_mode, cache, a.overlap_seconds)
                lm, am = last.get("load_ms", 0), last.get("audio_ms", 0)
                im = infer_ms_from_parsed(last)
                inf.append(im)
                tot.append(lm + am + im)
            im_avg = sum(inf) / len(inf)
            tm_avg = sum(tot) / len(tot)
            rtf_i, rtf_t = (im_avg / 1000) / dur, (tm_avg / 1000) / dur
            ct = str(last.get("decoded_text", ""))
            row = {
                "audio": wav.name,
                "duration_sec": round(dur, 4),
                "infer_ms_avg": round(im_avg, 2),
                "infer_ms_runs": [round(x, 2) for x in inf],
                "total_wall_ms_avg": round(tm_avg, 2),
                "total_wall_ms_runs": [round(x, 2) for x in tot],
                "load_ms_last_run": round(last.get("load_ms", 0), 2),
                "rtf_infer": round(rtf_i, 5),
                "rtf_total_wall": round(rtf_t, 5),
                "faster_than_realtime_x": round(1 / rtf_i, 1) if rtf_i else None,
                "precision": last.get("precision"),
                "quantize": last.get("quantize"),
                "chunk_count": last.get("chunk_count"),
                "coreml_text_preview": ct[:200],
            }
            if a.pytorch and pm:
                ptxt, psec = pt_once(pm, pp, str(wav), pdev)
                row["pytorch_infer_sec"] = round(psec, 3)
                row["pytorch_rtf"] = round(psec / dur, 5)
                row["text_match_exact"] = ptxt.strip() == ct.strip()
                row["text_match_normalized"] = norm(ptxt) == norm(ct)
            clips.append(row)
        by_compute[compute] = clips

    rep = {
        "note": (
            "Warmup runs load/compile caches; timed runs average infer_ms. "
            "infer_ms = frontend_ms_total+encoder_ms_total+decoder_ms_total (long audio) or per-stage ms without _total. "
            "total_wall_ms_avg = load_ms + audio_ms + infer_ms (from CLI counters). "
            "After warmup, load_ms should be small if compiled-cache-dir is stable."
        ),
        "warmup_runs": a.warmup_runs,
        "timed_runs": a.timed_runs,
        "overlap_seconds": a.overlap_seconds,
        "compiled_cache_dir": str(cache) if cache else None,
        "by_compute": by_compute,
    }
    outp = Path(a.output_json)
    outp.parent.mkdir(parents=True, exist_ok=True)
    outp.write_text(json.dumps(rep, indent=2))
    print(json.dumps(rep, indent=2))
    print("Saved", outp)
