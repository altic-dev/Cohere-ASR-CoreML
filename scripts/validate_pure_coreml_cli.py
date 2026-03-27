#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict, List

import librosa
import numpy as np
import soundfile as sf


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Validate pure CoreML Swift CLI exact parity against PyTorch.")
    p.add_argument("--audio-files", nargs="+", required=True)
    p.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--compute", default="cpu", choices=["cpu", "gpu", "ane", "all"])
    p.add_argument("--output-json", default=str(root / "reports" / "pure_coreml_cli_parity_summary.json"))
    p.add_argument("--reexport", action="store_true", help="Force re-export of pure CoreML pipeline before validation.")
    return p.parse_args()


def normalize_audio_16k(input_path: str, output_path: Path) -> None:
    audio, sr = sf.read(input_path, dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    if sr != 16000:
        audio = librosa.resample(audio, orig_sr=sr, target_sr=16000).astype(np.float32)
    sf.write(str(output_path), audio, 16000, subtype="PCM_16")


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    if not env.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be set for gated model access.")

    required = [
        root / "artifacts" / "coreml_manifest.json",
        root / "artifacts" / "cohere_frontend.mlpackage",
        root / "artifacts" / "cohere_encoder.mlpackage",
        root / "artifacts" / "cohere_decoder_fullseq_masked.mlpackage",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if args.reexport or missing:
        run_cmd(
            [
                str(root / ".venv" / "bin" / "python"),
                str(root / "scripts" / "export_coreml_pure_pipeline.py"),
                "--model-id",
                args.model_id,
                "--max-new-tokens",
                str(args.max_new_tokens),
            ],
            cwd=root,
            env=env,
        )

    results = []
    for idx, audio in enumerate(args.audio_files):
        audio_path = str(Path(audio).expanduser().resolve())
        tag = f"s{idx}"
        baseline_json = root / "artifacts" / f"baseline_pure_{tag}.json"
        pytorch_trace = root / "reports" / f"pytorch_pure_trace_{tag}.json"
        swift_trace = root / "reports" / f"swift_pure_trace_{tag}.json"

        with tempfile.TemporaryDirectory(prefix=f"pure_cli_{tag}_") as td:
            normalized_audio = Path(td) / "input_16k.wav"
            normalize_audio_16k(audio_path, normalized_audio)
            normalized_audio_path = str(normalized_audio)

            run_cmd(
                [
                    str(root / ".venv" / "bin" / "python"),
                    str(root / "scripts" / "run_baseline.py"),
                    "--model-id",
                    args.model_id,
                    "--audio-local",
                    normalized_audio_path,
                    "--output-json",
                    str(baseline_json),
                ],
                cwd=root,
                env=env,
            )
            run_cmd(
                [
                    str(root / ".venv" / "bin" / "python"),
                    str(root / "scripts" / "trace_pytorch_decode.py"),
                    "--model-id",
                    args.model_id,
                    "--baseline-json",
                    str(baseline_json),
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--output-json",
                    str(pytorch_trace),
                ],
                cwd=root,
                env=env,
            )

            swift_out = run_cmd(
                [
                    "swift",
                    "run",
                    "pure_coreml_asr_cli",
                    "--audio",
                    normalized_audio_path,
                    "--compute",
                    args.compute,
                    "--max-new-tokens",
                    str(args.max_new_tokens),
                    "--trace-json",
                    str(swift_trace),
                ],
                cwd=root / "swift_runner",
                env=env,
            )

        with open(pytorch_trace, "r", encoding="utf-8") as f:
            pyt = json.load(f)
        with open(swift_trace, "r", encoding="utf-8") as f:
            swf = json.load(f)

        with open(baseline_json, "r", encoding="utf-8") as f:
            baseline = json.load(f)
        baseline_text = baseline.get("text", "").strip()
        swift_text = swf.get("decoded_text", "").strip()
        out_text = re.search(r"^decoded_text=(.*)$", swift_out, flags=re.MULTILINE)
        cli_text = out_text.group(1).strip() if out_text else ""

        exact_ids = pyt.get("generated_ids") == swf.get("generated_ids")
        exact_text = pyt.get("decoded_text", "").strip() == swift_text
        exact_baseline_text = baseline_text == swift_text

        record = {
            "audio_file": audio_path,
            "normalized_audio_file": normalized_audio_path,
            "compute_mode": args.compute,
            "exact_ids": exact_ids,
            "exact_text": exact_text,
            "exact_baseline_text": exact_baseline_text,
            "pytorch_text": pyt.get("decoded_text", "").strip(),
            "swift_text": swift_text,
            "swift_stdout_text": cli_text,
        }
        results.append(record)

        if not exact_ids or not exact_text:
            raise RuntimeError(f"Parity failed on {audio_path}: {json.dumps(record, indent=2)}")

    summary = {
        "total_files": len(results),
        "all_exact_match": all(x["exact_ids"] and x["exact_text"] for x in results),
        "results": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
