#!/usr/bin/env python3
import argparse
import json
import os
import re
import subprocess
import sys
import tempfile
from pathlib import Path
from typing import Dict


ROOT = Path(__file__).resolve().parents[1]


def run_cmd(cmd: list[str], cwd: Path, env: Dict[str, str]) -> str:
    p = subprocess.run(cmd, cwd=str(cwd), env=env, text=True, capture_output=True)
    if p.returncode != 0:
        print(p.stdout)
        print(p.stderr, file=sys.stderr)
        raise RuntimeError(f"Command failed ({p.returncode}): {' '.join(cmd)}")
    return p.stdout


def extract_decoded_text(swift_output: str) -> str:
    m = re.search(r"^decoded_text=(.*)$", swift_output, flags=re.MULTILINE)
    return m.group(1).strip() if m else ""


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="End-to-end ASR run: audio in -> text out (PyTorch + CoreML fullseq).")
    p.add_argument("--audio", required=True, help="Path to local audio file")
    p.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output-json", default=str(ROOT / "reports" / "e2e_audio_to_text_result.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    env = os.environ.copy()
    if not env.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN is required for the gated Cohere model.")

    audio_path = str(Path(args.audio).expanduser().resolve())

    with tempfile.TemporaryDirectory(prefix="cohere_e2e_") as td:
        td_path = Path(td)
        baseline_json = td_path / "baseline.json"
        fullseq_input_json = td_path / "swift_fullseq_input.json"
        fullseq_report_json = td_path / "export_fullseq_report.json"

        run_cmd(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "scripts" / "run_baseline.py"),
                "--model-id",
                args.model_id,
                "--audio-local",
                audio_path,
                "--output-json",
                str(baseline_json),
            ],
            cwd=ROOT,
            env=env,
        )

        run_cmd(
            [
                str(ROOT / ".venv" / "bin" / "python"),
                str(ROOT / "scripts" / "export_coreml_fullseq.py"),
                "--model-id",
                args.model_id,
                "--baseline-json",
                str(baseline_json),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--output-input-json",
                str(fullseq_input_json),
                "--report-json",
                str(fullseq_report_json),
            ],
            cwd=ROOT,
            env=env,
        )

        swift_output = run_cmd(
            [
                "swift",
                "run",
                "swift_fullseq_runner",
                str(ROOT / "artifacts" / "cohere_decoder_fullseq.mlpackage"),
                str(fullseq_input_json),
                str(args.max_new_tokens),
            ],
            cwd=ROOT / "swift_runner",
            env=env,
        )

        with baseline_json.open("r", encoding="utf-8") as f:
            baseline = json.load(f)
        pytorch_text = baseline.get("text", "").strip()
        coreml_text = extract_decoded_text(swift_output)

        result = {
            "audio": audio_path,
            "model_id": args.model_id,
            "pytorch_text": pytorch_text,
            "coreml_fullseq_text": coreml_text,
            "exact_text_match": pytorch_text == coreml_text,
            "swift_raw_output": swift_output,
        }

        out = Path(args.output_json)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8") as f:
            json.dump(result, f, indent=2)

    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
