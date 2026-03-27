#!/usr/bin/env python3
import argparse
import json
import os
import subprocess
from pathlib import Path
from typing import Dict, List


def run_cmd(cmd: List[str], cwd: Path, env: Dict[str, str]) -> None:
    subprocess.run(cmd, cwd=str(cwd), env=env, check=True)


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Run fullseq CoreML parity over multiple local wav files.")
    p.add_argument("--audio-files", nargs="+", required=True)
    p.add_argument("--model-id", default="CohereLabs/cohere-transcribe-03-2026")
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output-json", default=str(root / "reports" / "fullseq_parity_set_summary.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    root = Path(__file__).resolve().parents[1]
    env = os.environ.copy()
    if not env.get("HF_TOKEN"):
        raise RuntimeError("HF_TOKEN must be set for gated model access.")

    results = []
    for idx, audio in enumerate(args.audio_files):
        audio_path = Path(audio).resolve()
        tag = f"s{idx}"
        baseline_json = root / "artifacts" / f"baseline_fullseq_{tag}.json"
        torch_trace = root / "reports" / f"pytorch_fullseq_trace_{tag}.json"
        swift_trace = root / "reports" / f"swift_fullseq_trace_{tag}.json"
        diff_json = root / "reports" / f"fullseq_trace_diff_{tag}.json"
        input_json = root / "artifacts" / f"swift_fullseq_input_{tag}.json"

        run_cmd(
            [
                str(root / ".venv" / "bin" / "python"),
                str(root / "scripts" / "run_baseline.py"),
                "--model-id",
                args.model_id,
                "--audio-local",
                str(audio_path),
                "--output-json",
                str(baseline_json),
            ],
            cwd=root,
            env=env,
        )
        run_cmd(
            [
                str(root / ".venv" / "bin" / "python"),
                str(root / "scripts" / "export_coreml_fullseq.py"),
                "--model-id",
                args.model_id,
                "--baseline-json",
                str(baseline_json),
                "--max-new-tokens",
                str(args.max_new_tokens),
                "--output-input-json",
                str(input_json),
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
                str(torch_trace),
            ],
            cwd=root,
            env=env,
        )
        run_cmd(
            [
                "swift",
                "run",
                "swift_fullseq_runner",
                str(root / "artifacts" / "cohere_decoder_fullseq.mlpackage"),
                str(input_json),
                str(args.max_new_tokens),
                str(swift_trace),
            ],
            cwd=root / "swift_runner",
            env=env,
        )
        run_cmd(
            [
                str(root / ".venv" / "bin" / "python"),
                str(root / "scripts" / "compare_decode_traces.py"),
                "--ref-json",
                str(torch_trace),
                "--cand-json",
                str(swift_trace),
                "--output-json",
                str(diff_json),
            ],
            cwd=root,
            env=env,
        )

        with open(diff_json, "r", encoding="utf-8") as f:
            diff = json.load(f)
        diff["audio_file"] = str(audio_path)
        results.append(diff)

    summary = {
        "total_files": len(results),
        "all_exact_match": all(bool(x.get("exact_match")) for x in results),
        "results": results,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
