#!/usr/bin/env python3
import argparse
import json
import time
from pathlib import Path
from typing import Any, Dict

import coremltools as ct
import numpy as np


def cosine_similarity(a: np.ndarray, b: np.ndarray) -> float:
    a = a.reshape(-1).astype(np.float64)
    b = b.reshape(-1).astype(np.float64)
    denom = (np.linalg.norm(a) * np.linalg.norm(b)) + 1e-12
    return float(np.dot(a, b) / denom)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    data = np.load(args.inputs_npz)
    encoder_hidden_states = data["encoder_hidden_states"].astype(np.float32)
    input_ids = data["input_ids"].astype(np.int32)
    ref_logits = data["ref_logits"].astype(np.float32)
    ref_top_token = int(np.argmax(ref_logits[0]))

    model = ct.models.MLModel(args.mlpackage, compute_units=ct.ComputeUnit.CPU_AND_NE)

    t0 = time.perf_counter()
    out = model.predict(
        {
            "encoder_hidden_states": encoder_hidden_states,
            "input_ids": input_ids,
        }
    )
    infer_seconds = time.perf_counter() - t0

    # pick first output tensor regardless of converted output name
    out_key = next(iter(out.keys()))
    got_logits = np.array(out[out_key], dtype=np.float32)
    got_top_token = int(np.argmax(got_logits[0]))

    result = {
        "mlpackage": args.mlpackage,
        "output_key": out_key,
        "infer_seconds": infer_seconds,
        "reference_top_token": ref_top_token,
        "coreml_top_token": got_top_token,
        "top_token_match": ref_top_token == got_top_token,
        "logits_cosine_similarity": cosine_similarity(ref_logits, got_logits),
    }
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="CoreML parity test against PyTorch reference logits.")
    p.add_argument("--mlpackage", default=str(root / "artifacts" / "cohere_first_step.mlpackage"))
    p.add_argument("--inputs-npz", default=str(root / "artifacts" / "coreml_inputs.npz"))
    p.add_argument("--report-json", default=str(root / "reports" / "python_parity_report.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    out = Path(args.report_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
