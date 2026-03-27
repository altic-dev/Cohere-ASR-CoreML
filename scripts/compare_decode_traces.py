#!/usr/bin/env python3
import argparse
import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Compare token decode traces.")
    p.add_argument("--ref-json", default=str(root / "reports" / "pytorch_decode_trace.json"))
    p.add_argument("--cand-json", default=str(root / "reports" / "swift_cached_trace.json"))
    p.add_argument("--output-json", default=str(root / "reports" / "decode_trace_diff.json"))
    return p.parse_args()


def first_mismatch(a: List[int], b: List[int]) -> Optional[int]:
    n = min(len(a), len(b))
    for i in range(n):
        if a[i] != b[i]:
            return i
    if len(a) != len(b):
        return n
    return None


def main() -> None:
    args = parse_args()
    with open(args.ref_json, "r", encoding="utf-8") as f:
        ref: Dict[str, Any] = json.load(f)
    with open(args.cand_json, "r", encoding="utf-8") as f:
        cand: Dict[str, Any] = json.load(f)

    ref_ids = [int(x) for x in ref.get("generated_ids", [])]
    cand_ids = [int(x) for x in cand.get("generated_ids", [])]
    m = first_mismatch(ref_ids, cand_ids)

    out: Dict[str, Any] = {
        "ref_json": args.ref_json,
        "cand_json": args.cand_json,
        "exact_match": m is None,
        "ref_len": len(ref_ids),
        "cand_len": len(cand_ids),
        "ref_text": ref.get("decoded_text", ""),
        "cand_text": cand.get("decoded_text", ""),
    }
    if m is not None:
        out["first_mismatch_index"] = m
        out["ref_token"] = ref_ids[m] if m < len(ref_ids) else None
        out["cand_token"] = cand_ids[m] if m < len(cand_ids) else None

    output = Path(args.output_json)
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("w", encoding="utf-8") as f:
        json.dump(out, f, indent=2)
    print(json.dumps(out, indent=2))


if __name__ == "__main__":
    main()
