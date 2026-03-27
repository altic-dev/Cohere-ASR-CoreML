#!/usr/bin/env python3
import argparse
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List

import torch
from huggingface_hub import hf_hub_download
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor, pipeline


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_AUDIO_REPO = "CohereLabs/cohere-transcribe-03-2026"
DEFAULT_AUDIO_FILE = "demo/voxpopuli_test_en_demo.wav"


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def transcribe_with_model(model: Any, processor: Any, audio_file: str, language: str, device: str) -> List[str]:
    if hasattr(model, "transcribe"):
        return model.transcribe(processor=processor, audio_files=[audio_file], language=language)
    asr = pipeline(
        task="automatic-speech-recognition",
        model=model,
        tokenizer=processor.tokenizer,
        feature_extractor=processor.feature_extractor,
        device="mps" if device == "mps" else -1,
    )
    out = asr(audio_file, generate_kwargs={"language": language} if language else None)
    text = out["text"] if isinstance(out, dict) and "text" in out else str(out)
    return [text]


def run(args: argparse.Namespace) -> Dict[str, Any]:
    token = os.getenv("HF_TOKEN")
    device = pick_device()

    t0 = time.perf_counter()
    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, token=token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, trust_remote_code=True, token=token).to(device)
    model.eval()
    load_seconds = time.perf_counter() - t0

    t1 = time.perf_counter()
    if args.audio_local:
        audio_file = args.audio_local
    else:
        audio_file = hf_hub_download(repo_id=args.audio_repo, filename=args.audio_file, token=token)
    download_seconds = time.perf_counter() - t1

    with torch.inference_mode():
        t2 = time.perf_counter()
        texts = transcribe_with_model(model, processor, audio_file, args.language, device)
        infer_seconds = time.perf_counter() - t2

    text = texts[0] if texts else ""
    result = {
        "model_id": args.model_id,
        "audio_repo": args.audio_repo,
        "audio_file_in_repo": args.audio_file,
        "audio_file_local": audio_file,
        "language": args.language,
        "device": device,
        "text": text,
        "metrics": {
            "load_seconds": load_seconds,
            "download_seconds": download_seconds,
            "infer_seconds": infer_seconds,
            "total_seconds": load_seconds + download_seconds + infer_seconds,
        },
    }
    return result


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Run baseline Cohere ASR inference and save transcript.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--audio-repo", default=DEFAULT_AUDIO_REPO)
    p.add_argument("--audio-file", default=DEFAULT_AUDIO_FILE)
    p.add_argument("--audio-local", default="", help="Local wav path. If set, skips hf_hub_download for audio.")
    p.add_argument("--language", default="en")
    p.add_argument(
        "--output-json",
        default=str(Path(__file__).resolve().parents[1] / "artifacts" / "baseline_result.json"),
    )
    return p.parse_args()


def main() -> None:
    args = parse_args()
    out_path = Path(args.output_json)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    result = run(args)
    with out_path.open("w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)

    print(result["text"])
    print(f"Saved: {out_path}")


if __name__ == "__main__":
    main()
