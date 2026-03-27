#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List

import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"


def load_audio_mono(path: str, target_sr: int = 16000):
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    data = data.astype(np.float32)
    if sr != target_sr:
        data = librosa.resample(data, orig_sr=sr, target_sr=target_sr).astype(np.float32)
        sr = target_sr
    return data, sr


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Emit per-step PyTorch decode trace.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--baseline-json", default=str(root / "artifacts" / "baseline_result.json"))
    p.add_argument("--language", default="en")
    p.add_argument("--punctuation", action="store_true", default=True)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output-json", default=str(root / "reports" / "pytorch_decode_trace.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    device = "mps" if torch.backends.mps.is_available() else "cpu"

    with open(args.baseline_json, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    audio_file = baseline["audio_file_local"]

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, token=token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, trust_remote_code=True, token=token).to(device).eval()

    audio, sr = load_audio_mono(audio_file)
    proc = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    input_features = proc["input_features"].to(device=device, dtype=torch.float32)

    prompt_text = model.build_prompt(language=args.language, punctuation=args.punctuation)
    prompt_inputs = processor(audio=[audio], text=[prompt_text], sampling_rate=sr, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"][0].to(device=device, dtype=torch.long)

    generated: List[int] = [int(x) for x in prompt_ids.tolist()]
    step_top_tokens: List[int] = []
    step_top5: List[List[int]] = []

    with torch.no_grad():
        for _ in range(args.max_new_tokens):
            cur = torch.tensor([generated], device=device, dtype=torch.long)
            out = model(input_features=input_features, decoder_input_ids=cur, use_cache=False, return_dict=True)
            logits = out.logits[:, -1, :]
            top5 = torch.topk(logits, k=5, dim=-1).indices[0].tolist()
            next_id = int(top5[0])
            step_top_tokens.append(next_id)
            step_top5.append([int(x) for x in top5])
            generated.append(next_id)
            eos = processor.tokenizer.eos_token_id
            if eos is not None and next_id == int(eos):
                break

    text = processor.tokenizer.decode(generated, skip_special_tokens=True)
    payload: Dict[str, Any] = {
        "prompt_ids": [int(x) for x in prompt_ids.tolist()],
        "generated_ids": generated,
        "step_top_tokens": step_top_tokens,
        "step_top5_tokens": step_top5,
        "decoded_text": text,
        "max_new_tokens": args.max_new_tokens,
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2)
    print(json.dumps({"trace": str(out), "decoded_text": text}, indent=2))


if __name__ == "__main__":
    main()
