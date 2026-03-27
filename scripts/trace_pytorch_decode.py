#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import librosa
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"


def decode_like_swift(ids: List[int], vocab: List[str], eos: Optional[int], pad: Optional[int]) -> str:
    pieces: List[str] = []
    for tid in ids:
        if eos is not None and tid == eos:
            break
        if pad is not None and tid == pad:
            continue
        if tid < 0 or tid >= len(vocab):
            continue
        piece = vocab[tid]
        if piece.startswith("<") and piece.endswith(">"):
            continue
        pieces.append(piece)
    return "".join(pieces).replace("▁", " ").strip()


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
    p.add_argument("--manifest-json", default=str(root / "artifacts" / "coreml_manifest.json"))
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
    # CoreML frontend export is deterministic and omits runtime dither noise.
    # Disable dither here to keep token-level parity checks stable.
    if hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "filterbank"):
        processor.feature_extractor.filterbank.dither = 0.0

    audio, sr = load_audio_mono(audio_file)
    proc = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    input_features = proc["input_features"].to(device=device, dtype=torch.float32)

    manifest_path = Path(args.manifest_json)
    prompt_ids = None
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as f:
            manifest = json.load(f)
        ids = manifest.get("prompt_ids")
        if isinstance(ids, list) and ids:
            prompt_ids = torch.tensor(ids, device=device, dtype=torch.long)

    if prompt_ids is None:
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

    vocab = processor.tokenizer.convert_ids_to_tokens(list(range(processor.tokenizer.vocab_size)))
    text = decode_like_swift(
        generated,
        vocab=vocab,
        eos=processor.tokenizer.eos_token_id,
        pad=processor.tokenizer.pad_token_id,
    )
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
