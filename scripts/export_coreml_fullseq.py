#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict

import coremltools as ct
import librosa
import numpy as np
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


class FullSeqDecoder(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, max_len: int):
        super().__init__()
        self.decoder = model.transf_decoder
        self.log_softmax = model.log_softmax
        self.max_len = max_len

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,  # [1,S,H]
        input_ids: torch.Tensor,  # [1,T]
        decoder_attention_mask: torch.Tensor,  # [1,T]
    ):
        input_ids = input_ids.to(torch.int64)
        decoder_attention_mask = decoder_attention_mask.to(torch.int64)
        bsz, t = input_ids.shape
        device = input_ids.device
        dtype = encoder_hidden_states.dtype

        positions = torch.arange(t, device=device, dtype=torch.int64).unsqueeze(0).expand(bsz, -1)
        q_pos = torch.arange(t, device=device).view(t, 1)
        k_pos = torch.arange(t, device=device).view(1, t)
        causal_bool = k_pos > q_pos
        self_attention_mask = torch.zeros((bsz, 1, t, t), device=device, dtype=dtype)
        self_attention_mask.masked_fill_(causal_bool.view(1, 1, t, t), float("-inf"))

        key_padding = (1.0 - decoder_attention_mask[:, None, None, :].to(dtype=dtype)) * -1e9
        self_attention_mask = self_attention_mask + key_padding

        outputs, _ = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=None,
            past_key_values=None,
            cache_position=None,
            kv_seq_len=None,
        )
        logits = self.log_softmax(outputs)
        return logits


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--baseline-json", default=str(root / "artifacts" / "baseline_result.json"))
    p.add_argument("--language", default="en")
    p.add_argument("--punctuation", action="store_true", default=True)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output-mlpackage", default=str(root / "artifacts" / "cohere_decoder_fullseq.mlpackage"))
    p.add_argument("--output-input-json", default=str(root / "artifacts" / "swift_fullseq_input.json"))
    p.add_argument("--report-json", default=str(root / "reports" / "export_fullseq_report.json"))
    return p.parse_args()


def main():
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    with open(args.baseline_json, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    audio_file = baseline["audio_file_local"]

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, token=token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, trust_remote_code=True, token=token).eval()

    audio, sr = load_audio_mono(audio_file)
    proc = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    input_features = proc["input_features"].to(torch.float32)
    with torch.no_grad():
        encoder_hidden_states, _ = model.encoder(input_features, None)
        if model.encoder_decoder_proj is not None:
            encoder_hidden_states = model.encoder_decoder_proj(encoder_hidden_states)

    prompt_text = model.build_prompt(language=args.language, punctuation=args.punctuation)
    prompt_inputs = processor(audio=[audio], text=[prompt_text], sampling_rate=sr, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"][0].to(torch.int32)

    max_len = int(prompt_ids.shape[0]) + args.max_new_tokens + 2
    input_ids = torch.full((1, max_len), fill_value=int(processor.tokenizer.pad_token_id), dtype=torch.int32)
    attention_mask = torch.zeros((1, max_len), dtype=torch.int32)
    plen = int(prompt_ids.shape[0])
    input_ids[0, :plen] = prompt_ids
    attention_mask[0, :plen] = 1

    wrapper = FullSeqDecoder(model, max_len=max_len).eval()
    traced = torch.jit.trace(wrapper, (encoder_hidden_states, input_ids, attention_mask), strict=False)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        inputs=[
            ct.TensorType(name="encoder_hidden_states", shape=tuple(encoder_hidden_states.shape), dtype=np.float32),
            ct.TensorType(name="input_ids", shape=tuple(input_ids.shape), dtype=np.int32),
            ct.TensorType(name="decoder_attention_mask", shape=tuple(attention_mask.shape), dtype=np.int32),
        ],
    )

    out_pkg = Path(args.output_mlpackage)
    out_pkg.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_pkg))

    tok = processor.tokenizer
    payload: Dict[str, Any] = {
        "encoder_hidden_states_shape": list(encoder_hidden_states.shape),
        "encoder_hidden_states_flat": encoder_hidden_states.detach().cpu().numpy().reshape(-1).tolist(),
        "prompt_ids": [int(x) for x in prompt_ids.tolist()],
        "eos_token_id": int(tok.eos_token_id) if tok.eos_token_id is not None else None,
        "pad_token_id": int(tok.pad_token_id) if tok.pad_token_id is not None else None,
        "id_to_token": tok.convert_ids_to_tokens(list(range(tok.vocab_size))),
        "max_len": max_len,
    }
    out_json = Path(args.output_input_json)
    with out_json.open("w", encoding="utf-8") as f:
        json.dump(payload, f)

    report = {
        "coreml_package": str(out_pkg),
        "swift_input_json": str(out_json),
        "max_len": max_len,
    }
    report_path = Path(args.report_json)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
