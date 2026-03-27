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


class DecoderStepWithCache(torch.nn.Module):
    def __init__(self, model: torch.nn.Module, max_len: int):
        super().__init__()
        self.embedding = model.transf_decoder._embedding
        self.decoder_layers = model.transf_decoder._decoder.layers
        self.final_ln = model.transf_decoder._decoder.final_layer_norm
        self.log_softmax = model.log_softmax
        self.max_len = max_len
        first_attn = self.decoder_layers[0].first_sub_layer
        self.num_heads = first_attn.num_heads
        self.head_dim = first_attn.head_dim

    def _manual_attn(self, q, k, v, scale: float, mask: torch.Tensor = None):
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores + mask
        probs = torch.softmax(scores, dim=-1)
        out = torch.matmul(probs, v)
        return out

    def _update_cache(self, cache: torch.Tensor, layer_idx: int, new_k: torch.Tensor, step_idx: torch.Tensor):
        # cache: [L, H, T, D], new_k: [1, H, 1, D]
        cur = cache[layer_idx : layer_idx + 1]  # [1,H,T,D]
        oh = torch.nn.functional.one_hot(step_idx, num_classes=self.max_len).to(dtype=cur.dtype)  # [1,T]
        oh = oh.view(1, 1, self.max_len, 1)
        expanded = new_k.expand(-1, -1, self.max_len, -1)
        updated = cur * (1.0 - oh) + expanded * oh
        return updated

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,  # [1,S,H]
        input_id: torch.Tensor,  # [1,1]
        cache_k: torch.Tensor,  # [L,H,T,D]
        cache_v: torch.Tensor,  # [L,H,T,D]
        step: torch.Tensor,  # [1]
    ):
        step_idx = step.to(torch.int64).reshape(1)
        tok = input_id.to(torch.int64)
        positions = step_idx.view(1, 1)

        hidden = self.embedding(tok, positions)
        new_cache_k = []
        new_cache_v = []

        for i, layer in enumerate(self.decoder_layers):
            # Self-attention
            residual = hidden
            x = layer.layer_norm_1(hidden)
            sa = layer.first_sub_layer
            q = sa._reshape(sa.query_net(x))
            k_new = sa._reshape(sa.key_net(x))
            v_new = sa._reshape(sa.value_net(x))

            k_upd = self._update_cache(cache_k, i, k_new, step_idx)
            v_upd = self._update_cache(cache_v, i, v_new, step_idx)
            pos = torch.arange(self.max_len, device=step.device, dtype=step_idx.dtype).view(1, 1, 1, self.max_len)
            valid = pos <= step_idx.view(1, 1, 1, 1)
            attn_mask = torch.where(
                valid,
                torch.zeros((1, 1, 1, self.max_len), device=k_upd.device, dtype=k_upd.dtype),
                torch.full((1, 1, 1, self.max_len), -1e9, device=k_upd.device, dtype=k_upd.dtype),
            )
            out = self._manual_attn(q, k_upd, v_upd, sa.scale, mask=attn_mask)
            out = out.transpose(1, 2).contiguous().view(1, 1, sa.hidden_size)
            hidden = residual + sa.out_projection(out)

            # Cross-attention (recompute K/V from encoder each step)
            residual = hidden
            x = layer.layer_norm_2(hidden)
            ca = layer.second_sub_layer
            q = ca._reshape(ca.query_net(x))
            k = ca._reshape(ca.key_net(encoder_hidden_states))
            v = ca._reshape(ca.value_net(encoder_hidden_states))
            out = self._manual_attn(q, k, v, ca.scale)
            out = out.transpose(1, 2).contiguous().view(1, 1, ca.hidden_size)
            hidden = residual + ca.out_projection(out)

            residual = hidden
            x = layer.layer_norm_3(hidden)
            hidden = residual + layer.third_sub_layer(x)

            new_cache_k.append(k_upd.squeeze(0))
            new_cache_v.append(v_upd.squeeze(0))

        hidden = self.final_ln(hidden)
        logits = self.log_softmax(hidden)[:, -1, :]
        cache_k_out = torch.stack(new_cache_k, dim=0)
        cache_v_out = torch.stack(new_cache_v, dim=0)
        return logits, cache_k_out, cache_v_out


def parse_args():
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser()
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--baseline-json", default=str(root / "artifacts" / "baseline_result.json"))
    p.add_argument("--language", default="en")
    p.add_argument("--punctuation", action="store_true", default=True)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--output-mlpackage", default=str(root / "artifacts" / "cohere_decoder_cached.mlpackage"))
    p.add_argument("--output-input-json", default=str(root / "artifacts" / "swift_cached_input.json"))
    p.add_argument("--report-json", default=str(root / "reports" / "export_cached_report.json"))
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

    bos = processor.tokenizer.bos_token_id
    if bos is None:
        bos = processor.tokenizer.cls_token_id
    if bos is None:
        raise RuntimeError("Tokenizer has no BOS/CLS token id.")

    dec = model.transf_decoder._decoder
    num_layers = len(dec.layers)
    num_heads = dec.layers[0].first_sub_layer.num_heads
    head_dim = dec.layers[0].first_sub_layer.head_dim
    max_len = args.max_new_tokens + int(prompt_ids.shape[0]) + 2

    input_id = prompt_ids[:1].reshape(1, 1)
    cache_k = torch.zeros((num_layers, num_heads, max_len, head_dim), dtype=torch.float32)
    cache_v = torch.zeros((num_layers, num_heads, max_len, head_dim), dtype=torch.float32)
    step = torch.tensor([0], dtype=torch.int32)

    wrapper = DecoderStepWithCache(model, max_len=max_len).eval()
    traced = torch.jit.trace(wrapper, (encoder_hidden_states, input_id, cache_k, cache_v, step), strict=False)

    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        inputs=[
            ct.TensorType(name="encoder_hidden_states", shape=tuple(encoder_hidden_states.shape), dtype=np.float32),
            ct.TensorType(name="input_id", shape=tuple(input_id.shape), dtype=np.int32),
            ct.TensorType(name="cache_k", shape=tuple(cache_k.shape), dtype=np.float32),
            ct.TensorType(name="cache_v", shape=tuple(cache_v.shape), dtype=np.float32),
            ct.TensorType(name="step", shape=tuple(step.shape), dtype=np.int32),
        ],
    )

    out_pkg = Path(args.output_mlpackage)
    out_pkg.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_pkg))
    spec = mlmodel.get_spec()
    output_names = [o.name for o in spec.description.output]
    cache_elem_count = int(np.prod(cache_k.shape))
    logits_output_name = None
    cache_output_names = []
    for out in spec.description.output:
        shape = list(out.type.multiArrayType.shape)
        elem_count = int(np.prod(shape)) if shape else 0
        if elem_count == cache_elem_count:
            cache_output_names.append(out.name)
        else:
            logits_output_name = out.name
    if logits_output_name is None or len(cache_output_names) != 2:
        raise RuntimeError(
            f"Could not infer output names cleanly. outputs={output_names}, cache_count={len(cache_output_names)}"
        )

    tok = processor.tokenizer
    vocab = tok.convert_ids_to_tokens(list(range(tok.vocab_size)))
    input_payload: Dict[str, Any] = {
        "encoder_hidden_states_shape": list(encoder_hidden_states.shape),
        "encoder_hidden_states_flat": encoder_hidden_states.detach().cpu().numpy().reshape(-1).tolist(),
        "prompt_ids": [int(x) for x in prompt_ids.tolist()],
        "bos_token_id": int(bos),
        "eos_token_id": int(tok.eos_token_id) if tok.eos_token_id is not None else None,
        "pad_token_id": int(tok.pad_token_id) if tok.pad_token_id is not None else None,
        "vocab_size": int(tok.vocab_size),
        "id_to_token": vocab,
        "spm_model_file": str(tok.spm_model_file),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "max_len": max_len,
        "logits_output_name": logits_output_name,
        "cache_k_output_name": cache_output_names[0],
        "cache_v_output_name": cache_output_names[1],
    }
    input_json = Path(args.output_input_json)
    with input_json.open("w", encoding="utf-8") as f:
        json.dump(input_payload, f)

    report = {
        "coreml_package": str(out_pkg),
        "swift_input_json": str(input_json),
        "num_layers": num_layers,
        "num_heads": num_heads,
        "head_dim": head_dim,
        "max_len": max_len,
        "logits_output_name": logits_output_name,
        "cache_k_output_name": cache_output_names[0],
        "cache_v_output_name": cache_output_names[1],
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
