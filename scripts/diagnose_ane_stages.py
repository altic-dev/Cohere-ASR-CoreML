#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from export_coreml_pure_pipeline import EncoderCore, FrontendCore, FullSeqDecoderMasked, patch_model_for_tracing


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Diagnose frontend/encoder/decoder drift between PyTorch and CoreML ANE.")
    p.add_argument("--audio", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--artifacts-dir", default=str(root / "artifacts"))
    p.add_argument("--decoder-step-index", type=int, default=23, help="Token index in generated_ids to inspect.")
    p.add_argument("--topk", type=int, default=10)
    p.add_argument("--output-json", default=str(root / "reports" / "ane_stage_diagnostic.json"))
    return p.parse_args()


def load_manifest(artifacts_dir: Path) -> Dict[str, Any]:
    return json.loads((artifacts_dir / "coreml_manifest.json").read_text())


def load_audio(path: Path, max_audio_samples: int) -> Tuple[np.ndarray, int]:
    audio, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    raw_length = min(len(audio), max_audio_samples)
    if len(audio) < max_audio_samples:
        audio = np.pad(audio, (0, max_audio_samples - len(audio)))
    else:
        audio = audio[:max_audio_samples]
    return audio.astype(np.float32), raw_length


def max_mean_abs_diff(a: np.ndarray, b: np.ndarray) -> Dict[str, float]:
    diff = np.abs(a.astype(np.float32) - b.astype(np.float32))
    return {
        "max_abs_diff": float(diff.max()),
        "mean_abs_diff": float(diff.mean()),
    }


def build_pytorch_modules(model_id: str, token: Optional[str]):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True, token=token).eval()
    patch_model_for_tracing(model)
    if hasattr(processor, "feature_extractor") and hasattr(processor.feature_extractor, "filterbank"):
        processor.feature_extractor.filterbank.dither = 0.0

    fb = processor.feature_extractor.filterbank
    frontend = FrontendCore(
        mel_fb=fb.fb.detach().to(torch.float32),
        window=fb.window.detach().to(torch.float32),
        n_fft=int(fb.n_fft),
        win_length=int(fb.win_length),
        hop_length=int(fb.hop_length),
        preemph=float(fb.preemph if fb.preemph is not None else 0.97),
        pad_to=int(fb.pad_to),
        log_zero_guard_value=float(fb.log_zero_guard_value_fn(torch.tensor(1.0))),
        mag_power=float(fb.mag_power),
    ).eval()
    encoder = EncoderCore(model).eval()
    decoder = FullSeqDecoderMasked(model).eval()
    return processor, frontend, encoder, decoder


def load_coreml_models(artifacts_dir: Path, manifest: Dict[str, Any], units: ct.ComputeUnit):
    return {
        "frontend": ct.models.MLModel(str(artifacts_dir / manifest["frontend"]["package"]), compute_units=units),
        "encoder": ct.models.MLModel(str(artifacts_dir / manifest["encoder"]["package"]), compute_units=units),
        "decoder": ct.models.MLModel(str(artifacts_dir / manifest["decoder"]["package"]), compute_units=units),
    }


def coreml_frontend_predict(model, manifest: Dict[str, Any], audio: np.ndarray, raw_length: int):
    out = model.predict(
        {
            manifest["frontend"]["inputs"][0]: audio[None, :],
            manifest["frontend"]["inputs"][1]: np.array([raw_length], dtype=np.int32),
        }
    )
    return out[manifest["frontend"]["outputs"][0]], out[manifest["frontend"]["outputs"][1]]


def coreml_encoder_predict(model, manifest: Dict[str, Any], feats: np.ndarray, feat_len: np.ndarray):
    out = model.predict(
        {
            manifest["encoder"]["inputs"][0]: feats.astype(np.float32),
            manifest["encoder"]["inputs"][1]: feat_len.astype(np.int32),
        }
    )
    return out[manifest["encoder"]["outputs"][0]], out[manifest["encoder"]["outputs"][1]]


def make_decoder_inputs(manifest: Dict[str, Any], encoder_hidden: np.ndarray, encoder_len: int):
    input_ids = np.full((1, manifest["decoder_max_len"]), manifest["pad_token_id"], dtype=np.int32)
    attn = np.zeros((1, manifest["decoder_max_len"]), dtype=np.int32)
    for i, tid in enumerate(manifest["prompt_ids"]):
        input_ids[0, i] = tid
        attn[0, i] = 1
    cross = np.full((1, 1, 1, manifest["max_encoder_frames"]), -1e9, dtype=np.float32)
    cross[0, 0, 0, :encoder_len] = 0.0
    return input_ids, attn, cross, len(manifest["prompt_ids"]) - 1


def decode_loop_coreml(model, manifest: Dict[str, Any], encoder_hidden: np.ndarray, encoder_len: int) -> Dict[str, Any]:
    input_ids, attn, cross, cur_idx = make_decoder_inputs(manifest, encoder_hidden, encoder_len)
    generated = list(manifest["prompt_ids"])
    logits_at_steps: Dict[int, np.ndarray] = {}

    while True:
        out = model.predict(
            {
                manifest["decoder"]["inputs"][0]: encoder_hidden.astype(np.float32),
                manifest["decoder"]["inputs"][1]: input_ids,
                manifest["decoder"]["inputs"][2]: attn,
                manifest["decoder"]["inputs"][3]: cross,
            }
        )
        logits = out[manifest["decoder"]["outputs"][0]]
        logits_at_steps[cur_idx] = logits[0, cur_idx, :].astype(np.float32).copy()
        next_id = int(np.argmax(logits[0, cur_idx, :]))
        generated.append(next_id)
        cur_idx += 1
        input_ids[0, cur_idx] = next_id
        attn[0, cur_idx] = 1
        if next_id == manifest["eos_token_id"] or cur_idx + 1 >= manifest["decoder_max_len"]:
            break

    return {
        "generated_ids": generated,
        "logits_at_steps": logits_at_steps,
    }


def decode_loop_torch(decoder, manifest: Dict[str, Any], encoder_hidden: np.ndarray, encoder_len: int) -> Dict[str, Any]:
    input_ids, attn, cross, cur_idx = make_decoder_inputs(manifest, encoder_hidden, encoder_len)
    ids_t = torch.from_numpy(input_ids)
    attn_t = torch.from_numpy(attn)
    cross_t = torch.from_numpy(cross)
    enc_t = torch.from_numpy(encoder_hidden.astype(np.float32))
    generated = list(manifest["prompt_ids"])
    logits_at_steps: Dict[int, np.ndarray] = {}

    with torch.no_grad():
        while True:
            logits = decoder(enc_t, ids_t, attn_t, cross_t)
            step_logits = logits[0, cur_idx, :].detach().cpu().numpy().astype(np.float32)
            logits_at_steps[cur_idx] = step_logits.copy()
            next_id = int(np.argmax(step_logits))
            generated.append(next_id)
            cur_idx += 1
            ids_t[0, cur_idx] = next_id
            attn_t[0, cur_idx] = 1
            if next_id == manifest["eos_token_id"] or cur_idx + 1 >= manifest["decoder_max_len"]:
                break

    return {
        "generated_ids": generated,
        "logits_at_steps": logits_at_steps,
    }


def topk_info(logits: np.ndarray, tokenizer, topk: int) -> List[Dict[str, Any]]:
    idx = np.argsort(-logits)[:topk].tolist()
    toks = tokenizer.convert_ids_to_tokens(idx)
    return [{"id": int(i), "token": tok, "value": float(logits[i])} for i, tok in zip(idx, toks)]


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    artifacts_dir = Path(args.artifacts_dir)
    manifest = load_manifest(artifacts_dir)
    audio_path = Path(args.audio).expanduser().resolve()
    audio, raw_length = load_audio(audio_path, manifest["max_audio_samples"])

    processor, frontend_t, encoder_t, decoder_t = build_pytorch_modules(args.model_id, token)
    with torch.no_grad():
        pyt_feat_t, pyt_feat_len_t = frontend_t(
            torch.from_numpy(audio[None, :].astype(np.float32)),
            torch.tensor([raw_length], dtype=torch.int32),
        )
        pyt_enc_t, pyt_enc_len_t = encoder_t(pyt_feat_t, pyt_feat_len_t)

    pyt_feat = pyt_feat_t.detach().cpu().numpy().astype(np.float32)
    pyt_feat_len = pyt_feat_len_t.detach().cpu().numpy().astype(np.int32)
    pyt_enc = pyt_enc_t.detach().cpu().numpy().astype(np.float32)
    pyt_enc_len = pyt_enc_len_t.detach().cpu().numpy().astype(np.int32)

    cpu_models = load_coreml_models(artifacts_dir, manifest, ct.ComputeUnit.CPU_ONLY)
    ane_models = load_coreml_models(artifacts_dir, manifest, ct.ComputeUnit.CPU_AND_NE)

    cpu_feat, cpu_feat_len = coreml_frontend_predict(cpu_models["frontend"], manifest, audio, raw_length)
    ane_feat, ane_feat_len = coreml_frontend_predict(ane_models["frontend"], manifest, audio, raw_length)

    cpu_enc_from_pyt, cpu_enc_len_from_pyt = coreml_encoder_predict(cpu_models["encoder"], manifest, pyt_feat, pyt_feat_len)
    ane_enc_from_pyt, ane_enc_len_from_pyt = coreml_encoder_predict(ane_models["encoder"], manifest, pyt_feat, pyt_feat_len)
    cpu_enc_from_cpu_front, cpu_enc_len_from_cpu_front = coreml_encoder_predict(cpu_models["encoder"], manifest, cpu_feat, np.asarray(cpu_feat_len, dtype=np.int32))
    ane_enc_from_ane_front, ane_enc_len_from_ane_front = coreml_encoder_predict(ane_models["encoder"], manifest, ane_feat, np.asarray(ane_feat_len, dtype=np.int32))
    cpu_enc_from_ane_front, cpu_enc_len_from_ane_front = coreml_encoder_predict(cpu_models["encoder"], manifest, ane_feat, np.asarray(ane_feat_len, dtype=np.int32))
    ane_enc_from_cpu_front, ane_enc_len_from_cpu_front = coreml_encoder_predict(ane_models["encoder"], manifest, cpu_feat, np.asarray(cpu_feat_len, dtype=np.int32))

    enc_len_value = int(round(float(pyt_enc_len[0])))
    pyt_dec = decode_loop_torch(decoder_t, manifest, pyt_enc, enc_len_value)
    cpu_dec = decode_loop_coreml(cpu_models["decoder"], manifest, pyt_enc, enc_len_value)
    ane_dec = decode_loop_coreml(ane_models["decoder"], manifest, pyt_enc, enc_len_value)
    cpu_full_chain = decode_loop_coreml(
        cpu_models["decoder"],
        manifest,
        cpu_enc_from_cpu_front.astype(np.float32),
        int(round(float(np.asarray(cpu_enc_len_from_cpu_front)[0]))),
    )
    ane_full_chain = decode_loop_coreml(
        ane_models["decoder"],
        manifest,
        ane_enc_from_ane_front.astype(np.float32),
        int(round(float(np.asarray(ane_enc_len_from_ane_front)[0]))),
    )
    cpu_dec_on_ane_enc = decode_loop_coreml(
        cpu_models["decoder"],
        manifest,
        ane_enc_from_ane_front.astype(np.float32),
        int(round(float(np.asarray(ane_enc_len_from_ane_front)[0]))),
    )
    ane_dec_on_cpu_enc = decode_loop_coreml(
        ane_models["decoder"],
        manifest,
        cpu_enc_from_cpu_front.astype(np.float32),
        int(round(float(np.asarray(cpu_enc_len_from_cpu_front)[0]))),
    )

    step = args.decoder_step_index
    payload: Dict[str, Any] = {
        "audio_file": str(audio_path),
        "decoder_step_index": step,
        "frontend": {
            "pytorch_vs_cpu_only": max_mean_abs_diff(pyt_feat, cpu_feat),
            "pytorch_vs_cpu_and_ne": max_mean_abs_diff(pyt_feat, ane_feat),
            "cpu_only_vs_cpu_and_ne": max_mean_abs_diff(cpu_feat, ane_feat),
            "pytorch_feature_len": pyt_feat_len.tolist(),
            "cpu_only_feature_len": np.asarray(cpu_feat_len).tolist(),
            "cpu_and_ne_feature_len": np.asarray(ane_feat_len).tolist(),
        },
        "encoder": {
            "pytorch_vs_cpu_only": max_mean_abs_diff(pyt_enc, cpu_enc_from_pyt),
            "pytorch_vs_cpu_and_ne": max_mean_abs_diff(pyt_enc, ane_enc_from_pyt),
            "cpu_only_vs_cpu_and_ne": max_mean_abs_diff(cpu_enc_from_pyt, ane_enc_from_pyt),
            "cpu_frontend_then_cpu_encoder_vs_pytorch_encoder": max_mean_abs_diff(pyt_enc, cpu_enc_from_cpu_front),
            "ane_frontend_then_ane_encoder_vs_pytorch_encoder": max_mean_abs_diff(pyt_enc, ane_enc_from_ane_front),
            "cpu_frontend_then_cpu_encoder_vs_ane_frontend_then_ane_encoder": max_mean_abs_diff(cpu_enc_from_cpu_front, ane_enc_from_ane_front),
            "cpu_frontend_then_ane_encoder_vs_cpu_frontend_then_cpu_encoder": max_mean_abs_diff(ane_enc_from_cpu_front, cpu_enc_from_cpu_front),
            "ane_frontend_then_cpu_encoder_vs_ane_frontend_then_ane_encoder": max_mean_abs_diff(cpu_enc_from_ane_front, ane_enc_from_ane_front),
            "pytorch_encoder_len": pyt_enc_len.tolist(),
            "cpu_only_encoder_len": np.asarray(cpu_enc_len_from_pyt).tolist(),
            "cpu_and_ne_encoder_len": np.asarray(ane_enc_len_from_pyt).tolist(),
            "cpu_chain_encoder_len": np.asarray(cpu_enc_len_from_cpu_front).tolist(),
            "ane_chain_encoder_len": np.asarray(ane_enc_len_from_ane_front).tolist(),
        },
        "decoder": {
            "pytorch_generated_ids": pyt_dec["generated_ids"],
            "cpu_only_generated_ids": cpu_dec["generated_ids"],
            "cpu_and_ne_generated_ids": ane_dec["generated_ids"],
            "cpu_full_chain_generated_ids": cpu_full_chain["generated_ids"],
            "ane_full_chain_generated_ids": ane_full_chain["generated_ids"],
            "cpu_decoder_on_ane_chain_encoder_ids": cpu_dec_on_ane_enc["generated_ids"],
            "ane_decoder_on_cpu_chain_encoder_ids": ane_dec_on_cpu_enc["generated_ids"],
            "pytorch_vs_cpu_only_exact_ids": pyt_dec["generated_ids"] == cpu_dec["generated_ids"],
            "pytorch_vs_cpu_and_ne_exact_ids": pyt_dec["generated_ids"] == ane_dec["generated_ids"],
            "cpu_only_vs_cpu_and_ne_exact_ids": cpu_dec["generated_ids"] == ane_dec["generated_ids"],
            "pytorch_vs_cpu_full_chain_exact_ids": pyt_dec["generated_ids"] == cpu_full_chain["generated_ids"],
            "pytorch_vs_ane_full_chain_exact_ids": pyt_dec["generated_ids"] == ane_full_chain["generated_ids"],
            "cpu_full_chain_vs_ane_full_chain_exact_ids": cpu_full_chain["generated_ids"] == ane_full_chain["generated_ids"],
            "cpu_decoder_on_ane_chain_encoder_exact_ids": pyt_dec["generated_ids"] == cpu_dec_on_ane_enc["generated_ids"],
            "ane_decoder_on_cpu_chain_encoder_exact_ids": pyt_dec["generated_ids"] == ane_dec_on_cpu_enc["generated_ids"],
            "inspect_step": {
                "step_index": step,
                "pytorch_topk": topk_info(pyt_dec["logits_at_steps"][step], processor.tokenizer, args.topk),
                "cpu_only_topk": topk_info(cpu_dec["logits_at_steps"][step], processor.tokenizer, args.topk),
                "cpu_and_ne_topk": topk_info(ane_dec["logits_at_steps"][step], processor.tokenizer, args.topk),
                "pytorch_vs_cpu_only": max_mean_abs_diff(pyt_dec["logits_at_steps"][step], cpu_dec["logits_at_steps"][step]),
                "pytorch_vs_cpu_and_ne": max_mean_abs_diff(pyt_dec["logits_at_steps"][step], ane_dec["logits_at_steps"][step]),
                "cpu_only_vs_cpu_and_ne": max_mean_abs_diff(cpu_dec["logits_at_steps"][step], ane_dec["logits_at_steps"][step]),
            },
        },
    }

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
