#!/usr/bin/env python3
import argparse
import json
import os
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor

from export_coreml_pure_pipeline import EncoderCore, FrontendCore, FullSeqDecoderMasked, patch_model_for_tracing


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Probe frontend/encoder/decoder boundary combinations.")
    p.add_argument("--audio", required=True)
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--artifacts-dir", default=str(root / "artifacts"))
    p.add_argument(
        "--probe",
        required=True,
        choices=[
            "torch_front_coreml_enc_coreml_dec",
            "coreml_front_coreml_enc_coreml_dec",
            "coreml_front_torch_enc_torch_dec",
        ],
    )
    p.add_argument("--compute", default="ane", choices=["cpu", "ane"])
    p.add_argument("--output-json", default=str(root / "reports" / "probe_chain_boundaries.json"))
    return p.parse_args()


def load_manifest(artifacts_dir: Path) -> Dict[str, Any]:
    return json.loads((artifacts_dir / "coreml_manifest.json").read_text())


def load_audio(path: Path, max_audio_samples: int) -> Tuple[np.ndarray, int]:
    audio, _ = sf.read(str(path), dtype="float32", always_2d=False)
    if isinstance(audio, np.ndarray) and audio.ndim == 2:
        audio = np.mean(audio, axis=1)
    audio = np.asarray(audio, dtype=np.float32)
    raw_length = min(len(audio), max_audio_samples)
    if len(audio) < max_audio_samples:
        audio = np.pad(audio, (0, max_audio_samples - len(audio)))
    else:
        audio = audio[:max_audio_samples]
    return audio.astype(np.float32), raw_length


def build_torch(model_id: str, token: Optional[str]):
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


def coreml_units(name: str):
    return ct.ComputeUnit.CPU_ONLY if name == "cpu" else ct.ComputeUnit.CPU_AND_NE


def load_coreml_models(artifacts_dir: Path, manifest: Dict[str, Any], compute: str):
    units = coreml_units(compute)
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


def make_decoder_inputs(manifest: Dict[str, Any], encoder_len: int):
    input_ids = np.full((1, manifest["decoder_max_len"]), manifest["pad_token_id"], dtype=np.int32)
    attn = np.zeros((1, manifest["decoder_max_len"]), dtype=np.int32)
    for i, tid in enumerate(manifest["prompt_ids"]):
        input_ids[0, i] = tid
        attn[0, i] = 1
    cross = np.full((1, 1, 1, manifest["max_encoder_frames"]), -1e9, dtype=np.float32)
    cross[0, 0, 0, :encoder_len] = 0.0
    return input_ids, attn, cross


def decode_coreml(model, manifest: Dict[str, Any], encoder_hidden: np.ndarray, encoder_len: int):
    input_ids, attn, cross = make_decoder_inputs(manifest, encoder_len)
    cur_idx = len(manifest["prompt_ids"]) - 1
    generated = list(manifest["prompt_ids"])
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
        next_id = int(np.argmax(logits[0, cur_idx, :]))
        generated.append(next_id)
        cur_idx += 1
        input_ids[0, cur_idx] = next_id
        attn[0, cur_idx] = 1
        if next_id == manifest["eos_token_id"] or cur_idx + 1 >= manifest["decoder_max_len"]:
            break
    return generated


def decode_torch(decoder, manifest: Dict[str, Any], encoder_hidden: np.ndarray, encoder_len: int):
    input_ids, attn, cross = make_decoder_inputs(manifest, encoder_len)
    ids_t = torch.from_numpy(input_ids)
    attn_t = torch.from_numpy(attn)
    cross_t = torch.from_numpy(cross)
    enc_t = torch.from_numpy(encoder_hidden.astype(np.float32))
    cur_idx = len(manifest["prompt_ids"]) - 1
    generated = list(manifest["prompt_ids"])
    with torch.no_grad():
        while True:
            logits = decoder(enc_t, ids_t, attn_t, cross_t)
            next_id = int(torch.argmax(logits[0, cur_idx, :]).item())
            generated.append(next_id)
            cur_idx += 1
            ids_t[0, cur_idx] = next_id
            attn_t[0, cur_idx] = 1
            if next_id == manifest["eos_token_id"] or cur_idx + 1 >= manifest["decoder_max_len"]:
                break
    return generated


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN")
    artifacts_dir = Path(args.artifacts_dir)
    manifest = load_manifest(artifacts_dir)
    audio_path = Path(args.audio).expanduser().resolve()
    audio, raw_length = load_audio(audio_path, manifest["max_audio_samples"])

    processor, torch_front, torch_enc, torch_dec = build_torch(args.model_id, token)
    coreml = load_coreml_models(artifacts_dir, manifest, args.compute)

    with torch.no_grad():
        torch_feat_t, torch_feat_len_t = torch_front(
            torch.from_numpy(audio[None, :].astype(np.float32)),
            torch.tensor([raw_length], dtype=torch.int32),
        )
    torch_feat = torch_feat_t.detach().cpu().numpy().astype(np.float32)
    torch_feat_len = torch_feat_len_t.detach().cpu().numpy().astype(np.int32)

    if args.probe == "torch_front_coreml_enc_coreml_dec":
        enc_h, enc_len = coreml_encoder_predict(coreml["encoder"], manifest, torch_feat, torch_feat_len)
        generated = decode_coreml(coreml["decoder"], manifest, enc_h.astype(np.float32), int(round(float(np.asarray(enc_len)[0]))))
    elif args.probe == "coreml_front_coreml_enc_coreml_dec":
        feat, feat_len = coreml_frontend_predict(coreml["frontend"], manifest, audio, raw_length)
        enc_h, enc_len = coreml_encoder_predict(coreml["encoder"], manifest, feat, np.asarray(feat_len, dtype=np.int32))
        generated = decode_coreml(coreml["decoder"], manifest, enc_h.astype(np.float32), int(round(float(np.asarray(enc_len)[0]))))
    else:
        feat, feat_len = coreml_frontend_predict(coreml["frontend"], manifest, audio, raw_length)
        with torch.no_grad():
            enc_h_t, enc_len_t = torch_enc(
                torch.from_numpy(feat.astype(np.float32)),
                torch.from_numpy(np.asarray(feat_len, dtype=np.int32)),
            )
        generated = decode_torch(
            torch_dec,
            manifest,
            enc_h_t.detach().cpu().numpy().astype(np.float32),
            int(round(float(enc_len_t.detach().cpu().numpy()[0]))),
        )

    text = processor.tokenizer.decode(generated, skip_special_tokens=True).strip()
    payload = {
        "audio_file": str(audio_path),
        "probe": args.probe,
        "compute_mode": args.compute,
        "generated_ids": generated,
        "decoded_text": text,
    }
    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
