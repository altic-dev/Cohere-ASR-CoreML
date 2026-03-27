#!/usr/bin/env python3
"""Stage-by-stage diff of PyTorch FrontendCore vs CoreML frontend mlpackage.

Runs both pipelines on the same audio and reports numeric divergence at each
sub-stage of the frontend: pre-emphasis, STFT magnitude, mel+log, normalization,
and final output.
"""
import argparse
import json
import os
from pathlib import Path

import coremltools as ct
import numpy as np
import torch

from export_coreml_pure_pipeline import FrontendCore, apply_preemph, patch_model_for_tracing
from probe_chain_boundaries import load_manifest, load_audio
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


def build_frontend(model_id: str, token):
    processor = AutoProcessor.from_pretrained(model_id, trust_remote_code=True, token=token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(model_id, trust_remote_code=True, token=token).eval()
    patch_model_for_tracing(model)
    fe = processor.feature_extractor
    fb = fe.filterbank
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
    return frontend


def pytorch_stages(frontend: FrontendCore, audio_t: torch.Tensor, audio_len_t: torch.Tensor):
    """Run FrontendCore forward step-by-step, returning intermediates.

    Expects pre-emphasized, zero-masked audio as input (same as FrontendCore).
    """
    x = audio_t.to(torch.float32)
    seq_len_time = audio_len_t.to(torch.int32)
    preemph_out = x.clone()

    mag = frontend._conv1d_stft_mag(x)
    if frontend.mag_power != 1.0:
        mag = mag.pow(frontend.mag_power)
    stft_mag = mag.clone()

    mels = torch.nn.functional.conv1d(mag, frontend.mel_conv_w)
    mels = torch.log(mels + frontend.log_zero_guard_value)
    mel_log = mels.clone()

    seq_len = frontend.get_seq_len(seq_len_time)
    max_time = mels.shape[2]
    valid = torch.arange(max_time, device=mels.device).unsqueeze(0) < seq_len.unsqueeze(1)

    mean_num = torch.where(valid.unsqueeze(1), mels, torch.zeros_like(mels)).sum(axis=2)
    mean_den = torch.clamp(seq_len.to(mels.dtype), min=1.0)
    mean = mean_num / mean_den.unsqueeze(1)
    var = torch.where(valid.unsqueeze(1), (mels - mean.unsqueeze(2)) ** 2, torch.zeros_like(mels)).sum(axis=2)
    var = var / torch.clamp(mean_den.unsqueeze(1) - 1.0, min=1.0)
    std = torch.sqrt(var)
    std = std.masked_fill(std.isnan(), 0.0) + 1e-5
    mels = (mels - mean.unsqueeze(2)) / std.unsqueeze(2)
    norm_out = mels.clone()

    mels = mels.masked_fill(~valid.unsqueeze(1), 0.0)
    if frontend.pad_to > 0:
        pad_amt = mels.shape[2] % frontend.pad_to
        if pad_amt != 0:
            mels = torch.nn.functional.pad(mels, (0, frontend.pad_to - pad_amt), value=0.0)
    final = mels.clone()

    return {
        "preemph": preemph_out.numpy(),
        "stft_mag": stft_mag.numpy(),
        "mel_log": mel_log.numpy(),
        "norm": norm_out.numpy(),
        "final": final.numpy(),
        "seq_len": seq_len.numpy(),
    }


def coreml_final(manifest, coreml_model, audio: np.ndarray, raw_length: int):
    out = coreml_model.predict({
        manifest["frontend"]["inputs"][0]: audio[None, :],
        manifest["frontend"]["inputs"][1]: np.array([raw_length], dtype=np.int32),
    })
    feat = np.asarray(out[manifest["frontend"]["outputs"][0]], dtype=np.float32)
    feat_len = np.asarray(out[manifest["frontend"]["outputs"][1]])
    return feat, feat_len


def diff_stats(a: np.ndarray, b: np.ndarray, label: str):
    d = np.abs(a.astype(np.float64) - b.astype(np.float64))
    return {
        "label": label,
        "shape_a": list(a.shape),
        "shape_b": list(b.shape),
        "max_abs_diff": float(d.max()),
        "mean_abs_diff": float(d.mean()),
        "median_abs_diff": float(np.median(d)),
        "frac_gt_1e-3": float((d > 1e-3).mean()),
        "frac_gt_0.1": float((d > 0.1).mean()),
        "frac_gt_1.0": float((d > 1.0).mean()),
    }


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--audio", required=True)
    parser.add_argument("--artifacts-dir", default=str(Path(__file__).resolve().parents[1] / "artifacts"))
    parser.add_argument("--output-json", default=str(Path(__file__).resolve().parents[1] / "reports" / "diff_frontend_stages.json"))
    args = parser.parse_args()

    token = os.getenv("HF_TOKEN")
    artifacts_dir = Path(args.artifacts_dir)
    manifest = json.loads((artifacts_dir / "coreml_manifest.json").read_text())

    audio, raw_length = load_audio(Path(args.audio), manifest["max_audio_samples"])
    frontend = build_frontend(manifest["model_id"], token)

    audio_t = torch.from_numpy(audio[None, :].astype(np.float32))
    audio_len_t = torch.tensor([raw_length], dtype=torch.int32)

    preemph_coeff = manifest.get("preemph", 0.97)
    preemph_audio_t = apply_preemph(audio_t, audio_len_t, preemph_coeff)
    preemph_audio_np = preemph_audio_t.numpy()[0]

    with torch.no_grad():
        pt = pytorch_stages(frontend, preemph_audio_t, audio_len_t)

    coreml_model = ct.models.MLModel(
        str(artifacts_dir / manifest["frontend"]["package"]),
        compute_units=ct.ComputeUnit.CPU_ONLY,
    )
    cm_feat, cm_len = coreml_final(manifest, coreml_model, preemph_audio_np, raw_length)

    report = {
        "audio": args.audio,
        "raw_length": raw_length,
        "pytorch_seq_len": int(pt["seq_len"][0]),
        "coreml_feat_len": int(cm_len.reshape(-1)[0]),
        "stages": {},
    }

    report["stages"]["final_vs_coreml"] = diff_stats(pt["final"], cm_feat, "PyTorch final vs CoreML final")

    traced_frontend = torch.jit.trace(frontend, (preemph_audio_t, audio_len_t), strict=False, check_trace=False)
    with torch.no_grad():
        traced_feat, traced_len = traced_frontend(preemph_audio_t, audio_len_t)
    traced_feat_np = traced_feat.numpy()
    report["stages"]["pytorch_eager_vs_jit_traced"] = diff_stats(pt["final"], traced_feat_np, "PyTorch eager vs JIT traced")
    report["stages"]["jit_traced_vs_coreml"] = diff_stats(traced_feat_np, cm_feat, "JIT traced vs CoreML final")

    max_audio_samples = manifest["max_audio_samples"]
    dummy_audio = torch.zeros((1, max_audio_samples), dtype=torch.float32)
    dummy_len = torch.tensor([max_audio_samples], dtype=torch.int32)
    traced_maxdummy = torch.jit.trace(frontend, (dummy_audio, dummy_len), strict=False, check_trace=False)
    with torch.no_grad():
        maxdummy_feat, maxdummy_len = traced_maxdummy(preemph_audio_t, audio_len_t)
    maxdummy_feat_np = maxdummy_feat.numpy()
    report["stages"]["pytorch_eager_vs_jit_maxdummy"] = diff_stats(pt["final"], maxdummy_feat_np, "PyTorch eager vs JIT traced (max-length dummy)")
    report["stages"]["jit_maxdummy_vs_coreml"] = diff_stats(maxdummy_feat_np, cm_feat, "JIT max-dummy vs CoreML final")

    out = Path(args.output_json)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
