#!/usr/bin/env python3
import argparse
import json
import os
import types
from pathlib import Path
from typing import Any, Dict, List

import coremltools as ct
import numpy as np
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"


def _build_stft_conv_weights(n_fft: int, win_length: int, window: torch.Tensor) -> torch.Tensor:
    """Build conv1d weights that implement windowed DFT.

    Returns weights of shape [2 * n_bins, 1, n_fft] where the first n_bins
    filters are the cosine (real) components and the next n_bins are the sine
    (imaginary) components, each pre-multiplied by the analysis window.
    """
    n_bins = n_fft // 2 + 1
    full_window = torch.zeros(n_fft, dtype=torch.float64)
    pad_left = (n_fft - win_length) // 2
    full_window[pad_left : pad_left + win_length] = window.to(torch.float64)

    k = torch.arange(n_bins, dtype=torch.float64)
    n = torch.arange(n_fft, dtype=torch.float64)
    angles = -2.0 * torch.pi * k.unsqueeze(1) * n.unsqueeze(0) / n_fft  # [n_bins, n_fft]
    cos_part = torch.cos(angles) * full_window.unsqueeze(0)  # [n_bins, n_fft]
    sin_part = torch.sin(angles) * full_window.unsqueeze(0)  # [n_bins, n_fft]
    weights = torch.cat([cos_part, sin_part], dim=0)  # [2*n_bins, n_fft]
    return weights.unsqueeze(1).to(torch.float32)  # [2*n_bins, 1, n_fft]


def apply_preemph(
    audio: torch.Tensor, length: torch.Tensor, preemph: float
) -> torch.Tensor:
    """Apply pre-emphasis and zero-mask beyond valid length.

    Must be called on raw audio *before* passing to FrontendCore so that
    CoreML's graph optimizer cannot fuse the multiply with the STFT conv1d
    (which causes non-determinism).

    Args:
        audio: [B, T] raw audio samples.
        length: [B] number of valid samples per utterance.
        preemph: pre-emphasis coefficient (typically 0.97).

    Returns:
        [B, T] pre-emphasized, zero-masked audio.
    """
    x = audio.to(torch.float32)
    mask = (
        torch.arange(x.shape[1], device=x.device).unsqueeze(0)
        < length.to(torch.int32).unsqueeze(1)
    ).to(x.dtype)
    x = torch.cat((x[:, 0].unsqueeze(1), x[:, 1:] - preemph * x[:, :-1]), dim=1)
    return x * mask


class FrontendCore(torch.nn.Module):
    def __init__(
        self,
        mel_fb: torch.Tensor,  # [1, n_mels, n_fft_bins]
        window: torch.Tensor,
        n_fft: int,
        win_length: int,
        hop_length: int,
        preemph: float,
        pad_to: int,
        log_zero_guard_value: float,
        mag_power: float,
    ):
        super().__init__()
        # mel_fb [1, n_mels, n_fft_bins] → conv1d weights [n_mels, n_fft_bins, 1]
        mel_fb_f32 = mel_fb.to(torch.float32)
        self.register_buffer("mel_conv_w", mel_fb_f32.squeeze(0).unsqueeze(2))
        n_fft = int(n_fft)
        win_length = int(win_length)
        stft_weights = _build_stft_conv_weights(n_fft, win_length, window)
        self.register_buffer("stft_weights", stft_weights)  # [2*n_bins, 1, n_fft]
        self.n_fft = n_fft
        self.n_bins = n_fft // 2 + 1
        self.win_length = win_length
        self.hop_length = int(hop_length)
        self.preemph = float(preemph)
        self.pad_to = int(pad_to)
        self.log_zero_guard_value = float(log_zero_guard_value)
        self.mag_power = float(mag_power)

    def get_seq_len(self, seq_len: torch.Tensor) -> torch.Tensor:
        return torch.floor_divide(seq_len, self.hop_length).to(dtype=torch.int32)

    def _conv1d_stft_mag(self, x: torch.Tensor) -> torch.Tensor:
        """Compute STFT magnitude via conv1d (deterministic in CoreML).

        Equivalent to:
            stft = torch.stft(x, n_fft, hop_length, win_length, center=True,
                              window=window, return_complex=True, pad_mode='constant')
            mag = sqrt(real**2 + imag**2)
        """
        pad_amount = self.n_fft // 2
        x_padded = torch.nn.functional.pad(x, (pad_amount, pad_amount), mode="constant", value=0.0)
        # x_padded: [B, L_padded] -> need [B, 1, L_padded] for conv1d
        spec = torch.nn.functional.conv1d(
            x_padded.unsqueeze(1), self.stft_weights, stride=self.hop_length
        )
        # spec: [B, 2*n_bins, num_frames]
        real = spec[:, : self.n_bins, :]
        imag = spec[:, self.n_bins :, :]
        return torch.sqrt(real * real + imag * imag)  # [B, n_bins, num_frames]

    def forward(self, audio_samples: torch.Tensor, audio_length: torch.Tensor):
        """Expects pre-emphasized, zero-masked audio (see ``apply_preemph``)."""
        x = audio_samples.to(torch.float32)
        seq_len_time = audio_length.to(torch.int32)

        mag = self._conv1d_stft_mag(x)
        if self.mag_power != 1.0:
            mag = mag.pow(self.mag_power)
        mels = torch.nn.functional.conv1d(mag, self.mel_conv_w)
        mels = torch.log(mels + self.log_zero_guard_value)

        seq_len = self.get_seq_len(seq_len_time)

        # Per-utterance normalization (mean/var over valid frames).
        # Implemented as sequential loop over mel channels to avoid non-deterministic
        # large reductions in CoreML. The loop is unrolled during tracing.
        n_mels = mels.shape[1]
        max_time = mels.shape[2]
        valid_f = (torch.arange(max_time, device=mels.device).unsqueeze(0) < seq_len.unsqueeze(1)).to(mels.dtype)
        count = torch.clamp(seq_len.to(mels.dtype), min=1.0)

        normed_channels = []
        for c in range(n_mels):
            ch = mels[:, c, :]  # [B, T]
            masked = ch * valid_f  # [B, T]
            ch_sum = masked.sum(dim=-1, keepdim=True)  # [B, 1]
            ch_mean = ch_sum / count.unsqueeze(1)
            diff = (ch - ch_mean) * valid_f
            var_sum = (diff * diff).sum(dim=-1, keepdim=True)
            ch_var = var_sum / torch.clamp(count.unsqueeze(1) - 1.0, min=1.0)
            ch_std = torch.sqrt(ch_var + 1e-10) + 1e-5
            normed = (ch - ch_mean) / ch_std * valid_f
            normed_channels.append(normed.unsqueeze(1))
        mels = torch.cat(normed_channels, dim=1)

        if self.pad_to > 0:
            pad_amt = mels.shape[2] % self.pad_to
            if pad_amt != 0:
                mels = torch.nn.functional.pad(mels, (0, self.pad_to - pad_amt), value=0.0)
        return mels, seq_len


class EncoderCore(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.encoder = model.encoder
        self.proj = model.encoder_decoder_proj

    def forward(self, input_features: torch.Tensor, feature_length: torch.Tensor):
        x, length = self.encoder(input_features, feature_length)
        if self.proj is not None:
            x = self.proj(x)
        return x, length.to(torch.int32)


class FullSeqDecoderMasked(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.decoder = model.transf_decoder
        self.classifier = model.log_softmax.mlp.layer0

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,  # [1,S,H]
        input_ids: torch.Tensor,  # [1,T]
        decoder_attention_mask: torch.Tensor,  # [1,T]
        cross_attention_mask: torch.Tensor,  # [1,1,1,S]
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
            cross_attention_mask=cross_attention_mask.to(dtype=dtype),
            past_key_values=None,
            cache_position=None,
            kv_seq_len=None,
        )
        return self.classifier(outputs)


class DecoderStepCached(torch.nn.Module):
    """Single-step decoder with KV cache for fast autoregressive generation.

    Instead of recomputing the full sequence each step, this processes only
    the new token and updates the self-attention KV cache in-place.
    Cross-attention K/V are recomputed from encoder_hidden_states each step
    (small cost since Q is a single token).
    """

    def __init__(self, model: torch.nn.Module, max_len: int):
        super().__init__()
        self.embedding = model.transf_decoder._embedding
        self.decoder_layers = model.transf_decoder._decoder.layers
        self.final_ln = model.transf_decoder._decoder.final_layer_norm
        self.classifier = model.log_softmax.mlp.layer0
        self.max_len = max_len
        first_attn = self.decoder_layers[0].first_sub_layer
        self.num_heads = first_attn.num_heads
        self.head_dim = first_attn.head_dim

    def _manual_attn(self, q, k, v, scale: float, mask=None):
        scores = torch.matmul(q, k.transpose(-1, -2)) * scale
        if mask is not None:
            scores = scores + mask
        return torch.matmul(torch.softmax(scores, dim=-1), v)

    def _update_cache(self, cache, layer_idx, new_kv, step_idx):
        cur = cache[layer_idx : layer_idx + 1]  # [1, H, T, D]
        oh = torch.nn.functional.one_hot(step_idx, num_classes=self.max_len).to(dtype=cur.dtype)
        oh = oh.view(1, 1, self.max_len, 1)
        return cur * (1.0 - oh) + new_kv.expand(-1, -1, self.max_len, -1) * oh

    def forward(
        self,
        encoder_hidden_states: torch.Tensor,  # [1, S, H]
        input_id: torch.Tensor,               # [1, 1]  int32
        cache_k: torch.Tensor,                # [L, H, T, D]
        cache_v: torch.Tensor,                # [L, H, T, D]
        step: torch.Tensor,                   # [1]     int32
        cross_attention_mask: torch.Tensor,    # [1, 1, 1, S]
    ):
        step_idx = step.to(torch.int64).reshape(1)
        hidden = self.embedding(input_id.to(torch.int64), step_idx.view(1, 1))
        new_ck, new_cv = [], []

        for i, layer in enumerate(self.decoder_layers):
            residual = hidden
            x = layer.layer_norm_1(hidden)
            sa = layer.first_sub_layer
            q = sa._reshape(sa.query_net(x))
            k_new = sa._reshape(sa.key_net(x))
            v_new = sa._reshape(sa.value_net(x))

            k_upd = self._update_cache(cache_k, i, k_new, step_idx)
            v_upd = self._update_cache(cache_v, i, v_new, step_idx)

            pos = torch.arange(self.max_len, device=step.device, dtype=step_idx.dtype).view(1, 1, 1, self.max_len)
            attn_mask = torch.where(
                pos <= step_idx.view(1, 1, 1, 1),
                torch.zeros((1, 1, 1, self.max_len), device=step.device, dtype=hidden.dtype),
                torch.full((1, 1, 1, self.max_len), -1e9, device=step.device, dtype=hidden.dtype),
            )
            out = self._manual_attn(q, k_upd, v_upd, sa.scale, mask=attn_mask)
            out = out.transpose(1, 2).contiguous().view(1, 1, sa.hidden_size)
            hidden = residual + sa.out_projection(out)

            residual = hidden
            x = layer.layer_norm_2(hidden)
            ca = layer.second_sub_layer
            q = ca._reshape(ca.query_net(x))
            k = ca._reshape(ca.key_net(encoder_hidden_states))
            v = ca._reshape(ca.value_net(encoder_hidden_states))
            out = self._manual_attn(q, k, v, ca.scale, mask=cross_attention_mask.to(dtype=hidden.dtype))
            out = out.transpose(1, 2).contiguous().view(1, 1, ca.hidden_size)
            hidden = residual + ca.out_projection(out)

            residual = hidden
            x = layer.layer_norm_3(hidden)
            hidden = residual + layer.third_sub_layer(x)

            new_ck.append(k_upd.squeeze(0))
            new_cv.append(v_upd.squeeze(0))

        hidden = self.final_ln(hidden)
        logits = self.classifier(hidden)[:, -1, :]  # [1, V]
        return logits, torch.stack(new_ck, dim=0), torch.stack(new_cv, dim=0)


def patch_model_for_tracing(model: torch.nn.Module) -> None:
    pre_encode = model.encoder.pre_encode
    pre_encode._needs_conv_split = types.MethodType(lambda self, x: False, pre_encode)
    pre_encode._conv_split_by_batch = types.MethodType(lambda self, x, lengths: self.conv(x, lengths), pre_encode)

    pos_enc = model.encoder.pos_enc

    def _materialize_pe(self, length: int, device: torch.device, dtype: torch.dtype):
        needed_size = 2 * length - 1
        if hasattr(self, "pe") and self.pe.size(1) >= needed_size:
            if self.pe.device != device:
                self.pe = self.pe.to(device=device)
            if self.pe.dtype != dtype:
                self.pe = self.pe.to(dtype=dtype)
            return
        effective_length = max(length, self.max_len)
        positions = torch.arange(
            effective_length - 1, -effective_length, -1, dtype=torch.float32, device=device
        ).unsqueeze(1)
        pe = self._create_pe(positions=positions, dtype=dtype)
        if hasattr(self, "pe"):
            self.pe = pe
        else:
            self.register_buffer("pe", pe, persistent=False)

    pos_enc._materialize_pe = types.MethodType(_materialize_pe, pos_enc)


def _pick_output_names(spec: Any) -> List[str]:
    return [o.name for o in spec.description.output]


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Export pure CoreML Cohere ASR pipeline: frontend + encoder + decoder.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--language", default="en")
    p.add_argument("--punctuation", action="store_true", default=True)
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--max-audio-seconds", type=float, default=30.0)
    p.add_argument("--artifacts-dir", default=str(root / "artifacts"))
    p.add_argument("--report-json", default=str(root / "reports" / "export_pure_coreml_report.json"))
    return p.parse_args()


def main() -> None:
    args = parse_args()
    token = os.getenv("HF_TOKEN")

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, token=token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, trust_remote_code=True, token=token).eval()
    patch_model_for_tracing(model)

    fe = processor.feature_extractor
    fb = fe.filterbank
    sample_rate = int(fe.sampling_rate)
    max_audio_samples = int(sample_rate * float(args.max_audio_seconds))

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

    preemph_coeff = frontend.preemph
    dummy_audio_raw = torch.zeros((1, max_audio_samples), dtype=torch.float32)
    dummy_audio_len = torch.tensor([max_audio_samples], dtype=torch.int32)
    dummy_audio = apply_preemph(dummy_audio_raw, dummy_audio_len, preemph_coeff)
    with torch.no_grad():
        dummy_features, dummy_feat_len = frontend(dummy_audio, dummy_audio_len)
    max_feature_frames = int(dummy_features.shape[2])

    encoder = EncoderCore(model).eval()
    with torch.no_grad():
        dummy_encoder_hidden, dummy_encoder_len = encoder(dummy_features, dummy_feat_len)
    max_encoder_frames = int(dummy_encoder_hidden.shape[1])
    encoder_hidden_size = int(dummy_encoder_hidden.shape[2])

    prompt_text = model.build_prompt(language=args.language, punctuation=args.punctuation)
    prompt_inputs = processor(audio=[np.zeros(1600, dtype=np.float32)], text=[prompt_text], sampling_rate=sample_rate, return_tensors="pt")
    prompt_ids = prompt_inputs["input_ids"][0].to(torch.int32)
    decoder_max_len = int(prompt_ids.shape[0]) + int(args.max_new_tokens) + 2

    decoder = FullSeqDecoderMasked(model).eval()
    dummy_input_ids = torch.full((1, decoder_max_len), fill_value=int(processor.tokenizer.pad_token_id), dtype=torch.int32)
    dummy_dec_mask = torch.zeros((1, decoder_max_len), dtype=torch.int32)
    dummy_cross_mask = torch.zeros((1, 1, 1, max_encoder_frames), dtype=torch.float32)
    with torch.no_grad():
        _ = decoder(dummy_encoder_hidden, dummy_input_ids, dummy_dec_mask, dummy_cross_mask)

    # Cached (single-step) decoder
    dec_layers = model.transf_decoder._decoder.layers
    num_dec_layers = len(dec_layers)
    num_heads = dec_layers[0].first_sub_layer.num_heads
    head_dim = dec_layers[0].first_sub_layer.head_dim
    decoder_cached = DecoderStepCached(model, max_len=decoder_max_len).eval()
    dummy_input_id = torch.tensor([[0]], dtype=torch.int32)
    dummy_cache_k = torch.zeros((num_dec_layers, num_heads, decoder_max_len, head_dim), dtype=torch.float32)
    dummy_cache_v = torch.zeros((num_dec_layers, num_heads, decoder_max_len, head_dim), dtype=torch.float32)
    dummy_step = torch.tensor([0], dtype=torch.int32)
    dummy_cached_cross = torch.zeros((1, 1, 1, max_encoder_frames), dtype=torch.float32)
    with torch.no_grad():
        _ = decoder_cached(dummy_encoder_hidden, dummy_input_id, dummy_cache_k, dummy_cache_v, dummy_step, dummy_cached_cross)

    front_traced = torch.jit.trace(frontend, (dummy_audio, dummy_audio_len), strict=False, check_trace=False)
    enc_traced = torch.jit.trace(encoder, (dummy_features, dummy_feat_len), strict=False, check_trace=False)
    dec_traced = torch.jit.trace(
        decoder,
        (dummy_encoder_hidden, dummy_input_ids, dummy_dec_mask, dummy_cross_mask),
        strict=False,
        check_trace=False,
    )
    dec_cached_traced = torch.jit.trace(
        decoder_cached,
        (dummy_encoder_hidden, dummy_input_id, dummy_cache_k, dummy_cache_v, dummy_step, dummy_cached_cross),
        strict=False,
        check_trace=False,
    )

    artifacts_dir = Path(args.artifacts_dir)
    artifacts_dir.mkdir(parents=True, exist_ok=True)

    frontend_pkg = artifacts_dir / "cohere_frontend.mlpackage"
    encoder_pkg = artifacts_dir / "cohere_encoder.mlpackage"
    decoder_pkg = artifacts_dir / "cohere_decoder_fullseq_masked.mlpackage"
    decoder_cached_pkg = artifacts_dir / "cohere_decoder_cached.mlpackage"
    manifest_path = artifacts_dir / "coreml_manifest.json"

    frontend_model = ct.convert(
        front_traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT32,
        inputs=[
            ct.TensorType(name="audio_samples", shape=tuple(dummy_audio.shape), dtype=np.float32),
            ct.TensorType(name="audio_length", shape=tuple(dummy_audio_len.shape), dtype=np.int32),
        ],
    )
    frontend_model.save(str(frontend_pkg))

    encoder_model = ct.convert(
        enc_traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT32,
        inputs=[
            ct.TensorType(name="input_features", shape=tuple(dummy_features.shape), dtype=np.float32),
            ct.TensorType(name="feature_length", shape=tuple(dummy_feat_len.shape), dtype=np.int32),
        ],
    )
    encoder_model.save(str(encoder_pkg))

    decoder_model = ct.convert(
        dec_traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT32,
        inputs=[
            ct.TensorType(name="encoder_hidden_states", shape=tuple(dummy_encoder_hidden.shape), dtype=np.float32),
            ct.TensorType(name="input_ids", shape=tuple(dummy_input_ids.shape), dtype=np.int32),
            ct.TensorType(name="decoder_attention_mask", shape=tuple(dummy_dec_mask.shape), dtype=np.int32),
            ct.TensorType(name="cross_attention_mask", shape=tuple(dummy_cross_mask.shape), dtype=np.float32),
        ],
    )
    decoder_model.save(str(decoder_pkg))

    decoder_cached_model = ct.convert(
        dec_cached_traced,
        convert_to="mlprogram",
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
        compute_precision=ct.precision.FLOAT32,
        inputs=[
            ct.TensorType(name="encoder_hidden_states", shape=tuple(dummy_encoder_hidden.shape), dtype=np.float32),
            ct.TensorType(name="input_id", shape=(1, 1), dtype=np.int32),
            ct.TensorType(name="cache_k", shape=tuple(dummy_cache_k.shape), dtype=np.float32),
            ct.TensorType(name="cache_v", shape=tuple(dummy_cache_v.shape), dtype=np.float32),
            ct.TensorType(name="step", shape=(1,), dtype=np.int32),
            ct.TensorType(name="cross_attention_mask", shape=(1, 1, 1, max_encoder_frames), dtype=np.float32),
        ],
    )
    decoder_cached_model.save(str(decoder_cached_pkg))
    cached_output_names = _pick_output_names(decoder_cached_model.get_spec())
    cache_elem_count = int(np.prod(dummy_cache_k.shape))
    logits_out_name = None
    cache_k_out_name = None
    cache_v_out_name = None
    for oname in cached_output_names:
        ospec = decoder_cached_model.get_spec()
        for o in ospec.description.output:
            if o.name == oname:
                shape = list(o.type.multiArrayType.shape)
                if int(np.prod(shape)) == cache_elem_count:
                    if cache_k_out_name is None:
                        cache_k_out_name = oname
                    else:
                        cache_v_out_name = oname
                else:
                    logits_out_name = oname

    manifest: Dict[str, Any] = {
        "model_id": args.model_id,
        "sample_rate": sample_rate,
        "preemph": preemph_coeff,
        "max_audio_samples": max_audio_samples,
        "max_feature_frames": max_feature_frames,
        "max_encoder_frames": max_encoder_frames,
        "encoder_hidden_size": encoder_hidden_size,
        "decoder_max_len": decoder_max_len,
        "default_max_new_tokens": int(args.max_new_tokens),
        "prompt_ids": [int(x) for x in prompt_ids.tolist()],
        "eos_token_id": int(processor.tokenizer.eos_token_id) if processor.tokenizer.eos_token_id is not None else None,
        "pad_token_id": int(processor.tokenizer.pad_token_id) if processor.tokenizer.pad_token_id is not None else None,
        "id_to_token": processor.tokenizer.convert_ids_to_tokens(list(range(processor.tokenizer.vocab_size))),
        "frontend": {
            "package": frontend_pkg.name,
            "inputs": ["audio_samples", "audio_length"],
            "outputs": _pick_output_names(frontend_model.get_spec()),
        },
        "encoder": {
            "package": encoder_pkg.name,
            "inputs": ["input_features", "feature_length"],
            "outputs": _pick_output_names(encoder_model.get_spec()),
        },
        "decoder": {
            "package": decoder_pkg.name,
            "inputs": ["encoder_hidden_states", "input_ids", "decoder_attention_mask", "cross_attention_mask"],
            "outputs": _pick_output_names(decoder_model.get_spec()),
        },
        "decoder_cached": {
            "package": decoder_cached_pkg.name,
            "inputs": ["encoder_hidden_states", "input_id", "cache_k", "cache_v", "step", "cross_attention_mask"],
            "outputs": cached_output_names,
            "logits_output": logits_out_name,
            "cache_k_output": cache_k_out_name,
            "cache_v_output": cache_v_out_name,
            "num_layers": num_dec_layers,
            "num_heads": num_heads,
            "head_dim": head_dim,
        },
    }
    with manifest_path.open("w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)

    report = {
        "frontend_package": str(frontend_pkg),
        "encoder_package": str(encoder_pkg),
        "decoder_package": str(decoder_pkg),
        "decoder_cached_package": str(decoder_cached_pkg),
        "manifest_json": str(manifest_path),
        "max_audio_samples": max_audio_samples,
        "max_feature_frames": max_feature_frames,
        "max_encoder_frames": max_encoder_frames,
        "decoder_max_len": decoder_max_len,
    }
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
