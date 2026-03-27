#!/usr/bin/env python3
import argparse
import inspect
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, Tuple

import coremltools as ct
import numpy as np
import soundfile as sf
import torch
from transformers import AutoModelForSpeechSeq2Seq, AutoProcessor


DEFAULT_MODEL = "CohereLabs/cohere-transcribe-03-2026"


class FirstStepWrapper(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.decoder = model.transf_decoder
        self.log_softmax = model.log_softmax

    def forward(self, encoder_hidden_states: torch.Tensor, input_ids: torch.Tensor) -> torch.Tensor:
        input_ids = input_ids.to(torch.int64)
        bsz, tgt_len = input_ids.shape
        device = input_ids.device
        dtype = encoder_hidden_states.dtype

        positions = torch.arange(tgt_len, device=device).unsqueeze(0).expand(bsz, -1)
        self_attention_mask = torch.zeros((bsz, 1, tgt_len, tgt_len), device=device, dtype=dtype)
        cross_attention_mask = torch.zeros(
            (bsz, 1, 1, encoder_hidden_states.shape[1]), device=device, dtype=dtype
        )
        outputs, _ = self.decoder(
            input_ids=input_ids,
            positions=positions,
            encoder_hidden_states=encoder_hidden_states,
            self_attention_mask=self_attention_mask,
            cross_attention_mask=cross_attention_mask,
            past_key_values=None,
            cache_position=None,
            kv_seq_len=None,
        )
        logits = self.log_softmax(outputs)
        return logits[:, -1, :]


def pick_device() -> str:
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def load_audio_mono(path: str) -> Tuple[np.ndarray, int]:
    data, sr = sf.read(path, dtype="float32", always_2d=False)
    if data.ndim == 2:
        data = np.mean(data, axis=1)
    return data.astype(np.float32), sr


def prepare_inputs(processor: Any, audio_path: str) -> Dict[str, torch.Tensor]:
    audio, sr = load_audio_mono(audio_path)
    proc_out = processor(audio=audio, sampling_rate=sr, return_tensors="pt")
    if "input_features" not in proc_out:
        raise RuntimeError("Processor output missing input_features.")
    input_features = proc_out["input_features"].to(torch.float32)

    bos = processor.tokenizer.bos_token_id
    if bos is None:
        bos = processor.tokenizer.cls_token_id
    if bos is None:
        raise RuntimeError("Tokenizer has no BOS/CLS token id for decoder bootstrap.")
    decoder_input_ids = torch.tensor([[bos]], dtype=torch.int64)
    return {"input_features": input_features, "decoder_input_ids": decoder_input_ids}


def save_swift_input_json(
    path: Path,
    encoder_hidden_states: np.ndarray,
    decoder_input_ids: np.ndarray,
    ref_top_token: int,
    tokenizer_meta: Dict[str, Any],
) -> None:
    payload = {
        "encoder_hidden_states_shape": list(encoder_hidden_states.shape),
        "encoder_hidden_states_flat": encoder_hidden_states.reshape(-1).tolist(),
        "input_ids_shape": list(decoder_input_ids.shape),
        "input_ids_flat": decoder_input_ids.reshape(-1).tolist(),
        "reference_top_token": int(ref_top_token),
        "tokenizer": tokenizer_meta,
    }
    with path.open("w", encoding="utf-8") as f:
        json.dump(payload, f)


def run(args: argparse.Namespace) -> Dict[str, Any]:
    token = os.getenv("HF_TOKEN")
    device = pick_device()

    processor = AutoProcessor.from_pretrained(args.model_id, trust_remote_code=True, token=token)
    model = AutoModelForSpeechSeq2Seq.from_pretrained(args.model_id, trust_remote_code=True, token=token).to(device)
    model.eval()

    with open(args.baseline_json, "r", encoding="utf-8") as f:
        baseline = json.load(f)
    audio_file = baseline["audio_file_local"]

    inputs = prepare_inputs(processor, audio_file)
    input_features = inputs["input_features"].to(device)
    decoder_input_ids = inputs["decoder_input_ids"].to(device).to(torch.int32)

    with torch.no_grad():
        encoder_hidden_states, _ = model.encoder(input_features, None)
        if model.encoder_decoder_proj is not None:
            encoder_hidden_states = model.encoder_decoder_proj(encoder_hidden_states)
    encoder_hidden_states = encoder_hidden_states.clone()

    wrapper = FirstStepWrapper(model).to(device).eval()

    with torch.no_grad():
        ref_logits = wrapper(encoder_hidden_states, decoder_input_ids).detach().cpu().numpy()
    ref_top_token = int(np.argmax(ref_logits[0]))

    traced = torch.jit.trace(
        wrapper,
        (encoder_hidden_states, decoder_input_ids),
        strict=False,
    )

    convert_start = time.perf_counter()
    mlmodel = ct.convert(
        traced,
        convert_to="mlprogram",
        inputs=[
            ct.TensorType(name="encoder_hidden_states", shape=tuple(encoder_hidden_states.shape), dtype=np.float32),
            ct.TensorType(name="input_ids", shape=tuple(decoder_input_ids.shape), dtype=np.int32),
        ],
        minimum_deployment_target=ct.target.macOS13,
        compute_units=ct.ComputeUnit.CPU_AND_NE,
    )
    convert_seconds = time.perf_counter() - convert_start

    out_mlpackage = Path(args.output_mlpackage)
    out_mlpackage.parent.mkdir(parents=True, exist_ok=True)
    mlmodel.save(str(out_mlpackage))

    quantized_path = None
    if args.quantize:
        try:
            from coremltools.optimize.coreml import OpLinearQuantizerConfig, OptimizationConfig, linear_quantize_weights

            q_cfg = OptimizationConfig(global_config=OpLinearQuantizerConfig(mode="linear_symmetric"))
            q_model = linear_quantize_weights(mlmodel, config=q_cfg)
            quantized_path = out_mlpackage.with_name(out_mlpackage.stem + "_int8.mlpackage")
            q_model.save(str(quantized_path))
        except Exception as e:  # pragma: no cover - best-effort optimization path
            print(f"[warn] quantization skipped: {e}")

    npz_path = Path(args.output_npz)
    npz_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        npz_path,
        encoder_hidden_states=encoder_hidden_states.detach().cpu().numpy().astype(np.float32),
        input_ids=decoder_input_ids.detach().cpu().numpy().astype(np.int32),
        ref_logits=ref_logits.astype(np.float32),
    )

    swift_input_path = Path(args.output_swift_input_json)
    tokenizer = processor.tokenizer
    vocab_size = int(getattr(tokenizer, "vocab_size"))
    id_to_token = tokenizer.convert_ids_to_tokens(list(range(vocab_size)))
    tokenizer_meta = {
        "bos_token_id": int(tokenizer.bos_token_id) if tokenizer.bos_token_id is not None else None,
        "eos_token_id": int(tokenizer.eos_token_id) if tokenizer.eos_token_id is not None else None,
        "pad_token_id": int(tokenizer.pad_token_id) if tokenizer.pad_token_id is not None else None,
        "vocab_size": vocab_size,
        "id_to_token": id_to_token,
    }
    save_swift_input_json(
        swift_input_path,
        encoder_hidden_states.detach().cpu().numpy().astype(np.float32),
        decoder_input_ids.detach().cpu().numpy().astype(np.int32),
        ref_top_token=ref_top_token,
        tokenizer_meta=tokenizer_meta,
    )

    # Reference greedy decode for full-sequence parity checks in Swift.
    with torch.no_grad():
        bos = tokenizer.bos_token_id
        if bos is None:
            bos = tokenizer.cls_token_id
        if bos is None:
            raise RuntimeError("Tokenizer has no BOS/CLS token id for reference generation.")
        model.generation_config.decoder_start_token_id = bos
        model.generation_config.bos_token_id = bos
        if tokenizer.eos_token_id is not None:
            model.generation_config.eos_token_id = tokenizer.eos_token_id
        if tokenizer.pad_token_id is not None:
            model.generation_config.pad_token_id = tokenizer.pad_token_id
        dec_start = torch.tensor([[bos]], dtype=torch.long, device=device)
        generated_ids = model.generate(
            input_features=input_features,
            decoder_input_ids=dec_start,
            max_new_tokens=args.max_new_tokens,
            do_sample=False,
        )
    ref_ids = generated_ids[0].detach().cpu().tolist()
    ref_text = tokenizer.decode(ref_ids, skip_special_tokens=True)
    reference_decode = {
        "max_new_tokens": args.max_new_tokens,
        "generated_ids": ref_ids,
        "decoded_text": ref_text,
    }
    ref_decode_path = Path(args.output_reference_decode_json)
    ref_decode_path.parent.mkdir(parents=True, exist_ok=True)
    with ref_decode_path.open("w", encoding="utf-8") as f:
        json.dump(reference_decode, f, indent=2)

    result = {
        "model_id": args.model_id,
        "device": device,
        "forward_signature": str(inspect.signature(model.forward)),
        "coreml_package": str(out_mlpackage),
        "quantized_coreml_package": str(quantized_path) if quantized_path else None,
        "npz_inputs": str(npz_path),
        "swift_input_json": str(swift_input_path),
        "reference_decode_json": str(ref_decode_path),
        "reference_top_token": ref_top_token,
        "conversion_seconds": convert_seconds,
    }
    return result


def parse_args() -> argparse.Namespace:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description="Export first-step decoder CoreML model for Cohere ASR.")
    p.add_argument("--model-id", default=DEFAULT_MODEL)
    p.add_argument("--baseline-json", default=str(root / "artifacts" / "baseline_result.json"))
    p.add_argument("--output-mlpackage", default=str(root / "artifacts" / "cohere_first_step.mlpackage"))
    p.add_argument("--output-npz", default=str(root / "artifacts" / "coreml_inputs.npz"))
    p.add_argument("--output-swift-input-json", default=str(root / "artifacts" / "swift_input.json"))
    p.add_argument("--output-reference-decode-json", default=str(root / "artifacts" / "reference_decode.json"))
    p.add_argument("--report-json", default=str(root / "reports" / "export_report.json"))
    p.add_argument("--max-new-tokens", type=int, default=96)
    p.add_argument("--quantize", action="store_true", help="Attempt int8 linear quantization.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    report = run(args)
    report_path = Path(args.report_json)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with report_path.open("w", encoding="utf-8") as f:
        json.dump(report, f, indent=2)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
