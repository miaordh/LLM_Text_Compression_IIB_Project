import argparse
import gc
import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_codec_drift_test import DriftAwareLLMCodec, DriftTestCodecConfig, cleanup_codec, summarize_drift_rows


if os.environ.get("TRANSFORMERS_CACHE") and not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
    os.environ.pop("TRANSFORMERS_CACHE", None)


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _preferred_accelerator_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(dtype_name: str, device: str):
    if dtype_name == "auto":
        if device == "cuda":
            return torch.float16
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[dtype_name]


def _load_model_tokenizer(settings: Dict[str, Any], device: str):
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(settings.get("trust_remote_code", False)),
        "low_cpu_mem_usage": True,
    }
    revision = settings.get("revision")
    if revision:
        model_kwargs["revision"] = revision

    dtype = _resolve_torch_dtype(str(settings.get("torch_dtype", "auto")), device)
    if dtype != "auto":
        model_kwargs["dtype"] = dtype

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_id"],
        trust_remote_code=bool(settings.get("trust_remote_code", False)),
        revision=revision,
    )
    if bool(settings.get("ignore_model_max_length_warning", True)):
        tokenizer.model_max_length = int(1e30)

    try:
        model = AutoModelForCausalLM.from_pretrained(settings["model_id"], **model_kwargs)
    except TypeError:
        if "dtype" not in model_kwargs:
            raise
        fallback_kwargs = dict(model_kwargs)
        fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
        model = AutoModelForCausalLM.from_pretrained(settings["model_id"], **fallback_kwargs)

    return tokenizer, model


def _build_codec_config(
    trial: Dict[str, Any],
    diagnostics_prefix: Optional[str],
    trace_prefix: Optional[str],
    drift_prefix: Optional[str],
) -> DriftTestCodecConfig:
    return DriftTestCodecConfig(
        precision=int(trial.get("precision", 32)),
        slots=int(trial.get("slots", 1 << 24)),
        context_window=int(trial.get("context_window", 1024)),
        margin=int(trial.get("margin", 128)),
        strategy=str(trial.get("strategy", "rolling")),
        use_legacy_counts=bool(trial.get("use_legacy_counts", False)),
        quant=bool(trial.get("quant", False)),
        logit_round_decimals=int(trial.get("logit_round_decimals", 2)),
        prob_round_decimals=int(trial.get("prob_round_decimals", 5)),
        diagnostics_csv_prefix=diagnostics_prefix,
        determinism_mode=trial.get("determinism_mode"),
        drift_correction_enabled=bool(trial.get("drift_correction_enabled", True)),
        drift_measurements_csv_prefix=drift_prefix,
        encoder_trace_csv_prefix=trace_prefix,
        emit_full_reference_trace=bool(trial.get("emit_full_reference_trace", True)),
    )


def _run_trial_on_file(
    trial: Dict[str, Any],
    file_path: Path,
    artifact_dir: Path,
    global_settings: Dict[str, Any],
) -> Dict[str, Any]:
    trial_id = str(trial.get("trial_id", "trial"))
    text_encoding = str(global_settings.get("text_encoding", "utf-8"))
    text = file_path.read_text(encoding=text_encoding, errors="replace")

    safe_mode = bool(global_settings.get("safe_mode", True))
    max_decode_tokens = global_settings.get("max_decode_tokens")
    show_progress = bool(global_settings.get("show_progress", False))

    device_mode = str(global_settings.get("device_mode", "single_device")).strip().lower()
    base_device = _resolve_device(str(global_settings.get("device", "auto")))

    if device_mode == "cross_device":
        encode_device = "cpu"
        decode_device = _preferred_accelerator_device()
    else:
        encode_device = base_device
        decode_device = base_device

    diagnostics_prefix = str(artifact_dir / f"{trial_id}_diagnostics")
    trace_prefix = str(artifact_dir / f"{trial_id}_reference")
    drift_prefix = str(artifact_dir / f"{trial_id}_drift")

    start_total = time.time()
    encode_codec = None
    decode_codec = None

    try:
        enc_tokenizer, enc_model = _load_model_tokenizer(global_settings, encode_device)
        encode_codec = DriftAwareLLMCodec(
            tokenizer=enc_tokenizer,
            model=enc_model,
            device=encode_device,
            config=_build_codec_config(trial, diagnostics_prefix, trace_prefix, drift_prefix),
        )

        t0 = time.time()
        encoded_bytes, num_tokens, reference_rows = encode_codec.encode_with_trace(
            text,
            safe_mode=safe_mode,
            return_token_count=True,
            show_progress=show_progress,
        )
        encode_seconds = time.time() - t0

        dec_tokenizer, dec_model = _load_model_tokenizer(global_settings, decode_device)
        decode_codec = DriftAwareLLMCodec(
            tokenizer=dec_tokenizer,
            model=dec_model,
            device=decode_device,
            config=_build_codec_config(trial, diagnostics_prefix, trace_prefix, drift_prefix),
        )

        t1 = time.time()
        decoded_text, drift_rows = decode_codec.decode_with_reference(
            encoded_bytes=encoded_bytes,
            reference_rows=reference_rows,
            max_decode_tokens=max_decode_tokens,
            safe_mode=safe_mode,
            expected_num_tokens=num_tokens if safe_mode else None,
            show_progress=show_progress,
        )
        decode_seconds = time.time() - t1

        (artifact_dir / f"{trial_id}_decoded.txt").write_text(decoded_text, encoding=text_encoding, errors="replace")

        drift_summary = summarize_drift_rows(drift_rows)
        encoded_size_bytes = len(encoded_bytes)
        original_size_bytes = len(text.encode(text_encoding, errors="replace"))
        compression_ratio = (
            (encoded_size_bytes * 8.0) / original_size_bytes if original_size_bytes > 0 else float("nan")
        )

        ok = decoded_text == text
        return {
            "file": str(file_path),
            "trial_id": trial_id,
            "status": "ok" if ok else "mismatch",
            "device_mode": device_mode,
            "encode_device": encode_device,
            "decode_device": decode_device,
            "determinism_mode": str(trial.get("determinism_mode")),
            "quant": bool(trial.get("quant", False)),
            "slots": int(trial.get("slots", 1 << 24)),
            "logit_round_decimals": int(trial.get("logit_round_decimals", 2)),
            "prob_round_decimals": int(trial.get("prob_round_decimals", 5)),
            "drift_correction_enabled": bool(trial.get("drift_correction_enabled", True)),
            "input_size_bytes": original_size_bytes,
            "encoded_size_bytes": encoded_size_bytes,
            "compression_ratio": compression_ratio,
            "num_tokens": int(num_tokens) if num_tokens is not None else None,
            "encode_seconds": encode_seconds,
            "decode_seconds": decode_seconds,
            "total_seconds": time.time() - start_total,
            "decoded_chars": len(decoded_text),
            "original_chars": len(text),
            "drift_events": drift_summary["drift_events"],
            "corrections_applied": drift_summary["corrections_applied"],
            "mean_abs_interval_low_delta": drift_summary["mean_abs_interval_low_delta"],
            "mean_abs_interval_high_delta": drift_summary["mean_abs_interval_high_delta"],
            "mean_distance_D_to_reference_interval": drift_summary["mean_distance_D_to_reference_interval"],
            "max_distance_D_to_reference_interval": drift_summary["max_distance_D_to_reference_interval"],
            "error": "",
        }
    except Exception as exc:
        return {
            "file": str(file_path),
            "trial_id": trial_id,
            "status": "error",
            "device_mode": device_mode,
            "encode_device": encode_device,
            "decode_device": decode_device,
            "determinism_mode": str(trial.get("determinism_mode")),
            "quant": bool(trial.get("quant", False)),
            "slots": int(trial.get("slots", 1 << 24)),
            "logit_round_decimals": int(trial.get("logit_round_decimals", 2)),
            "prob_round_decimals": int(trial.get("prob_round_decimals", 5)),
            "drift_correction_enabled": bool(trial.get("drift_correction_enabled", True)),
            "input_size_bytes": None,
            "encoded_size_bytes": None,
            "compression_ratio": None,
            "num_tokens": None,
            "encode_seconds": None,
            "decode_seconds": None,
            "total_seconds": time.time() - start_total,
            "decoded_chars": None,
            "original_chars": None,
            "drift_events": None,
            "corrections_applied": None,
            "mean_abs_interval_low_delta": None,
            "mean_abs_interval_high_delta": None,
            "mean_distance_D_to_reference_interval": None,
            "max_distance_D_to_reference_interval": None,
            "error": str(exc),
        }
    finally:
        cleanup_codec(encode_codec)
        cleanup_codec(decode_codec)
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Drift search worker")
    parser.add_argument("--config", required=True, help="Path to drift search config JSON")
    args = parser.parse_args()

    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))

    files = [Path(p) for p in config["files"]]
    trials = list(config["trials"])
    settings = dict(config.get("settings", {}))
    output_csv = Path(config.get("output_csv", "drift_search_results.csv")).resolve()
    artifact_root = Path(config.get("artifact_root", ".drift_search_artifacts")).resolve()

    artifact_root.mkdir(parents=True, exist_ok=True)

    rows: List[Dict[str, Any]] = []
    for trial in trials:
        trial_id = str(trial.get("trial_id", "trial"))
        trial_dir = artifact_root / trial_id
        trial_dir.mkdir(parents=True, exist_ok=True)

        for file_path in files:
            rows.append(_run_trial_on_file(trial, file_path, trial_dir, settings))

    pd.DataFrame(rows).to_csv(output_csv, index=False)
    print(f"Wrote search results to: {output_csv}")


if __name__ == "__main__":
    main()
