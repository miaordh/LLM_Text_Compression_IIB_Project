import argparse
import importlib.util
import os
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_codec_deterministic import DeterministicCodecConfig, DeterministicLLMCodec

# Transformers warns that TRANSFORMERS_CACHE is deprecated; prefer HF_HOME.
if os.environ.get("TRANSFORMERS_CACHE") and not os.environ.get("HF_HOME"):
    os.environ["HF_HOME"] = os.environ["TRANSFORMERS_CACHE"]
    os.environ.pop("TRANSFORMERS_CACHE", None)


ARTIFACT_ROOT = Path(".roundtrip_artifacts")
ENCODE_ATTEMPT_MARKER = "__ROUNDTRIP_ENCODE_ATTEMPT__"


def _read_text(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding, errors="replace")


def _encoding_for_file(settings: Dict[str, Any], file_path: Path) -> str:
    default_encoding = settings.get("text_encoding", "utf-8")
    overrides = settings.get("file_encoding_overrides", {}) or {}
    candidates = [
        str(file_path),
        str(file_path.name),
        str(file_path.as_posix()),
    ]
    try:
        rel = file_path.resolve().relative_to(Path.cwd().resolve())
        candidates.append(str(rel.as_posix()))
    except Exception:
        pass

    for key in candidates:
        if key in overrides:
            return overrides[key]
    return default_encoding


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_device_mode(settings: Dict[str, Any]) -> str:
    mode = str(settings.get("device_mode", "single_device")).strip().lower()
    if mode not in {"single_device", "cross_device"}:
        raise ValueError("Unsupported device_mode. Expected 'single_device' or 'cross_device'.")
    return mode


def _preferred_accelerator_device() -> str:
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    raise RuntimeError("cross_device mode requires an accelerator for decode, but neither CUDA nor MPS is available")


def _resolve_determinism_mode(settings: Dict[str, Any]):
    mode = settings.get("determinism_mode")
    if mode is None and "use_batch_invariant_ops" in settings:
        return "batch_invariant_ops" if bool(settings.get("use_batch_invariant_ops")) else None
    if mode is None:
        return None
    text = str(mode).strip().lower()
    if text in {"", "none", "off", "false", "0"}:
        return None
    return text


def _resolve_torch_dtype(dtype_name: str, device: str):
    if dtype_name == "auto":
        # On consumer GPUs, fp16 is typically required for 1B+ models.
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


def _triton_available() -> bool:
    return importlib.util.find_spec("triton") is not None


def _should_use_vllm(settings: Dict[str, Any], device: str, determinism_mode: Optional[str]) -> bool:
    backend = str(settings.get("inference_backend", "auto")).strip().lower()
    if backend in {"huggingface", "hf", "transformers"}:
        return False
    if backend == "vllm":
        return True
    if backend not in {"", "auto"}:
        raise ValueError("Unsupported inference_backend. Expected one of: auto, huggingface, vllm")

    return bool(determinism_mode in {"batch_invariant_ops", "tbik"} and device == "cuda" and _triton_available())


def _is_cuda_oom(exc: BaseException) -> bool:
    text = str(exc).lower()
    return "out of memory" in text and ("cuda" in text or "cudnn" in text)


def _build_oom_fallback_settings(settings: Dict[str, Any]) -> Dict[str, Any]:
    fallback = dict(settings)
    fallback["torch_dtype"] = "float16"
    fallback["strategy"] = str(settings.get("oom_fallback_strategy", "block"))
    fallback_window = int(settings.get("oom_fallback_context_window", 512))
    fallback_margin = int(settings.get("oom_fallback_margin", 64))
    fallback["context_window"] = max(64, fallback_window)
    fallback["margin"] = max(0, min(fallback_margin, fallback["context_window"] - 1))
    return fallback


def _emit_encode_attempt_marker(attempt_index: int, total_attempts: int):
    print(
        f"{ENCODE_ATTEMPT_MARKER} attempt={attempt_index + 1} total={total_attempts} fallback={int(attempt_index > 0)}",
        file=sys.stderr,
        flush=True,
    )


def _parse_encode_attempt_info(stderr_text: str):
    attempt_count = 0
    fallback_attempted = False
    for line in (stderr_text or "").splitlines():
        if ENCODE_ATTEMPT_MARKER not in line:
            continue
        # Expected format: __ROUNDTRIP_ENCODE_ATTEMPT__ attempt=1 total=2 fallback=0
        parts = line.strip().split()
        kv = {}
        for part in parts[1:]:
            if "=" in part:
                k, v = part.split("=", 1)
                kv[k] = v
        try:
            attempt_count = max(attempt_count, int(kv.get("attempt", "0")))
        except ValueError:
            pass
        fallback_attempted = fallback_attempted or kv.get("fallback") == "1"
    return attempt_count, fallback_attempted


def _sanitize_error_text(stderr_text: str) -> str:
    if not stderr_text:
        return ""

    lines = []
    previous_blank = False
    for raw_line in stderr_text.splitlines():
        line = raw_line.replace("\r", "").rstrip()
        if ENCODE_ATTEMPT_MARKER in line:
            continue
        if "Loading weights:" in line:
            continue

        if not line:
            if previous_blank:
                continue
            previous_blank = True
            lines.append("")
            continue

        previous_blank = False
        lines.append(line)

    return "\n".join(lines).strip()


def _load_codec(settings: Dict[str, Any]) -> DeterministicLLMCodec:
    device = _resolve_device(settings.get("device", "auto"))
    determinism_mode = _resolve_determinism_mode(settings)
    use_vllm = _should_use_vllm(settings, device, determinism_mode)

    if device == "cuda":
        # Helps with long-running fragmentation-heavy workloads.
        os.environ.setdefault("PYTORCH_CUDA_ALLOC_CONF", "expandable_segments:True")

    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(settings.get("trust_remote_code", False)),
        "low_cpu_mem_usage": True,
    }

    torch_dtype_name = str(settings.get("torch_dtype", "auto"))
    torch_dtype = _resolve_torch_dtype(torch_dtype_name, device)
    if torch_dtype != "auto":
        # Newer Transformers prefers `dtype`; older versions still accept `torch_dtype`.
        model_kwargs["dtype"] = torch_dtype

    revision = settings.get("revision")
    if revision:
        model_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_id"],
        trust_remote_code=bool(settings.get("trust_remote_code", False)),
        revision=revision,
    )
    if bool(settings.get("ignore_model_max_length_warning", True)):
        # We intentionally encode full files and enforce context limits ourselves.
        tokenizer.model_max_length = int(1e30)

    slots = int(settings.get("slots", 1 << 24))
    vocab_size = len(tokenizer)
    if slots < vocab_size:
        raise ValueError(
            f"Invalid slots={slots} for vocab_size={vocab_size}. "
            "Use slots >= vocab size (for example 1<<18 or 1<<20 for ~150k vocab)."
        )

    model = None
    if not use_vllm:
        try:
            model = AutoModelForCausalLM.from_pretrained(settings["model_id"], **model_kwargs)
        except TypeError:
            if "dtype" not in model_kwargs:
                raise
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(settings["model_id"], **fallback_kwargs)

    config = DeterministicCodecConfig(
        precision=int(settings.get("precision", 32)),
        slots=int(settings.get("slots", 1 << 24)),
        context_window=int(settings.get("context_window", 2048)),
        margin=int(settings.get("margin", 128)),
        strategy=str(settings.get("strategy", "rolling")),
        use_legacy_counts=bool(settings.get("use_legacy_counts", False)),
        quant=bool(settings.get("quant", False)),
        logit_round_decimals=int(settings.get("logit_round_decimals", 2)),
        prob_round_decimals=int(settings.get("prob_round_decimals", 5)),
        diagnostics_csv_prefix=settings.get("diagnostics_csv_prefix"),
        determinism_mode=determinism_mode,
        inference_backend="vllm" if use_vllm else "huggingface",
        model_id=str(settings.get("model_id")),
        trust_remote_code=bool(settings.get("trust_remote_code", False)),
        revision=revision,
        torch_dtype=torch_dtype_name,
        vllm_tensor_parallel_size=int(settings.get("vllm_tensor_parallel_size", 1)),
        vllm_gpu_memory_utilization=float(settings.get("vllm_gpu_memory_utilization", 0.9)),
        vllm_max_logprobs=settings.get("vllm_max_logprobs"),
    )

    return DeterministicLLMCodec(tokenizer=tokenizer, model=model, device=device, config=config)


def _phase_encode(config: Dict[str, Any], file_path: Path, artifact_dir: Path):
    settings = config["settings"]
    safe_mode = bool(settings.get("safe_mode", True))
    text_encoding = _encoding_for_file(settings, file_path)
    device_mode = _resolve_device_mode(settings)
    diagnostics_enabled = bool(settings.get("diagnostics_enabled", False))
    demo_mode = bool(settings.get("demo_mode", False))
    speed_demo = bool(settings.get("speed_demo", False))
    memory_demo = bool(settings.get("memory_demo", False))
    memory_sample_interval = float(settings.get("memory_sample_interval", 0.05))

    artifact_dir.mkdir(parents=True, exist_ok=True)

    text = _read_text(file_path, text_encoding)
    base_attempt = dict(settings)
    if device_mode == "cross_device":
        base_attempt["device"] = "cpu"
    attempts = [base_attempt]

    if bool(settings.get("enable_oom_fallback", True)):
        fallback_attempt = _build_oom_fallback_settings(settings)
        if fallback_attempt != base_attempt:
            attempts.append(fallback_attempt)

    last_exc = None
    for idx, attempt_settings in enumerate(attempts):
        codec = None
        try:
            _emit_encode_attempt_marker(idx, len(attempts))
            if diagnostics_enabled and not attempt_settings.get("diagnostics_csv_prefix"):
                attempt_settings["diagnostics_csv_prefix"] = str(artifact_dir / f"diagnostics_attempt_{idx + 1}")
            codec = _load_codec(attempt_settings)

            start = time.time()
            encoded_result = codec.encode(
                text,
                safe_mode=safe_mode,
                return_token_count=safe_mode,
                show_progress=False,
                demo=demo_mode,
                demo_csv_path=str(artifact_dir / "demo_encode.csv"),
                speed_demo=speed_demo,
                speed_csv_path=str(artifact_dir / "speed_encode.csv"),
                memory_demo=memory_demo,
                memory_csv_path=str(artifact_dir / "memory_encode.csv"),
                memory_sample_interval=memory_sample_interval,
            )
            encode_seconds = time.time() - start

            if safe_mode:
                encoded_bytes, num_tokens = encoded_result
            else:
                encoded_bytes = encoded_result
                num_tokens = None

            original_size_bytes = len(text.encode(text_encoding, errors="replace"))
            (artifact_dir / "encoded.bin").write_bytes(encoded_bytes)
            metadata = {
                "input_file": str(file_path),
                "safe_mode": safe_mode,
                "num_tokens": num_tokens,
                "encode_seconds": encode_seconds,
                "encoded_size_bytes": len(encoded_bytes),
                "original_size_bytes": original_size_bytes,
                "text_encoding": text_encoding,
                "device_mode": device_mode,
                "demo_mode": demo_mode,
                "speed_demo": speed_demo,
                "memory_demo": memory_demo,
                "memory_sample_interval": memory_sample_interval,
                "divergence_window": int(settings.get("divergence_window", 5)),
                "effective_settings": {
                    "device": attempt_settings.get("device", "auto"),
                    "torch_dtype": attempt_settings.get("torch_dtype", "auto"),
                    "inference_backend": attempt_settings.get("inference_backend", "auto"),
                    "vllm_tensor_parallel_size": int(attempt_settings.get("vllm_tensor_parallel_size", 1)),
                    "vllm_gpu_memory_utilization": float(
                        attempt_settings.get("vllm_gpu_memory_utilization", 0.9)
                    ),
                    "vllm_max_logprobs": attempt_settings.get("vllm_max_logprobs"),
                    "context_window": int(attempt_settings.get("context_window", 2048)),
                    "margin": int(attempt_settings.get("margin", 128)),
                    "strategy": str(attempt_settings.get("strategy", "rolling")),
                    "quant": bool(attempt_settings.get("quant", False)),
                    "logit_round_decimals": int(attempt_settings.get("logit_round_decimals", 2)),
                    "prob_round_decimals": int(attempt_settings.get("prob_round_decimals", 5)),
                    "diagnostics_csv_prefix": attempt_settings.get("diagnostics_csv_prefix"),
                    "use_legacy_counts": bool(attempt_settings.get("use_legacy_counts", False)),
                    "determinism_mode": _resolve_determinism_mode(attempt_settings),
                },
                "used_oom_fallback": idx > 0,
            }
            (artifact_dir / "encode_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
            return
        except RuntimeError as exc:
            last_exc = exc
            has_next_attempt = (idx + 1) < len(attempts)
            if not (_is_cuda_oom(exc) and has_next_attempt):
                raise
        finally:
            if codec is not None:
                del codec
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()

    if last_exc is not None:
        raise last_exc


def _phase_decode(config: Dict[str, Any], artifact_dir: Path):
    settings = dict(config["settings"])
    safe_mode = bool(settings.get("safe_mode", True))
    device_mode = _resolve_device_mode(settings)
    decode_device_override = settings.get("decode_device_override")
    diagnostics_enabled = bool(settings.get("diagnostics_enabled", False))
    demo_mode = bool(settings.get("demo_mode", False))
    speed_demo = bool(settings.get("speed_demo", False))
    memory_demo = bool(settings.get("memory_demo", False))
    memory_sample_interval = float(settings.get("memory_sample_interval", 0.05))
    divergence_window = int(settings.get("divergence_window", 5))

    encode_meta_path = artifact_dir / "encode_metadata.json"
    encoded_path = artifact_dir / "encoded.bin"

    if not encode_meta_path.exists() or not encoded_path.exists():
        raise FileNotFoundError(f"Missing encode artifacts in {artifact_dir}")

    encode_meta = json.loads(encode_meta_path.read_text(encoding="utf-8"))
    effective_settings = encode_meta.get("effective_settings")
    if isinstance(effective_settings, dict):
        settings.update(effective_settings)

    # Preserve run-level demo settings even if encode metadata includes effective settings.
    demo_mode = bool(encode_meta.get("demo_mode", demo_mode))
    speed_demo = bool(encode_meta.get("speed_demo", speed_demo))
    memory_demo = bool(encode_meta.get("memory_demo", memory_demo))
    memory_sample_interval = float(encode_meta.get("memory_sample_interval", memory_sample_interval))
    divergence_window = int(encode_meta.get("divergence_window", divergence_window))

    if device_mode == "cross_device":
        settings["device"] = _preferred_accelerator_device()

    if decode_device_override is not None:
        settings["device"] = str(decode_device_override)

    if diagnostics_enabled and not settings.get("diagnostics_csv_prefix"):
        settings["diagnostics_csv_prefix"] = str(artifact_dir / "diagnostics")

    text_encoding = encode_meta.get("text_encoding", settings.get("text_encoding", "utf-8"))
    codec = None
    try:
        codec = _load_codec(settings)
        encoded_bytes = encoded_path.read_bytes()

        decode_kwargs: Dict[str, Any] = {
            "max_decode_tokens": settings.get("max_decode_tokens"),
            "safe_mode": safe_mode,
            "show_progress": False,
            "demo": demo_mode,
            "demo_csv_path": str(artifact_dir / "demo_decode.csv"),
            "speed_demo": speed_demo,
            "speed_csv_path": str(artifact_dir / "speed_decode.csv"),
            "memory_demo": memory_demo,
            "memory_csv_path": str(artifact_dir / "memory_decode.csv"),
            "memory_sample_interval": memory_sample_interval,
            "divergence_window": divergence_window,
        }
        if safe_mode:
            decode_kwargs["expected_num_tokens"] = int(encode_meta["num_tokens"])

        # Demo mode includes forensic-style divergence reporting by comparing
        # decoded tokens against reference tokenization of the original input.
        if demo_mode:
            input_file = encode_meta.get("input_file")
            if input_file:
                input_path = Path(str(input_file))
                if input_path.exists() and input_path.is_file():
                    source_text = _read_text(input_path, text_encoding)
                    try:
                        reference_ids = list(codec.tokenizer.encode(source_text, add_special_tokens=False))
                    except TypeError:
                        reference_ids = list(codec.tokenizer.encode(source_text))
                    if not safe_mode:
                        reference_ids = reference_ids + [int(codec.eof_token_id)]
                    decode_kwargs["reference_token_ids"] = reference_ids

        start = time.time()
        decoded_text = codec.decode(encoded_bytes, **decode_kwargs)
        decode_seconds = time.time() - start

        (artifact_dir / "decoded.txt").write_text(decoded_text, encoding=text_encoding, errors="replace")

        decode_meta = {
            "decode_seconds": decode_seconds,
            "decoded_chars": len(decoded_text),
            "demo_mode": demo_mode,
            "speed_demo": speed_demo,
            "memory_demo": memory_demo,
        }
        (artifact_dir / "decode_metadata.json").write_text(json.dumps(decode_meta, indent=2), encoding="utf-8")
    finally:
        if codec is not None:
            del codec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _job_id(path: Path) -> str:
    digest = hashlib.sha1(str(path).encode("utf-8")).hexdigest()[:10]
    return f"{path.stem}_{digest}"


def _run_subprocess(args: List[str], timeout_seconds: int = 0):
    timeout = None if timeout_seconds <= 0 else timeout_seconds
    try:
        completed = subprocess.run(
            args,
            stdout=subprocess.DEVNULL,
            stderr=subprocess.PIPE,
            text=True,
            timeout=timeout,
        )
        return completed.returncode, completed.stderr
    except subprocess.TimeoutExpired as exc:
        msg = f"Timed out after {timeout_seconds}s: {' '.join(args)}"
        if exc.stderr:
            msg += f"\n{exc.stderr}"
        return 124, msg


def _cleanup_job_dir(job_dir: Path):
    if not job_dir.exists():
        return
    for child in job_dir.glob("*"):
        if child.is_file():
            child.unlink()
    job_dir.rmdir()


def _cleanup_memory():
    # Defensive cleanup in the orchestrator process between files.
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()


def _run_orchestrator(config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    files = [Path(p) for p in config["files"]]
    output_csv = Path(config.get("output_csv", "deterministic_roundtrip_results.csv"))
    artifact_root = Path(config.get("artifact_root", str(ARTIFACT_ROOT)))
    keep_artifacts = bool(config["settings"].get("keep_artifacts", False))
    diagnostics_enabled = bool(config["settings"].get("diagnostics_enabled", False))
    diagnostics_csv_prefix = config["settings"].get("diagnostics_csv_prefix")
    stop_on_file_error = bool(config["settings"].get("stop_on_file_error", True))
    phase_timeout_seconds = int(config["settings"].get("phase_timeout_seconds", 0) or 0)

    # When diagnostics are enabled and no explicit prefix is set, per-token CSVs are
    # written under each job artifact directory. Keep those artifacts so diagnostics survive.
    preserve_job_artifacts = keep_artifacts or (
        diagnostics_enabled and not diagnostics_csv_prefix
    )

    artifact_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for file_path in files:
        job_dir = artifact_root / _job_id(file_path)
        try:
            if not file_path.exists() or not file_path.is_file():
                rows.append(
                    {
                        "file": str(file_path),
                        "status": "skipped",
                        "error": "File does not exist or is not a file",
                        "fallback_attempted": False,
                        "attempt_count": 0,
                    }
                )
                continue

            if job_dir.exists():
                _cleanup_job_dir(job_dir)
            job_dir.mkdir(parents=True, exist_ok=True)

            encode_cmd = [
                sys.executable,
                __file__,
                "--phase",
                "encode",
                "--config",
                str(config_path),
                "--file",
                str(file_path),
                "--artifact-dir",
                str(job_dir),
            ]
            rc_enc, err_enc = _run_subprocess(encode_cmd, timeout_seconds=phase_timeout_seconds)
            if rc_enc != 0:
                attempt_count, fallback_attempted = _parse_encode_attempt_info(err_enc or "")
                rows.append(
                    {
                        "file": str(file_path),
                        "status": "encode_failed",
                        "error": _sanitize_error_text(err_enc or "Encode phase failed"),
                        "fallback_attempted": fallback_attempted,
                        "attempt_count": attempt_count,
                    }
                )
                if not preserve_job_artifacts:
                    _cleanup_job_dir(job_dir)
                continue

            decode_cmd = [
                sys.executable,
                __file__,
                "--phase",
                "decode",
                "--config",
                str(config_path),
                "--artifact-dir",
                str(job_dir),
            ]
            rc_dec, err_dec = _run_subprocess(decode_cmd, timeout_seconds=phase_timeout_seconds)
            if rc_dec != 0:
                fallback_attempted = False
                attempt_count = 0
                encode_meta_for_decode = job_dir / "encode_metadata.json"
                if encode_meta_for_decode.exists():
                    try:
                        meta = json.loads(encode_meta_for_decode.read_text(encoding="utf-8"))
                        fallback_attempted = bool(meta.get("used_oom_fallback", False))
                        attempt_count = 2 if fallback_attempted else 1
                    except Exception:
                        pass
                rows.append(
                    {
                        "file": str(file_path),
                        "status": "decode_failed",
                        "error": _sanitize_error_text(err_dec or "Decode phase failed"),
                        "fallback_attempted": fallback_attempted,
                        "attempt_count": attempt_count,
                    }
                )
                if not preserve_job_artifacts:
                    _cleanup_job_dir(job_dir)
                continue

            encode_meta = json.loads((job_dir / "encode_metadata.json").read_text(encoding="utf-8"))
            file_encoding = encode_meta.get("text_encoding", config["settings"].get("text_encoding", "utf-8"))

            original_text = _read_text(file_path, file_encoding)
            decoded_text = (job_dir / "decoded.txt").read_text(
                encoding=file_encoding,
                errors="replace",
            )

            decode_meta = json.loads((job_dir / "decode_metadata.json").read_text(encoding="utf-8"))

            is_match = original_text == decoded_text
            row = {
                "file": str(file_path),
                "status": "ok" if is_match else "mismatch",
                "inference_backend": str(
                    (encode_meta.get("effective_settings") or {}).get("inference_backend", "auto")
                ),
                "input_size_bytes": encode_meta.get("original_size_bytes", 0),
                "safe_mode": encode_meta["safe_mode"],
                "num_tokens": encode_meta["num_tokens"],
                "encoded_size_bytes": encode_meta["encoded_size_bytes"],
                "compression_ratio": (
                    (encode_meta["encoded_size_bytes"] * 8) / encode_meta.get("original_size_bytes", 1)
                    if encode_meta.get("original_size_bytes", 0) > 0
                    else None
                ),
                "encode_seconds": encode_meta["encode_seconds"],
                "decode_seconds": decode_meta["decode_seconds"],
                "original_chars": len(original_text),
                "decoded_chars": decode_meta["decoded_chars"],
                "fallback_attempted": bool(encode_meta.get("used_oom_fallback", False)),
                "attempt_count": 2 if bool(encode_meta.get("used_oom_fallback", False)) else 1,
                "error": "",
            }
            rows.append(row)

            if not preserve_job_artifacts:
                _cleanup_job_dir(job_dir)
        finally:
            # Always clear parent-process memory/cache before moving to next file,
            # including timeout/error paths and any early continue.
            _cleanup_memory()

    if stop_on_file_error:
        print(
            "Note: stop_on_file_error is enabled but per-file errors now only stop that file; "
            "the runner continues with remaining files.",
            file=sys.stderr,
        )

    pd.DataFrame(rows).to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Deterministic round-trip worker")
    parser.add_argument("--phase", choices=["run", "encode", "decode"], default="run")
    parser.add_argument("--config", required=True, help="Path to runner-generated JSON config")
    parser.add_argument("--file", help="Input file path (encode phase)")
    parser.add_argument("--artifact-dir", help="Artifact directory for intermediate files")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))

    try:
        if args.phase == "run":
            _run_orchestrator(cfg_path)
        elif args.phase == "encode":
            if not args.file or not args.artifact_dir:
                raise ValueError("--file and --artifact-dir are required for encode phase")
            _phase_encode(cfg, Path(args.file).resolve(), Path(args.artifact_dir).resolve())
        elif args.phase == "decode":
            if not args.artifact_dir:
                raise ValueError("--artifact-dir is required for decode phase")
            _phase_decode(cfg, Path(args.artifact_dir).resolve())
    except Exception as exc:
        print(f"Worker failed: {exc}", file=sys.stderr)
        sys.exit(1)
