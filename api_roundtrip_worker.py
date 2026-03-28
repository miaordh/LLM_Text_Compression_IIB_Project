import argparse
import hashlib
import json
import os
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd

from llm_codec_api import APICodecConfig, APILLMCodec

ARTIFACT_ROOT = Path(".api_roundtrip_artifacts")

ROUNDTRIP_RESULT_COLUMNS = [
    "file",
    "status",
    "input_size_bytes",
    "safe_mode",
    "num_tokens",
    "encoded_size_bytes",
    "compression_ratio",
    "encode_seconds",
    "decode_seconds",
    "original_chars",
    "decoded_chars",
    "error",
]


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


def _tokenizer_vocab_size(tokenizer) -> int:
    try:
        size = int(len(tokenizer))
        if size > 0:
            return size
    except Exception:
        pass

    for attr in ("vocab_size", "n_vocab"):
        value = getattr(tokenizer, attr, None)
        if isinstance(value, int) and value > 0:
            return int(value)

    decoder = getattr(tokenizer, "decoder", None)
    if isinstance(decoder, dict) and decoder:
        return max(int(k) for k in decoder.keys()) + 1

    raise RuntimeError(
        "Could not determine tokenizer vocab size for slot validation."
    )


def _load_codec(settings: Dict[str, Any]) -> APILLMCodec:
    tokenizer_backend = str(settings.get("tokenizer_backend", "qwen_tokenizer")).strip().lower()

    if tokenizer_backend == "qwen_tokenizer":
        try:
            import qwen_tokenizer
        except Exception as exc:
            raise RuntimeError(
                "qwen-tokenizer package is required for API mode tokenizer. "
                "Install with: pip install qwen-tokenizer"
            ) from exc

        tokenizer_name = str(settings.get("tokenizer_name", "qwen-max"))
        try:
            tokenizer = qwen_tokenizer.get_tokenizer(tokenizer_name)
        except Exception as exc:
            available = []
            try:
                available = list(qwen_tokenizer.list_tokenizers())
            except Exception:
                pass
            raise RuntimeError(
                f"Failed to load qwen_tokenizer tokenizer_name={tokenizer_name!r}. "
                f"Available tokenizers: {available}"
            ) from exc
    elif tokenizer_backend == "huggingface":
        # Optional legacy fallback path.
        from transformers import AutoTokenizer

        revision = settings.get("revision")
        tokenizer_model_id = settings.get("tokenizer_model_id", settings.get("model_id"))
        if not tokenizer_model_id:
            raise ValueError(
                "Missing tokenizer model id. Provide settings['tokenizer_model_id'] "
                "with a valid Hugging Face tokenizer."
            )
        tokenizer = AutoTokenizer.from_pretrained(
            tokenizer_model_id,
            trust_remote_code=bool(settings.get("trust_remote_code", False)),
            revision=revision,
        )
        if bool(settings.get("ignore_model_max_length_warning", True)):
            tokenizer.model_max_length = int(1e30)
    else:
        raise ValueError(
            "Unsupported tokenizer_backend. Expected 'qwen_tokenizer' or 'huggingface'."
        )

    slots = int(settings.get("slots", 1 << 24))
    vocab_size = _tokenizer_vocab_size(tokenizer)
    if slots < vocab_size:
        raise ValueError(
            f"Invalid slots={slots} for vocab_size={vocab_size}. "
            "Use slots >= vocab size (for example 1<<18 or 1<<20 for ~150k vocab)."
        )

    resolved_api_key = settings.get("api_key")
    if not resolved_api_key:
        api_key_env = settings.get("api_key_env")
        if api_key_env:
            resolved_api_key = os.getenv(str(api_key_env))

    config = APICodecConfig(
        precision=int(settings.get("precision", 32)),
        slots=slots,
        context_window=int(settings.get("context_window", 2048)),
        margin=int(settings.get("margin", 128)),
        strategy=str(settings.get("strategy", "rolling")),
        use_legacy_counts=bool(settings.get("use_legacy_counts", False)),
        quant=bool(settings.get("quant", False)),
        logit_round_decimals=int(settings.get("logit_round_decimals", 2)),
        prob_round_decimals=int(settings.get("prob_round_decimals", 5)),
        diagnostics_csv_prefix=settings.get("diagnostics_csv_prefix"),
        determinism_mode=settings.get("determinism_mode"),
        model=str(settings.get("api_model", "qwen-plus")),
        api_key=resolved_api_key,
        base_url=str(settings.get("api_base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")),
        top_k=int(settings.get("api_top_k", 20)),
        temperature=float(settings.get("api_temperature", 0.0)),
        enable_api_cache_hints=bool(settings.get("enable_api_cache_hints", True)),
        api_attention_sink=int(settings.get("api_attention_sink", 4)),
        deterministic_strict=bool(settings.get("deterministic_strict", True)),
        api_request_mode=str(settings.get("api_request_mode", "chat")),
        api_seed=settings.get("api_seed", 1),
        strict_single_id_mapping=bool(settings.get("strict_single_id_mapping", True)),
    )

    return APILLMCodec(tokenizer=tokenizer, config=config)


def _phase_encode(config: Dict[str, Any], file_path: Path, artifact_dir: Path):
    settings = config["settings"]
    safe_mode = bool(settings.get("safe_mode", True))
    text_encoding = _encoding_for_file(settings, file_path)
    diagnostics_enabled = bool(settings.get("diagnostics_enabled", False))
    artifact_dir.mkdir(parents=True, exist_ok=True)

    text = _read_text(file_path, text_encoding)
    local_settings = dict(settings)
    if diagnostics_enabled and not local_settings.get("diagnostics_csv_prefix"):
        local_settings["diagnostics_csv_prefix"] = str(artifact_dir / "diagnostics")

    codec = _load_codec(local_settings)

    start = time.time()
    encoded_result = codec.encode(
        text,
        safe_mode=safe_mode,
        return_token_count=True,
        show_progress=False,
    )
    encode_seconds = time.time() - start

    if isinstance(encoded_result, tuple):
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
        "effective_settings": {
            "context_window": int(local_settings.get("context_window", 2048)),
            "margin": int(local_settings.get("margin", 128)),
            "strategy": str(local_settings.get("strategy", "rolling")),
            "quant": bool(local_settings.get("quant", False)),
            "logit_round_decimals": int(local_settings.get("logit_round_decimals", 2)),
            "prob_round_decimals": int(local_settings.get("prob_round_decimals", 5)),
            "diagnostics_csv_prefix": local_settings.get("diagnostics_csv_prefix"),
            "use_legacy_counts": bool(local_settings.get("use_legacy_counts", False)),
            "api_model": str(local_settings.get("api_model", "qwen-plus")),
            "api_base_url": str(local_settings.get("api_base_url", "https://dashscope-intl.aliyuncs.com/compatible-mode/v1")),
            "api_top_k": int(local_settings.get("api_top_k", 20)),
            "api_temperature": float(local_settings.get("api_temperature", 0.0)),
            "enable_api_cache_hints": bool(local_settings.get("enable_api_cache_hints", True)),
            "api_attention_sink": int(local_settings.get("api_attention_sink", 4)),
            "deterministic_strict": bool(local_settings.get("deterministic_strict", True)),
            "api_request_mode": str(local_settings.get("api_request_mode", "chat")),
            "api_seed": local_settings.get("api_seed", 1),
            "strict_single_id_mapping": bool(local_settings.get("strict_single_id_mapping", True)),
            "tokenizer_backend": str(local_settings.get("tokenizer_backend", "qwen_tokenizer")),
            "tokenizer_name": str(local_settings.get("tokenizer_name", "qwen-max")),
        },
    }
    (artifact_dir / "encode_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")


def _phase_decode(config: Dict[str, Any], artifact_dir: Path):
    settings = dict(config["settings"])
    safe_mode = bool(settings.get("safe_mode", True))
    diagnostics_enabled = bool(settings.get("diagnostics_enabled", False))

    encode_meta_path = artifact_dir / "encode_metadata.json"
    encoded_path = artifact_dir / "encoded.bin"

    if not encode_meta_path.exists() or not encoded_path.exists():
        raise FileNotFoundError(f"Missing encode artifacts in {artifact_dir}")

    encode_meta = json.loads(encode_meta_path.read_text(encoding="utf-8"))
    effective_settings = encode_meta.get("effective_settings")
    if isinstance(effective_settings, dict):
        settings.update(effective_settings)

    if diagnostics_enabled and not settings.get("diagnostics_csv_prefix"):
        settings["diagnostics_csv_prefix"] = str(artifact_dir / "diagnostics")

    text_encoding = encode_meta.get("text_encoding", settings.get("text_encoding", "utf-8"))
    codec = _load_codec(settings)

    start = time.time()
    encoded_bytes = encoded_path.read_bytes()

    decode_kwargs: Dict[str, Any] = {
        "max_decode_tokens": settings.get("max_decode_tokens"),
        "safe_mode": safe_mode,
        "show_progress": False,
    }
    if safe_mode:
        decode_kwargs["expected_num_tokens"] = int(encode_meta["num_tokens"])

    decoded_text = codec.decode(encoded_bytes, **decode_kwargs)
    decode_seconds = time.time() - start

    (artifact_dir / "decoded.txt").write_text(decoded_text, encoding=text_encoding, errors="replace")

    decode_meta = {
        "decode_seconds": decode_seconds,
        "decoded_chars": len(decoded_text),
    }
    (artifact_dir / "decode_metadata.json").write_text(json.dumps(decode_meta, indent=2), encoding="utf-8")


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


def _run_orchestrator(config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    files = [Path(p) for p in config["files"]]
    output_csv = Path(config.get("output_csv", "api_roundtrip_results.csv"))
    artifact_root = Path(config.get("artifact_root", str(ARTIFACT_ROOT)))
    keep_artifacts = bool(config["settings"].get("keep_artifacts", False))
    diagnostics_enabled = bool(config["settings"].get("diagnostics_enabled", False))
    diagnostics_csv_prefix = config["settings"].get("diagnostics_csv_prefix")
    stop_on_file_error = bool(config["settings"].get("stop_on_file_error", True))
    phase_timeout_seconds = int(config["settings"].get("phase_timeout_seconds", 0) or 0)

    preserve_job_artifacts = keep_artifacts or (
        diagnostics_enabled and not diagnostics_csv_prefix
    )

    artifact_root.mkdir(parents=True, exist_ok=True)

    rows = []
    for file_path in files:
        job_dir = artifact_root / _job_id(file_path)

        if not file_path.exists() or not file_path.is_file():
            rows.append(
                {
                    "file": str(file_path),
                    "status": "skipped",
                    "num_tokens": None,
                    "error": "File does not exist or is not a file",
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
            rows.append(
                {
                    "file": str(file_path),
                    "status": "encode_failed",
                    "num_tokens": None,
                    "error": (err_enc or "Encode phase failed").strip(),
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
            rows.append(
                {
                    "file": str(file_path),
                    "status": "decode_failed",
                    "num_tokens": None,
                    "error": (err_dec or "Decode phase failed").strip(),
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
            "input_size_bytes": encode_meta.get("original_size_bytes", 0),
            "safe_mode": encode_meta.get("safe_mode", True),
            "num_tokens": encode_meta.get("num_tokens"),
            "encoded_size_bytes": encode_meta.get("encoded_size_bytes", 0),
            "compression_ratio": (
                (
                    encode_meta.get("encoded_size_bytes", 0)
                    * 8
                )
                / encode_meta.get("original_size_bytes", 1)
                if encode_meta.get("original_size_bytes", 0) > 0
                else None
            ),
            "encode_seconds": encode_meta.get("encode_seconds"),
            "decode_seconds": decode_meta.get("decode_seconds"),
            "original_chars": len(original_text),
            "decoded_chars": decode_meta.get("decoded_chars"),
            "error": "",
        }
        rows.append(row)

        if not preserve_job_artifacts:
            _cleanup_job_dir(job_dir)

    if stop_on_file_error:
        print(
            "Note: stop_on_file_error is enabled but per-file errors only stop that file; "
            "the runner continues with remaining files.",
            file=sys.stderr,
        )

    pd.DataFrame(rows).reindex(columns=ROUNDTRIP_RESULT_COLUMNS).to_csv(output_csv, index=False)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="API round-trip worker")
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
