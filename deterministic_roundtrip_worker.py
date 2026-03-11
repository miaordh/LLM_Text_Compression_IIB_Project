import argparse
import gc
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_codec_deterministic import DeterministicCodecConfig, DeterministicLLMCodec


ARTIFACT_ROOT = Path(".roundtrip_artifacts")


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


def _resolve_torch_dtype(dtype_name: str):
    if dtype_name == "auto":
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch_dtype: {dtype_name}")
    return mapping[dtype_name]


def _load_codec(settings: Dict[str, Any]) -> DeterministicLLMCodec:
    model_kwargs: Dict[str, Any] = {
        "trust_remote_code": bool(settings.get("trust_remote_code", False)),
    }

    torch_dtype = _resolve_torch_dtype(settings.get("torch_dtype", "auto"))
    if torch_dtype != "auto":
        model_kwargs["torch_dtype"] = torch_dtype

    revision = settings.get("revision")
    if revision:
        model_kwargs["revision"] = revision

    tokenizer = AutoTokenizer.from_pretrained(
        settings["model_id"],
        trust_remote_code=bool(settings.get("trust_remote_code", False)),
        revision=revision,
    )
    model = AutoModelForCausalLM.from_pretrained(settings["model_id"], **model_kwargs)

    config = DeterministicCodecConfig(
        precision=int(settings.get("precision", 32)),
        slots=int(settings.get("slots", 1 << 24)),
        use_kv_cache=bool(settings.get("use_kv_cache", False)),
        quant=bool(settings.get("quant", False)),
        logit_round_decimals=int(settings.get("logit_round_decimals", 2)),
        prob_round_decimals=int(settings.get("prob_round_decimals", 5)),
        use_batch_invariant_ops=bool(settings.get("use_batch_invariant_ops", True)),
    )

    device = _resolve_device(settings.get("device", "auto"))
    return DeterministicLLMCodec(tokenizer=tokenizer, model=model, device=device, config=config)


def _phase_encode(config: Dict[str, Any], file_path: Path, artifact_dir: Path):
    settings = config["settings"]
    safe_mode = bool(settings.get("safe_mode", True))
    text_encoding = _encoding_for_file(settings, file_path)

    artifact_dir.mkdir(parents=True, exist_ok=True)

    codec = None
    try:
        codec = _load_codec(settings)
        text = _read_text(file_path, text_encoding)

        start = time.time()
        encoded_result = codec.encode(
            text,
            safe_mode=safe_mode,
            return_token_count=safe_mode,
            show_progress=False,
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
        }
        (artifact_dir / "encode_metadata.json").write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    finally:
        if codec is not None:
            del codec
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def _phase_decode(config: Dict[str, Any], artifact_dir: Path):
    settings = config["settings"]
    safe_mode = bool(settings.get("safe_mode", True))

    encode_meta_path = artifact_dir / "encode_metadata.json"
    encoded_path = artifact_dir / "encoded.bin"

    if not encode_meta_path.exists() or not encoded_path.exists():
        raise FileNotFoundError(f"Missing encode artifacts in {artifact_dir}")

    encode_meta = json.loads(encode_meta_path.read_text(encoding="utf-8"))

    text_encoding = encode_meta.get("text_encoding", settings.get("text_encoding", "utf-8"))
    codec = None
    try:
        codec = _load_codec(settings)
        encoded_bytes = encoded_path.read_bytes()

        decode_kwargs: Dict[str, Any] = {
            "max_decode_tokens": settings.get("max_decode_tokens"),
            "safe_mode": safe_mode,
        }
        if safe_mode:
            decode_kwargs["expected_num_tokens"] = int(encode_meta["num_tokens"])

        start = time.time()
        decoded_text = codec.decode(encoded_bytes, **decode_kwargs)
        decode_seconds = time.time() - start

        (artifact_dir / "decoded.txt").write_text(decoded_text, encoding=text_encoding, errors="replace")

        decode_meta = {
            "decode_seconds": decode_seconds,
            "decoded_chars": len(decoded_text),
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


def _run_orchestrator(config_path: Path):
    config = json.loads(config_path.read_text(encoding="utf-8"))
    files = [Path(p) for p in config["files"]]
    output_csv = Path(config.get("output_csv", "deterministic_roundtrip_results.csv"))
    keep_artifacts = bool(config["settings"].get("keep_artifacts", False))
    stop_on_file_error = bool(config["settings"].get("stop_on_file_error", True))
    phase_timeout_seconds = int(config["settings"].get("phase_timeout_seconds", 0) or 0)

    ARTIFACT_ROOT.mkdir(parents=True, exist_ok=True)

    rows = []
    for file_path in files:
        if not file_path.exists() or not file_path.is_file():
            rows.append(
                {
                    "file": str(file_path),
                    "status": "skipped",
                    "error": "File does not exist or is not a file",
                }
            )
            continue

        job_dir = ARTIFACT_ROOT / _job_id(file_path)
        if job_dir.exists():
            for child in job_dir.glob("*"):
                if child.is_file():
                    child.unlink()
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
                    "error": (err_enc or "Encode phase failed").strip(),
                }
            )
            if not keep_artifacts:
                for child in job_dir.glob("*"):
                    if child.is_file():
                        child.unlink()
                job_dir.rmdir()
            if stop_on_file_error:
                break
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
                    "error": (err_dec or "Decode phase failed").strip(),
                }
            )
            if not keep_artifacts:
                for child in job_dir.glob("*"):
                    if child.is_file():
                        child.unlink()
                job_dir.rmdir()
            if stop_on_file_error:
                break
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
            "error": "",
        }
        rows.append(row)

        if not keep_artifacts:
            for child in job_dir.glob("*"):
                if child.is_file():
                    child.unlink()
            job_dir.rmdir()

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
