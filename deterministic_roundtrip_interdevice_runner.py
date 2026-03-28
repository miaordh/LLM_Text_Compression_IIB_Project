import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List

import pandas as pd


# ---------------------------------------
# Inter-device runner settings (edit me)
# ---------------------------------------
# encode: create artifacts/config on source machine
# decode: consume existing artifacts/config on destination machine
RUN_PHASE = "encode"  # encode | decode

# For encode: None -> auto-generated unique tag.
# For decode: must be set to the encode run tag.
RUN_TAG = None

MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = None
TRUST_REMOTE_CODE = False
TORCH_DTYPE = "float16"  # auto | float32 | float16 | bfloat16
DEVICE = "cuda"  # auto | cpu | cuda | mps

# Decode-side override: use this device for decoding regardless of encode device.
# Set to None to use encoded settings without an explicit decode-device override.
DECODE_DEVICE_OVERRIDE = "auto"  # None | auto | cpu | cuda | mps

IGNORE_MODEL_MAX_LENGTH_WARNING = True
ENABLE_OOM_FALLBACK = True
OOM_FALLBACK_STRATEGY = "block"  # rolling | block | no_kv_cache
OOM_FALLBACK_CONTEXT_WINDOW = 128
OOM_FALLBACK_MARGIN = 16

SAFE_MODE = True
PRECISION = 32
SLOTS = 1 << 24
CONTEXT_WINDOW = 1024
MARGIN = 128
STRATEGY = "rolling"  # rolling | block | no_kv_cache
USE_LEGACY_COUNTS = False
QUANT = False
LOGIT_ROUND_DECIMALS = 2
PROB_ROUND_DECIMALS = 5
# None | batch_invariant_ops | tbik
# `tbik` is CUDA/Triton/vLLM-oriented for this stack.
DETERMINISM_MODE = None
INFERENCE_BACKEND = "auto"  # auto | huggingface | vllm
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_MAX_LOGPROBS = None
VLLM_MAX_MODEL_LEN = None
MAX_DECODE_TOKENS = None
DIAGNOSTICS_ENABLED = False
# Optional diagnostics CSV prefix. If None and diagnostics is enabled, worker writes
# per-file CSVs under that file's artifact directory.
DIAGNOSTICS_CSV_PREFIX = None

# Optional demo modes in llm_codec_deterministic.
# When enabled, worker writes per-file CSVs under each file artifact directory.
DEMO_MODE = False
SPEED_DEMO = False
MEMORY_DEMO = False
MEMORY_SAMPLE_INTERVAL = 0.05
DIVERGENCE_WINDOW = 5

TEXT_ENCODING = "utf-8"
FILE_ENCODING_OVERRIDES = {
    "cp.html": "windows-1252",
}

# Inter-device flow requires keeping encode artifacts.
KEEP_ARTIFACTS = True
PHASE_TIMEOUT_SECONDS = 18000  # 0 disables timeout
STOP_ON_FILE_ERROR = True

# Encode-side file selection:
CANTRBRY_FILE_SELECTION = None
CURRENT_FOLDER_TEXT_SELECTION = None
MY_CORPUS_FILE_SELECTION = None
ARTIFICIAL_CORPUS_FILE_SELECTION = None

CANTRBRY_DIR = Path("cantrbry")
MY_CORPUS_DIR = Path("my_corpus")
ARTIFICIAL_CORPUS_DIR = Path("artificial_corpus")
OUTPUT_CSV_ENCODE = Path("results/roundtrip/deterministic_roundtrip_encode_results.csv")
OUTPUT_CSV_DECODE = Path("results/roundtrip/deterministic_roundtrip_decode_results.csv")
WORKER_SCRIPT = Path("deterministic_roundtrip_worker.py")
CONFIG_PATH = Path("results/roundtrip/deterministic_roundtrip_config.json")
DECODE_CONFIG_PATH = Path("results/roundtrip/deterministic_roundtrip_decode_config.json")
ARTIFACT_ROOT_BASE = Path(".roundtrip_artifacts")
ENCODE_ATTEMPT_MARKER = "__ROUNDTRIP_ENCODE_ATTEMPT__"

ENCODE_RESULT_COLUMNS = [
    "file",
    "status",
    "input_size_bytes",
    "safe_mode",
    "num_tokens",
    "encoded_size_bytes",
    "compression_ratio",
    "encode_seconds",
    "fallback_attempted",
    "attempt_count",
    "error",
]

DECODE_RESULT_COLUMNS = [
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
    "fallback_attempted",
    "attempt_count",
    "error",
]


def _with_tag(path: Path, tag: str) -> Path:
    return path.with_name(f"{path.stem}_{tag}{path.suffix}")


def _resolve_run_tag_for_phase() -> str:
    phase = RUN_PHASE.strip().lower()
    if phase == "encode":
        if RUN_TAG:
            return str(RUN_TAG)
        return f"{socket.gethostname()}_{os.getpid()}_{time.time_ns()}"
    if phase == "decode":
        if not RUN_TAG:
            raise ValueError("RUN_TAG must be set for decode phase.")
        return str(RUN_TAG)
    raise ValueError("RUN_PHASE must be either 'encode' or 'decode'.")


def _select_files(base_dir: Path, selection, txt_only: bool = False) -> List[Path]:
    if selection is None:
        return []

    if selection == "all":
        all_files = [p for p in base_dir.iterdir() if p.is_file()]
        if txt_only:
            all_files = [p for p in all_files if p.suffix.lower() == ".txt"]
        return sorted(all_files)

    if not isinstance(selection, list):
        raise ValueError("Selection must be None, 'all', or a list of filenames.")

    selected = []
    for rel in selection:
        candidate = (base_dir / rel).resolve()
        if not candidate.exists() or not candidate.is_file():
            raise FileNotFoundError(f"Selected file does not exist: {candidate}")
        if txt_only and candidate.suffix.lower() != ".txt":
            raise ValueError(f"Selected file is not a .txt file: {candidate}")
        selected.append(candidate)
    return selected


def _job_id(path: Path) -> str:
    import hashlib

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


def _parse_encode_attempt_info(stderr_text: str):
    attempt_count = 0
    fallback_attempted = False
    for line in (stderr_text or "").splitlines():
        if ENCODE_ATTEMPT_MARKER not in line:
            continue
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


def _read_text(path: Path, encoding: str) -> str:
    return path.read_text(encoding=encoding, errors="replace")


def _build_settings() -> Dict[str, Any]:
    return {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "torch_dtype": TORCH_DTYPE,
        "device": DEVICE,
        "ignore_model_max_length_warning": IGNORE_MODEL_MAX_LENGTH_WARNING,
        "enable_oom_fallback": ENABLE_OOM_FALLBACK,
        "oom_fallback_strategy": OOM_FALLBACK_STRATEGY,
        "oom_fallback_context_window": OOM_FALLBACK_CONTEXT_WINDOW,
        "oom_fallback_margin": OOM_FALLBACK_MARGIN,
        "safe_mode": SAFE_MODE,
        "precision": PRECISION,
        "slots": SLOTS,
        "context_window": CONTEXT_WINDOW,
        "margin": MARGIN,
        "strategy": STRATEGY,
        "use_legacy_counts": USE_LEGACY_COUNTS,
        "quant": QUANT,
        "logit_round_decimals": LOGIT_ROUND_DECIMALS,
        "prob_round_decimals": PROB_ROUND_DECIMALS,
        "determinism_mode": DETERMINISM_MODE,
        "inference_backend": INFERENCE_BACKEND,
        "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
        "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
        "vllm_max_logprobs": VLLM_MAX_LOGPROBS,
        "vllm_max_model_len": VLLM_MAX_MODEL_LEN,
        "max_decode_tokens": MAX_DECODE_TOKENS,
        "diagnostics_enabled": DIAGNOSTICS_ENABLED,
        "diagnostics_csv_prefix": DIAGNOSTICS_CSV_PREFIX,
        "demo_mode": DEMO_MODE,
        "speed_demo": SPEED_DEMO,
        "memory_demo": MEMORY_DEMO,
        "memory_sample_interval": MEMORY_SAMPLE_INTERVAL,
        "divergence_window": DIVERGENCE_WINDOW,
        "text_encoding": TEXT_ENCODING,
        "file_encoding_overrides": FILE_ENCODING_OVERRIDES,
        "keep_artifacts": KEEP_ARTIFACTS,
        "phase_timeout_seconds": PHASE_TIMEOUT_SECONDS,
        "stop_on_file_error": STOP_ON_FILE_ERROR,
    }


def _run_encode_phase(project_root: Path, worker: Path, run_tag: str):
    cantrbry_dir = (project_root / CANTRBRY_DIR).resolve()
    my_corpus_dir = (project_root / MY_CORPUS_DIR).resolve()
    artificial_corpus_dir = (project_root / ARTIFICIAL_CORPUS_DIR).resolve()
    if CANTRBRY_FILE_SELECTION is not None:
        if not cantrbry_dir.exists() or not cantrbry_dir.is_dir():
            raise FileNotFoundError(f"Cantrbry folder not found: {cantrbry_dir}")
    if MY_CORPUS_FILE_SELECTION is not None:
        if not my_corpus_dir.exists() or not my_corpus_dir.is_dir():
            raise FileNotFoundError(f"my_corpus folder not found: {my_corpus_dir}")
    if ARTIFICIAL_CORPUS_FILE_SELECTION is not None:
        if not artificial_corpus_dir.exists() or not artificial_corpus_dir.is_dir():
            raise FileNotFoundError(f"artificial_corpus folder not found: {artificial_corpus_dir}")

    cantrbry_files = _select_files(cantrbry_dir, CANTRBRY_FILE_SELECTION)
    my_corpus_files = _select_files(my_corpus_dir, MY_CORPUS_FILE_SELECTION, txt_only=True)
    artificial_corpus_files = _select_files(
        artificial_corpus_dir,
        ARTIFICIAL_CORPUS_FILE_SELECTION,
        txt_only=True,
    )
    current_folder_text_files = _select_files(project_root, CURRENT_FOLDER_TEXT_SELECTION, txt_only=True)
    files = sorted(
        set(
            cantrbry_files
            + my_corpus_files
            + artificial_corpus_files
            + current_folder_text_files
        )
    )
    if not files:
        raise RuntimeError("No files selected for encode phase.")

    settings = _build_settings()
    artifact_root = (project_root / ARTIFACT_ROOT_BASE / run_tag).resolve()
    output_csv_path = (project_root / _with_tag(OUTPUT_CSV_ENCODE, run_tag)).resolve()
    config_path = (project_root / _with_tag(CONFIG_PATH, run_tag)).resolve()

    config = {
        "files": [str(p) for p in files],
        "settings": settings,
        "model_name": MODEL_ID,
        "output_csv": str(output_csv_path),
        "artifact_root": str(artifact_root),
        "run_tag": run_tag,
    }
    config_path.parent.mkdir(parents=True, exist_ok=True)
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    rows = []
    phase_timeout_seconds = int(settings.get("phase_timeout_seconds", 0) or 0)
    artifact_root.mkdir(parents=True, exist_ok=True)

    for file_path in files:
        job_dir = artifact_root / _job_id(file_path)
        if job_dir.exists():
            for child in job_dir.glob("*"):
                if child.is_file():
                    child.unlink()
        job_dir.mkdir(parents=True, exist_ok=True)

        cmd = [
            sys.executable,
            str(worker),
            "--phase",
            "encode",
            "--config",
            str(config_path),
            "--file",
            str(file_path),
            "--artifact-dir",
            str(job_dir),
        ]
        rc, err = _run_subprocess(cmd, timeout_seconds=phase_timeout_seconds)
        if rc != 0:
            attempt_count, fallback_attempted = _parse_encode_attempt_info(err or "")
            rows.append(
                {
                    "file": str(file_path),
                    "status": "encode_failed",
                    "num_tokens": None,
                    "error": _sanitize_error_text(err or "Encode phase failed"),
                    "fallback_attempted": fallback_attempted,
                    "attempt_count": attempt_count,
                }
            )
            continue

        encode_meta = json.loads((job_dir / "encode_metadata.json").read_text(encoding="utf-8"))
        rows.append(
            {
                "file": str(file_path),
                "status": "encoded",
                "input_size_bytes": encode_meta.get("original_size_bytes", 0),
                "safe_mode": encode_meta.get("safe_mode", True),
                "num_tokens": encode_meta.get("num_tokens"),
                "encoded_size_bytes": encode_meta.get("encoded_size_bytes", 0),
                "compression_ratio": (
                    (encode_meta.get("encoded_size_bytes", 0) * 8) / encode_meta.get("original_size_bytes", 1)
                    if encode_meta.get("original_size_bytes", 0) > 0
                    else None
                ),
                "encode_seconds": encode_meta.get("encode_seconds", None),
                "fallback_attempted": bool(encode_meta.get("used_oom_fallback", False)),
                "attempt_count": 2 if bool(encode_meta.get("used_oom_fallback", False)) else 1,
                "error": "",
            }
        )

    pd.DataFrame(rows).reindex(columns=ENCODE_RESULT_COLUMNS).to_csv(output_csv_path, index=False)

    print(f"Run tag: {run_tag}")
    print(f"Encode config written to: {config_path}")
    print(f"Encode artifacts root: {artifact_root}")
    print(f"Encode status CSV written to: {output_csv_path}")


def _run_decode_phase(project_root: Path, worker: Path, run_tag: str):
    config_path = (project_root / _with_tag(CONFIG_PATH, run_tag)).resolve()
    if not config_path.exists():
        raise FileNotFoundError(f"Encode config not found for run tag {run_tag}: {config_path}")

    base_config = json.loads(config_path.read_text(encoding="utf-8"))
    files = [Path(p) for p in base_config.get("files", [])]
    if not files:
        raise RuntimeError("No files found in encode config.")

    settings = dict(base_config.get("settings", {}))
    if DECODE_DEVICE_OVERRIDE is not None:
        settings["decode_device_override"] = str(DECODE_DEVICE_OVERRIDE)

    decode_tag = f"{run_tag}_{socket.gethostname()}"
    decode_csv_path = (project_root / _with_tag(OUTPUT_CSV_DECODE, decode_tag)).resolve()
    decode_config_path = (project_root / _with_tag(DECODE_CONFIG_PATH, decode_tag)).resolve()

    decode_config = dict(base_config)
    decode_config["settings"] = settings
    decode_config["model_name"] = base_config.get("model_name", MODEL_ID)
    decode_config["output_csv"] = str(decode_csv_path)
    decode_config["decode_host"] = socket.gethostname()
    decode_config_path.parent.mkdir(parents=True, exist_ok=True)
    decode_csv_path.parent.mkdir(parents=True, exist_ok=True)
    decode_config_path.write_text(json.dumps(decode_config, indent=2), encoding="utf-8")

    artifact_root = Path(base_config.get("artifact_root", str(project_root / ARTIFACT_ROOT_BASE / run_tag)))
    phase_timeout_seconds = int(settings.get("phase_timeout_seconds", 0) or 0)

    if not artifact_root.exists():
        raise FileNotFoundError(
            "Decode machine cannot see encode artifacts root: "
            f"{artifact_root}. Ensure shared storage is mounted and RUN_TAG is correct."
        )
    if not artifact_root.is_dir():
        raise NotADirectoryError(f"Artifact root is not a directory: {artifact_root}")
    if not os.access(artifact_root, os.R_OK | os.X_OK):
        raise PermissionError(
            f"Artifact root is not accessible on this machine: {artifact_root}"
        )

    rows = []
    for file_path in files:
        job_dir = artifact_root / _job_id(file_path)
        if not job_dir.exists():
            rows.append(
                {
                    "file": str(file_path),
                    "status": "decode_failed",
                    "num_tokens": None,
                    "error": f"Missing artifact directory: {job_dir}",
                    "fallback_attempted": False,
                    "attempt_count": 0,
                }
            )
            continue

        cmd = [
            sys.executable,
            str(worker),
            "--phase",
            "decode",
            "--config",
            str(decode_config_path),
            "--artifact-dir",
            str(job_dir),
        ]
        rc, err = _run_subprocess(cmd, timeout_seconds=phase_timeout_seconds)
        if rc != 0:
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
                    "num_tokens": None,
                    "error": _sanitize_error_text(err or "Decode phase failed"),
                    "fallback_attempted": fallback_attempted,
                    "attempt_count": attempt_count,
                }
            )
            continue

        encode_meta = json.loads((job_dir / "encode_metadata.json").read_text(encoding="utf-8"))
        decode_meta = json.loads((job_dir / "decode_metadata.json").read_text(encoding="utf-8"))

        file_encoding = encode_meta.get("text_encoding", settings.get("text_encoding", "utf-8"))
        original_text = _read_text(file_path, file_encoding)
        decoded_text = (job_dir / "decoded.txt").read_text(encoding=file_encoding, errors="replace")
        is_match = original_text == decoded_text

        rows.append(
            {
                "file": str(file_path),
                "status": "ok" if is_match else "mismatch",
                "input_size_bytes": encode_meta.get("original_size_bytes", 0),
                "safe_mode": encode_meta.get("safe_mode", True),
                "num_tokens": encode_meta.get("num_tokens"),
                "encoded_size_bytes": encode_meta.get("encoded_size_bytes", 0),
                "compression_ratio": (
                    (encode_meta.get("encoded_size_bytes", 0) * 8) / encode_meta.get("original_size_bytes", 1)
                    if encode_meta.get("original_size_bytes", 0) > 0
                    else None
                ),
                "encode_seconds": encode_meta.get("encode_seconds", None),
                "decode_seconds": decode_meta.get("decode_seconds", None),
                "original_chars": len(original_text),
                "decoded_chars": decode_meta.get("decoded_chars", 0),
                "fallback_attempted": bool(encode_meta.get("used_oom_fallback", False)),
                "attempt_count": 2 if bool(encode_meta.get("used_oom_fallback", False)) else 1,
                "error": "",
            }
        )

    pd.DataFrame(rows).reindex(columns=DECODE_RESULT_COLUMNS).to_csv(decode_csv_path, index=False)

    print(f"Run tag: {run_tag}")
    print(f"Decode config used: {decode_config_path}")
    print(f"Decode status CSV written to: {decode_csv_path}")
    print(f"Artifact root used: {artifact_root}")


def main():
    project_root = Path(__file__).resolve().parent
    worker = (project_root / WORKER_SCRIPT).resolve()
    if not worker.exists():
        raise FileNotFoundError(f"Worker script not found: {worker}")

    run_tag = _resolve_run_tag_for_phase()
    phase = RUN_PHASE.strip().lower()

    if phase == "encode":
        _run_encode_phase(project_root, worker, run_tag)
        return

    if phase == "decode":
        _run_decode_phase(project_root, worker, run_tag)
        return

    raise ValueError("RUN_PHASE must be either 'encode' or 'decode'.")


if __name__ == "__main__":
    main()
