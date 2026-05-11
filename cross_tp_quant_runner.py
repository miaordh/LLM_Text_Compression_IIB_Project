import csv
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


def _env_str(name: str, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    if isinstance(default, str):
        return value
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return value


def _env_int(name: str, default: int) -> int:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return int(value)


def _env_float(name: str, default: float) -> float:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return float(value)


def _env_bool(name: str, default: bool) -> bool:
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def _env_optional_int(name: str, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return int(value)


def _parse_modes(value: Optional[str]) -> List[Optional[str]]:
    if not value:
        return [None, "tbik"]
    modes: List[Optional[str]] = []
    for item in value.split(","):
        text = item.strip().lower()
        if not text:
            continue
        if text in {"none", "null", "off", "0"}:
            modes.append(None)
        else:
            modes.append(text)
    return modes or [None, "tbik"]


def _parse_int_list(value: Optional[str], default: List[int]) -> List[int]:
    if not value:
        return default
    return [int(item.strip()) for item in value.split(",") if item.strip()]


def _parse_rounding_pairs(value: Optional[str]) -> List[Tuple[int, int]]:
    if not value:
        return [
            (15, 5),
            (10, 3),
            (5, 2),
            (2, 1),
            (1, 1),
            (0, 0),
        ]
    pairs: List[Tuple[int, int]] = []
    for item in value.split(","):
        text = item.strip()
        if not text:
            continue
        left, right = text.split(":", 1)
        pairs.append((int(left.strip()), int(right.strip())))
    return pairs


def _resolve_run_tag() -> str:
    run_tag = os.environ.get("CODEC_RUN_TAG")
    if run_tag:
        return str(run_tag)
    return f"{socket.gethostname()}_{os.getpid()}_{time.time_ns()}"


def _with_tag(path: Path, run_tag: str) -> Path:
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


def _select_files(project_root: Path) -> List[Path]:
    raw = os.environ.get("CODEC_CROSS_TP_FILES")
    if raw:
        files = [(project_root / item.strip()).resolve() for item in raw.split(",") if item.strip()]
    else:
        files = [(project_root / "my_corpus" / "sonnet.txt").resolve()]
    for path in files:
        if not path.exists() or not path.is_file():
            raise FileNotFoundError(f"Selected file does not exist: {path}")
    return files


def _build_trials() -> List[Dict[str, Any]]:
    modes = _parse_modes(os.environ.get("CODEC_CROSS_TP_DETERMINISM_MODES"))
    slots_values = _parse_int_list(
        os.environ.get("CODEC_CROSS_TP_SLOTS_VALUES"),
        [1 << 24, 1 << 20, 1 << 18, 1 << 16],
    )
    rounding_pairs = _parse_rounding_pairs(os.environ.get("CODEC_CROSS_TP_ROUNDING_PAIRS"))
    include_unquantized = _env_bool("CODEC_CROSS_TP_INCLUDE_UNQUANTIZED", True)
    max_trials = _env_optional_int("CODEC_CROSS_TP_MAX_TRIALS", None)

    trials: List[Dict[str, Any]] = []
    trial_index = 0
    for mode in modes:
        if include_unquantized:
            trial_index += 1
            trials.append(
                {
                    "trial_id": f"trial_{trial_index:04d}",
                    "determinism_mode": mode,
                    "quant": False,
                    "slots": 1 << 24,
                    "logit_round_decimals": -1,
                    "prob_round_decimals": -1,
                }
            )

        for slots in slots_values:
            for logit_round_decimals, prob_round_decimals in rounding_pairs:
                trial_index += 1
                trials.append(
                    {
                        "trial_id": f"trial_{trial_index:04d}",
                        "determinism_mode": mode,
                        "quant": True,
                        "slots": int(slots),
                        "logit_round_decimals": int(logit_round_decimals),
                        "prob_round_decimals": int(prob_round_decimals),
                    }
                )

    if max_trials is not None:
        trials = trials[: int(max_trials)]
    return trials


def _base_settings(trial: Dict[str, Any]) -> Dict[str, Any]:
    encode_tp = _env_int("CODEC_ENCODE_VLLM_TENSOR_PARALLEL_SIZE", 1)
    decode_tp = _env_int("CODEC_DECODE_VLLM_TENSOR_PARALLEL_SIZE", 2)
    attention_backend = _env_str("CODEC_VLLM_ATTENTION_BACKEND", "TRITON_ATTN")
    use_v1 = _env_str("CODEC_VLLM_USE_V1", "1")

    return {
        "model_id": _env_str("CODEC_MODEL_ID", "deepseek-ai/deepseek-coder-1.3b-base"),
        "revision": _env_str("CODEC_REVISION", None),
        "trust_remote_code": False,
        "torch_dtype": _env_str("CODEC_TORCH_DTYPE", "auto"),
        "device": _env_str("CODEC_DEVICE", "cuda"),
        "device_mode": "single_device",
        "ignore_model_max_length_warning": True,
        "enable_oom_fallback": False,
        "safe_mode": True,
        "precision": _env_int("CODEC_PRECISION", 32),
        "slots": int(trial["slots"]),
        "context_window": _env_int("CODEC_CONTEXT_WINDOW", 100),
        "margin": _env_int("CODEC_MARGIN", 16),
        "strategy": _env_str("CODEC_STRATEGY", "rolling"),
        "use_legacy_counts": _env_bool("CODEC_USE_LEGACY_COUNTS", False),
        "quant": bool(trial["quant"]),
        "logit_round_decimals": int(trial["logit_round_decimals"]),
        "prob_round_decimals": int(trial["prob_round_decimals"]),
        "determinism_mode": trial["determinism_mode"],
        "inference_backend": "vllm",
        "vllm_tensor_parallel_size": encode_tp,
        "decode_vllm_tensor_parallel_size": decode_tp,
        "vllm_gpu_memory_utilization": _env_float("CODEC_VLLM_GPU_MEMORY_UTILIZATION", 0.9),
        "vllm_attention_backend": attention_backend,
        "decode_vllm_attention_backend": attention_backend,
        "vllm_use_v1": use_v1,
        "decode_vllm_use_v1": use_v1,
        "vllm_max_logprobs": _env_optional_int("CODEC_VLLM_MAX_LOGPROBS", None),
        "vllm_max_model_len": _env_optional_int("CODEC_VLLM_MAX_MODEL_LEN", None),
        "max_decode_tokens": None,
        "diagnostics_enabled": _env_bool("CODEC_DIAGNOSTICS_ENABLED", False),
        "diagnostics_csv_prefix": None,
        "demo_mode": _env_bool("CODEC_DEMO_MODE", False),
        "speed_demo": _env_bool("CODEC_SPEED_DEMO", False),
        "memory_demo": _env_bool("CODEC_MEMORY_DEMO", False),
        "memory_sample_interval": 0.05,
        "divergence_window": 5,
        "text_encoding": "utf-8",
        "file_encoding_overrides": {"cp.html": "windows-1252"},
        "keep_artifacts": True,
        "phase_timeout_seconds": _env_int("CODEC_PHASE_TIMEOUT_SECONDS", 0),
        "stop_on_file_error": True,
    }


def _read_result_rows(path: Path) -> List[Dict[str, Any]]:
    if not path.exists():
        return []
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _write_rows(path: Path, rows: List[Dict[str, Any]]):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames: List[str] = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def main():
    project_root = Path(__file__).resolve().parent
    worker = (project_root / "deterministic_roundtrip_worker.py").resolve()
    if not worker.exists():
        raise FileNotFoundError(f"Worker script not found: {worker}")

    run_tag = _resolve_run_tag()
    files = _select_files(project_root)
    trials = _build_trials()
    if not trials:
        raise RuntimeError("No cross-TP quantization trials were generated.")

    output_csv = (
        project_root
        / _with_tag(Path("results/cross_tp_quant/cross_tp_quant_results.csv"), run_tag)
    ).resolve()
    config_root = (project_root / "results/cross_tp_quant" / f"configs_{run_tag}").resolve()
    artifact_root = (project_root / ".cross_tp_quant_artifacts" / run_tag).resolve()
    config_root.mkdir(parents=True, exist_ok=True)
    artifact_root.mkdir(parents=True, exist_ok=True)

    aggregate_rows: List[Dict[str, Any]] = []
    for trial in trials:
        trial_id = str(trial["trial_id"])
        settings = _base_settings(trial)
        trial_output_csv = (config_root / f"{trial_id}_results.csv").resolve()
        trial_artifact_root = (artifact_root / trial_id).resolve()
        trial_config_path = (config_root / f"{trial_id}_config.json").resolve()

        config = {
            "files": [str(path) for path in files],
            "settings": settings,
            "model_name": settings["model_id"],
            "output_csv": str(trial_output_csv),
            "artifact_root": str(trial_artifact_root),
            "run_tag": f"{run_tag}_{trial_id}",
        }
        trial_config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

        env = os.environ.copy()
        if settings.get("vllm_attention_backend"):
            env["VLLM_ATTENTION_BACKEND"] = str(settings["vllm_attention_backend"])
        if settings.get("vllm_use_v1") is not None:
            env["VLLM_USE_V1"] = str(settings["vllm_use_v1"])

        cmd = [
            sys.executable,
            str(worker),
            "--phase",
            "run",
            "--config",
            str(trial_config_path),
        ]
        started = time.time()
        completed = subprocess.run(cmd, env=env)
        elapsed = time.time() - started

        trial_rows = _read_result_rows(trial_output_csv)
        if not trial_rows:
            trial_rows = [
                {
                    "file": "",
                    "status": "runner_failed",
                    "error": f"worker exit code {completed.returncode}",
                }
            ]

        for row in trial_rows:
            enriched = {
                "trial_id": trial_id,
                "trial_seconds": f"{elapsed:.6f}",
                "trial_returncode": completed.returncode,
                "determinism_mode": trial["determinism_mode"],
                "quant": trial["quant"],
                "slots": trial["slots"],
                "logit_round_decimals": trial["logit_round_decimals"],
                "prob_round_decimals": trial["prob_round_decimals"],
                "encode_tp": settings["vllm_tensor_parallel_size"],
                "decode_tp": settings["decode_vllm_tensor_parallel_size"],
                "vllm_attention_backend": settings["vllm_attention_backend"],
                "vllm_use_v1": settings["vllm_use_v1"],
                "trial_config": str(trial_config_path),
                "trial_artifact_root": str(trial_artifact_root),
            }
            enriched.update(row)
            aggregate_rows.append(enriched)
        _write_rows(output_csv, aggregate_rows)

    print(f"Run tag: {run_tag}")
    print(f"Trials: {len(trials)}")
    print(f"Results CSV: {output_csv}")
    print(f"Config root: {config_root}")
    print(f"Artifact root: {artifact_root}")


if __name__ == "__main__":
    main()
