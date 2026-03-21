import itertools
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import Dict, List


# ---------------------------
# Drift search settings
# ---------------------------
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"
REVISION = None
TRUST_REMOTE_CODE = False
TORCH_DTYPE = "float16"  # auto | float32 | float16 | bfloat16
DEVICE = "mps"  # auto | cpu | cuda | mps
DEVICE_MODE = "cross_device"  # single_device | cross_device
INFERENCE_BACKEND = "auto"  # auto | huggingface | vllm
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_MAX_LOGPROBS = None

SAFE_MODE = True
MAX_DECODE_TOKENS = None
SHOW_PROGRESS = False
TEXT_ENCODING = "utf-8"
PASS_REQUIRES_ZERO_CORRECTION = True

# Dataset selection
CURRENT_FOLDER_TEXT_SELECTION = ["my_corpus/Shall I Compare Thee To a Summer's Day.txt"]

# File selection for my_corpus:
# - None: run nothing from MY_CORPUS_DIR
# - "all": run every .txt file under MY_CORPUS_DIR
# - list[str]: explicit filenames relative to MY_CORPUS_DIR
MY_CORPUS_FILE_SELECTION = None

# File selection for artificial_corpus:
# - None: run nothing from ARTIFICIAL_CORPUS_DIR
# - "all": run every .txt file under ARTIFICIAL_CORPUS_DIR
# - list[str]: explicit filenames relative to ARTIFICIAL_CORPUS_DIR
ARTIFICIAL_CORPUS_FILE_SELECTION = None

# Search grid knobs
DETERMINISM_MODES = [None]
QUANT_VALUES = [True]
SLOTS_VALUES = [1 << 15]
# Pair logit/prob rounding together to reduce the search loop depth.
# Format: (logit_round_decimals, prob_round_decimals)
ROUNDING_PAIRS = [
    (10, 2),
    (5, 1),
    (5, 0),
]

# Codec behavior knobs
PRECISION = 32
CONTEXT_WINDOW = 1024
MARGIN = 128
STRATEGY = "rolling"
USE_LEGACY_COUNTS = False
DRIFT_CORRECTION_ENABLED = True
EMIT_FULL_REFERENCE_TRACE = True

# Optional cap for quick experiments
MAX_TRIALS = None

OUTPUT_CSV = Path("results/drift_search/drift_search_results.csv")
CONFIG_PATH = Path("results/drift_search/drift_search_config.json")
WORKER_SCRIPT = Path("drift_search_worker.py")
ARTIFACT_ROOT_BASE = Path(".drift_search_artifacts")
MY_CORPUS_DIR = Path("my_corpus")
ARTIFICIAL_CORPUS_DIR = Path("artificial_corpus")


def _resolve_run_tag() -> str:
    return f"{socket.gethostname()}_{os.getpid()}_{time.time_ns()}"


def _with_tag(path: Path, run_tag: str) -> Path:
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


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


def _build_trials() -> List[Dict]:
    trials: List[Dict] = []
    trial_index = 0
    canonical_slots = 1 << 24

    # Keep search compact while still scanning determinism/quant knobs.
    for determinism_mode, quant in itertools.product(DETERMINISM_MODES, QUANT_VALUES):
        if quant:
            iterator = itertools.product(SLOTS_VALUES, ROUNDING_PAIRS)
            iter_rows = [
                (slots, logit_round_decimals, prob_round_decimals)
                for slots, (logit_round_decimals, prob_round_decimals) in iterator
            ]
        else:
            # Quant-off baseline: run once only with fixed slots and no rounding.
            iter_rows = [(canonical_slots, -1, -1)]

        for slots, logit_round_decimals, prob_round_decimals in iter_rows:
            trial_index += 1
            trial = {
                "trial_id": f"trial_{trial_index:04d}",
                "determinism_mode": determinism_mode,
                "quant": quant,
                "slots": int(slots),
                "logit_round_decimals": int(logit_round_decimals),
                "prob_round_decimals": int(prob_round_decimals),
                "precision": int(PRECISION),
                "context_window": int(CONTEXT_WINDOW),
                "margin": int(MARGIN),
                "strategy": str(STRATEGY),
                "use_legacy_counts": bool(USE_LEGACY_COUNTS),
                "drift_correction_enabled": bool(DRIFT_CORRECTION_ENABLED),
                "emit_full_reference_trace": bool(EMIT_FULL_REFERENCE_TRACE),
            }
            trials.append(trial)

    if MAX_TRIALS is not None:
        trials = trials[: int(MAX_TRIALS)]

    return trials


def main():
    project_root = Path(__file__).resolve().parent
    run_tag = _resolve_run_tag()
    my_corpus_dir = (project_root / MY_CORPUS_DIR).resolve()
    artificial_corpus_dir = (project_root / ARTIFICIAL_CORPUS_DIR).resolve()

    worker = (project_root / WORKER_SCRIPT).resolve()
    if not worker.exists():
        raise FileNotFoundError(f"Worker script not found: {worker}")

    if MY_CORPUS_FILE_SELECTION is not None:
        if not my_corpus_dir.exists() or not my_corpus_dir.is_dir():
            raise FileNotFoundError(f"my_corpus folder not found: {my_corpus_dir}")
    if ARTIFICIAL_CORPUS_FILE_SELECTION is not None:
        if not artificial_corpus_dir.exists() or not artificial_corpus_dir.is_dir():
            raise FileNotFoundError(f"artificial_corpus folder not found: {artificial_corpus_dir}")

    current_folder_files = _select_files(project_root, CURRENT_FOLDER_TEXT_SELECTION, txt_only=True)
    my_corpus_files = _select_files(my_corpus_dir, MY_CORPUS_FILE_SELECTION, txt_only=True)
    artificial_corpus_files = _select_files(
        artificial_corpus_dir,
        ARTIFICIAL_CORPUS_FILE_SELECTION,
        txt_only=True,
    )
    files = [str(p) for p in sorted(set(current_folder_files + my_corpus_files + artificial_corpus_files))]
    if not files:
        raise RuntimeError("No files selected for drift search.")

    trials = _build_trials()
    if not trials:
        raise RuntimeError("No trials generated. Check search grid settings.")

    output_csv_path = (project_root / _with_tag(OUTPUT_CSV, run_tag)).resolve()
    config_path = (project_root / _with_tag(CONFIG_PATH, run_tag)).resolve()
    artifact_root = (project_root / ARTIFACT_ROOT_BASE / run_tag).resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "files": files,
        "trials": trials,
        "model_name": MODEL_ID,
        "settings": {
            "model_id": MODEL_ID,
            "revision": REVISION,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "torch_dtype": TORCH_DTYPE,
            "device": DEVICE,
            "device_mode": DEVICE_MODE,
            "inference_backend": INFERENCE_BACKEND,
            "vllm_tensor_parallel_size": VLLM_TENSOR_PARALLEL_SIZE,
            "vllm_gpu_memory_utilization": VLLM_GPU_MEMORY_UTILIZATION,
            "vllm_max_logprobs": VLLM_MAX_LOGPROBS,
            "safe_mode": SAFE_MODE,
            "max_decode_tokens": MAX_DECODE_TOKENS,
            "show_progress": SHOW_PROGRESS,
            "text_encoding": TEXT_ENCODING,
            "pass_requires_zero_correction": PASS_REQUIRES_ZERO_CORRECTION,
            "ignore_model_max_length_warning": True,
        },
        "output_csv": str(output_csv_path),
        "artifact_root": str(artifact_root),
        "run_tag": run_tag,
    }

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(worker),
        "--config",
        str(config_path),
    ]
    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"drift_search_worker failed with exit code {completed.returncode}")

    print(f"Run tag: {run_tag}")
    print(f"Search config written to: {config_path}")
    print(f"Search artifacts root: {artifact_root}")
    print(f"Search results CSV: {output_csv_path}")


if __name__ == "__main__":
    main()
