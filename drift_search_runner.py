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

SAFE_MODE = True
MAX_DECODE_TOKENS = None
SHOW_PROGRESS = False
TEXT_ENCODING = "utf-8"
PASS_REQUIRES_ZERO_CORRECTION = True

# Dataset selection
CURRENT_FOLDER_TEXT_SELECTION = ["Shall I Compare Thee To a Summer's Day.txt"]

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

OUTPUT_CSV = Path("drift_search_results.csv")
CONFIG_PATH = Path("drift_search_config.json")
WORKER_SCRIPT = Path("drift_search_worker.py")
ARTIFACT_ROOT_BASE = Path(".drift_search_artifacts")


def _resolve_run_tag() -> str:
    return f"{socket.gethostname()}_{os.getpid()}_{time.time_ns()}"


def _with_tag(path: Path, run_tag: str) -> Path:
    return path.with_name(f"{path.stem}_{run_tag}{path.suffix}")


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

    worker = (project_root / WORKER_SCRIPT).resolve()
    if not worker.exists():
        raise FileNotFoundError(f"Worker script not found: {worker}")

    files = []
    for rel in CURRENT_FOLDER_TEXT_SELECTION:
        p = (project_root / rel).resolve()
        if not p.exists() or not p.is_file():
            raise FileNotFoundError(f"Selected file not found: {p}")
        files.append(str(p))

    trials = _build_trials()
    if not trials:
        raise RuntimeError("No trials generated. Check search grid settings.")

    output_csv_path = (project_root / _with_tag(OUTPUT_CSV, run_tag)).resolve()
    config_path = (project_root / _with_tag(CONFIG_PATH, run_tag)).resolve()
    artifact_root = (project_root / ARTIFACT_ROOT_BASE / run_tag).resolve()

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
