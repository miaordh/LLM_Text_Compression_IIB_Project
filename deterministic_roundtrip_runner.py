import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List


# ---------------------------
# Runner settings (edit me)
# ---------------------------
MODEL_ID = "deepseek-ai/deepseek-coder-1.3b-base"
REVISION = None
TRUST_REMOTE_CODE = False
TORCH_DTYPE = "auto"  # auto | float32 | float16 | bfloat16
DEVICE = "cpu"  # auto | cpu | cuda | mps

SAFE_MODE = True
PRECISION = 32
SLOTS = 1 << 24
USE_KV_CACHE = False
QUANT = False
LOGIT_ROUND_DECIMALS = 2
PROB_ROUND_DECIMALS = 5
USE_BATCH_INVARIANT_OPS = True
MAX_DECODE_TOKENS = None

TEXT_ENCODING = "utf-8"
# Optional per-file text encoding overrides.
# Keys can be bare filenames (e.g. "cp.html") or relative paths.
FILE_ENCODING_OVERRIDES = {
    "cp.html": "windows-1252",
}

KEEP_ARTIFACTS = False
PHASE_TIMEOUT_SECONDS = 10800 # 0 disables timeout
STOP_ON_FILE_ERROR = True

# Parallel-run safety:
# - RUN_TAG = None -> auto-generate unique tag per run (recommended)
# - RUN_TAG = "mytag" -> fixed tag (you manage collisions)
RUN_TAG = None

# File selection for cantrbry:
# - None: run nothing from CANTRBRY_DIR
# - "all": run every file under CANTRBRY_DIR
# - list[str]: explicit relative filenames under CANTRBRY_DIR
CANTRBRY_FILE_SELECTION = ["fields.c"]

# File selection for project root text files:
# - None: run nothing from project root
# - "all": run every .txt file directly under project root
# - list[str]: explicit relative filenames under project root
CURRENT_FOLDER_TEXT_SELECTION = ["Shall I Compare Thee To a Summer's Day.txt"]

CANTRBRY_DIR = Path("cantrbry")
OUTPUT_CSV = Path("deterministic_roundtrip_results.csv")
WORKER_SCRIPT = Path("deterministic_roundtrip_worker.py")
CONFIG_PATH = Path("deterministic_roundtrip_config.json")
ARTIFACT_ROOT_BASE = Path(".roundtrip_artifacts")


def _resolve_run_tag() -> str:
    if RUN_TAG:
        return str(RUN_TAG)
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


def main():
    project_root = Path(__file__).resolve().parent
    run_tag = _resolve_run_tag()
    cantrbry_dir = (project_root / CANTRBRY_DIR).resolve()
    worker = (project_root / WORKER_SCRIPT).resolve()

    if CANTRBRY_FILE_SELECTION is not None:
        if not cantrbry_dir.exists() or not cantrbry_dir.is_dir():
            raise FileNotFoundError(f"Cantrbry folder not found: {cantrbry_dir}")
    if not worker.exists():
        raise FileNotFoundError(f"Worker script not found: {worker}")

    cantrbry_files = _select_files(cantrbry_dir, CANTRBRY_FILE_SELECTION)
    current_folder_text_files = _select_files(
        project_root,
        CURRENT_FOLDER_TEXT_SELECTION,
        txt_only=True,
    )

    files = sorted(set(cantrbry_files + current_folder_text_files))
    if not files:
        raise RuntimeError("No files selected for round-trip test from either location.")

    settings = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "torch_dtype": TORCH_DTYPE,
        "device": DEVICE,
        "safe_mode": SAFE_MODE,
        "precision": PRECISION,
        "slots": SLOTS,
        "use_kv_cache": USE_KV_CACHE,
        "quant": QUANT,
        "logit_round_decimals": LOGIT_ROUND_DECIMALS,
        "prob_round_decimals": PROB_ROUND_DECIMALS,
        "use_batch_invariant_ops": USE_BATCH_INVARIANT_OPS,
        "max_decode_tokens": MAX_DECODE_TOKENS,
        "text_encoding": TEXT_ENCODING,
        "file_encoding_overrides": FILE_ENCODING_OVERRIDES,
        "keep_artifacts": KEEP_ARTIFACTS,
        "phase_timeout_seconds": PHASE_TIMEOUT_SECONDS,
        "stop_on_file_error": STOP_ON_FILE_ERROR,
    }

    output_csv_path = (project_root / _with_tag(OUTPUT_CSV, run_tag)).resolve()
    config_path = (project_root / _with_tag(CONFIG_PATH, run_tag)).resolve()
    artifact_root = (project_root / ARTIFACT_ROOT_BASE / run_tag).resolve()

    config = {
        "files": [str(p) for p in files],
        "settings": settings,
        "output_csv": str(output_csv_path),
        "artifact_root": str(artifact_root),
        "run_tag": run_tag,
    }

    config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")

    cmd = [
        sys.executable,
        str(worker),
        "--phase",
        "run",
        "--config",
        str(config_path),
    ]

    completed = subprocess.run(cmd)
    if completed.returncode != 0:
        raise RuntimeError(f"Worker failed with exit code {completed.returncode}")

    print(f"Run tag: {run_tag}")
    print(f"Round-trip results written to: {output_csv_path}")
    print(f"Run config written to: {config_path}")
    print(f"Run artifacts root: {artifact_root}")


if __name__ == "__main__":
    main()
