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
MODEL_ID = "Qwen/Qwen2.5-0.5B"
REVISION = None
TRUST_REMOTE_CODE = False
TORCH_DTYPE = "float32"  # auto | float32 | float16 | bfloat16
DEVICE = "cpu"  # auto | cpu | cuda | mps
DEVICE_MODE = "single_device"  # single_device | cross_device

# Large-file safety knobs (used by worker):
# - ignore_model_max_length_warning suppresses benign tokenizer warnings when
#   full-file tokenization is intentional.
# - enable_oom_fallback retries encode with smaller KV settings on CUDA OOM.
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
LOGIT_ROUND_DECIMALS = 15
PROB_ROUND_DECIMALS = 1
# None | batch_invariant_ops | tbik
# In cross_device mode, worker encodes on CPU and decodes on accelerator.
DETERMINISM_MODE = "tbik"
MAX_DECODE_TOKENS = None
DIAGNOSTICS_ENABLED = True
# Optional diagnostics CSV prefix. If None and diagnostics is enabled, worker writes
# per-file CSVs under that file's artifact directory.
DIAGNOSTICS_CSV_PREFIX = None

# Optional demo modes in llm_codec_deterministic.
# When enabled, worker writes per-file CSVs under each file artifact directory.
DEMO_MODE = True
SPEED_DEMO = True
MEMORY_DEMO = True
MEMORY_SAMPLE_INTERVAL = 0.05
DIVERGENCE_WINDOW = 5

TEXT_ENCODING = "utf-8"
# Optional per-file text encoding overrides.
# Keys can be bare filenames (e.g. "cp.html") or relative paths.
FILE_ENCODING_OVERRIDES = {
    "cp.html": "windows-1252",
}

KEEP_ARTIFACTS = False
PHASE_TIMEOUT_SECONDS = 18000 # 0 disables timeout
# Per-file behavior: when True, a file stops immediately on encode/decode error,
# but the run still continues with remaining files.
STOP_ON_FILE_ERROR = True

# Parallel-run safety:
# - RUN_TAG = None -> auto-generate unique tag per run (recommended)
# - RUN_TAG = "mytag" -> fixed tag (you manage collisions)
RUN_TAG = None

# File selection for cantrbry:
# - None: run nothing from CANTRBRY_DIR
# - "all": run every file under CANTRBRY_DIR
# - list[str]: explicit relative filenames under CANTRBRY_DIR
# CANTRBRY_FILE_SELECTION = ["alice29.txt", "asyoulik.txt", "cp.html", "kennedy.xls", "lcet10.txt", "plrabn12.txt", "ptt5"]
CANTRBRY_FILE_SELECTION = None

# File selection for text files using project-relative paths:
# - None: run nothing from project-relative selections
# - "all": run every .txt file directly under project root
# - list[str]: explicit relative filenames (for example under my_corpus/)
# CURRENT_FOLDER_TEXT_SELECTION = ["my_corpus/Shall I Compare Thee To a Summer's Day.txt", "my_corpus/再别康桥.txt"]
CURRENT_FOLDER_TEXT_SELECTION = [
    "my_corpus/Shall I Compare Thee To a Summer's Day.txt",
    "my_corpus/再别康桥.txt",
]
# CURRENT_FOLDER_TEXT_SELECTION = None

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

CANTRBRY_DIR = Path("cantrbry")
MY_CORPUS_DIR = Path("my_corpus")
ARTIFICIAL_CORPUS_DIR = Path("artificial_corpus")
OUTPUT_CSV = Path("results/roundtrip/deterministic_roundtrip_results.csv")
WORKER_SCRIPT = Path("deterministic_roundtrip_worker.py")
CONFIG_PATH = Path("results/roundtrip/deterministic_roundtrip_config.json")
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
    my_corpus_dir = (project_root / MY_CORPUS_DIR).resolve()
    artificial_corpus_dir = (project_root / ARTIFICIAL_CORPUS_DIR).resolve()
    worker = (project_root / WORKER_SCRIPT).resolve()

    if CANTRBRY_FILE_SELECTION is not None:
        if not cantrbry_dir.exists() or not cantrbry_dir.is_dir():
            raise FileNotFoundError(f"Cantrbry folder not found: {cantrbry_dir}")
    if MY_CORPUS_FILE_SELECTION is not None:
        if not my_corpus_dir.exists() or not my_corpus_dir.is_dir():
            raise FileNotFoundError(f"my_corpus folder not found: {my_corpus_dir}")
    if ARTIFICIAL_CORPUS_FILE_SELECTION is not None:
        if not artificial_corpus_dir.exists() or not artificial_corpus_dir.is_dir():
            raise FileNotFoundError(f"artificial_corpus folder not found: {artificial_corpus_dir}")
    if not worker.exists():
        raise FileNotFoundError(f"Worker script not found: {worker}")

    cantrbry_files = _select_files(cantrbry_dir, CANTRBRY_FILE_SELECTION)
    my_corpus_files = _select_files(my_corpus_dir, MY_CORPUS_FILE_SELECTION, txt_only=True)
    artificial_corpus_files = _select_files(
        artificial_corpus_dir,
        ARTIFICIAL_CORPUS_FILE_SELECTION,
        txt_only=True,
    )
    current_folder_text_files = _select_files(
        project_root,
        CURRENT_FOLDER_TEXT_SELECTION,
        txt_only=True,
    )

    files = sorted(
        set(
            cantrbry_files
            + my_corpus_files
            + artificial_corpus_files
            + current_folder_text_files
        )
    )
    if not files:
        raise RuntimeError("No files selected for round-trip test from either location.")

    settings = {
        "model_id": MODEL_ID,
        "revision": REVISION,
        "trust_remote_code": TRUST_REMOTE_CODE,
        "torch_dtype": TORCH_DTYPE,
        "device": DEVICE,
        "device_mode": DEVICE_MODE,
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

    output_csv_path = (project_root / _with_tag(OUTPUT_CSV, run_tag)).resolve()
    config_path = (project_root / _with_tag(CONFIG_PATH, run_tag)).resolve()
    artifact_root = (project_root / ARTIFACT_ROOT_BASE / run_tag).resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    config = {
        "files": [str(p) for p in files],
        "settings": settings,
        "model_name": MODEL_ID,
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
