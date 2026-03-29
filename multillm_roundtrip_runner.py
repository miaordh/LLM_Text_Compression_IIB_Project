"""
Multi-LLM Roundtrip Runner
Runs roundtrip tests for a single text file using multiple LLMs, aggregating results into a single CSV.
"""
import json
import os
import socket
import subprocess
import sys
import time
from pathlib import Path
from typing import List

# ---------------------------
# User settings (edit me)
# ---------------------------
MODEL_ID_LIST = [
    "deepseek-ai/deepseek-coder-1.3b-base",
    "Qwen/Qwen2.5-0.5B",
    "Qwen/Qwen3-0.6B",
    "google/gemma-2b",
    "openai-community/gpt2",
    "openai-community/gpt2-medium"
    # Add more model IDs here
]
REVISION = None
TRUST_REMOTE_CODE = False
TORCH_DTYPE = "float32"
DEVICE = "cpu"
DEVICE_MODE = "single_device"

TEXT_FILE = "my_corpus/sonnet.txt"  # Path to the single file to test
TEXT_ENCODING = "utf-8"
FILE_ENCODING_OVERRIDES = {}

KEEP_ARTIFACTS = True
PHASE_TIMEOUT_SECONDS = 0  # 0 means no timeout
RUN_TAG = None

OUTPUT_CSV = Path("results/multillm_results/multillm_roundtrip_results.csv")
WORKER_SCRIPT = Path("multillm_roundtrip_worker.py")
CONFIG_PATH = Path("results/multillm_results/multillm_roundtrip_config.json")
ARTIFACT_ROOT_BASE = Path(".multillm_roundtrip_artifacts")

# Other settings can be added as needed
# Additional settings from deterministic_roundtrip_runner.py
IGNORE_MODEL_MAX_LENGTH_WARNING = True
ENABLE_OOM_FALLBACK = True
OOM_FALLBACK_STRATEGY = "block"  # rolling | block | no_kv_cache
OOM_FALLBACK_CONTEXT_WINDOW = 16
OOM_FALLBACK_MARGIN = 2

SAFE_MODE = True
PRECISION = 32
SLOTS = 1 << 24
CONTEXT_WINDOW = 512
MARGIN = 32
STRATEGY = "rolling"  # rolling | block | no_kv_cache
USE_LEGACY_COUNTS = False
QUANT = False
LOGIT_ROUND_DECIMALS = 15
PROB_ROUND_DECIMALS = 1
DETERMINISM_MODE = None
INFERENCE_BACKEND = "huggingface"  # auto | huggingface | vllm
VLLM_TENSOR_PARALLEL_SIZE = 1
VLLM_GPU_MEMORY_UTILIZATION = 0.9
VLLM_MAX_LOGPROBS = None
VLLM_MAX_MODEL_LEN = None
MAX_DECODE_TOKENS = None
DIAGNOSTICS_ENABLED = False
DIAGNOSTICS_CSV_PREFIX = None
DEMO_MODE = False
SPEED_DEMO = False
MEMORY_DEMO = False
MEMORY_SAMPLE_INTERVAL = 0.05
DIVERGENCE_WINDOW = 5
STOP_ON_FILE_ERROR = True


def _resolve_run_tag() -> str:
    if RUN_TAG:
        return str(RUN_TAG)
    return f"{socket.gethostname()}_{os.getpid()}_{time.time_ns()}"


def main():
    project_root = Path(__file__).resolve().parent
    run_tag = _resolve_run_tag()
    worker = (project_root / WORKER_SCRIPT).resolve()
    text_file = (project_root / TEXT_FILE).resolve()
    if not text_file.exists():
        raise FileNotFoundError(f"Text file not found: {text_file}")
    if not worker.exists():
        raise FileNotFoundError(f"Worker script not found: {worker}")

    output_csv_path = (project_root / OUTPUT_CSV).resolve()
    config_path = (project_root / CONFIG_PATH).resolve()
    artifact_root = (project_root / ARTIFACT_ROOT_BASE / run_tag).resolve()
    output_csv_path.parent.mkdir(parents=True, exist_ok=True)
    config_path.parent.mkdir(parents=True, exist_ok=True)

    # Aggregate results from all models
    all_results = []

    for model_id in MODEL_ID_LIST:
        settings = {
            "model_id": model_id,
            "revision": REVISION,
            "trust_remote_code": TRUST_REMOTE_CODE,
            "torch_dtype": TORCH_DTYPE,
            "device": DEVICE,
            "device_mode": DEVICE_MODE,
            "text_encoding": TEXT_ENCODING,
            "file_encoding_overrides": FILE_ENCODING_OVERRIDES,
            "keep_artifacts": KEEP_ARTIFACTS,
            "phase_timeout_seconds": PHASE_TIMEOUT_SECONDS,
            "run_tag": run_tag,
        }
        config = {
            "file": str(text_file),
            "settings": settings,
            "model_name": model_id,
            "artifact_root": str(artifact_root / model_id.replace('/', '_')),
            "run_tag": run_tag,
        }
        config_path.write_text(json.dumps(config, indent=2), encoding="utf-8")
        cmd = [
            sys.executable,
            str(worker),
            "--config",
            str(config_path),
        ]
        completed = subprocess.run(cmd, capture_output=True, text=True)
        if completed.returncode != 0:
            print(f"Worker failed for model {model_id}: {completed.stderr}", file=sys.stderr)
            continue
        # Worker writes results to a temp CSV, read and append
        result_csv = artifact_root / model_id.replace('/', '_') / "result.csv"
        if result_csv.exists():
            import pandas as pd
            df = pd.read_csv(result_csv)
            all_results.append(df)

    # Aggregate and write final CSV
    if all_results:
        import pandas as pd
        final_df = pd.concat(all_results, ignore_index=True)
        final_df.to_csv(output_csv_path, index=False)
        print(f"Aggregated results written to: {output_csv_path}")
    else:
        print("No results to aggregate.")

    print(f"Run tag: {run_tag}")
    print(f"Artifacts root: {artifact_root}")

if __name__ == "__main__":
    main()
