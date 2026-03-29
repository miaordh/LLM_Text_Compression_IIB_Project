"""
Multi-LLM Roundtrip Worker
Runs roundtrip encode/decode for a single file and a single model, writes result to result.csv in artifact dir.
"""
import argparse
import gc
import json
import os
import sys
from pathlib import Path

import pandas as pd
import torch
from deterministic_roundtrip_worker import _phase_encode, _phase_decode, _cleanup_memory

ROUNDTRIP_RESULT_COLUMNS = [
    "model_id",
    "file",
    "status",
    "inference_backend",
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

def main():
    parser = argparse.ArgumentParser(description="Multi-LLM roundtrip worker")
    parser.add_argument("--config", required=True, help="Path to runner-generated JSON config")
    args = parser.parse_args()

    cfg_path = Path(args.config).resolve()
    cfg = json.loads(cfg_path.read_text(encoding="utf-8"))
    file_path = Path(cfg["file"])
    artifact_root = Path(cfg["artifact_root"])
    artifact_root.mkdir(parents=True, exist_ok=True)
    model_id = cfg["settings"]["model_id"]
    keep_artifacts = bool(cfg["settings"].get("keep_artifacts", False))
    phase_timeout_seconds = int(cfg["settings"].get("phase_timeout_seconds", 0) or 0)

    job_dir = artifact_root
    if job_dir.exists():
        for child in job_dir.glob("*"):
            if child.is_file():
                child.unlink()
    job_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    try:
        # Encode
        _phase_encode(cfg, file_path, job_dir)
        # Decode
        _phase_decode(cfg, job_dir)
        # Collect results
        encode_meta = json.loads((job_dir / "encode_metadata.json").read_text(encoding="utf-8"))
        file_encoding = encode_meta.get("text_encoding", cfg["settings"].get("text_encoding", "utf-8"))
        original_text = file_path.read_text(encoding=file_encoding, errors="replace")
        decoded_text = (job_dir / "decoded.txt").read_text(encoding=file_encoding, errors="replace")
        decode_meta = json.loads((job_dir / "decode_metadata.json").read_text(encoding="utf-8"))
        is_match = original_text == decoded_text
        row = {
            "model_id": model_id,
            "file": str(file_path),
            "status": "ok" if is_match else "mismatch",
            "inference_backend": str((encode_meta.get("effective_settings") or {}).get("inference_backend", "auto")),
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
    except Exception as exc:
        rows.append({
            "model_id": model_id,
            "file": str(file_path),
            "status": "failed",
            "inference_backend": "",
            "input_size_bytes": None,
            "safe_mode": None,
            "num_tokens": None,
            "encoded_size_bytes": None,
            "compression_ratio": None,
            "encode_seconds": None,
            "decode_seconds": None,
            "original_chars": None,
            "decoded_chars": None,
            "fallback_attempted": None,
            "attempt_count": None,
            "error": str(exc),
        })
    finally:
        _cleanup_memory()

    # Write result CSV in artifact dir
    df = pd.DataFrame(rows)
    df = df.reindex(columns=ROUNDTRIP_RESULT_COLUMNS)
    (job_dir / "result.csv").write_text(df.to_csv(index=False), encoding="utf-8")

if __name__ == "__main__":
    main()
