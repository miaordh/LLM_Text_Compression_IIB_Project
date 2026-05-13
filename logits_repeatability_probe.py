import argparse
import csv
import json
import math
import os
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_codec_deterministic import DeterministicCodecConfig, DeterministicLLMCodec


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


def _env_optional_int(name: str, default):
    value = os.environ.get(name)
    if value is None or value == "":
        return default
    lowered = value.strip().lower()
    if lowered in {"none", "null"}:
        return None
    return int(value)


def _normalize_mode(mode: Optional[str]) -> Optional[str]:
    if mode is None:
        return None
    text = str(mode).strip().lower()
    if text in {"", "none", "off", "false", "0", "null"}:
        return None
    return text


def _resolve_device(device: str) -> str:
    if device != "auto":
        return device
    if torch.cuda.is_available():
        return "cuda"
    if torch.backends.mps.is_available():
        return "mps"
    return "cpu"


def _resolve_torch_dtype(dtype_name: str, device: str):
    dtype_name = str(dtype_name).strip().lower()
    if dtype_name == "auto":
        if device == "cuda":
            return torch.float16
        return "auto"
    mapping = {
        "float32": torch.float32,
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
    }
    if dtype_name not in mapping:
        raise ValueError(f"Unsupported torch dtype: {dtype_name}")
    return mapping[dtype_name]


def _apply_vllm_env(attention_backend: Optional[str], use_v1: Optional[str]):
    if attention_backend:
        os.environ["VLLM_ATTENTION_BACKEND"] = str(attention_backend)
    if use_v1 is not None and str(use_v1).strip() != "":
        os.environ["VLLM_USE_V1"] = str(use_v1)


def _load_codec(args: argparse.Namespace) -> DeterministicLLMCodec:
    device = _resolve_device(args.device)
    determinism_mode = _normalize_mode(args.determinism_mode)
    backend = str(args.inference_backend).strip().lower()

    _apply_vllm_env(args.vllm_attention_backend, args.vllm_use_v1)

    tokenizer = AutoTokenizer.from_pretrained(
        args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
        revision=args.revision,
    )
    tokenizer.model_max_length = int(1e30)

    model = None
    if backend in {"huggingface", "hf", "transformers"}:
        model_kwargs: Dict[str, Any] = {
            "trust_remote_code": bool(args.trust_remote_code),
            "low_cpu_mem_usage": True,
        }
        if args.revision:
            model_kwargs["revision"] = args.revision

        torch_dtype = _resolve_torch_dtype(args.torch_dtype, device)
        if torch_dtype != "auto":
            model_kwargs["dtype"] = torch_dtype

        try:
            model = AutoModelForCausalLM.from_pretrained(args.model_id, **model_kwargs)
        except TypeError:
            if "dtype" not in model_kwargs:
                raise
            fallback_kwargs = dict(model_kwargs)
            fallback_kwargs["torch_dtype"] = fallback_kwargs.pop("dtype")
            model = AutoModelForCausalLM.from_pretrained(args.model_id, **fallback_kwargs)
    elif backend != "vllm":
        raise ValueError("inference_backend must be one of: huggingface, vllm")

    config = DeterministicCodecConfig(
        context_window=max(256, int(args.max_model_len or 256)),
        determinism_mode=determinism_mode,
        inference_backend="vllm" if backend == "vllm" else "huggingface",
        model_id=args.model_id,
        trust_remote_code=bool(args.trust_remote_code),
        revision=args.revision,
        torch_dtype=args.torch_dtype,
        vllm_tensor_parallel_size=int(args.vllm_tensor_parallel_size),
        vllm_gpu_memory_utilization=float(args.vllm_gpu_memory_utilization),
        vllm_attention_backend=args.vllm_attention_backend,
        vllm_use_v1=args.vllm_use_v1,
        vllm_max_logprobs=args.vllm_max_logprobs,
        vllm_max_model_len=args.max_model_len,
    )
    return DeterministicLLMCodec(
        tokenizer=tokenizer,
        model=model,
        device=device,
        config=config,
    )


def _to_numpy(logits: torch.Tensor) -> np.ndarray:
    return logits.detach().to(device="cpu", dtype=torch.float64).numpy().reshape(-1)


def _finite_stats(diff: np.ndarray) -> Dict[str, Any]:
    finite = np.isfinite(diff)
    if not finite.any():
        return {
            "max_abs_diff": math.nan,
            "mean_abs_diff": math.nan,
            "rms_diff": math.nan,
            "nonzero_count": int(diff.size),
        }
    abs_diff = np.abs(diff[finite])
    return {
        "max_abs_diff": float(np.max(abs_diff)),
        "mean_abs_diff": float(np.mean(abs_diff)),
        "rms_diff": float(np.sqrt(np.mean(np.square(diff[finite])))),
        "nonzero_count": int(np.count_nonzero(diff)),
    }


def _write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = []
    for row in rows:
        for key in row:
            if key not in fieldnames:
                fieldnames.append(key)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def run_probe(args: argparse.Namespace):
    output_dir = Path(args.output_dir).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)

    codec = _load_codec(args)
    try:
        try:
            prompt_ids = codec.tokenizer.encode(args.prompt, add_special_tokens=False)
        except TypeError:
            prompt_ids = codec.tokenizer.encode(args.prompt)

        metadata = {
            "model_id": args.model_id,
            "prompt": args.prompt,
            "prompt_ids": [int(x) for x in prompt_ids],
            "repeats": int(args.repeats),
            "device": str(codec.device),
            "inference_backend": codec.inference_backend,
            "determinism_mode": _normalize_mode(args.determinism_mode),
            "torch_dtype": args.torch_dtype,
            "vllm_tensor_parallel_size": int(args.vllm_tensor_parallel_size),
            "vllm_attention_backend": args.vllm_attention_backend,
            "vllm_use_v1": args.vllm_use_v1,
            "vllm_max_logprobs": args.vllm_max_logprobs,
            "max_model_len": args.max_model_len,
        }
        (output_dir / "metadata.json").write_text(
            json.dumps(metadata, indent=2),
            encoding="utf-8",
        )

        baseline = None
        baseline_top = None
        rows = []
        topk_rows = []

        for idx in range(int(args.repeats)):
            start = time.perf_counter()
            logits = _to_numpy(codec._logits_for_prefix(prompt_ids))
            elapsed = time.perf_counter() - start

            if baseline is None:
                baseline = logits.copy()
                baseline_top = np.argsort(-baseline)[: int(args.top_k)]

            assert baseline is not None
            diff = logits - baseline
            stats = _finite_stats(diff)
            top1 = int(np.argmax(logits))
            base_top1 = int(np.argmax(baseline))
            same_top1 = top1 == base_top1
            all_equal = bool(np.array_equal(logits, baseline))
            allclose_1e6 = bool(np.allclose(logits, baseline, rtol=1e-6, atol=1e-6))
            allclose_1e8 = bool(np.allclose(logits, baseline, rtol=1e-8, atol=1e-8))

            row = {
                "repeat_index": idx,
                "elapsed_seconds": f"{elapsed:.9f}",
                "all_equal_to_first": all_equal,
                "allclose_1e-6": allclose_1e6,
                "allclose_1e-8": allclose_1e8,
                "same_top1_as_first": same_top1,
                "top1_token_id": top1,
                "first_top1_token_id": base_top1,
                **stats,
            }
            rows.append(row)

            if baseline_top is not None:
                for rank, token_id in enumerate(baseline_top):
                    token_id = int(token_id)
                    topk_rows.append(
                        {
                            "repeat_index": idx,
                            "first_rank": rank + 1,
                            "token_id": token_id,
                            "first_logit": float(baseline[token_id]),
                            "current_logit": float(logits[token_id]),
                            "diff": float(logits[token_id] - baseline[token_id]),
                        }
                    )

        _write_csv(output_dir / "repeatability_summary.csv", rows)
        _write_csv(output_dir / "topk_diffs.csv", topk_rows)

        max_diff = max(float(row["max_abs_diff"]) for row in rows)
        max_nonzero = max(int(row["nonzero_count"]) for row in rows)
        all_equal = all(str(row["all_equal_to_first"]) == "True" or row["all_equal_to_first"] is True for row in rows)
        print(f"Output dir: {output_dir}")
        print(f"Prompt token count: {len(prompt_ids)}")
        print(f"Repeats: {args.repeats}")
        print(f"All repeats exactly equal to first: {all_equal}")
        print(f"Maximum absolute difference vs first: {max_diff:.12g}")
        print(f"Maximum nonzero entries vs first: {max_nonzero}")
    finally:
        del codec
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(
        description="Repeat one fixed prompt and measure next-logit differences."
    )
    parser.add_argument("--model-id", default=_env_str("CODEC_MODEL_ID", "deepseek-ai/deepseek-coder-1.3b-base"))
    parser.add_argument("--revision", default=_env_str("CODEC_REVISION", None))
    parser.add_argument("--trust-remote-code", action="store_true")
    parser.add_argument("--prompt", default=_env_str("LOGITS_PROBE_PROMPT", "def add(a, b):"))
    parser.add_argument("--repeats", type=int, default=_env_int("LOGITS_PROBE_REPEATS", 100))
    parser.add_argument("--top-k", type=int, default=_env_int("LOGITS_PROBE_TOP_K", 20))
    parser.add_argument("--output-dir", default=_env_str("LOGITS_PROBE_OUTPUT_DIR", "results/logits_repeatability/hf_single_gpu_none"))
    parser.add_argument("--device", default=_env_str("CODEC_DEVICE", "cuda"))
    parser.add_argument("--inference-backend", default=_env_str("CODEC_INFERENCE_BACKEND", "huggingface"))
    parser.add_argument("--determinism-mode", default=_env_str("CODEC_DETERMINISM_MODE", None))
    parser.add_argument("--torch-dtype", default=_env_str("CODEC_TORCH_DTYPE", "auto"))
    parser.add_argument("--vllm-tensor-parallel-size", type=int, default=_env_int("CODEC_VLLM_TENSOR_PARALLEL_SIZE", 1))
    parser.add_argument("--vllm-gpu-memory-utilization", type=float, default=_env_float("CODEC_VLLM_GPU_MEMORY_UTILIZATION", 0.9))
    parser.add_argument("--vllm-attention-backend", default=_env_str("CODEC_VLLM_ATTENTION_BACKEND", "TRITON_ATTN"))
    parser.add_argument("--vllm-use-v1", default=_env_str("CODEC_VLLM_USE_V1", "1"))
    parser.add_argument("--vllm-max-logprobs", type=int, default=_env_optional_int("CODEC_VLLM_MAX_LOGPROBS", None))
    parser.add_argument("--max-model-len", type=int, default=_env_optional_int("CODEC_VLLM_MAX_MODEL_LEN", 256))
    args = parser.parse_args()
    run_probe(args)


if __name__ == "__main__":
    main()
