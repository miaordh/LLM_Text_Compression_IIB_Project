from dataclasses import dataclass
import importlib
from typing import Any, Dict, Optional


@dataclass
class VLLMDeterministicConfig:
    model: str
    dtype: str = "float32"
    tensor_parallel_size: int = 1
    max_num_seqs: int = 1
    enforce_eager: bool = True
    disable_log_stats: bool = True


def build_vllm_engine_kwargs(cfg: VLLMDeterministicConfig) -> Dict[str, Any]:
    return {
        "model": cfg.model,
        "dtype": cfg.dtype,
        "tensor_parallel_size": cfg.tensor_parallel_size,
        "max_num_seqs": cfg.max_num_seqs,
        "enforce_eager": cfg.enforce_eager,
        "disable_log_stats": cfg.disable_log_stats,
    }


def create_vllm_llm(cfg: VLLMDeterministicConfig):
    try:
        vllm_mod = importlib.import_module("vllm")
    except ImportError as exc:
        raise ImportError(
            "vLLM is not installed. Install with `pip install vllm` in a compatible environment."
        ) from exc

    kwargs = build_vllm_engine_kwargs(cfg)
    return vllm_mod.LLM(**kwargs)


def deterministic_sampling_params(max_tokens: int = 1):
    try:
        vllm_mod = importlib.import_module("vllm")
    except ImportError as exc:
        raise ImportError("vLLM is not installed.") from exc

    return vllm_mod.SamplingParams(
        temperature=0.0,
        top_p=1.0,
        top_k=-1,
        max_tokens=max_tokens,
        logprobs=0,
    )
