# Deterministic Kernel Path (Experimental)

This path is isolated from the original codec implementation.

## Added files

- `deterministic_ops.cpp`: CPU float64 deterministic kernels for:
  - softmax
  - matmul
  - RMSNorm
- `setup_deterministic_ops.py`: build script for `deterministic_ops_cpp`.
- `deterministic_runtime.py`: deterministic runtime configuration + monkeypatch layer wrappers.
- `llm_codec_deterministic.py`: standalone codec using deterministic runtime and kernels.
- `vllm_deterministic_backend.py`: optional vLLM constructor with deterministic-friendly defaults.
- `determinism_harness.py`: CPU->MPS roundtrip checker.

## Why this addresses paper sources

- **Batch invariance**: codec runs a fixed single-request path (batch size 1) with explicit masks and positions.
- **RMSNorm reduction variance**: RMSNorm can be routed through a deterministic CPU float64 kernel.
- **Matmul split-K variance**: linear layers can be routed through deterministic CPU float64 matmul.
- **Attention kernel path variance**: SDPA can be replaced by a deterministic eager implementation.
- **KV-cache boundary variance**: default uses `use_kv_cache=False` to remove dynamic cache split behavior.

## Build and use

1. Build extension:

```bash
python setup_deterministic_ops.py install
```

2. Run harness:

```bash
python determinism_harness.py \
  --model google/gemma-2-2b \
  --text "Determinism test." \
  --determinism-mode strict_cpu \
  --encode-device cpu \
  --decode-device mps
```

## Determinism modes

- `strict_cpu` (default)
  - Patches Linear/RMSNorm/Attention to deterministic CPU reduction paths.
  - Strongest cross-platform reproducibility, slowest runtime.
- `gpu_best_effort`
  - Keeps model ops on GPU where possible, still enables torch deterministic settings.
  - Faster runtime, but weaker cross-platform guarantee (CPU↔MPS/CUDA may still diverge).

## Notes

- This is intentionally slow because it prioritizes deterministic reductions over speed.
- The Python fallback in `deterministic_runtime.py` works without compiling C++ extension.
- vLLM integration is optional and currently limited to deterministic engine setup scaffolding.
