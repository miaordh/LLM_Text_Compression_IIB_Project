# Deterministic Kernel Plan (Current Status)

## Summary

The project now uses `batch_invariant_ops` for deterministic/batch-invariant behavior in the standalone deterministic codec path.

## Kept files

- `llm_codec_deterministic.py`: deterministic codec, now using `batch_invariant_ops` instead of custom runtime/kernels.
- `determinism_harness.py`: encode/decode roundtrip harness.
- `batch_invariant_ops/`: external implementation integrated in this workspace.

## Removed legacy files

The following files were intentionally removed during cleanup because they belonged to the previous custom kernel implementation:

- `deterministic_ops.cpp`
- `deterministic_runtime.py`
- `exact_softmax.cpp`
- `setup_deterministic_ops.py`
- `vllm_deterministic_backend.py`
- `setup.py`

## Implications

- Any setup instructions referring to building local C++ extensions are obsolete.
- Any references to `strict_cpu` / `gpu_best_effort` modes are obsolete.
- Deterministic path selection is now capability-based inside `llm_codec_deterministic.py`.

## Run harness

```bash
python determinism_harness.py \
  --model google/gemma-2-2b \
  --text "Determinism test." \
  --encode-device cuda \
  --decode-device cpu
```
