import gc
import importlib.util
import math
import sys
import time
from contextlib import nullcontext
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence

import numpy as np
import torch
from tqdm import tqdm
from determinism_utils import (
    VLLMLogitsBackend,
    configure_determinism_runtime,
    normalize_determinism_mode,
)
from diagnostics import (
    DiagnosticsWriter,
    maybe_write_divergence_rows,
    start_memory_monitor,
    stop_memory_monitor,
    token_text_for_diag,
    write_csv_rows,
)

from bitReadWrite import BitReader, BitWriter
from decoder import Decoder
from encoder import Encoder


def _load_project_utils_module():
    module_name = "_project_utils_local"
    if module_name in sys.modules:
        return sys.modules[module_name]

    utils_path = Path(__file__).resolve().with_name("utils.py")
    spec = importlib.util.spec_from_file_location(module_name, str(utils_path))
    if spec is None or spec.loader is None:
        raise ImportError(f"Failed to load project utils module at {utils_path}")

    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    sys.modules[module_name] = module
    return module


_project_utils = _load_project_utils_module()
counts_to_cum_desc = _project_utils.counts_to_cum_desc
probs_to_counts = _project_utils.probs_to_counts
probs_to_counts_legacy = _project_utils.probs_to_counts_legacy

try:
    from transformers.cache_utils import DynamicCache

    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False


@dataclass
class DeterministicCodecConfig:
    precision: int = 32
    slots: int = (1 << 24)

    context_window: int = 2048
    margin: int = 128
    strategy: str = "rolling"  # rolling | block | no_kv_cache
    use_legacy_counts: bool = False

    quant: bool = False
    logit_round_decimals: int = 2
    prob_round_decimals: int = 5

    # Optional diagnostics CSV prefix/path. When set, codec writes
    # per-token diagnostics to <prefix>_encode.csv and <prefix>_decode.csv.
    diagnostics_csv_prefix: Optional[str] = None

    determinism_mode: Optional[str] = None
    inference_backend: str = "auto"  # auto | huggingface | vllm
    model_id: Optional[str] = None
    trust_remote_code: bool = False
    revision: Optional[str] = None
    torch_dtype: str = "auto"
    vllm_tensor_parallel_size: int = 1
    vllm_gpu_memory_utilization: float = 0.9
    vllm_max_logprobs: Optional[int] = None
    vllm_max_model_len: Optional[int] = None


class DeterministicLLMCodec:
    def __init__(
        self,
        tokenizer,
        model=None,
        device: str = "auto",
        config: Optional[DeterministicCodecConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or DeterministicCodecConfig()
        # DiagnosticsWriter is not used unless diagnostics are enabled

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self._determinism_mode = normalize_determinism_mode(self.config.determinism_mode)
        self._inference_backend = self._resolve_inference_backend()

        self.model = None
        self._vllm_backend = None
        if self._inference_backend == "huggingface":
            if model is None:
                raise ValueError("Hugging Face backend requires a loaded model instance")
            self.model = model.to(self.device)
            self.model.eval()

        if self.config.strategy not in {"rolling", "block", "no_kv_cache"}:
            raise ValueError(
                "Unsupported strategy. Expected one of: rolling, block, no_kv_cache"
            )
        if self.config.context_window <= 0:
            raise ValueError("context_window must be > 0")
        if self.config.margin < 0:
            raise ValueError("margin must be >= 0")

        # Initialize diagnostics writer if diagnostics are enabled
        if self.config.diagnostics_csv_prefix:
            self._diag_writer = DiagnosticsWriter(self.config.diagnostics_csv_prefix)
        else:
            self._diag_writer = None
        self.use_kv_cache = self.config.strategy != "no_kv_cache"

        self.block_stride = max(1, self.config.context_window - self.config.margin)

        # Remove <EOF> special token logic; assume EOF is handled in codec logic
        self.eof_token_id = self._resolve_existing_eof_token_id()

        self.dec_prec = max(50, int(math.ceil(self.config.precision * math.log10(2))) + 10)

        self._batch_invariant_enabled = False
        self._log_softmax_fn = torch.log_softmax
        self._batch_invariant_ctx = nullcontext

        if self.model is not None:
            try:
                self.model.config._attn_implementation = "eager"
            except Exception:
                pass

        (
            self._batch_invariant_enabled,
            self._batch_invariant_ctx,
            self._log_softmax_fn,
        ) = configure_determinism_runtime(self.device, self._determinism_mode)
        if self._inference_backend == "vllm":
            model_id = self.config.model_id
            if not model_id:
                raise ValueError("vLLM backend requires config.model_id")
            effective_max_model_len = self.config.vllm_max_model_len
            if effective_max_model_len is None:
                # Keep KV cache sized to the codec's actual context needs by default.
                effective_max_model_len = max(256, int(self.config.context_window))
            self._vllm_backend = VLLMLogitsBackend(
                model_id=str(model_id),
                tokenizer=self.tokenizer,
                device=self.device,
                trust_remote_code=bool(self.config.trust_remote_code),
                revision=self.config.revision,
                torch_dtype=str(self.config.torch_dtype),
                tensor_parallel_size=int(self.config.vllm_tensor_parallel_size),
                gpu_memory_utilization=float(self.config.vllm_gpu_memory_utilization),
                max_logprobs=self.config.vllm_max_logprobs,
                max_model_len=effective_max_model_len,
            )

    def _resolve_inference_backend(self) -> str:
        backend = str(self.config.inference_backend or "auto").strip().lower()
        if backend in {"hf", "transformers", "huggingface"}:
            return "huggingface"
        if backend in {"vllm"}:
            return "vllm"
        if backend not in {"", "auto"}:
            raise ValueError("Unsupported inference_backend. Expected one of: auto, huggingface, vllm")

        if self._determinism_mode is None:
            return "huggingface"
            
        # As requested: forcefully try vLLM if determinism_mode is set.
        # Fallback to HuggingFace only occurs via error catching during initialization elsewhere, 
        # or if determinism_mode is None.
        return "vllm"

    def _resolve_existing_eof_token_id(self) -> int:
        for token_attr in ("eos_token_id", "eod_id", "im_end_id"):
            token_id = getattr(self.tokenizer, token_attr, None)
            if isinstance(token_id, int) and token_id >= 0:
                return int(token_id)

        token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")
        if isinstance(token_id, int) and token_id >= 0:
            return int(token_id)

        pad_id = getattr(self.tokenizer, "pad_token_id", None)
        if isinstance(pad_id, int) and pad_id >= 0:
            return int(pad_id)
        return 0

    @property
    def inference_backend(self) -> str:
        return self._inference_backend

    def _invariant_context(self):
        if self._batch_invariant_enabled:
            return self._batch_invariant_ctx(True)
        return nullcontext()

    @staticmethod
    def _position_ids_from_mask(attention_mask: torch.Tensor) -> torch.Tensor:
        position_ids = attention_mask.long().cumsum(-1) - 1
        position_ids.masked_fill_(attention_mask == 0, 0)
        return position_ids

    def _logits_for_prefix(self, prefix_ids):
        if self._vllm_backend is not None:
            return self._vllm_backend.next_logits(prefix_ids)

        if len(prefix_ids) == 0:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                bos = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            input_ids = torch.tensor([[bos]], dtype=torch.long, device=self.device)
        else:
            input_ids = torch.tensor([prefix_ids], dtype=torch.long, device=self.device)

        attention_mask = torch.ones_like(input_ids)
        position_ids = self._position_ids_from_mask(attention_mask)

        with torch.no_grad():
            out = self.model(
                input_ids,
                attention_mask=attention_mask,
                position_ids=position_ids,
                use_cache=self.use_kv_cache,
            )
        return out.logits[0, -1, :]

    def _init_cache_state(self):
        return {
            "past_key_values": None,
            "next_logits": None,
            "cached_token_count": 0,
        }

    def _ensure_dynamic_cache(self, past_kv):
        if past_kv is None:
            return None
        if hasattr(past_kv, "key_cache"):
            return past_kv
        if isinstance(past_kv, tuple) and HAS_DYNAMIC_CACHE:
            try:
                return DynamicCache.from_legacy_cache(past_kv)
            except Exception:
                return past_kv
        return past_kv

    def _hard_reset_cache_and_warmup(self, full_sequence, current_index, warmup_length):
        if self._vllm_backend is not None:
            return self._init_cache_state()

        gc.collect()
        if self.device.type == "mps":
            torch.mps.empty_cache()

        start = max(0, current_index - warmup_length)
        warmup_tokens = full_sequence[start:current_index]
        if len(warmup_tokens) == 0:
            return self._init_cache_state()

        input_ids = torch.tensor([warmup_tokens], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)

        return {
            "past_key_values": outputs.past_key_values,
            "next_logits": outputs.logits[0, -1, :],
            "cached_token_count": len(warmup_tokens),
        }

    def _truncate_cache_rolling(self, cache_state):
        past_kv = cache_state["past_key_values"]
        limit = self.config.context_window
        sink = 4

        if past_kv is None:
            return cache_state

        if hasattr(past_kv, "key_cache"):
            current_len = past_kv.key_cache[0].size(2)
            if current_len <= limit:
                return cache_state

            keep_recent = max(1, limit - sink)
            new_keys = []
            new_values = []
            for key, value in zip(past_kv.key_cache, past_kv.value_cache):
                key_sink = key[:, :, :sink, :]
                key_recent = key[:, :, -keep_recent:, :]
                value_sink = value[:, :, :sink, :]
                value_recent = value[:, :, -keep_recent:, :]
                new_keys.append(torch.cat([key_sink, key_recent], dim=2))
                new_values.append(torch.cat([value_sink, value_recent], dim=2))

            past_kv.key_cache = new_keys
            past_kv.value_cache = new_values
            if hasattr(past_kv, "_seen_tokens"):
                past_kv._seen_tokens = limit

            cache_state["past_key_values"] = past_kv
            cache_state["cached_token_count"] = limit
            return cache_state

        if isinstance(past_kv, tuple):
            current_len = past_kv[0][0].size(2)
            if current_len <= limit:
                return cache_state

            keep_recent = max(1, limit - sink)
            new_past = []
            for key, value in past_kv:
                new_key = torch.cat([key[:, :, :sink, :], key[:, :, -keep_recent:, :]], dim=2)
                new_value = torch.cat([value[:, :, :sink, :], value[:, :, -keep_recent:, :]], dim=2)
                new_past.append((new_key, new_value))

            cache_state["past_key_values"] = tuple(new_past)
            cache_state["cached_token_count"] = limit
            return cache_state

        return cache_state

    def _get_logits(self, current_idx, full_token_sequence, cache_state):
        if self._vllm_backend is not None:
            return self._vllm_backend.next_logits(full_token_sequence[:current_idx])

        if not self.use_kv_cache:
            start = max(0, current_idx - self.config.context_window)
            context = full_token_sequence[start:current_idx]
            if len(context) == 0:
                bos = self.tokenizer.bos_token_id
                if bos is None:
                    bos = self.tokenizer.pad_token_id
                if bos is None:
                    return torch.log(torch.ones(self.tokenizer.vocab_size).to(self.device))
                context = [bos]

            input_ids = torch.tensor([context], dtype=torch.long).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, use_cache=False)
            return outputs.logits[0, -1, :]

        if cache_state["next_logits"] is not None:
            return cache_state["next_logits"]

        if current_idx == 0:
            bos = self.tokenizer.bos_token_id
            if bos is None:
                bos = self.tokenizer.pad_token_id
            if bos is not None:
                input_ids = torch.tensor([[bos]], dtype=torch.long).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids, use_cache=True)
                return outputs.logits[0, -1, :]

        return torch.log(torch.ones(self.tokenizer.vocab_size).to(self.device))

    def _advance_state(self, just_encoded_token_id, cache_state):
        if self._vllm_backend is not None:
            return cache_state

        if not self.use_kv_cache:
            return cache_state

        past_kv = self._ensure_dynamic_cache(cache_state["past_key_values"])
        input_ids = torch.tensor([[just_encoded_token_id]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(input_ids, past_key_values=past_kv, use_cache=True)

        cache_state["past_key_values"] = outputs.past_key_values
        cache_state["next_logits"] = outputs.logits[0, -1, :]
        cache_state["cached_token_count"] += 1
        return cache_state

    def _raw_and_effective_probs(self, logits: torch.Tensor):
        logits_2d = logits.view(1, -1).detach().to(dtype=torch.float32)
        raw_t = torch.exp(self._log_softmax_fn(logits_2d, dim=-1))[0]
        raw_probs = raw_t.detach().cpu().numpy().astype(np.float64, copy=False)

        raw_sum = raw_probs.sum()
        if raw_sum <= 0:
            raw_probs = np.full_like(raw_probs, 1.0 / len(raw_probs), dtype=np.float64)
        else:
            raw_probs = raw_probs / raw_sum

        if not self.config.quant:
            return raw_probs, raw_probs.copy()

        logits_np = (
            logits.detach()
            .to(device="cpu", dtype=torch.float32)
            .numpy()
            .reshape(-1)
            .astype(np.float64, copy=False)
        )

        if self.config.logit_round_decimals >= 0:
            scale = 10 ** int(self.config.logit_round_decimals)
            logits_np = np.rint(logits_np * scale) / float(scale)

        max_logit = float(np.max(logits_np))
        exp_shifted = np.exp(logits_np - max_logit)
        eff_probs = exp_shifted / np.sum(exp_shifted)

        if self.config.prob_round_decimals >= 0:
            scale = 10 ** int(self.config.prob_round_decimals)
            fixed = np.rint(eff_probs * scale).astype(np.int64)
            fixed = np.clip(fixed, 0, None)
            fixed_sum = int(fixed.sum())

            if fixed_sum <= 0:
                fixed = np.zeros_like(fixed)
                fixed[int(np.argmax(eff_probs))] = 1
                fixed_sum = 1

            eff_probs = fixed.astype(np.float64) / float(fixed_sum)

        eff_probs = np.clip(eff_probs, 0.0, None)
        eff_sum = float(np.sum(eff_probs))
        if eff_sum <= 0.0:
            eff_probs = np.full_like(eff_probs, 1.0 / len(eff_probs), dtype=np.float64)
        else:
            eff_probs = eff_probs / eff_sum

        return raw_probs, eff_probs

    def _probs(self, logits: torch.Tensor) -> np.ndarray:
        _, effective_probs = self._raw_and_effective_probs(logits)
        return effective_probs

    def _counts_from_probs(self, probs: np.ndarray):
        if self.config.use_legacy_counts:
            return probs_to_counts_legacy(probs, self.config.slots, self.dec_prec)
        return probs_to_counts(probs, self.config.slots, self.dec_prec)

    def encode(
        self,
        text: str,
        safe_mode: bool = False,
        return_token_count: bool = False,
        show_progress: bool = True,
        demo: bool = False,
        demo_csv_path: str = "compression_stats.csv",
        speed_demo: bool = False,
        speed_csv_path: str = "speed_encode.csv",
        memory_demo: bool = False,
        memory_csv_path: str = "memory_encode.csv",
        memory_sample_interval: float = 0.05,
    ):
        # Standard tokenisation; fallback logic removed
        token_ids = self.tokenizer.encode(text)
        if not safe_mode:
            token_ids = token_ids + [self.eof_token_id]


        writer = BitWriter()
        enc = Encoder(writer, b=self.config.precision)
        iterator = enumerate(token_ids)
        if show_progress:
            iterator = tqdm(iterator, total=len(token_ids), desc="Deterministic Encode")

        cache_state = self._init_cache_state()
        diag_handle = diag_writer = None
        demo_rows: List[Dict[str, Any]] = []
        speed_rows: List[Dict[str, Any]] = []
        monitor_state = start_memory_monitor(memory_demo, memory_sample_interval)

        context_window = int(self.config.context_window)
        vocab_size = len(self.tokenizer)
        try:
            with self._invariant_context():
                if self.use_kv_cache:
                    cache_state = self._hard_reset_cache_and_warmup([], 0, 0)
                    cache_state["next_logits"] = self._get_logits(0, token_ids, cache_state)

                idx = 0
                while idx < len(token_ids):
                    # Determine the chunk size: never exceed context_window
                    chunk_end = min(idx + context_window, len(token_ids))
                    chunk = token_ids[idx:chunk_end]
                    for chunk_idx, token_id in enumerate(chunk):
                        global_idx = idx + chunk_idx
                        iter_start = time.perf_counter()
                        if not (0 <= int(token_id) < vocab_size):
                            raise ValueError(f"Out-of-range token_id {token_id} at position {global_idx} (vocab_size={vocab_size})")
                        if self.config.strategy == "block" and global_idx > 0 and (global_idx % self.block_stride == 0):
                            cache_state = self._hard_reset_cache_and_warmup(
                                token_ids,
                                current_index=global_idx,
                                warmup_length=self.config.margin,
                            )

                        logits = self._get_logits(global_idx, token_ids, cache_state)
                        raw_probs, probs = self._raw_and_effective_probs(logits)
                        counts = self._counts_from_probs(probs)

                        if demo:
                            p_raw = float(raw_probs[int(token_id)])
                            p_eff = float(probs[int(token_id)])
                            safe_raw = max(p_raw, 1e-12)
                            demo_rows.append(
                                {
                                    "pos": int(global_idx),
                                    "token_id": int(token_id),
                                    "token": token_text_for_diag(self.tokenizer, int(token_id)),
                                    "prob_raw": p_raw,
                                    "prob_effective": p_eff,
                                    "perplexity_raw": 1.0 / safe_raw,
                                    "surprisal_bits_raw": -math.log2(safe_raw),
                                }
                            )

                        coder_L_before = int(enc.L)
                        coder_R_before = int(enc.R)
                        if self._diag_writer is not None:
                            self._diag_writer.write_token_diagnostic(
                                diag_writer,
                                phase="encode",
                                step_idx=global_idx,
                                token_id=int(token_id),
                                raw_probs=raw_probs,
                                effective_probs=probs,
                                counts=counts,
                                coder_L_before=coder_L_before,
                                coder_R_before=coder_R_before,
                                coder_D_before=None,
                            )

                        enc.encode_symbol(token_id, counts_to_cum_desc(counts))

                        if global_idx < len(token_ids) - 1:
                            cache_state = self._advance_state(token_id, cache_state)
                            if (
                                self.config.strategy == "rolling"
                                and cache_state["cached_token_count"]
                                > (self.config.context_window + self.config.margin)
                            ):
                                cache_state = self._truncate_cache_rolling(cache_state)

                        if speed_demo or memory_demo:
                            speed_rows.append({"pos": int(global_idx), "time": time.perf_counter() - iter_start})
                    idx = chunk_end
        finally:
            if diag_handle is not None:
                diag_handle.close()
            if memory_demo:
                memory_rows = stop_memory_monitor(monitor_state, speed_rows)
                write_csv_rows(memory_csv_path, memory_rows)

        enc.finish()
        writer.flush()
        encoded_bytes = writer.getvalue()

        if speed_demo:
            write_csv_rows(speed_csv_path, speed_rows)
        if demo:
            write_csv_rows(demo_csv_path, demo_rows)

        if safe_mode or return_token_count:
            return encoded_bytes, len(token_ids)
        return encoded_bytes

    def decode(
        self,
        encoded_bytes: bytes,
        max_decode_tokens: Optional[int] = None,
        safe_mode: bool = False,
        expected_num_tokens: Optional[int] = None,
        show_progress: bool = True,
        demo: bool = False,
        demo_csv_path: str = "demo_decode.csv",
        speed_demo: bool = False,
        speed_csv_path: str = "speed_decode.csv",
        memory_demo: bool = False,
        memory_csv_path: str = "memory_decode.csv",
        memory_sample_interval: float = 0.05,
        reference_token_ids: Optional[Sequence[int]] = None,
        divergence_window: int = 5,
    ) -> str:
        dec = Decoder(BitReader(encoded_bytes), b=self.config.precision)
        decoded_ids = []
        cache_state = self._init_cache_state()
        if self._diag_writer is not None:
            diag_handle, diag_writer = self._diag_writer.open_writer("decode")
        else:
            diag_handle, diag_writer = None, None
        demo_rows: List[Dict[str, Any]] = []
        speed_rows: List[Dict[str, Any]] = []
        monitor_state = start_memory_monitor(memory_demo, memory_sample_interval)
        decoded_text = ""
        vocab_size = len(self.tokenizer)

        if safe_mode and expected_num_tokens is None:
            raise ValueError("safe_mode=True requires expected_num_tokens to be provided.")

        try:
            with self._invariant_context():
                if self.use_kv_cache:
                    cache_state = self._hard_reset_cache_and_warmup([], 0, 0)
                    cache_state["next_logits"] = self._get_logits(0, [], cache_state)

                if safe_mode:
                    target_tokens = int(expected_num_tokens)
                    iterator = range(target_tokens)
                    if show_progress:
                        iterator = tqdm(iterator, total=target_tokens, desc="Deterministic Decode")

                    for _ in iterator:
                        iter_start = time.perf_counter()
                        idx = len(decoded_ids)
                        if max_decode_tokens is not None and idx >= max_decode_tokens:
                            raise RuntimeError(
                                f"Decoding exceeded max_decode_tokens={max_decode_tokens} before target token count."
                            )

                        if self.config.strategy == "block" and idx > 0 and (idx % self.block_stride == 0):
                            cache_state = self._hard_reset_cache_and_warmup(
                                decoded_ids,
                                current_index=idx,
                                warmup_length=self.config.margin,
                            )

                        logits = self._get_logits(idx, decoded_ids, cache_state)
                        raw_probs, probs = self._raw_and_effective_probs(logits)
                        counts = self._counts_from_probs(probs)

                        coder_L_before = int(dec.L)
                        coder_R_before = int(dec.R)
                        coder_D_before = int(dec.D)
                        token_id = dec.decode_symbol(counts_to_cum_desc(counts))
                        if not (0 <= int(token_id) < vocab_size):
                            print(f"[llm_codec_deterministic] ERROR: Out-of-range decoded token_id {token_id} at position {idx} (vocab_size={vocab_size})", file=sys.stderr)
                            raise ValueError(f"Out-of-range decoded token_id {token_id} at position {idx} (vocab_size={vocab_size})")

                        if self._diag_writer is not None:
                            self._diag_writer.write_token_diagnostic(
                                diag_writer,
                                phase="decode",
                                step_idx=idx,
                                token_id=int(token_id),
                                raw_probs=raw_probs,
                                effective_probs=probs,
                                counts=counts,
                                coder_L_before=coder_L_before,
                                coder_R_before=coder_R_before,
                                coder_D_before=coder_D_before,
                            )

                        if demo:
                            p_raw = float(raw_probs[int(token_id)])
                            p_eff = float(probs[int(token_id)])
                            safe_raw = max(p_raw, 1e-12)
                            row: Dict[str, Any] = {
                                "pos": int(idx),
                                "decoded_token_id": int(token_id),
                                "decoded_token": token_text_for_diag(self.tokenizer, int(token_id)),
                                "prob_raw": p_raw,
                                "prob_effective": p_eff,
                                "perplexity_raw": 1.0 / safe_raw,
                                "surprisal_bits_raw": -math.log2(safe_raw),
                            }
                            if reference_token_ids is not None and idx < len(reference_token_ids):
                                ref_id = int(reference_token_ids[idx])
                                row["reference_token_id"] = ref_id
                                row["reference_token"] = token_text_for_diag(self.tokenizer, ref_id)
                                row["match"] = int(ref_id == int(token_id))
                            demo_rows.append(row)

                        decoded_ids.append(token_id)

                        cache_state = self._advance_state(token_id, cache_state)
                        if (
                            self.config.strategy == "rolling"
                            and cache_state["cached_token_count"]
                            > (self.config.context_window + self.config.margin)
                        ):
                            cache_state = self._truncate_cache_rolling(cache_state)

                        if speed_demo or memory_demo:
                            speed_rows.append({"pos": int(idx), "time": time.perf_counter() - iter_start})

                    decoded_text = self.tokenizer.decode(decoded_ids, skip_special_tokens=True)
                else:
                    token_id = None
                    decode_iterator = None
                    if show_progress:
                        decode_iterator = tqdm(desc="Deterministic Decode", unit="tok")
                    while token_id != self.eof_token_id:
                        iter_start = time.perf_counter()
                        idx = len(decoded_ids)
                        if max_decode_tokens is not None and idx >= max_decode_tokens:
                            raise RuntimeError(
                                f"Decoding exceeded max_decode_tokens={max_decode_tokens} before EOF."
                            )

                        if self.config.strategy == "block" and idx > 0 and (idx % self.block_stride == 0):
                            cache_state = self._hard_reset_cache_and_warmup(
                                decoded_ids,
                                current_index=idx,
                                warmup_length=self.config.margin,
                            )

                        logits = self._get_logits(idx, decoded_ids, cache_state)
                        raw_probs, probs = self._raw_and_effective_probs(logits)
                        counts = self._counts_from_probs(probs)

                        coder_L_before = int(dec.L)
                        coder_R_before = int(dec.R)
                        coder_D_before = int(dec.D)
                        token_id = dec.decode_symbol(counts_to_cum_desc(counts))
                        if not (0 <= int(token_id) < vocab_size):
                            print(f"[llm_codec_deterministic] ERROR: Out-of-range decoded token_id {token_id} at position {idx} (vocab_size={vocab_size})", file=sys.stderr)
                            raise ValueError(f"Out-of-range decoded token_id {token_id} at position {idx} (vocab_size={vocab_size})")

                        if self._diag_writer is not None:
                            self._diag_writer.write_token_diagnostic(
                                diag_writer,
                                phase="decode",
                                step_idx=idx,
                                token_id=int(token_id),
                                raw_probs=raw_probs,
                                effective_probs=probs,
                                counts=counts,
                                coder_L_before=coder_L_before,
                                coder_R_before=coder_R_before,
                                coder_D_before=coder_D_before,
                            )

                        if demo:
                            p_raw = float(raw_probs[int(token_id)])
                            p_eff = float(probs[int(token_id)])
                            safe_raw = max(p_raw, 1e-12)
                            row = {
                                "pos": int(idx),
                                "decoded_token_id": int(token_id),
                                "decoded_token": token_text_for_diag(self.tokenizer, int(token_id)),
                                "prob_raw": p_raw,
                                "prob_effective": p_eff,
                                "perplexity_raw": 1.0 / safe_raw,
                                "surprisal_bits_raw": -math.log2(safe_raw),
                            }
                            if reference_token_ids is not None and idx < len(reference_token_ids):
                                ref_id = int(reference_token_ids[idx])
                                row["reference_token_id"] = ref_id
                                row["reference_token"] = token_text_for_diag(self.tokenizer, ref_id)
                                row["match"] = int(ref_id == int(token_id))
                            demo_rows.append(row)

                        decoded_ids.append(token_id)

                        if token_id != self.eof_token_id:
                            cache_state = self._advance_state(token_id, cache_state)
                            if (
                                self.config.strategy == "rolling"
                                and cache_state["cached_token_count"]
                                > (self.config.context_window + self.config.margin)
                            ):
                                cache_state = self._truncate_cache_rolling(cache_state)
                        if decode_iterator is not None:
                            decode_iterator.update(1)

                        if speed_demo or memory_demo:
                            speed_rows.append({"pos": int(idx), "time": time.perf_counter() - iter_start})

                    if decode_iterator is not None:
                        decode_iterator.close()

                    decoded_text = self.tokenizer.decode(decoded_ids[:-1], skip_special_tokens=True)
        finally:
            if diag_handle is not None:
                diag_handle.close()
            if memory_demo:
                memory_rows = stop_memory_monitor(monitor_state, speed_rows)
                write_csv_rows(memory_csv_path, memory_rows)

        if speed_demo:
            write_csv_rows(speed_csv_path, speed_rows)
        if demo:
            write_csv_rows(demo_csv_path, demo_rows)
            maybe_write_divergence_rows(
                self.tokenizer,
                demo_csv_path=demo_csv_path,
                decoded_ids=decoded_ids,
                reference_token_ids=reference_token_ids,
                divergence_window=divergence_window,
            )

        return decoded_text
