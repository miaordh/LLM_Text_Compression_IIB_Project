import gc
import math
from contextlib import nullcontext
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from arithmetic_coding import Coder
from bitReadWrite import BitReader, BitWriter
from decoder import Decoder
from encoder import Encoder
from utils import counts_to_cum_desc, probs_to_counts, probs_to_counts_legacy

try:
    from transformers.cache_utils import DynamicCache

    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False

try:
    from batch_invariant_ops import set_batch_invariant_mode, log_softmax
except ImportError:
    try:
        from batch_invariant_ops.batch_invariant_ops import set_batch_invariant_mode, log_softmax
    except ImportError:
        set_batch_invariant_mode = None
        log_softmax = None


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

    use_batch_invariant_ops: bool = True


class DeterministicLLMCodec:
    def __init__(
        self,
        tokenizer,
        model,
        device: str = "auto",
        config: Optional[DeterministicCodecConfig] = None,
    ):
        self.tokenizer = tokenizer
        self.config = config or DeterministicCodecConfig()

        if device == "auto":
            if torch.cuda.is_available():
                self.device = torch.device("cuda")
            elif torch.backends.mps.is_available():
                self.device = torch.device("mps")
            else:
                self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

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

        self.use_kv_cache = self.config.strategy != "no_kv_cache"

        self.block_stride = max(1, self.config.context_window - self.config.margin)

        if "<EOF>" not in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.eof_token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")

        self.dec_prec = max(50, int(math.ceil(self.config.precision * math.log10(2))) + 10)

        self._batch_invariant_enabled = False
        self._log_softmax_fn = torch.log_softmax
        self._batch_invariant_ctx = nullcontext

        try:
            self.model.config._attn_implementation = "eager"
        except Exception:
            pass

        self._configure_batch_invariant_runtime()

    def _configure_batch_invariant_runtime(self):
        if not self.config.use_batch_invariant_ops:
            return
        if set_batch_invariant_mode is None or log_softmax is None:
            return

        try:
            sample = torch.zeros((1, 4), dtype=torch.float32, device=self.device)
            with set_batch_invariant_mode(True):
                _ = log_softmax(sample, dim=-1)
            self._batch_invariant_enabled = True
            self._batch_invariant_ctx = set_batch_invariant_mode
            self._log_softmax_fn = log_softmax
        except Exception:
            self._batch_invariant_enabled = False
            self._batch_invariant_ctx = nullcontext
            self._log_softmax_fn = torch.log_softmax

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

    def _probs(self, logits: torch.Tensor) -> np.ndarray:
        if self.config.quant:
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
            probs = exp_shifted / np.sum(exp_shifted)

            if self.config.prob_round_decimals >= 0:
                scale = 10 ** int(self.config.prob_round_decimals)
                fixed = np.rint(probs * scale).astype(np.int64)
                fixed = np.clip(fixed, 0, None)
                fixed_sum = int(fixed.sum())

                if fixed_sum <= 0:
                    fixed = np.zeros_like(fixed)
                    fixed[int(np.argmax(probs))] = 1
                    fixed_sum = 1

                probs = fixed.astype(np.float64) / float(fixed_sum)

            probs = np.clip(probs, 0.0, None)
            probs_sum = float(np.sum(probs))
            if probs_sum <= 0.0:
                probs = np.full_like(probs, 1.0 / len(probs), dtype=np.float64)
            else:
                probs = probs / probs_sum
            return probs

        logits_2d = logits.view(1, -1).detach().to(dtype=torch.float32)
        probs_t = torch.exp(self._log_softmax_fn(logits_2d, dim=-1))[0]
        probs = probs_t.detach().cpu().numpy().astype(np.float64, copy=False)

        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs[:] = 1.0 / len(probs)
        else:
            probs /= probs_sum
        return probs

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
    ):
        token_ids = self.tokenizer.encode(text)
        if not safe_mode:
            token_ids = token_ids + [self.eof_token_id]

        writer = BitWriter()
        enc = Encoder(Coder(b=self.config.precision), writer)
        iterator = enumerate(token_ids)
        if show_progress:
            iterator = tqdm(iterator, total=len(token_ids), desc="Deterministic Encode")

        cache_state = self._init_cache_state()

        with self._invariant_context():
            if self.use_kv_cache:
                cache_state = self._hard_reset_cache_and_warmup([], 0, 0)
                cache_state["next_logits"] = self._get_logits(0, token_ids, cache_state)

            for idx, token_id in iterator:
                if self.config.strategy == "block" and idx > 0 and (idx % self.block_stride == 0):
                    cache_state = self._hard_reset_cache_and_warmup(
                        token_ids,
                        current_index=idx,
                        warmup_length=self.config.margin,
                    )

                logits = self._get_logits(idx, token_ids, cache_state)
                probs = self._probs(logits)
                counts = self._counts_from_probs(probs)
                enc.encode_symbol(token_id, counts_to_cum_desc(counts))

                if idx < len(token_ids) - 1:
                    cache_state = self._advance_state(token_id, cache_state)
                    if (
                        self.config.strategy == "rolling"
                        and cache_state["cached_token_count"]
                        > (self.config.context_window + self.config.margin)
                    ):
                        cache_state = self._truncate_cache_rolling(cache_state)

        enc.finish()
        writer.flush()
        encoded_bytes = writer.getvalue()

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
    ) -> str:
        dec = Decoder(Coder(b=self.config.precision), BitReader(encoded_bytes))
        decoded_ids = []
        cache_state = self._init_cache_state()

        if safe_mode and expected_num_tokens is None:
            raise ValueError("safe_mode=True requires expected_num_tokens to be provided.")

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
                    probs = self._probs(logits)
                    counts = self._counts_from_probs(probs)

                    token_id = dec.decode_symbol(counts_to_cum_desc(counts))
                    decoded_ids.append(token_id)

                    cache_state = self._advance_state(token_id, cache_state)
                    if (
                        self.config.strategy == "rolling"
                        and cache_state["cached_token_count"]
                        > (self.config.context_window + self.config.margin)
                    ):
                        cache_state = self._truncate_cache_rolling(cache_state)

                return self.tokenizer.decode(decoded_ids, skip_special_tokens=True)

            token_id = None
            decode_iterator = None
            if show_progress:
                decode_iterator = tqdm(desc="Deterministic Decode", unit="tok")
            while token_id != self.eof_token_id:
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
                probs = self._probs(logits)
                counts = self._counts_from_probs(probs)

                token_id = dec.decode_symbol(counts_to_cum_desc(counts))
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

            if decode_iterator is not None:
                decode_iterator.close()

        return self.tokenizer.decode(decoded_ids[:-1], skip_special_tokens=True)
