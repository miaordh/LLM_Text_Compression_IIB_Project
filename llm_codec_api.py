import csv
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import numpy as np
from tqdm import tqdm

from arithmetic_coding import Coder
from bitReadWrite import BitReader, BitWriter
from decoder import Decoder
from encoder import Encoder
from utils import counts_to_cum_desc, probs_to_counts, probs_to_counts_legacy


@dataclass
class APICodecConfig:
    # Arithmetic-coding / quantization knobs (aligned with deterministic codec).
    precision: int = 32
    slots: int = (1 << 24)

    context_window: int = 2048
    margin: int = 128
    strategy: str = "rolling"  # rolling | block | no_kv_cache
    use_legacy_counts: bool = False

    quant: bool = False
    logit_round_decimals: int = 2
    prob_round_decimals: int = 5

    diagnostics_csv_prefix: Optional[str] = None
    determinism_mode: Optional[str] = None

    # API knobs.
    model: str = "qwen-plus"
    api_key: Optional[str] = None
    base_url: str = "https://dashscope-intl.aliyuncs.com/compatible-mode/v1"
    top_k: int = 20
    temperature: float = 0.0

    # If enabled, the codec will try sending cache hints to API (and retry
    # without hints if API rejects unknown fields).
    enable_api_cache_hints: bool = True
    api_attention_sink: int = 4

    # Determinism knobs (no trace replay).
    deterministic_strict: bool = True
    api_request_mode: str = "chat"  # chat | completions | auto
    api_seed: Optional[int] = 1
    strict_single_id_mapping: bool = True


class APILLMCodec:
    """
    API-backed arithmetic codec.

    Probability source:
    1. Request top-k next-token logprobs from API for the current prefix.
    2. Map returned token strings to local tokenizer vocab IDs.
    3. Assign mapped masses to those IDs.
    4. Distribute the remaining mass uniformly over all unassigned vocab IDs.
    """

    def __init__(self, tokenizer, config: Optional[APICodecConfig] = None):
        self.tokenizer = tokenizer
        self.config = config or APICodecConfig()

        if self.config.strategy not in {"rolling", "block", "no_kv_cache"}:
            raise ValueError("Unsupported strategy. Expected one of: rolling, block, no_kv_cache")
        if self.config.context_window <= 0:
            raise ValueError("context_window must be > 0")
        if self.config.margin < 0:
            raise ValueError("margin must be >= 0")
        if self.config.top_k <= 0:
            raise ValueError("top_k must be > 0")

        self.block_stride = max(1, self.config.context_window - self.config.margin)
        self.dec_prec = max(50, int(math.ceil(self.config.precision * math.log10(2))) + 10)

        if self.config.determinism_mode not in (None, "", "none", "off", "false", "0"):
            # This codec intentionally does not support deterministic backends.
            raise ValueError(
                "APILLMCodec does not support determinism_mode backends. "
                "Set determinism_mode=None/off."
            )

        self.eof_token_id = self._resolve_eof_token_id()

        self.vocab_size = self._resolve_vocab_size()

        self._decoded_text_to_ids: Optional[Dict[str, List[int]]] = None
        self._init_api_client()

    def _resolve_eof_token_id(self) -> int:
        # Prefer explicit <EOF> token if tokenizer supports that interface.
        try:
            specials = getattr(self.tokenizer, "all_special_tokens", [])
            if "<EOF>" in specials:
                token_id = int(self.tokenizer.convert_tokens_to_ids("<EOF>"))
                if token_id >= 0:
                    return token_id
        except Exception:
            pass

        try:
            if hasattr(self.tokenizer, "add_special_tokens"):
                self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
                token_id = int(self.tokenizer.convert_tokens_to_ids("<EOF>"))
                if token_id >= 0:
                    return token_id
        except Exception:
            pass

        # qwen_tokenizer exposes an end-of-document token id.
        for attr in ("eod_id", "im_end_id", "eos_token_id"):
            value = getattr(self.tokenizer, attr, None)
            if isinstance(value, int) and value >= 0:
                return int(value)

        raise RuntimeError(
            "Could not resolve EOF token id from tokenizer. "
            "Provide a tokenizer exposing <EOF> or eod/eos token id."
        )

    def _resolve_vocab_size(self) -> int:
        # Hugging Face tokenizers generally expose __len__.
        try:
            size = int(len(self.tokenizer))
            if size > 0:
                return size
        except Exception:
            pass

        # Some tokenizers expose explicit vocab size attributes.
        for attr in ("vocab_size", "n_vocab"):
            value = getattr(self.tokenizer, attr, None)
            if isinstance(value, int) and value > 0:
                return int(value)

        # qwen_tokenizer exposes an id->bytes map as `decoder`.
        decoder = getattr(self.tokenizer, "decoder", None)
        if isinstance(decoder, dict) and decoder:
            max_id = max(int(k) for k in decoder.keys())
            return max_id + 1

        raise RuntimeError(
            "Could not determine tokenizer vocab size. "
            "Expected one of: __len__, vocab_size/n_vocab, or decoder dict."
        )

    def _init_api_client(self):
        try:
            from openai import OpenAI
        except Exception as exc:
            raise RuntimeError(
                "openai package is required for APILLMCodec. Install with: pip install openai"
            ) from exc

        api_key = (
            self.config.api_key
            or os.getenv("DASHSCOPE_API_KEY")
            or os.getenv("QWEN_API_KEY")
            or os.getenv("OPENAI_API_KEY")
        )
        if not api_key:
            raise RuntimeError(
                "Missing API key. Set APICodecConfig.api_key or env DASHSCOPE_API_KEY/QWEN_API_KEY/OPENAI_API_KEY."
            )

        self.client = OpenAI(api_key=api_key, base_url=self.config.base_url)

    def _build_token_text_index(self):
        # Token text can be non-unique; keep all candidate IDs.
        self._decoded_text_to_ids = {}
        for token_id in range(self.vocab_size):
            token_text = self._token_text_for_id(token_id)
            self._decoded_text_to_ids.setdefault(token_text, []).append(token_id)

    def _ensure_token_text_index(self):
        if self._decoded_text_to_ids is None:
            self._build_token_text_index()

    def _token_text_for_id(self, token_id: int) -> str:
        try:
            return self.tokenizer.decode([int(token_id)], skip_special_tokens=False)
        except TypeError:
            return self.tokenizer.decode([int(token_id)])
        except Exception:
            return ""

    def _encode_text(self, text: str) -> List[int]:
        try:
            return list(self.tokenizer.encode(text, add_special_tokens=False))
        except TypeError:
            return list(self.tokenizer.encode(text))

    def _decode_ids(self, token_ids: Sequence[int], skip_special_tokens: bool = False) -> str:
        ids = [int(t) for t in token_ids]
        try:
            return self.tokenizer.decode(ids, skip_special_tokens=skip_special_tokens)
        except TypeError:
            if skip_special_tokens:
                specials = {
                    x
                    for x in (
                        getattr(self.tokenizer, "eod_id", None),
                        getattr(self.tokenizer, "im_start_id", None),
                        getattr(self.tokenizer, "im_end_id", None),
                    )
                    if isinstance(x, int)
                }
                ids = [x for x in ids if x not in specials]
            return self.tokenizer.decode(ids)

    def _context_ids_for_index(self, full_token_sequence: Sequence[int], current_idx: int) -> List[int]:
        prefix = list(full_token_sequence[:current_idx])

        if self.config.strategy == "block":
            block_start = (current_idx // self.block_stride) * self.block_stride
            start = max(0, block_start - self.config.margin)
            return prefix[start:]

        if len(prefix) <= self.config.context_window:
            return prefix

        if self.config.strategy == "rolling":
            sink = max(0, int(self.config.api_attention_sink))
            sink = min(sink, self.config.context_window)
            keep_recent = max(0, self.config.context_window - sink)
            if sink == 0:
                return prefix[-self.config.context_window :]
            return prefix[:sink] + prefix[-keep_recent:]

        # no_kv_cache: just sliding-window context.
        return prefix[-self.config.context_window :]

    def _prefix_text_for_index(self, full_token_sequence: Sequence[int], current_idx: int) -> str:
        context_ids = self._context_ids_for_index(full_token_sequence, current_idx)
        if len(context_ids) == 0:
            return ""
        return self._decode_ids(context_ids, skip_special_tokens=False)

    def _diagnostics_enabled(self) -> bool:
        return bool(self.config.diagnostics_csv_prefix)

    def _diagnostics_csv_path(self, phase: str) -> Path:
        base = Path(str(self.config.diagnostics_csv_prefix))
        if base.suffix.lower() == ".csv":
            return base.with_name(f"{base.stem}_{phase}.csv")
        return Path(str(base) + f"_{phase}.csv")

    @staticmethod
    def _diagnostics_fieldnames():
        return [
            "phase",
            "step_idx",
            "token_id",
            "token_text",
            "coder_D_before",
            "selected_interval_contains_D",
            "raw_cum_prob",
            "effective_cum_prob",
            "raw_lower_prob",
            "raw_upper_prob",
            "effective_lower_prob",
            "effective_upper_prob",
            "coder_L_before",
            "coder_R_before",
            "rank_raw",
            "rank_effective",
            "top1_raw_token_id",
            "top1_raw_prob",
            "top1_effective_token_id",
            "top1_effective_prob",
            "symbol_count",
            "counts_total",
        ]

    def _open_diagnostics_writer(self, phase: str):
        if not self._diagnostics_enabled():
            return None, None

        path = self._diagnostics_csv_path(phase)
        path.parent.mkdir(parents=True, exist_ok=True)
        handle = path.open("w", encoding="utf-8", newline="")
        writer = csv.DictWriter(handle, fieldnames=self._diagnostics_fieldnames())
        writer.writeheader()
        return handle, writer

    @staticmethod
    def _rank_from_probs(probs: np.ndarray, token_id: int) -> int:
        token_prob = float(probs[int(token_id)])
        return int(np.count_nonzero(probs > token_prob) + 1)

    @staticmethod
    def _interval_bounds_from_probs(
        probs: np.ndarray,
        token_id: int,
        coder_L: int,
        coder_R: int,
    ):
        token_id = int(token_id)
        lower_prob = float(np.sum(probs[:token_id]))
        token_prob = float(probs[token_id])
        upper_prob = lower_prob + token_prob

        low = int(coder_L + math.floor(coder_R * lower_prob))
        high = int(coder_L + math.floor(coder_R * upper_prob) - 1)
        return lower_prob, upper_prob, low, high

    @staticmethod
    def _interval_bounds_from_counts(
        counts: np.ndarray,
        token_id: int,
        coder_L: int,
        coder_R: int,
    ):
        token_id = int(token_id)
        total = int(np.sum(counts))
        lower_count = int(np.sum(counts[:token_id]))
        symbol_count = int(counts[token_id])
        upper_count = lower_count + symbol_count

        lower_prob = (lower_count / total) if total > 0 else 0.0
        upper_prob = (upper_count / total) if total > 0 else 0.0

        low = int(coder_L + (coder_R * lower_count) // total) if total > 0 else coder_L
        high = int(coder_L + (coder_R * upper_count) // total - 1) if total > 0 else (coder_L - 1)
        return lower_prob, upper_prob, low, high, symbol_count, total

    def _write_token_diagnostic(
        self,
        writer,
        phase: str,
        step_idx: int,
        token_id: int,
        raw_probs: np.ndarray,
        effective_probs: np.ndarray,
        counts,
        coder_L_before: int,
        coder_R_before: int,
        coder_D_before: Optional[int] = None,
    ):
        if writer is None:
            return

        token_id = int(token_id)
        counts_np = np.asarray(counts, dtype=np.int64)

        raw_lower, raw_upper, _, _ = self._interval_bounds_from_probs(
            raw_probs,
            token_id,
            coder_L_before,
            coder_R_before,
        )
        eff_lower, eff_upper, eff_low, eff_high, symbol_count, counts_total = self._interval_bounds_from_counts(
            counts_np,
            token_id,
            coder_L_before,
            coder_R_before,
        )

        contains_d = ""
        if coder_D_before is not None:
            contains_d = int(eff_low <= int(coder_D_before) <= eff_high)

        top1_raw = int(np.argmax(raw_probs))
        top1_eff = int(np.argmax(effective_probs))

        writer.writerow(
            {
                "phase": phase,
                "step_idx": int(step_idx),
                "token_id": token_id,
                "token_text": self._token_text_for_id(token_id).replace("\n", "\\n"),
                "coder_D_before": "" if coder_D_before is None else int(coder_D_before),
                "selected_interval_contains_D": contains_d,
                "raw_cum_prob": raw_upper,
                "effective_cum_prob": eff_upper,
                "raw_lower_prob": raw_lower,
                "raw_upper_prob": raw_upper,
                "effective_lower_prob": eff_lower,
                "effective_upper_prob": eff_upper,
                "coder_L_before": int(coder_L_before),
                "coder_R_before": int(coder_R_before),
                "rank_raw": self._rank_from_probs(raw_probs, token_id),
                "rank_effective": self._rank_from_probs(effective_probs, token_id),
                "top1_raw_token_id": top1_raw,
                "top1_raw_prob": float(raw_probs[top1_raw]),
                "top1_effective_token_id": top1_eff,
                "top1_effective_prob": float(effective_probs[top1_eff]),
                "symbol_count": symbol_count,
                "counts_total": counts_total,
            }
        )

    def _parse_top_logprobs(self, response) -> List[Tuple[str, float]]:
        """Return list[(token_text, logprob)] for one next-token position."""
        out: List[Tuple[str, float]] = []

        # OpenAI-compatible completions format.
        try:
            choice = response.choices[0]
            lp = choice.logprobs
            if lp is not None:
                top_lp = getattr(lp, "top_logprobs", None)
                if top_lp and len(top_lp) > 0:
                    first = top_lp[0]
                    if isinstance(first, dict):
                        for tok, val in first.items():
                            out.append((str(tok), float(val)))
                        if out:
                            return out
        except Exception:
            pass

        # OpenAI-compatible chat.completions logprobs format.
        try:
            choice = response.choices[0]
            content = choice.logprobs.content
            if content and len(content) > 0:
                first = content[0]
                for item in first.top_logprobs:
                    tok = getattr(item, "token", None)
                    lp = getattr(item, "logprob", None)
                    if tok is not None and lp is not None:
                        out.append((str(tok), float(lp)))
        except Exception:
            pass

        return out

    def _extract_top_logprobs_from_api(self, prefix_text: str) -> List[Tuple[str, float]]:
        requested_top_k = max(1, int(self.config.top_k))
        effective_top_k = min(requested_top_k, 5)
        request_mode = str(self.config.api_request_mode).strip().lower()
        if request_mode not in {"chat", "completions", "auto"}:
            raise ValueError("Unsupported api_request_mode. Expected one of: chat, completions, auto")

        extra_body = {}
        if self.config.enable_api_cache_hints:
            extra_body = {
                "use_cache": self.config.strategy != "no_kv_cache",
                "enable_prefix_caching": self.config.strategy != "no_kv_cache",
                "enable_rolling_cache": self.config.strategy == "rolling",
                "attention_sink_size": int(self.config.api_attention_sink),
            }

        last_exc = None

        def _completions_call(with_extra_body: bool):
            kwargs = {
                "model": self.config.model,
                "prompt": prefix_text,
                "max_tokens": 1,
                "temperature": float(self.config.temperature),
                "logprobs": int(effective_top_k),
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            if self.config.api_seed is not None:
                kwargs["seed"] = int(self.config.api_seed)
            if with_extra_body and extra_body:
                kwargs["extra_body"] = extra_body
            return self.client.completions.create(**kwargs)

        def _chat_call(with_extra_body: bool, with_seed: bool):
            kwargs = {
                "model": self.config.model,
                "messages": [{"role": "user", "content": prefix_text}],
                "max_tokens": 1,
                "temperature": float(self.config.temperature),
                "logprobs": True,
                "top_logprobs": int(effective_top_k),
                "top_p": 1,
                "frequency_penalty": 0,
                "presence_penalty": 0,
            }
            if with_seed and self.config.api_seed is not None:
                kwargs["seed"] = int(self.config.api_seed)
            if with_extra_body and extra_body:
                kwargs["extra_body"] = extra_body
            return self.client.chat.completions.create(**kwargs)

        if self.config.deterministic_strict:
            if request_mode == "completions":
                try:
                    response = _completions_call(with_extra_body=bool(extra_body))
                    parsed = self._parse_top_logprobs(response)
                    if parsed:
                        return parsed
                except Exception as exc:
                    last_exc = exc
                    if extra_body:
                        try:
                            response = _completions_call(with_extra_body=False)
                            parsed = self._parse_top_logprobs(response)
                            if parsed:
                                return parsed
                        except Exception as retry_exc:
                            last_exc = retry_exc

                if last_exc is not None:
                    raise last_exc
                raise RuntimeError("Deterministic completions request returned no parseable top-k logprobs.")

            # chat or auto -> keep one request family in strict mode.
            try:
                response = _chat_call(with_extra_body=bool(extra_body), with_seed=True)
                parsed = self._parse_top_logprobs(response)
                if parsed:
                    return parsed
            except Exception as exc:
                last_exc = exc
                try:
                    response = _chat_call(with_extra_body=False, with_seed=False)
                    parsed = self._parse_top_logprobs(response)
                    if parsed:
                        return parsed
                except Exception as retry_exc:
                    last_exc = retry_exc

            if last_exc is not None:
                raise last_exc
            raise RuntimeError("Deterministic chat request returned no parseable top-k logprobs.")

        # Non-strict mode: use configured request mode with fallback.
        if request_mode in {"completions", "auto"}:
            try:
                response = _completions_call(with_extra_body=bool(extra_body))
                parsed = self._parse_top_logprobs(response)
                if parsed:
                    return parsed
            except Exception as exc:
                last_exc = exc
                if extra_body:
                    try:
                        response = _completions_call(with_extra_body=False)
                        parsed = self._parse_top_logprobs(response)
                        if parsed:
                            return parsed
                    except Exception as retry_exc:
                        last_exc = retry_exc

        if request_mode in {"chat", "auto"}:
            try:
                response = _chat_call(with_extra_body=bool(extra_body), with_seed=True)
                parsed = self._parse_top_logprobs(response)
                if parsed:
                    return parsed
            except Exception as exc:
                last_exc = exc
                if extra_body:
                    try:
                        response = _chat_call(with_extra_body=False, with_seed=True)
                        parsed = self._parse_top_logprobs(response)
                        if parsed:
                            return parsed
                    except Exception as retry_exc:
                        last_exc = retry_exc

        if last_exc is not None:
            raise last_exc

        raise RuntimeError(
            "API call did not return parseable top-k logprobs. "
            "Check model capability for logprobs/top_logprobs."
        )

    def _candidate_ids_for_token_text(self, token_text: str) -> List[int]:
        # Fast path: re-tokenize and accept single-id mappings only.
        try:
            ids = self._encode_text(token_text)
        except Exception:
            ids = []
        if len(ids) == 1 and 0 <= int(ids[0]) < self.vocab_size:
            return [int(ids[0])]

        if self.config.strict_single_id_mapping:
            return []

        # Slow fallback: lookup by decoded token text index.
        self._ensure_token_text_index()
        if token_text in self._decoded_text_to_ids:
            return self._decoded_text_to_ids[token_text]
        return []

    @staticmethod
    def _stable_logprob_to_prob(logprob: float) -> float:
        if logprob <= -700.0:
            return 0.0
        if logprob >= 0.0:
            # Guard against invalid positive "log-prob" values.
            return 1.0
        return float(math.exp(logprob))

    def _build_probs_from_topk(self, topk_logprobs: Sequence[Tuple[str, float]]) -> np.ndarray:
        probs = np.zeros(self.vocab_size, dtype=np.float64)
        assigned = np.zeros(self.vocab_size, dtype=bool)

        for token_text, logprob in topk_logprobs:
            token_prob = self._stable_logprob_to_prob(float(logprob))
            if token_prob <= 0.0:
                continue
            candidates = self._candidate_ids_for_token_text(token_text)
            if not candidates:
                continue

            # If multiple IDs decode to same text, split the top-k mass equally.
            share = token_prob / float(len(candidates))
            for token_id in candidates:
                probs[token_id] += share
                assigned[token_id] = True

        mapped_mass = float(np.sum(probs))
        if mapped_mass >= 1.0:
            if mapped_mass <= 0.0:
                probs[:] = 1.0 / float(self.vocab_size)
                return probs
            probs /= mapped_mass
            return probs

        remaining_mass = 1.0 - mapped_mass
        unassigned_count = int(np.count_nonzero(~assigned))
        if unassigned_count > 0:
            fill_value = remaining_mass / float(unassigned_count)
            probs[~assigned] = fill_value
        else:
            # Degenerate case: everything assigned but mass < 1 due to numerical loss.
            probs += remaining_mass / float(self.vocab_size)

        probs = np.clip(probs, 0.0, None)
        total = float(np.sum(probs))
        if total <= 0.0:
            probs[:] = 1.0 / float(self.vocab_size)
        else:
            probs /= total
        return probs

    def _raw_and_effective_probs(self, topk_logprobs: Sequence[Tuple[str, float]]):
        raw_probs = self._build_probs_from_topk(topk_logprobs)
        if not self.config.quant:
            return raw_probs, raw_probs.copy()

        rounded_topk = []
        if self.config.logit_round_decimals >= 0:
            scale = 10 ** int(self.config.logit_round_decimals)
            for token_text, lp in topk_logprobs:
                rounded_topk.append((token_text, float(np.rint(lp * scale) / float(scale))))
        else:
            rounded_topk = list(topk_logprobs)

        eff_probs = self._build_probs_from_topk(rounded_topk)

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

    def _counts_from_probs(self, probs: np.ndarray):
        if self.config.use_legacy_counts:
            return probs_to_counts_legacy(probs, self.config.slots, self.dec_prec)
        return probs_to_counts(probs, self.config.slots, self.dec_prec)

    def _probs_for_position(self, token_sequence: Sequence[int], idx: int):
        prefix_text = self._prefix_text_for_index(token_sequence, idx)
        topk_logprobs = self._extract_top_logprobs_from_api(prefix_text)
        return self._raw_and_effective_probs(topk_logprobs)

    def encode(
        self,
        text: str,
        safe_mode: bool = False,
        return_token_count: bool = False,
        show_progress: bool = True,
    ):
        token_ids = self._encode_text(text)
        if not safe_mode:
            token_ids = token_ids + [self.eof_token_id]

        writer = BitWriter()
        enc = Encoder(Coder(b=self.config.precision), writer)
        iterator = enumerate(token_ids)
        if show_progress:
            iterator = tqdm(iterator, total=len(token_ids), desc="API Encode")

        diag_handle, diag_writer = self._open_diagnostics_writer("encode")
        try:
            for idx, token_id in iterator:
                raw_probs, probs = self._probs_for_position(token_ids, idx)
                counts = self._counts_from_probs(probs)

                coder_L_before = int(enc.coder.L)
                coder_R_before = int(enc.coder.R)
                self._write_token_diagnostic(
                    diag_writer,
                    phase="encode",
                    step_idx=idx,
                    token_id=int(token_id),
                    raw_probs=raw_probs,
                    effective_probs=probs,
                    counts=counts,
                    coder_L_before=coder_L_before,
                    coder_R_before=coder_R_before,
                    coder_D_before=None,
                )

                enc.encode_symbol(int(token_id), counts_to_cum_desc(counts))
        finally:
            if diag_handle is not None:
                diag_handle.close()

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
        decoded_ids: List[int] = []
        diag_handle, diag_writer = self._open_diagnostics_writer("decode")

        if safe_mode and expected_num_tokens is None:
            raise ValueError("safe_mode=True requires expected_num_tokens to be provided.")

        try:
            if safe_mode:
                target_tokens = int(expected_num_tokens)
                iterator = range(target_tokens)
                if show_progress:
                    iterator = tqdm(iterator, total=target_tokens, desc="API Decode")

                for _ in iterator:
                    idx = len(decoded_ids)
                    if max_decode_tokens is not None and idx >= max_decode_tokens:
                        raise RuntimeError(
                            f"Decoding exceeded max_decode_tokens={max_decode_tokens} before target token count."
                        )

                    raw_probs, probs = self._probs_for_position(decoded_ids, idx)
                    counts = self._counts_from_probs(probs)

                    coder_L_before = int(dec.coder.L)
                    coder_R_before = int(dec.coder.R)
                    coder_D_before = int(dec.coder.D)
                    token_id = dec.decode_symbol(counts_to_cum_desc(counts))

                    self._write_token_diagnostic(
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

                    decoded_ids.append(int(token_id))

                return self._decode_ids(decoded_ids, skip_special_tokens=True)

            token_id = None
            decode_iterator = None
            if show_progress:
                decode_iterator = tqdm(desc="API Decode", unit="tok")
            while token_id != self.eof_token_id:
                idx = len(decoded_ids)
                if max_decode_tokens is not None and idx >= max_decode_tokens:
                    raise RuntimeError(
                        f"Decoding exceeded max_decode_tokens={max_decode_tokens} before EOF."
                    )

                raw_probs, probs = self._probs_for_position(decoded_ids, idx)
                counts = self._counts_from_probs(probs)

                coder_L_before = int(dec.coder.L)
                coder_R_before = int(dec.coder.R)
                coder_D_before = int(dec.coder.D)
                token_id = dec.decode_symbol(counts_to_cum_desc(counts))

                self._write_token_diagnostic(
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

                decoded_ids.append(int(token_id))
                if decode_iterator is not None:
                    decode_iterator.update(1)

            if decode_iterator is not None:
                decode_iterator.close()
        finally:
            if diag_handle is not None:
                diag_handle.close()

        return self._decode_ids(decoded_ids[:-1], skip_special_tokens=True)