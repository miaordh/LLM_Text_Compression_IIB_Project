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
from utils import counts_to_cum_desc, probs_to_counts

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
    use_kv_cache: bool = False

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
                use_cache=self.config.use_kv_cache,
            )
        return out.logits[0, -1, :]

    def _probs(self, logits: torch.Tensor) -> np.ndarray:
        if self.config.quant:
            logits_np = logits.detach().to(dtype=torch.float64).cpu().numpy().reshape(-1)

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
        probs_t = torch.exp(self._log_softmax_fn(logits_2d, dim=-1))[0].to(dtype=torch.float64)
        probs = probs_t.detach().cpu().numpy()

        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs[:] = 1.0 / len(probs)
        else:
            probs /= probs_sum
        return probs

    def _counts_from_probs(self, probs: np.ndarray):
        return probs_to_counts(probs, self.config.slots, self.dec_prec)

    def encode(self, text: str):
        token_ids = self.tokenizer.encode(text) + [self.eof_token_id]

        writer = BitWriter()
        enc = Encoder(Coder(b=self.config.precision), writer)

        with self._invariant_context():
            for idx, token_id in tqdm(enumerate(token_ids), total=len(token_ids), desc="Deterministic Encode"):
                prefix = token_ids[:idx]
                logits = self._logits_for_prefix(prefix)
                probs = self._probs(logits)
                counts = self._counts_from_probs(probs)
                enc.encode_symbol(token_id, counts_to_cum_desc(counts))

        enc.finish()
        writer.flush()
        return writer.getvalue()

    def decode(self, encoded_bytes: bytes, max_decode_tokens: Optional[int] = None) -> str:
        dec = Decoder(Coder(b=self.config.precision), BitReader(encoded_bytes))
        decoded_ids = []

        with self._invariant_context():
            token_id = None
            while token_id != self.eof_token_id:
                idx = len(decoded_ids)
                if max_decode_tokens is not None and idx >= max_decode_tokens:
                    raise RuntimeError(
                        f"Decoding exceeded max_decode_tokens={max_decode_tokens} before EOF."
                    )

                logits = self._logits_for_prefix(decoded_ids)
                probs = self._probs(logits)
                counts = self._counts_from_probs(probs)

                token_id = dec.decode_symbol(counts_to_cum_desc(counts))
                decoded_ids.append(token_id)

        return self.tokenizer.decode(decoded_ids[:-1], skip_special_tokens=True)
