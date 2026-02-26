import math
from dataclasses import dataclass
from typing import Optional

import numpy as np
import torch
from tqdm import tqdm

from arithmetic_coding import Coder
from bitReadWrite import BitReader, BitWriter
from decoder import Decoder
from encoder import Encoder
from deterministic_runtime import DeterministicKernelConfig, deterministic_kernel_context, deterministic_softmax
from utils import counts_to_cum_desc, probs_to_counts, probs_to_counts_legacy


@dataclass
class DeterministicCodecConfig:
    determinism_mode: str = "strict_cpu"
    precision: int = 32
    slots: int = (1 << 24)
    use_legacy_counts: bool = True
    use_kv_cache: bool = False
    seed: int = 0
    patch_linear: bool = True
    patch_rmsnorm: bool = True
    patch_attention: bool = True


class DeterministicLLMCodec:
    def __init__(self, tokenizer, model, device: str = "auto", config: Optional[DeterministicCodecConfig] = None):
        self.tokenizer = tokenizer
        self.config = config or DeterministicCodecConfig()

        if self.config.determinism_mode not in {"strict_cpu", "gpu_best_effort"}:
            raise ValueError("determinism_mode must be 'strict_cpu' or 'gpu_best_effort'")

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

        try:
            self.model.config._attn_implementation = "eager"
        except Exception:
            pass

        self.kernel_config = DeterministicKernelConfig(
            mode=self.config.determinism_mode,
            seed=self.config.seed,
            patch_linear=self.config.patch_linear,
            patch_rmsnorm=self.config.patch_rmsnorm,
            patch_attention=self.config.patch_attention,
            force_cpu_kernels=(self.config.determinism_mode == "strict_cpu"),
            single_thread_cpu=(self.config.determinism_mode == "strict_cpu"),
        )

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
        probs_t = deterministic_softmax(logits, dim=-1, mode=self.config.determinism_mode).detach().cpu().to(torch.float64)
        probs = probs_t.numpy()

        probs_sum = probs.sum()
        if probs_sum <= 0:
            probs[:] = 1.0 / len(probs)
        else:
            probs /= probs_sum
        return probs

    def _counts_from_probs(self, probs: np.ndarray):
        if self.config.use_legacy_counts:
            return probs_to_counts_legacy(probs.tolist(), self.config.slots, self.dec_prec)
        return probs_to_counts(probs, self.config.slots, self.dec_prec)

    def encode(self, text: str):
        token_ids = self.tokenizer.encode(text) + [self.eof_token_id]

        writer = BitWriter()
        enc = Encoder(Coder(b=self.config.precision), writer)

        with deterministic_kernel_context(self.model, self.kernel_config):
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

        with deterministic_kernel_context(self.model, self.kernel_config):
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
