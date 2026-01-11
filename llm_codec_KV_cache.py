import csv
import math
from typing import Optional

import torch
from bitReadWrite import BitWriter, BitReader
from arithmetic_coding import Coder
from encoder import Encoder
from decoder import Decoder
from utils import counts_to_cum_desc, probs_to_counts
from tqdm import tqdm


class _KVCacheMixin:
    """Shared helpers for KV cache management with block-based resets."""

    def _init_cache_state(self):
        return {
            "past_key_values": None,
            "next_logits": None,
            "cached_token_count": 0,
        }

    def _reset_cache(self, cache_state):
        cache_state["past_key_values"] = None
        cache_state["next_logits"] = None
        cache_state["cached_token_count"] = 0
        return cache_state

    def _is_block_start(self, idx):
        if idx == 0:
            return True
        if self.block_length is None:
            return False
        return idx % self.block_length == 0

    def _block_start_index(self, idx):
        if self.block_length is None:
            return 0
        return (idx // self.block_length) * self.block_length

    def _context_slice(self, idx, token_sequence):
        block_start = self._block_start_index(idx)
        start = block_start
        max_positions = getattr(self.model.config, "max_position_embeddings", None)
        if max_positions is not None and max_positions > 0:
            max_context = max_positions - 1
            start = max(block_start, idx - max_context)
        return token_sequence[start:idx]

    def _ensure_next_logits(self, idx, token_sequence, cache_state):
        if cache_state["next_logits"] is not None:
            return cache_state["next_logits"]

        context_ids = self._context_slice(idx, token_sequence)
        if len(context_ids) == 0:
            if self.tokenizer.pad_token_id is not None:
                input_ids = torch.tensor([[self.tokenizer.pad_token_id]], dtype=torch.long).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    cache_state["next_logits"] = outputs.logits[0, -1, :]
            else:
                vocab_size = self.tokenizer.vocab_size
                probs_tensor = torch.ones(vocab_size, dtype=torch.float32, device=self.device) / vocab_size
                cache_state["next_logits"] = torch.log(probs_tensor)
            cache_state["past_key_values"] = None
            cache_state["cached_token_count"] = 0
            return cache_state["next_logits"]

        input_ids = torch.tensor([context_ids], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=True)
        cache_state["past_key_values"] = outputs.past_key_values
        cache_state["next_logits"] = outputs.logits[0, -1, :]
        cache_state["cached_token_count"] = len(context_ids)
        return cache_state["next_logits"]

    def _advance_cache(self, token_id, cache_state):
        input_ids = torch.tensor([[token_id]], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=cache_state["past_key_values"],
                use_cache=True,
            )
        cache_state["past_key_values"] = outputs.past_key_values
        cache_state["next_logits"] = outputs.logits[0, -1, :]
        cache_state["cached_token_count"] += 1
        return cache_state

class LLM_Encode_KV_Cache(_KVCacheMixin):
    """Encode text into compressed bitstream using LLM-based probabilities."""

    def __init__(
        self,
        tokenizer,
        model,
        precision=32,
        block_length: Optional[int] = 1024,
    ):
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        # add eof special token if it hasn't been added
        if "<EOF>" not in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.precision = precision
        if block_length is not None and block_length <= 0:
            raise ValueError("block_length must be positive or None")
        self.block_length = block_length
        self.model.eval()

    def encode(self, text, demo: bool = False, demo_csv_path: str = "llm_encode_probs.csv"):
        # Break text into tokens
        token_ids = self.tokenizer.encode(text)
        # Append EOF token ID
        eof_token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")
        token_ids.append(eof_token_id)

        # Prepare for encoder
        bit_writer = BitWriter()
        coder_enc = Coder(b=self.precision)
        enc = Encoder(coder_enc, bit_writer)
        slots = coder_enc.tb

        dec_prec = max(50, int(math.ceil(self.precision * math.log10(2))) + 10)

        demo_rows = []
        cache_state = self._init_cache_state()

        print("Encoding tokens. Progress:")
        for i, token_id in tqdm(enumerate(token_ids), total=len(token_ids)):
            if self._is_block_start(i):
                cache_state = self._reset_cache(cache_state)

            logits = self._ensure_next_logits(i, token_ids, cache_state).detach()
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            vocab_size = len(probs)
            counts = probs_to_counts(probs=probs, total=slots, dec_prec=dec_prec)
            cum_desc = counts_to_cum_desc(counts)
            enc.encode_symbol(token_id, cum_desc)

            if demo:
                demo_rows.append(
                    {
                        "position": i,
                        "token_id": int(token_id),
                        "token_text": self.tokenizer.decode([token_id], skip_special_tokens=False),
                        "probability": float(probs[token_id]),
                    }
                )

            if i < len(token_ids) - 1:
                cache_state = self._advance_cache(token_id, cache_state)
        
        enc.finish()
        bit_writer.flush(padbit=0)
        encoded = bit_writer.getvalue()

        if demo and demo_rows:
            with open(demo_csv_path, "w", newline="", encoding="utf-8") as csvfile:
                fieldnames = ["position", "token_id", "token_text", "probability"]
                writer = csv.DictWriter(csvfile, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerows(demo_rows)

        return encoded

class LLM_Decode_KV_Cache(_KVCacheMixin):
    def __init__(
        self,
        tokenizer,
        model,
        precision=32,
        block_length: Optional[int] = 1024,
    ):
        self.tokenizer = tokenizer
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        # add eof special token if it hasn't been added
        if "<EOF>" not in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.precision = precision
        if block_length is not None and block_length <= 0:
            raise ValueError("block_length must be positive or None")
        self.block_length = block_length

        self.model.eval()

    def decode(self, encoded_bytes):
        # Prepare for decoder:
        bit_reader = BitReader(encoded_bytes)
        coder_dec = Coder(b=self.precision)
        dec = Decoder(coder_dec, bit_reader)

        decoded_token_ids = []
        slots = coder_dec.tb

        dec_prec = max(50, int(math.ceil(self.precision * math.log10(2))) + 10)

        cache_state = self._init_cache_state()
        token_id = None
        eof_token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")
        while token_id is None or token_id != eof_token_id:
            position = len(decoded_token_ids)
            if self._is_block_start(position):
                cache_state = self._reset_cache(cache_state)

            logits = self._ensure_next_logits(position, decoded_token_ids, cache_state).detach()
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            vocab_size = len(probs)
            counts = probs_to_counts(probs=probs, total=slots, dec_prec=dec_prec)
            cum_desc = counts_to_cum_desc(counts)
            token_id = dec.decode_symbol(cum_desc)
            decoded_token_ids.append(token_id)

            if token_id != eof_token_id:
                cache_state = self._advance_cache(token_id, cache_state)

        # convert token ids back to text
        decoded_text = self.tokenizer.decode(
            decoded_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded_text
