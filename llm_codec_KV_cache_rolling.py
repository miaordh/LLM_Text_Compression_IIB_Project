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
    """Shared helpers for Rolling KV cache management with Attention Sinks."""

    def _init_cache_state(self):
        return {
            "past_key_values": None,
            "next_logits": None,
            "cached_token_count": 0,
        }

    def _truncate_cache(self, cache_state, target_window, sink_size=4):
        """
        Implements Rolling Cache with Attention Sinks.
        Truncates the cache down to 'target_window' size.
        """
        past_kv = cache_state["past_key_values"]
        if past_kv is None:
            return cache_state

        # --- DynamicCache Support ---
        if hasattr(past_kv, "key_cache") and hasattr(past_kv, "value_cache"):
            current_len = past_kv.key_cache[0].size(2)
            if current_len <= target_window:
                return cache_state

            # We calculate how many recent tokens to keep
            keep_recent = target_window - sink_size
            
            new_keys = []
            new_values = []
            
            for k, v in zip(past_kv.key_cache, past_kv.value_cache):
                # [Sinks] + [Most Recent]
                k_sinks = k[:, :, :sink_size, :]
                k_recent = k[:, :, -keep_recent:, :]
                new_keys.append(torch.cat([k_sinks, k_recent], dim=2))
                
                v_sinks = v[:, :, :sink_size, :]
                v_recent = v[:, :, -keep_recent:, :]
                new_values.append(torch.cat([v_sinks, v_recent], dim=2))
            
            past_kv.key_cache = new_keys
            past_kv.value_cache = new_values
            
            if hasattr(past_kv, "_seen_tokens"):
                past_kv._seen_tokens = target_window
                
            cache_state["past_key_values"] = past_kv
            cache_state["cached_token_count"] = target_window
            return cache_state

        # --- Tuple Support ---
        if isinstance(past_kv, tuple):
            current_len = past_kv[0][0].size(2)
            if current_len <= target_window:
                return cache_state

            keep_recent = target_window - sink_size
            new_past_kv = []
            
            for layer_past in past_kv:
                keys, values = layer_past
                k_sinks = keys[:, :, :sink_size, :]
                k_recent = keys[:, :, -keep_recent:, :]
                
                v_sinks = values[:, :, :sink_size, :]
                v_recent = values[:, :, -keep_recent:, :]
                
                new_past_kv.append((
                    torch.cat([k_sinks, k_recent], dim=2),
                    torch.cat([v_sinks, v_recent], dim=2)
                ))
                
            cache_state["past_key_values"] = tuple(new_past_kv)
            cache_state["cached_token_count"] = target_window
            return cache_state
            
        return cache_state

    def _ensure_next_logits(self, idx, token_sequence, cache_state):
        if cache_state["next_logits"] is not None:
            return cache_state["next_logits"]

        # Initialization logic (Start of file)
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

    def _advance_cache(self, token_id, cache_state, global_position_idx):
        input_ids = torch.tensor([[token_id]], dtype=torch.long).to(self.device)
        position_ids = torch.tensor([[global_position_idx]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=cache_state["past_key_values"],
                use_cache=True,
                position_ids=position_ids 
            )
        
        cache_state["past_key_values"] = outputs.past_key_values
        cache_state["next_logits"] = outputs.logits[0, -1, :]
        cache_state["cached_token_count"] += 1

        # --- OPTIMIZATION: Lazy Truncation ---
        # Allow cache to grow 10% larger than limit before truncating.
        # This reduces memory copy overhead by ~50x.
        overflow_buffer = 128  # Buffer size
        
        if self.context_window and cache_state["cached_token_count"] > (self.context_window + overflow_buffer):
            # When we finally truncate, we cut it back down to exactly self.context_window
            cache_state = self._truncate_cache(cache_state, target_window=self.context_window, sink_size=4)
            
        return cache_state


class LLM_Encode_KV_Cache(_KVCacheMixin):
    """Encode text into compressed bitstream using Rolling KV Cache."""

    def __init__(
        self,
        tokenizer,
        model,
        precision=32,
        context_window: Optional[int] = 2048, # Renamed from block_length
    ):
        self.tokenizer = tokenizer
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using NVIDIA GPU (CUDA)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        self.model = model.to(self.device)
        
        if "<EOF>" not in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        self.precision = precision
        if context_window is not None and context_window <= 4:
            # Need at least >4 for sink logic
            raise ValueError("context_window must be > 4 for attention sinks")
        self.context_window = context_window
        self.model.eval()

    def encode(self, text, demo: bool = False, demo_csv_path: str = "llm_encode_probs.csv"):
        token_ids = self.tokenizer.encode(text)
        eof_token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")
        token_ids.append(eof_token_id)

        bit_writer = BitWriter()
        coder_enc = Coder(b=self.precision)
        enc = Encoder(coder_enc, bit_writer)
        
        # Use Safe Slots (1 << 24) to prevent Register Underflow with precision=32
        slots = 1 << 24 

        dec_prec = max(50, int(math.ceil(self.precision * math.log10(2))) + 10)

        demo_rows = []
        cache_state = self._init_cache_state()

        print("Encoding tokens (Rolling Window)...")
        for i, token_id in tqdm(enumerate(token_ids), total=len(token_ids)):
            
            # 1. Get Logits
            # Note: For i > 0, this returns logits cached from the previous loop's _advance_cache
            logits = self._ensure_next_logits(i, token_ids, cache_state).detach()
            
            # 2. Probability & Arithmetic Coding
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            counts = probs_to_counts(probs=probs, total=slots, dec_prec=dec_prec)
            cum_desc = counts_to_cum_desc(counts)
            enc.encode_symbol(token_id, cum_desc)

            if demo:
                demo_rows.append({
                    "position": i,
                    "token_id": int(token_id),
                    "token_text": self.tokenizer.decode([token_id], skip_special_tokens=False),
                    "probability": float(probs[token_id]),
                })

            # 3. Advance Cache
            # We must pass the GLOBAL POSITION index.
            # We just encoded the token at 'token_ids[i]'. 
            # The model predicts 'i+1' based on input 'i'. 
            # So the position of the input we are feeding is 'i'.
            if i < len(token_ids) - 1:
                cache_state = self._advance_cache(token_id, cache_state, global_position_idx=i)
        
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
        context_window: Optional[int] = 2048,
    ):
        self.tokenizer = tokenizer
        if torch.backends.mps.is_available():
            self.device = torch.device("mps")
            print("Using Apple Silicon GPU (MPS)")
        elif torch.cuda.is_available():
            self.device = torch.device("cuda")
            print("Using NVIDIA GPU (CUDA)")
        else:
            self.device = torch.device("cpu")
            print("Using CPU")
        self.model = model.to(self.device)
        
        if "<EOF>" not in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
            
        self.precision = precision
        if context_window is not None and context_window <= 4:
            raise ValueError("context_window must be > 4 for attention sinks")
        self.context_window = context_window
        self.model.eval()

    def decode(self, encoded_bytes):
        bit_reader = BitReader(encoded_bytes)
        coder_dec = Coder(b=self.precision)
        dec = Decoder(coder_dec, bit_reader)

        decoded_token_ids = []
        # Must match Encoder slots exactly
        slots = 1 << 24 

        dec_prec = max(50, int(math.ceil(self.precision * math.log10(2))) + 10)

        cache_state = self._init_cache_state()
        token_id = None
        eof_token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")
        
        pbar = tqdm(desc="Decoding (Rolling)", unit="tok")
        
        while token_id != eof_token_id:
            current_idx = len(decoded_token_ids)

            # 1. Get Logits
            logits = self._ensure_next_logits(current_idx, decoded_token_ids, cache_state).detach()
            
            # 2. Probability & Arithmetic Decoding
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            counts = probs_to_counts(probs=probs, total=slots, dec_prec=dec_prec)
            cum_desc = counts_to_cum_desc(counts)
            token_id = dec.decode_symbol(cum_desc)
            decoded_token_ids.append(token_id)
            pbar.update(1)

            # 3. Advance Cache
            # The token we just decoded is at 'len(decoded) - 1' (which is current_idx)
            if token_id != eof_token_id:
                cache_state = self._advance_cache(token_id, cache_state, global_position_idx=current_idx)

        pbar.close()

        decoded_text = self.tokenizer.decode(
            decoded_token_ids[:-1], # Exclude EOF
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded_text