import csv
import math
import time
import torch
import numpy as np
import gc
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

# --- COMPATIBILITY FIX ---
try:
    from transformers.cache_utils import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False

# Assumed local imports
from bitReadWrite import BitWriter, BitReader
from arithmetic_coding import Coder
from encoder import Encoder
from decoder import Decoder
from utils import counts_to_cum_desc, probs_to_counts, probs_to_counts_legacy, stabilize_probs

class LLM_Codec_Base:
    """
    Base configuration class.
    Strategies:
      - 'rolling': Continuous KV cache with lazy truncation. (Fastest)
      - 'block':   KV cache with hard resets every stride. (Stable for huge files)
      - 'no_kv_cache': Sliding window with NO cache. (Slow baseline)
    """
    def __init__(
        self,
        tokenizer,
        model,
        precision: int = 32,
        context_window: int = 2048,
        margin: int = 128,
        strategy: str = "rolling",
        device: str = "auto",
        use_legacy_counts: bool = False
    ):
        self.tokenizer = tokenizer
        self.precision = precision
        self.context_window = context_window
        self.margin = margin
        self.strategy = strategy
        self.use_legacy_counts = use_legacy_counts
        
        if self.strategy == "no_kv_cache":
            self.use_kv_cache = False
        else:
            self.use_kv_cache = True

        self.block_stride = self.context_window - self.margin

        # Device Setup
        if device == "auto":
            if torch.cuda.is_available(): self.device = torch.device("cuda")
            elif torch.backends.mps.is_available(): self.device = torch.device("mps")
            else: self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        
        # Ensure Special Tokens
        if "<EOF>" not in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.eof_token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")

        self.slots = 1 << 24 
        self.dec_prec = max(50, int(math.ceil(self.precision * math.log10(2))) + 10)


class _KVCacheMixin:
    """
    Handles Cache Logic & State Management.
    Includes Fix for 'AttributeError: tuple object has no attribute get_seq_length'
    """
    
    def _init_cache_state(self):
        return {
            "past_key_values": None,
            "next_logits": None,
            "cached_token_count": 0,
        }
        
    def _ensure_dynamic_cache(self, past_kv):
        """
        CRITICAL FIX: Converts legacy tuple cache to DynamicCache if required.
        """
        if past_kv is None:
            return None
            
        # If it's already a DynamicCache (or similar object with key_cache), return it
        if hasattr(past_kv, "key_cache"):
            return past_kv
            
        # If it's a tuple and we have DynamicCache available, upgrade it
        if isinstance(past_kv, tuple) and HAS_DYNAMIC_CACHE:
            try:
                return DynamicCache.from_legacy_cache(past_kv)
            except Exception:
                # Fallback if conversion fails (e.g. structure mismatch)
                return past_kv
                
        return past_kv

    def _hard_reset_cache_and_warmup(self, full_sequence, current_index, warmup_length):
        gc.collect()
        if self.device.type == "mps": torch.mps.empty_cache()
            
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
            "cached_token_count": len(warmup_tokens)
        }

    def _truncate_cache_rolling(self, cache_state):
        past_kv = cache_state["past_key_values"]
        limit = self.context_window
        sink = 4 
        
        if past_kv is None: return cache_state
        
        # 1. Handle DynamicCache (New Transformers)
        if hasattr(past_kv, "key_cache"):
            # Simple in-place list slicing for DynamicCache
            current_len = past_kv.key_cache[0].size(2)
            if current_len <= limit: return cache_state
            
            keep_recent = limit - sink
            new_keys = []
            new_values = []
            
            for k, v in zip(past_kv.key_cache, past_kv.value_cache):
                k_sink = k[:, :, :sink, :]
                k_recent = k[:, :, -keep_recent:, :]
                v_sink = v[:, :, :sink, :]
                v_recent = v[:, :, -keep_recent:, :]
                
                new_keys.append(torch.cat([k_sink, k_recent], dim=2))
                new_values.append(torch.cat([v_sink, v_recent], dim=2))
                
            past_kv.key_cache = new_keys
            past_kv.value_cache = new_values
            if hasattr(past_kv, "_seen_tokens"):
                past_kv._seen_tokens = limit # Hack to reset internal counter
                
            cache_state["past_key_values"] = past_kv
            cache_state["cached_token_count"] = limit
            return cache_state

        # 2. Handle Tuple (Legacy)
        elif isinstance(past_kv, tuple):
            current_len = past_kv[0][0].size(2)
            if current_len <= limit: return cache_state
            
            keep_recent = limit - sink
            new_past_kv = []
            for k, v in past_kv:
                new_k = torch.cat([k[:, :, :sink, :], k[:, :, -keep_recent:, :]], dim=2)
                new_v = torch.cat([v[:, :, :sink, :], v[:, :, -keep_recent:, :]], dim=2)
                new_past_kv.append((new_k, new_v))
            
            cache_state["past_key_values"] = tuple(new_past_kv)
            cache_state["cached_token_count"] = limit
            return cache_state
            
        return cache_state

    def _get_logits(self, current_idx, full_token_sequence, cache_state):
        if not self.use_kv_cache:
            start = max(0, current_idx - self.context_window)
            context = full_token_sequence[start:current_idx]
            if len(context) == 0:
                bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.pad_token_id
                if bos is None: return torch.log(torch.ones(self.tokenizer.vocab_size).to(self.device))
                context = [bos]
            input_ids = torch.tensor([context], dtype=torch.long).to(self.device)
            with torch.no_grad():
                outputs = self.model(input_ids, use_cache=False)
                return outputs.logits[0, -1, :]

        if cache_state["next_logits"] is not None:
            return cache_state["next_logits"]
        
        # Cold Start
        if current_idx == 0:
             bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id else self.tokenizer.pad_token_id
             if bos:
                 input_ids = torch.tensor([[bos]], dtype=torch.long).to(self.device)
                 with torch.no_grad():
                     outputs = self.model(input_ids, use_cache=True)
                     return outputs.logits[0, -1, :]
        
        return torch.log(torch.ones(self.tokenizer.vocab_size).to(self.device))

    def _advance_state(self, just_encoded_token_id, cache_state, global_pos_idx):
        if not self.use_kv_cache: return cache_state 

        # --- FIX: Ensure Cache is DynamicCache if needed ---
        past_kv = self._ensure_dynamic_cache(cache_state["past_key_values"])
        # ---------------------------------------------------

        input_ids = torch.tensor([[just_encoded_token_id]], dtype=torch.long).to(self.device)

        with torch.no_grad():
            outputs = self.model(
                input_ids,
                past_key_values=past_kv,
                use_cache=True
            )

        cache_state["past_key_values"] = outputs.past_key_values
        cache_state["next_logits"] = outputs.logits[0, -1, :]
        cache_state["cached_token_count"] += 1
        
        return cache_state


class LLM_Encoder(LLM_Codec_Base, _KVCacheMixin):
    def __init__(
        self,
        *args,
        parallel: bool = False,
        parallel_batch_size: int = 16,
        **kwargs,
    ):
        super().__init__(*args, **kwargs)
        self.parallel = parallel
        self.parallel_batch_size = max(1, int(parallel_batch_size))

    def _get_pad_token_id(self) -> int:
        if self.tokenizer.pad_token_id is not None:
            return self.tokenizer.pad_token_id
        if self.tokenizer.eos_token_id is not None:
            return self.tokenizer.eos_token_id
        if self.tokenizer.bos_token_id is not None:
            return self.tokenizer.bos_token_id
        return 0

    def _get_bos_logits(self) -> torch.Tensor:
        bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
        if bos is None:
            return torch.log(torch.ones(self.tokenizer.vocab_size).to(self.device))
        input_ids = torch.tensor([[bos]], dtype=torch.long).to(self.device)
        with torch.no_grad():
            outputs = self.model(input_ids, use_cache=False)
        return outputs.logits[0, -1, :]

    def _encode_one_token(
        self,
        idx: int,
        token_id: int,
        logits: torch.Tensor,
        enc: Encoder,
        demo: bool,
        demo_rows: List[Dict[str, Any]],
    ):
        # [CHANGE] Use deterministic_softmax instead of standard softmax
        # This returns a Numpy array directly, ensuring Float64 precision was used for the sum.
        probs = stabilize_probs(logits)

        if demo:
            p = float(probs[token_id])
            safe_p = max(p, 1e-12)
            demo_rows.append({
                "pos": idx,
                "token_id": token_id,
                "token": self.tokenizer.decode([token_id]),
                "prob": p,
                "perplexity": 1.0 / safe_p,
                "surprisal_bits": -math.log2(safe_p)
            })

        if self.use_legacy_counts:
            counts = probs_to_counts_legacy(probs, self.slots, self.dec_prec)
        else:
            counts = probs_to_counts(probs, self.slots, self.dec_prec)
        cum_desc = counts_to_cum_desc(counts)
        enc.encode_symbol(token_id, cum_desc)

    def _encode_parallel_no_kv(
        self,
        token_ids: List[int],
        enc: Encoder,
        demo: bool,
        speed_demo: bool,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
        speed_rows: List[Dict[str, float]] = []
        demo_rows: List[Dict[str, Any]] = []
        total = len(token_ids)
        if total == 0:
            return speed_rows, demo_rows

        pad_id = self._get_pad_token_id()
        batch_size = self.parallel_batch_size
        pbar = tqdm(total=total)

        t0 = time.perf_counter()
        self._encode_one_token(0, token_ids[0], self._get_bos_logits(), enc, demo, demo_rows)
        if speed_demo:
            speed_rows.append({"pos": 0, "time": time.perf_counter() - t0})
        pbar.update(1)

        for batch_start in range(1, total, batch_size):
            batch_pos = list(range(batch_start, min(batch_start + batch_size, total)))
            contexts = [token_ids[max(0, i - self.context_window):i] for i in batch_pos]
            lengths = [len(c) for c in contexts]
            max_len = max(lengths)
            input_ids = torch.full((len(batch_pos), max_len), pad_id, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(batch_pos), max_len), dtype=torch.long, device=self.device)

            for row_idx, context in enumerate(contexts):
                context_len = len(context)
                if context_len > 0:
                    input_ids[row_idx, :context_len] = torch.tensor(context, dtype=torch.long, device=self.device)
                    attention_mask[row_idx, :context_len] = 1

            batch_t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
            elapsed = time.perf_counter() - batch_t0
            per_token_elapsed = elapsed / max(1, len(batch_pos))

            for row_idx, pos in enumerate(batch_pos):
                token_t0 = time.perf_counter()
                logits = outputs.logits[row_idx, lengths[row_idx] - 1, :]
                self._encode_one_token(pos, token_ids[pos], logits, enc, demo, demo_rows)
                if speed_demo:
                    overhead = time.perf_counter() - token_t0
                    speed_rows.append({"pos": pos, "time": per_token_elapsed + overhead})
                pbar.update(1)

        pbar.close()
        return speed_rows, demo_rows

    def _encode_parallel_block(
        self,
        token_ids: List[int],
        enc: Encoder,
        demo: bool,
        speed_demo: bool,
    ) -> Tuple[List[Dict[str, float]], List[Dict[str, Any]]]:
        speed_rows: List[Dict[str, float]] = []
        demo_rows: List[Dict[str, Any]] = []
        total = len(token_ids)
        if total == 0:
            return speed_rows, demo_rows

        pbar = tqdm(total=total)
        t0 = time.perf_counter()
        self._encode_one_token(0, token_ids[0], self._get_bos_logits(), enc, demo, demo_rows)
        if speed_demo:
            speed_rows.append({"pos": 0, "time": time.perf_counter() - t0})
        pbar.update(1)

        block_items = []
        for block_start in range(0, total, self.block_stride):
            block_end = min(block_start + self.block_stride, total)
            first_target = max(block_start, 1)
            if first_target >= block_end:
                continue

            warmup_start = max(0, block_start - self.margin)
            warmup_tokens = token_ids[warmup_start:block_start]
            prefix_tokens = warmup_tokens + token_ids[block_start:block_end - 1]
            if len(prefix_tokens) == 0:
                continue

            warmup_len = len(warmup_tokens)
            mapping = []
            for pos in range(first_target, block_end):
                out_idx = warmup_len - 1 + (pos - block_start)
                mapping.append((pos, out_idx))

            block_items.append({"seq": prefix_tokens, "mapping": mapping})

        if not block_items:
            pbar.close()
            return speed_rows, demo_rows

        pad_id = self._get_pad_token_id()
        batch_size = self.parallel_batch_size
        for batch_start in range(0, len(block_items), batch_size):
            batch_items = block_items[batch_start:batch_start + batch_size]
            lengths = [len(item["seq"]) for item in batch_items]
            max_len = max(lengths)

            input_ids = torch.full((len(batch_items), max_len), pad_id, dtype=torch.long, device=self.device)
            attention_mask = torch.zeros((len(batch_items), max_len), dtype=torch.long, device=self.device)
            for row_idx, item in enumerate(batch_items):
                seq = item["seq"]
                seq_len = len(seq)
                input_ids[row_idx, :seq_len] = torch.tensor(seq, dtype=torch.long, device=self.device)
                attention_mask[row_idx, :seq_len] = 1

            batch_t0 = time.perf_counter()
            with torch.no_grad():
                outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=False)
            elapsed = time.perf_counter() - batch_t0

            token_count = sum(len(item["mapping"]) for item in batch_items)
            per_token_elapsed = elapsed / max(1, token_count)

            for row_idx, item in enumerate(batch_items):
                for pos, out_idx in item["mapping"]:
                    token_t0 = time.perf_counter()
                    logits = outputs.logits[row_idx, out_idx, :]
                    self._encode_one_token(pos, token_ids[pos], logits, enc, demo, demo_rows)
                    if speed_demo:
                        overhead = time.perf_counter() - token_t0
                        speed_rows.append({"pos": pos, "time": per_token_elapsed + overhead})
                    pbar.update(1)

        pbar.close()
        return speed_rows, demo_rows

    def encode(
        self, 
        text, 
        demo: bool = False, 
        demo_csv_path: str = "compression_stats.csv",
        speed_demo: bool = False, 
        speed_csv_path: str = "speed_encode.csv"
    ):
        token_ids = self.tokenizer.encode(text)
        token_ids.append(self.eof_token_id)
        
        bit_writer = BitWriter()
        coder = Coder(b=self.precision)
        enc = Encoder(coder, bit_writer)
        
        speed_rows = []
        demo_rows = []
        
        mode_label = "PARALLEL" if self.parallel else "SEQUENTIAL"
        print(f"Encoding {len(token_ids)} tokens via '{self.strategy.upper()}' strategy [{mode_label}]...")

        # --- PARALLEL PATH ---
        if self.parallel:
            if self.strategy == "rolling":
                print("Parallel mode is not supported for 'rolling'; falling back to sequential mode.")
            
            elif self.strategy == "block":
                speed_rows, demo_rows = self._encode_parallel_block(token_ids, enc, demo, speed_demo)
                
                # [FIXED] Must finish encoder to flush buffered bits!
                enc.finish()
                bit_writer.flush(padbit=0)

                if speed_demo and speed_rows:
                    with open(speed_csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["pos", "time"])
                        writer.writeheader()
                        writer.writerows(speed_rows)
                if demo and demo_rows:
                    with open(demo_csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=demo_rows[0].keys())
                        writer.writeheader()
                        writer.writerows(demo_rows)
                return bit_writer.getvalue()

            elif self.strategy == "no_kv_cache":
                speed_rows, demo_rows = self._encode_parallel_no_kv(token_ids, enc, demo, speed_demo)
                
                # [FIXED] Must finish encoder to flush buffered bits!
                enc.finish()
                bit_writer.flush(padbit=0)

                if speed_demo and speed_rows:
                    with open(speed_csv_path, "w", newline="") as f:
                        writer = csv.DictWriter(f, fieldnames=["pos", "time"])
                        writer.writeheader()
                        writer.writerows(speed_rows)
                if demo and demo_rows:
                    with open(demo_csv_path, "w", newline="", encoding="utf-8") as f:
                        writer = csv.DictWriter(f, fieldnames=demo_rows[0].keys())
                        writer.writeheader()
                        writer.writerows(demo_rows)
                return bit_writer.getvalue()

        # --- SEQUENTIAL / FALLBACK LOOP ---
        cache_state = self._init_cache_state()
        if self.use_kv_cache:
            cache_state = self._hard_reset_cache_and_warmup([], 0, 0)
            cache_state["next_logits"] = self._get_logits(0, token_ids, cache_state)

        for i, token_id in tqdm(enumerate(token_ids), total=len(token_ids)):
            t_start = time.perf_counter()
            
            # --- BLOCK STRATEGY RESET ---
            if self.strategy == "block" and i > 0 and (i % self.block_stride == 0):
                cache_state = self._hard_reset_cache_and_warmup(
                    token_ids, current_index=i, warmup_length=self.margin
                )

            logits = self._get_logits(i, token_ids, cache_state)
            
            # [CRITICAL] Use the deterministic softmax from utils.py
            # This handles the Cross-Device rounding logic
            probs = stabilize_probs(logits)
            
            # Demo Stats
            if demo:
                p = float(probs[token_id])
                safe_p = max(p, 1e-12)
                demo_rows.append({
                    "pos": i, "token_id": token_id, "token": self.tokenizer.decode([token_id]),
                    "prob": p, "perplexity": 1.0/safe_p, "surprisal_bits": -math.log2(safe_p)
                })

            if self.use_legacy_counts:
                counts = probs_to_counts_legacy(probs, self.slots, self.dec_prec)
            else:
                counts = probs_to_counts(probs, self.slots, self.dec_prec)
            cum_desc = counts_to_cum_desc(counts)
            enc.encode_symbol(token_id, cum_desc)
            
            # Advance
            if i < len(token_ids) - 1:
                cache_state = self._advance_state(token_id, cache_state, i)
                if self.strategy == "rolling":
                    if cache_state["cached_token_count"] > (self.context_window + self.margin):
                        cache_state = self._truncate_cache_rolling(cache_state)
            
            t_end = time.perf_counter()
            if speed_demo:
                speed_rows.append({"pos": i, "time": t_end - t_start})

        enc.finish()
        bit_writer.flush(padbit=0)
        
        if speed_demo and speed_rows:
            with open(speed_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["pos", "time"])
                writer.writeheader()
                writer.writerows(speed_rows)
        if demo and demo_rows:
            with open(demo_csv_path, "w", newline="", encoding="utf-8") as f:
                writer = csv.DictWriter(f, fieldnames=demo_rows[0].keys())
                writer.writeheader()
                writer.writerows(demo_rows)
                
        return bit_writer.getvalue()

class LLM_Decoder(LLM_Codec_Base, _KVCacheMixin):
    def decode(self, encoded_bytes, speed_demo=False, speed_csv_path="speed_decode.csv"):
        bit_reader = BitReader(encoded_bytes)
        coder = Coder(b=self.precision)
        dec = Decoder(coder, bit_reader)
        
        decoded_ids = []
        cache_state = self._init_cache_state()
        token_id = None
        speed_rows = []
        
        if self.use_kv_cache:
            cache_state = self._hard_reset_cache_and_warmup([], 0, 0)
            cache_state["next_logits"] = self._get_logits(0, [], cache_state)
        
        pbar = tqdm(desc=f"Decoding ({self.strategy})", unit="tok")
        
        while token_id != self.eof_token_id:
            t_start = time.perf_counter()
            curr_idx = len(decoded_ids)
            
            if self.strategy == "block" and curr_idx > 0 and (curr_idx % self.block_stride == 0):
                cache_state = self._hard_reset_cache_and_warmup(
                    decoded_ids, current_index=curr_idx, warmup_length=self.margin
                )
            
            logits = self._get_logits(curr_idx, decoded_ids, cache_state)
            
            probs = stabilize_probs(logits)
            
            if self.use_legacy_counts:
                counts = probs_to_counts_legacy(probs, self.slots, self.dec_prec)
            else:
                counts = probs_to_counts(probs, self.slots, self.dec_prec)
                
            cum_desc = counts_to_cum_desc(counts)
            token_id = dec.decode_symbol(cum_desc)
            decoded_ids.append(token_id)
            
            if token_id != self.eof_token_id:
                cache_state = self._advance_state(token_id, cache_state, curr_idx)
                if self.strategy == "rolling":
                    if cache_state["cached_token_count"] > (self.context_window + self.margin):
                        cache_state = self._truncate_cache_rolling(cache_state)
            
            pbar.update(1)
            t_end = time.perf_counter()
            if speed_demo:
                speed_rows.append({"pos": curr_idx, "time": t_end - t_start})

        pbar.close()
        
        if speed_demo:
            with open(speed_csv_path, "w", newline="") as f:
                writer = csv.DictWriter(f, fieldnames=["pos", "time"])
                writer.writeheader()
                writer.writerows(speed_rows)
                
        return self.tokenizer.decode(decoded_ids[:-1], skip_special_tokens=True)