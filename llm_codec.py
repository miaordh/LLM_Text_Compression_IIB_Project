import csv
import math
import time
import torch
import numpy as np
import gc
from typing import Optional, Dict, Any, List, Tuple
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from transformers.cache_utils import DynamicCache
    HAS_DYNAMIC_CACHE = True
except ImportError:
    HAS_DYNAMIC_CACHE = False

from bitReadWrite import BitWriter, BitReader
from arithmetic_coding import Coder
from encoder import Encoder
from decoder import Decoder
from utils import counts_to_cum_desc, probs_to_counts, probs_to_counts_legacy, stabilize_probs

class LLM_Codec_Base:
    def __init__(self, tokenizer, model, precision: int = 32, context_window: int = 2048, 
                 margin: int = 128, strategy: str = "rolling", device: str = "auto", use_legacy_counts: bool = False):
        self.tokenizer = tokenizer
        self.precision = precision
        self.context_window = context_window
        self.margin = margin
        self.strategy = strategy
        self.use_legacy_counts = use_legacy_counts
        self.use_kv_cache = (self.strategy != "no_kv_cache")
        self.block_stride = self.context_window - self.margin

        if device == "auto":
            if torch.cuda.is_available(): self.device = torch.device("cuda")
            elif torch.backends.mps.is_available(): self.device = torch.device("mps")
            else: self.device = torch.device("cpu")
        else:
            self.device = torch.device(device)

        self.model = model.to(self.device)
        self.model.eval()
        
        if "<EOF>" not in self.tokenizer.all_special_tokens:
            self.tokenizer.add_special_tokens({"additional_special_tokens": ["<EOF>"]})
            self.model.resize_token_embeddings(len(self.tokenizer))
        self.eof_token_id = self.tokenizer.convert_tokens_to_ids("<EOF>")

        self.slots = 1 << 24 
        self.dec_prec = max(50, int(math.ceil(self.precision * math.log10(2))) + 10)

class _KVCacheMixin:
    def _init_cache_state(self):
        return {"past_key_values": None, "next_logits": None, "cached_token_count": 0}
        
    def _ensure_dynamic_cache(self, past_kv):
        if past_kv is None: return None
        if hasattr(past_kv, "key_cache"): return past_kv
        if isinstance(past_kv, tuple) and HAS_DYNAMIC_CACHE:
            try: return DynamicCache.from_legacy_cache(past_kv)
            except: return past_kv
        return past_kv

    def _hard_reset_cache_and_warmup(self, full_sequence, current_index, warmup_length):
        gc.collect()
        if self.device.type == "mps": torch.mps.empty_cache()
        start = max(0, current_index - warmup_length)
        warmup_tokens = full_sequence[start:current_index]
        if not warmup_tokens: return self._init_cache_state()
        
        input_ids = torch.tensor([warmup_tokens], dtype=torch.long).to(self.device)
        # [FIX] Force Attention Mask even during warmup to match Parallel behavior
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = self.model(input_ids, attention_mask=attention_mask, use_cache=True)
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
        
        if hasattr(past_kv, "key_cache"):
            if past_kv.key_cache[0].size(2) <= limit: return cache_state
            keep = limit - sink
            past_kv.key_cache = [torch.cat([k[:,:,:sink], k[:,:,-keep:]], 2) for k in past_kv.key_cache]
            past_kv.value_cache = [torch.cat([v[:,:,:sink], v[:,:,-keep:]], 2) for v in past_kv.value_cache]
            if hasattr(past_kv, "_seen_tokens"): past_kv._seen_tokens = limit
            cache_state["cached_token_count"] = limit
            
        elif isinstance(past_kv, tuple):
            if past_kv[0][0].size(2) <= limit: return cache_state
            keep = limit - sink
            cache_state["past_key_values"] = tuple(
                (torch.cat([k[:,:,:sink], k[:,:,-keep:]], 2), torch.cat([v[:,:,:sink], v[:,:,-keep:]], 2))
                for k, v in past_kv
            )
            cache_state["cached_token_count"] = limit
        return cache_state

    def _get_logits(self, current_idx, full_token_sequence, cache_state):
        if not self.use_kv_cache:
            start = max(0, current_idx - self.context_window)
            ctx = full_token_sequence[start:current_idx]
            if not ctx: 
                bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
                ctx = [bos] if bos is not None else [0]
                
            input_ids = torch.tensor([ctx], dtype=torch.long).to(self.device)
            
            # [CRITICAL FIX] 
            # Sequential mode was missing 'attention_mask'. Parallel mode uses it.
            # This causes "Path Divergence" inside the model kernels.
            # We must explicitely pass all-ones mask here to force the same kernel usage.
            attention_mask = torch.ones_like(input_ids)
            
            with torch.no_grad():
                return self.model(input_ids, attention_mask=attention_mask, use_cache=False).logits[0, -1, :]

        if cache_state["next_logits"] is not None: return cache_state["next_logits"]
        if current_idx == 0:
             bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
             if bos is not None:
                 input_ids = torch.tensor([[bos]], device=self.device)
                 attention_mask = torch.ones_like(input_ids)
                 with torch.no_grad():
                     return self.model(input_ids, attention_mask=attention_mask, use_cache=True).logits[0, -1, :]
        return torch.zeros(self.tokenizer.vocab_size, device=self.device) 

    def _advance_state(self, token_id, cache_state, global_pos_idx):
        if not self.use_kv_cache: return cache_state 
        past_kv = self._ensure_dynamic_cache(cache_state["past_key_values"])
        
        input_ids = torch.tensor([[token_id]], device=self.device)
        attention_mask = torch.ones_like(input_ids)
        
        with torch.no_grad():
            outputs = self.model(
                input_ids,
                attention_mask=attention_mask, # [FIX] Add mask here too
                past_key_values=past_kv, 
                use_cache=True
            )
        cache_state["past_key_values"] = outputs.past_key_values
        cache_state["next_logits"] = outputs.logits[0, -1, :]
        cache_state["cached_token_count"] += 1
        return cache_state

class LLM_Encoder(LLM_Codec_Base, _KVCacheMixin):
    def __init__(self, *args, parallel: bool = False, parallel_batch_size: int = 16, **kwargs):
        super().__init__(*args, **kwargs)
        self.parallel = parallel
        self.parallel_batch_size = max(1, int(parallel_batch_size))

    def _encode_token(self, token_id, logits, enc):
        probs = stabilize_probs(logits)
        if self.use_legacy_counts:
            counts = probs_to_counts_legacy(probs, self.slots, self.dec_prec)
        else:
            counts = probs_to_counts(probs, self.slots, self.dec_prec)
        enc.encode_symbol(token_id, counts_to_cum_desc(counts))

    def encode(self, text, demo=False, speed_demo=False, **kwargs):
        token_ids = self.tokenizer.encode(text) + [self.eof_token_id]
        bit_writer = BitWriter()
        enc = Encoder(Coder(b=self.precision), bit_writer)
        
        if self.parallel and self.strategy == "no_kv_cache":
            # --- PARALLEL ENCODING (NO KV) ---
            total = len(token_ids)
            
            # [FIX] Ensure pad_id is valid (never None)
            pad_id = self.tokenizer.pad_token_id if self.tokenizer.pad_token_id is not None else 0
            
            # 1. First Token (BOS)
            bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
            if bos is not None:
                dummy_in = torch.tensor([[bos]], device=self.device)
                dummy_mask = torch.ones_like(dummy_in)
                logits = self.model(dummy_in, attention_mask=dummy_mask).logits[0, -1, :]
            else:
                logits = torch.zeros(self.tokenizer.vocab_size).to(self.device)
            self._encode_token(token_ids[0], logits, enc)

            # 2. Parallel Loop
            bs = self.parallel_batch_size
            for i in tqdm(range(1, total, bs), desc="Parallel Encode"):
                batch_indices = range(i, min(i + bs, total))
                
                # Construct contexts
                contexts = []
                for idx in batch_indices:
                    start = max(0, idx - self.context_window)
                    # [CRITICAL PARITY FIX] 
                    # Sequential: ctx = tokens[start:idx]
                    # Parallel:   ctx = tokens[start:idx]
                    c = token_ids[start:idx]
                    if not c: # Should rarely happen given i starts at 1
                         bos = self.tokenizer.bos_token_id if self.tokenizer.bos_token_id is not None else self.tokenizer.pad_token_id
                         c = [bos] if bos is not None else [0]
                    contexts.append(c)

                max_len = max(len(c) for c in contexts)
                
                input_ids = torch.full((len(batch_indices), max_len), pad_id, dtype=torch.long, device=self.device)
                mask = torch.zeros((len(batch_indices), max_len), dtype=torch.long, device=self.device)
                
                for r, ctx in enumerate(contexts):
                    l = len(ctx)
                    # Right Padding
                    input_ids[r, :l] = torch.tensor(ctx, device=self.device)
                    mask[r, :l] = 1
                
                with torch.no_grad():
                    out = self.model(input_ids, attention_mask=mask)
                
                for r, idx in enumerate(batch_indices):
                    # For right padding, the last valid token is at index len(ctx) - 1
                    valid_len = len(contexts[r])
                    logits = out.logits[r, valid_len - 1, :]
                    self._encode_token(token_ids[idx], logits, enc)
            
            enc.finish()
            bit_writer.flush()
            return bit_writer.getvalue()

        # --- SEQUENTIAL ENCODING ---
        cache = self._init_cache_state()
        if self.use_kv_cache:
            cache = self._hard_reset_cache_and_warmup([], 0, 0)
            cache["next_logits"] = self._get_logits(0, token_ids, cache)

        for i, tid in tqdm(enumerate(token_ids), total=len(token_ids)):
            if self.strategy == "block" and i > 0 and i % self.block_stride == 0:
                cache = self._hard_reset_cache_and_warmup(token_ids, i, self.margin)
            
            logits = self._get_logits(i, token_ids, cache)
            self._encode_token(tid, logits, enc)
            
            if i < len(token_ids) - 1:
                cache = self._advance_state(tid, cache, i)
                if self.strategy == "rolling" and cache["cached_token_count"] > self.context_window + self.margin:
                    cache = self._truncate_cache_rolling(cache)

        enc.finish()
        bit_writer.flush()
        return bit_writer.getvalue()

class LLM_Decoder(LLM_Codec_Base, _KVCacheMixin):
    def decode(self, encoded_bytes, speed_demo=False, **kwargs):
        dec = Decoder(Coder(b=self.precision), BitReader(encoded_bytes))
        decoded_ids = []
        cache = self._init_cache_state()
        token_id = None
        
        if self.use_kv_cache:
            cache = self._hard_reset_cache_and_warmup([], 0, 0)
            cache["next_logits"] = self._get_logits(0, [], cache)
            
        pbar = tqdm(desc="Decoding")
        while token_id != self.eof_token_id:
            curr_idx = len(decoded_ids)
            if self.strategy == "block" and curr_idx > 0 and curr_idx % self.block_stride == 0:
                cache = self._hard_reset_cache_and_warmup(decoded_ids, curr_idx, self.margin)
            
            logits = self._get_logits(curr_idx, decoded_ids, cache)
            
            # [FIX] Use same stabilizer
            probs = stabilize_probs(logits)
            
            if self.use_legacy_counts:
                counts = probs_to_counts_legacy(probs, self.slots, self.dec_prec)
            else:
                counts = probs_to_counts(probs, self.slots, self.dec_prec)
                
            token_id = dec.decode_symbol(counts_to_cum_desc(counts))
            decoded_ids.append(token_id)
            
            if token_id != self.eof_token_id:
                cache = self._advance_state(token_id, cache, curr_idx)
                if self.strategy == "rolling" and cache["cached_token_count"] > self.context_window + self.margin:
                    cache = self._truncate_cache_rolling(cache)
            pbar.update(1)
            
        return self.tokenizer.decode(decoded_ids[:-1], skip_special_tokens=True)