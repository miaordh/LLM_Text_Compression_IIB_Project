import torch
from bitReadWrite import BitWriter, BitReader
from arithmetic_coding import Coder
from encoder import Encoder
from decoder import Decoder
from utils import counts_to_cum_desc, probs_to_counts, get_context_slice
from tqdm import tqdm
import math

class LLM_Encode:
    """Encode text into compressed bitstream using LLM-based probabilities."""
    def __init__(self, tokenizer, model, precision=32):
        self.tokenizer = tokenizer
        self.precision = precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def encode(self, text):
        # Break text into tokens
        token_ids = self.tokenizer.encode(text)
        text_length = len(token_ids)
        
        # Prepare for encoder:
        bit_writer = BitWriter()
        coder_enc = Coder(b=self.precision)
        enc = Encoder(coder_enc, bit_writer)
        slots = coder_enc.tb

        dec_prec = max(50, int(math.ceil(self.precision * math.log10(2))) + 10)


        print("Encoding tokens. Progress:")
        for i, token_id in tqdm(enumerate(token_ids), total=len(token_ids)):
            context_ids = get_context_slice(i, self.model, token_ids)
            if len(context_ids) == 0:
                # Some models may not accept empty input_ids; fall back to using a pad token if available, else uniform
                if self.tokenizer.pad_token_id is not None:
                    input_ids = torch.tensor([[self.tokenizer.pad_token_id]], dtype=torch.long).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        logits = outputs.logits[0, -1, :]
                else:
                    vocab_size = self.tokenizer.vocab_size
                    probs_tensor = torch.ones(vocab_size, dtype=torch.float32) / vocab_size
                    logits = torch.log(probs_tensor)
            else:
                input_ids = torch.tensor([context_ids], dtype=torch.long).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits[0, -1, :]

            # Ensure logits does not require grad before converting to numpy
            logits = logits.detach()
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            vocab_size = len(probs)
            counts = probs_to_counts(probs=probs, total=slots, dec_prec=dec_prec)
            cum_desc = counts_to_cum_desc(counts)
            enc.encode_symbol(token_id, cum_desc)
        
        enc.finish()
        bit_writer.flush(padbit=0)
        encoded = bit_writer.getvalue()
        return encoded, text_length
    
class LLM_Decode:
    def __init__(self, tokenizer, model, precision=32):
        self.tokenizer = tokenizer
        self.precision = precision
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model = model.to(self.device)
        self.model.eval()

    def decode(self, encoded_bytes, text_length):
        # Prepare for decoder:
        bit_reader = BitReader(encoded_bytes)
        coder_dec = Coder(b=self.precision)
        dec = Decoder(coder_dec, bit_reader)

        decoded_token_ids = []
        slots = coder_dec.tb

        dec_prec = max(50, int(math.ceil(self.precision * math.log10(2))) + 10)
        
        print("Decoding tokens. Progress:")
        for i in tqdm(range(text_length)):
            context_ids = get_context_slice(i, self.model, decoded_token_ids)
            if len(context_ids) == 0:
                # Some models may not accept empty input_ids; fall back to using a pad token if available, else uniform
                if self.tokenizer.pad_token_id is not None:
                    input_ids = torch.tensor([[self.tokenizer.pad_token_id]], dtype=torch.long).to(self.device)
                    with torch.no_grad():
                        outputs = self.model(input_ids)
                        logits = outputs.logits[0, -1, :]
                else:
                    vocab_size = self.tokenizer.vocab_size
                    probs_tensor = torch.ones(vocab_size, dtype=torch.float32) / vocab_size
                    logits = torch.log(probs_tensor)
            else:
                input_ids = torch.tensor([context_ids], dtype=torch.long).to(self.device)
                with torch.no_grad():
                    outputs = self.model(input_ids)
                    logits = outputs.logits[0, -1, :]

            # Ensure logits does not require grad before converting to numpy
            logits = logits.detach()
            probs = torch.softmax(logits, dim=-1).detach().cpu().numpy()
            vocab_size = len(probs)
            counts = probs_to_counts(probs=probs, total=slots, dec_prec=dec_prec)
            cum_desc = counts_to_cum_desc(counts)
            token_id = dec.decode_symbol(cum_desc)
            decoded_token_ids.append(token_id)

        # convert token ids back to text
        decoded_text = self.tokenizer.decode(
            decoded_token_ids,
            skip_special_tokens=True,
            clean_up_tokenization_spaces=True,
        )
        return decoded_text
