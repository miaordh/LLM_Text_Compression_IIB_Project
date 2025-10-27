from typing import List
from arithmetic_coding import Coder
from bitReadWrite import BitReader

class Decoder:
    def __init__(self, coder: Coder, reader: BitReader):
        self.coder = coder
        self.reader = reader
        self.coder.start_decode(reader)

    def decode_symbol(self, cum_desc: List[int]) -> int:
        """
        Decode one symbol and update coder state.
        Returns symbol_index.
        """
        total = cum_desc[0]
        L = self.coder.L
        R = self.coder.R
        D = self.coder.D

        # Find symbol by computing absolute intervals same as encoder.
        s_found = None
        for s in range(len(cum_desc) - 1):
            l = cum_desc[s + 1]; h = cum_desc[s]
            lower = total - h
            upper = total - l
            new_low_s = L + (R * lower) // total
            new_high_s = L + (R * upper) // total - 1
            if new_low_s <= D <= new_high_s:
                s_found = s
                break
        if s_found is None:
            # fallback (shouldn't happen if arithmetic & bit I/O are consistent)
            s_found = len(cum_desc) - 2

        # Now update coder using the same scaled absolute interval
        l = cum_desc[s_found + 1]; h = cum_desc[s_found]
        lower = total - h
        upper = total - l
        new_low = self.coder.L + (self.coder.R * lower) // total
        new_high = self.coder.L + (self.coder.R * upper) // total - 1
        self.coder.set_interval_and_renorm_decode(new_low, new_high)
        return s_found