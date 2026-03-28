from typing import List, Optional
from arithmetic_coding import Coder
from bitReadWrite import BitReader

class Decoder(Coder):
    def __init__(self, reader: BitReader, b: int = 16):
        super().__init__(b)
        self.reader = reader
        self.input: Optional[BitReader] = reader
        
        self.D = 0
        # prime D with first b bits (MSB-first)
        for _ in range(self.b):
            self.D = ((self.D << 1) & self.mask) + self.input.read_bit()
        self.L = 0
        self.R = self.tb + 1

    def _discard_bits(self) -> None:
        assert self.input is not None, "_discard_bits: no BitReader"
        while self.R <= self.lb:
            if self.L >= self.hb:
                self.L -= self.hb
                self.D -= self.hb
            elif (self.L + self.R) <= self.hb:
                # lower half -> nothing
                pass
            else:
                self.L -= self.lb
                self.D -= self.lb
            self.L = (self.L << 1) & self.mask
            self.R = (self.R << 1) & self.mask
            # bring in next bit
            self.D = ((self.D << 1) & self.mask) + self.input.read_bit()
            if self.R == 0:
                raise RuntimeError("_discard_bits: R became zero")

    def set_interval_and_renorm_decode(self, new_low: int, new_high: int) -> None:
        """Set absolute interval [new_low, new_high] and renormalise for decoder."""
        width = int(new_high) - int(new_low) + 1
        if width <= 0:
            raise RuntimeError(
                f"set_interval_and_renorm_decode: invalid interval [{new_low}, {new_high}] (width={width})"
            )
        if width > (self.tb + 1):
            raise RuntimeError(
                f"set_interval_and_renorm_decode: interval width {width} exceeds coder range {self.tb + 1}"
            )

        self.L = new_low & self.mask
        self.R = width
        self._discard_bits()

    def decode_symbol(self, cum_desc: List[int]) -> int:
        """
        Decode one symbol and update coder state.
        Returns symbol_index.
        """
        total = cum_desc[0]
        L = self.L
        R = self.R
        D = self.D

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
        new_low = self.L + (self.R * lower) // total
        new_high = self.L + (self.R * upper) // total - 1
        self.set_interval_and_renorm_decode(new_low, new_high)
        return s_found