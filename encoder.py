from typing import List
from arithmetic_coding import Coder
from bitReadWrite import BitWriter

class Encoder:
    def __init__(self, coder: Coder, writer: BitWriter):
        self.coder = coder
        self.writer = writer
        self.coder.start_encode(writer)

    def encode_symbol(self, symbol_index: int, cum_desc: List[int]) -> None:
        """
        Encode a symbol given cum_desc (descending cumulative counts).
        cum_desc[0] == total, cum_desc[-1] == 0.
        symbol_index is index into alphabet (0..n-1) where cum_desc maps accordingly.
        """
        total = cum_desc[0]
        # descending cum --> ascending lower/upper
        l = cum_desc[symbol_index + 1]
        h = cum_desc[symbol_index]
        lower = total - h
        upper = total - l

        rng = self.coder.R
        new_low = self.coder.L + (rng * lower) // total
        new_high = self.coder.L + (rng * upper) // total - 1

        # set and renormalise via coder
        self.coder.set_interval_and_renorm_encode(new_low, new_high)

    def finish(self) -> None:
        self.coder.finish_encode()
        # caller should call writer.flush(padbit=0) after finish if desired