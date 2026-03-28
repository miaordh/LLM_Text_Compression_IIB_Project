from typing import List, Optional
from arithmetic_coding import Coder
from bitReadWrite import BitWriter

class Encoder(Coder):
    def __init__(self, writer: BitWriter, b: int = 16):
        super().__init__(b)
        self.writer = writer
        self.bits_waiting = 0
        self.output: Optional[BitWriter] = writer
        
        self.L = 0
        # initial range must be 2^b (i.e. tb + 1)
        self.R = self.tb + 1
        self.bits_waiting = 0

    def _output_bits(self) -> None:
        assert self.output is not None, "_output_bits: no BitWriter"
        # loop while range too small
        while self.R <= self.lb:
            if (self.L + self.R) <= self.hb:
                # E1: lower half
                self._output_all(0)
            elif self.L >= self.hb:
                # E2: upper half
                self._output_all(1)
                self.L -= self.hb
            else:
                # E3: middle half (postpone)
                self.bits_waiting += 1
                self.L -= self.lb
            # shift left 1 bit, keep within b bits
            self.L = (self.L << 1) & self.mask
            self.R = (self.R << 1) & self.mask
            if self.R == 0:
                raise RuntimeError("_output_bits: R became zero")

    def _output_all(self, bit: int) -> None:
        assert self.output is not None
        self.output.write_bit(bit)
        while self.bits_waiting > 0:
            self.output.write_bit(1 - bit)
            self.bits_waiting -= 1

    def set_interval_and_renorm_encode(self, new_low: int, new_high: int) -> None:
        """Set absolute interval [new_low, new_high] and renormalise for encoder."""
        width = int(new_high) - int(new_low) + 1
        if width <= 0:
            raise RuntimeError(
                f"set_interval_and_renorm_encode: invalid interval [{new_low}, {new_high}] (width={width})"
            )
        if width > (self.tb + 1):
            raise RuntimeError(
                f"set_interval_and_renorm_encode: interval width {width} exceeds coder range {self.tb + 1}"
            )

        self.L = new_low & self.mask
        self.R = width
        self._output_bits()

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

        rng = self.R
        new_low = self.L + (rng * lower) // total
        new_high = self.L + (rng * upper) // total - 1

        # set and renormalise
        self.set_interval_and_renorm_encode(new_low, new_high)

    def finish(self) -> None:
        assert self.output is not None
        # Emulate textbook finalisation: drive interval to restore full range
        MAX_ITERS = 5_000_000
        it = 0
        while True:
            it += 1
            if it > MAX_ITERS:
                raise RuntimeError("finish: reached max iterations")
            if self.L + (self.R >> 1) >= self.hb:
                self._output_all(1)
                if self.L < self.hb:
                    # adjust R so range equals hb - L (in original loop)
                    self.R = (self.R - (self.hb - self.L)) & self.mask
                    self.L = 0
                else:
                    self.L = (self.L - self.hb) & self.mask
            else:
                self._output_all(0)
                if self.L + self.R > self.hb:
                    self.R = (self.hb - self.L) & self.mask
            if self.R == 0:
                raise RuntimeError("finish: R==0")
            if self.R == self.hb:
                break
            self.L = (self.L << 1) & self.mask
            self.R = (self.R << 1) & self.mask
        # caller should call writer.flush(padbit=0) after finish if desired