from typing import Optional
from bitReadWrite import BitReader, BitWriter
class Coder:
    def __init__(self, b: int = 16):
        self.b = b
        self.lb = 1 << (self.b - 2)         # lower quarter index
        self.hb = 1 << (self.b - 1)         # half index (midpoint)
        self.tb = (1 << self.b) - 1         # top value (2^b - 1)
        self.mask = (1 << self.b) - 1

        # working state (integers)
        self.R = 0
        self.L = 0
        self.D = 0

        self.bits_waiting = 0
        self.output: Optional[BitWriter] = None
        self.input: Optional[BitReader] = None

    # --- encoder renormalisation: emit bits to writer ---
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

    # --- decoder renormalisation: consume bits from reader ---
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

    # --- public methods to start/finish encoding and decoding ---
    def start_encode(self, output: BitWriter) -> None:
        self.output = output
        self.L = 0
        # initial range must be 2^b (i.e. tb + 1)
        self.R = self.tb + 1
        self.bits_waiting = 0

    def start_decode(self, input: BitReader) -> None:
        self.input = input
        self.D = 0
        # prime D with first b bits (MSB-first)
        for _ in range(self.b):
            self.D = ((self.D << 1) & self.mask) + self.input.read_bit()
        self.L = 0
        self.R = self.tb + 1

    def finish_encode(self) -> None:
        assert self.output is not None
        # Emulate textbook finalisation: drive interval to restore full range
        MAX_ITERS = 5_000_000
        it = 0
        while True:
            it += 1
            if it > MAX_ITERS:
                raise RuntimeError("finish_encode: reached max iterations")
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
                raise RuntimeError("finish_encode: R==0")
            if self.R == self.hb:
                break
            self.L = (self.L << 1) & self.mask
            self.R = (self.R << 1) & self.mask

    def set_interval_and_renorm_encode(self, new_low: int, new_high: int) -> None:
        """Set absolute interval [new_low, new_high] and renormalise for encoder."""
        self.L = new_low & self.mask
        self.R = ((new_high - new_low + 1) & self.mask)
        self._output_bits()

    def set_interval_and_renorm_decode(self, new_low: int, new_high: int) -> None:
        """Set absolute interval [new_low, new_high] and renormalise for decoder."""
        self.L = new_low & self.mask
        self.R = ((new_high - new_low + 1) & self.mask)
        self._discard_bits()
