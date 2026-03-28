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

