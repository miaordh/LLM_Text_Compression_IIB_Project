# -----------------------
# MSB-first bit IO
# -----------------------
class BitWriter:
    def __init__(self):
        self._acc = 0
        self._nbits = 0
        self._out = bytearray()

    def write_bit(self, bit: int) -> None:
        bit &= 1
        # append as next MSB in accumulator
        self._acc = (self._acc << 1) | bit
        self._nbits += 1
        if self._nbits == 8:
            self._out.append(self._acc & 0xFF)
            self._acc = 0
            self._nbits = 0

    def flush(self, padbit: int = 0) -> None:
        if self._nbits > 0:
            pad_len = 8 - self._nbits
            if padbit == 0:
                padded = (self._acc << pad_len) & 0xFF
            else:
                padded = ((self._acc << pad_len) | ((1 << pad_len) - 1)) & 0xFF
            self._out.append(padded)
            self._acc = 0
            self._nbits = 0

    def getvalue(self) -> bytes:
        return bytes(self._out)


class BitReader:
    def __init__(self, data: bytes):
        self._data = data
        self._pos = 0
        self._acc = 0
        self._nbits = 0

    def read_bit(self) -> int:
        if self._nbits == 0:
            if self._pos >= len(self._data):
                # No more bits -> return 0 (safe padding)
                return 0
            self._acc = self._data[self._pos]
            self._pos += 1
            self._nbits = 8
        bit = (self._acc >> (self._nbits - 1)) & 1
        self._nbits -= 1
        return int(bit)
