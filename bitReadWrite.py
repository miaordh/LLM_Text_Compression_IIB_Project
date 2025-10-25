from typing import Optional
import io

class BitWriter:
    """Simple bit-level writer (MSB-first inside byte)."""
    def __init__(self, buf: Optional[io.BytesIO] = None):
        self.buf = buf or io.BytesIO()
        self._acc = 0
        self._nbits = 0

    def write_bit(self, bit: int) -> None:
        bit &= 1
        self._acc = (self._acc << 1) | bit
        self._nbits += 1
        if self._nbits == 8:
            self.buf.write(bytes([self._acc & 0xFF]))
            self._acc = 0
            self._nbits = 0

    def flush(self, padbit: int = 0) -> None:
        """Pad the final partial byte with padbit (0 or 1)."""
        if self._nbits > 0:
            self._acc = (self._acc << (8 - self._nbits)) | (0 if padbit == 0 else ((1 << (8 - self._nbits)) - 1))
            self.buf.write(bytes([self._acc & 0xFF]))
            self._acc = 0
            self._nbits = 0

    def getvalue(self) -> bytes:
        return self.buf.getvalue()


class BitReader:
    """Simple bit-level reader (MSB-first inside byte)."""
    def __init__(self, data: bytes):
        self.buf = io.BytesIO(data)
        self._acc = 0
        self._nbits = 0

    def read_bit(self) -> int:
        if self._nbits == 0:
            b = self.buf.read(1)
            if not b:
                # EOF -> return 0 (padding)
                return 0
            self._acc = b[0]
            self._nbits = 8
        bit = (self._acc >> (self._nbits - 1)) & 1
        self._nbits -= 1
        return bit