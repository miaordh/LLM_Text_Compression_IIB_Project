import io
from typing import Optional
from bitReadWrite import BitReader, BitWriter

class Coder:
    def __init__(self, b=32):
        self.b = b
        self.lb = 1 << (self.b - 2) # Index of lower quarter
        self.hb = 1 << (self.b - 1) # Index of midpoint
        self.tb = (1 << self.b) - 1 # Index of top point
        self.mask = self.tb

        self.R = 0 # Current range of coding interval
        self.L = 0    # Low index of coding interval
        self.D = 0 # Target location in coding interval

        self.bits_waiting = 0 # Number of opposite-valued bits queued.
        self.output: Optional[BitWriter] = None # Output stream
        self.input: Optional[BitReader] = None # Input stream

    def _output_bits(self) -> None:
        """Outputs encoder's processed bits."""
        while (self.R <= self.lb):
            if (self.L + self.R) <= self.hb:
                self._output_all(0)
            elif (self.L >= self.hb):
                self._output_all(1)
                self.L -= self.hb
            else:
                self.bits_waiting += 1
                self.L -= self.lb
            self.L <<= 1
            self.R <<= 1

    def _output_all(self, bit: int) -> None:
        """Writes a bit, followed by bits_waiting bits of opposite value."""
        self.output.write_bit(bit)
        while self.bits_waiting > 0:
            self.output.write_bit(1 - bit)
            self.bits_waiting -= 1

    def _narrow_region(self, l: int, h: int) -> None:
        """Sets a region."""
        self.L += l
        self.R = h - l

    def _discard_bits(self) -> None:
        """Discards decoder's processed bits."""
        while (self.R <= self.lb):
            if (self.L >= self.hb):
                self.L -= self.hb
                self.D -= self.hb
            elif (self.L + self.R) <= self.hb:
                pass
            else:
                self.L -= self.lb
                self.D -= self.lb
            self.L <<= 1
            self.R <<= 1
            self.D <<= 1
            self.D &= self.mask
            self.D += self.input.read_bit()

    def loadRegion(self, l: int, h: int) -> None:
        """Loads a region"""
        self._narrow_region(l, h)
        self._discard_bits()

    def getTarget(self) -> int:
        """Returns a target pointer."""
        return self.D - self.L
    
    def getRange(self) -> int:
        """Returns the coding range."""
        return self.R
    
    def storeRegion(self, l: int, h: int) -> None:
        """Encodes a region."""
        self._narrow_region(l, h)
        self._output_bits()

    def start_encode(self, output: BitWriter) -> None:
        """Initializes the encoder."""
        self.output = output
        self.L = 0
        self.R = self.tb
        self.bits_waiting = 0

    def start_decode(self, input: BitReader) -> None:
        """Initializes the decoder."""
        self.input = input
        self.D = 0
        for k in range(self.b):
            self.D <<= 1
            self.D += self.input.read_bit()
        self.L = 0
        self.R = self.tb

    def finish_encode(self) -> None:
        """Finalizes the encoding process."""
        while True:
            if self.L + (self.R >> 1) >= self.hb:
                self._output_all(1)
                if self.L < self.hb:
                    self.R -= self.hb - self.L
                    self.L = 0
                else:
                    self.L -= self.hb
            else:
                self._output_all(0)
                if self.L + self.R > self.hb:
                    self.R = self.hb - self.L
            if self.R == self.hb:
                break
            self.L <<= 1
            self.R <<= 1
    def finish_decode(self) -> None:
        """Finalizes the decoding process."""
        pass

def bits_to_bitstring(bts: bytes, bitlen: int = None):
    s = ''.join(f'{byte:08b}' for byte in bts)
    return s if bitlen is None else s[:bitlen]

# Test 1: BitWriter / BitReader basic roundtrip
bw = BitWriter()
pattern = [1,0,1,1,0,0,1,0]  # one byte pattern
for bit in pattern:
    bw.write_bit(bit)
bw.flush(padbit=0)
data = bw.getvalue()
br = BitReader(data)
read_back = [br.read_bit() for _ in range(8)]
print("TEST1 BitWriter/BitReader")
print("  written bits:", pattern)
print("  data bytes  :", data, "hex:", data.hex())
print("  read bits   :", read_back)
print()

# Test 2: store_region that triggers output_all(0) branch
bw2 = BitWriter()
coder2 = Coder(b=8)
coder2.start_encode(bw2)
# choose l=0,h=64 so R = 64 <= lb (64), L=0 -> L+R = 64 <= hb (128) => output_all(0)
coder2.storeRegion(0, 64)
bw2.flush(padbit=0)
print("TEST2 store_region -> output_all(0)")
print("  output bytes:", bw2.getvalue(), "hex:", bw2.getvalue().hex())
print("  bits        :", bits_to_bitstring(bw2.getvalue(), bitlen=8))
print()

# Test 3: store_region that triggers output_all(1) branch by setting L >= hb beforehand
bw3 = BitWriter()
coder3 = Coder(b=8)
coder3.start_encode(bw3)
# artificially set L to hb to force output_all(1)
coder3.L = coder3.hb
coder3.storeRegion(0, 64)  # R=64, L >= hb => output_all(1)
bw3.flush(padbit=0)
print("TEST3 store_region -> output_all(1) (L forced to hb)")
print("  L initial   :", coder3.hb)
print("  output bytes:", bw3.getvalue(), "hex:", bw3.getvalue().hex())
print("  bits        :", bits_to_bitstring(bw3.getvalue(), bitlen=8))
print()

# Test 4: bits_waiting (underflow) branch
bw4 = BitWriter()
coder4 = Coder(b=8)
coder4.start_encode(bw4)
# set L in the middle zone such that L >= lb and L < hb and L+R > hb
coder4.L = coder4.lb + 6  # e.g., 64 + 6 = 70
coder4.storeRegion(0, 64)  # R=64, will trigger bits_waiting += 1
# No immediate bits emitted; bits_waiting should be > 0; flush may pad nothing written
bw4.flush(padbit=0)
print("TEST4 bits_waiting (underflow E3)")
print("  L initial   :", coder4.lb + 6)
print("  bits_waiting:", coder4.bits_waiting)
print("  output bytes:", bw4.getvalue(), "hex:", bw4.getvalue().hex())
print()

# Test 5: start_decode priming D with known bitstream
# Prepare a byte where the first b=8 bits are 0b10100101 (0xA5)
data5 = bytes([0xA5])
br5 = BitReader(data5)
coder5 = Coder(b=8)
coder5.start_decode(br5)
print("TEST5 start_decode priming D")
print("  input byte  :", data5.hex())
print("  D primed    :", coder5.D)
print("  D binary    :", format(coder5.D, '08b'))
print()

# Test 6: load_region + get_target + get_range (decoder)
coder6 = Coder(b=8)
coder6.start_decode(BitReader(bytes([0xA5])))  # primed D
print("TEST6 before load_region: D, L, R:", coder6.D, coder6.L, coder6.R)
coder6.loadRegion(0, 64)
print("TEST6 after load_region(0,64):")
print("  D           :", coder6.D)
print("  get_target() :", coder6.getTarget())
print("  get_range()  :", coder6.getRange())
print()

print("All tests completed.")