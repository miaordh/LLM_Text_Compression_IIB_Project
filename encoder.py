class Encoder:
    def __init__(self, bit: int = 32, current_low: int = 0, current_high: int = None,
                 eof_symbol: str = "<EOF>", eof_index: int = 0):
        self.precision = int(bit)
        self.TOP_VALUE = (1 << self.precision) - 1
        self.current_low = int(current_low)
        self.current_high = self.TOP_VALUE if current_high is None else int(current_high)

        self.eof_symbol = eof_symbol
        self.eof_reached = False

        # Quarter/half/three-quarter thresholds (standard)
        self.FIRST_QTR = self.TOP_VALUE // 4 + 1         # integer division of 2^P / 4
        self.HALF = self.FIRST_QTR * 2
        self.THREE_QTR = self.FIRST_QTR * 3

        # for E3 underflow handling
        self.bits_to_follow = 0

        # --- bit/byte output buffers ---
        self._bit_buffer = 0            # accumulates bits (MSB-first)
        self._bits_in_buffer = 0        # 0..7
        self.output_bytes = bytearray()
        self.total_bits_emitted = 0     # exact number of bits emitted

    # -------------------------
    # low-level bit output
    # -------------------------
    def _output_bit(self, bit: int) -> None:
        """Append a single bit (0/1) to byte buffer, MSB-first packing in each byte."""
        assert bit in (0, 1)
        self._bit_buffer = (self._bit_buffer << 1) | bit
        self._bits_in_buffer += 1
        self.total_bits_emitted += 1
        if self._bits_in_buffer == 8:
            self.output_bytes.append(self._bit_buffer & 0xFF)
            self._bit_buffer = 0
            self._bits_in_buffer = 0

    def _bit_plus_follow(self, bit: int) -> None:
        """
        Emit 'bit' and then emit bits_to_follow bits which are the complement of 'bit'
        (standard arithmetic coding trick for underflow).
        """
        # primary bit
        self._output_bit(bit)
        # emit complements for each previously postponed bit
        while self.bits_to_follow > 0:
            self._output_bit(1 - bit)
            self.bits_to_follow -= 1

    def _flush_output(self) -> None:
        """
        Pad remaining bits (if any) with zeros to complete the final byte,
        and append it to output_bytes. This must be called at finalization.
        """
        if self._bits_in_buffer > 0:
            # pad the remainder with zeros on the least-significant side
            padded = self._bit_buffer << (8 - self._bits_in_buffer)
            self.output_bytes.append(padded & 0xFF)
            self._bit_buffer = 0
            self._bits_in_buffer = 0

    # -------------------------
    # encoding core (renormalisation fixes)
    # -------------------------
    def encode(self, symbol: str, symbol_index = int, probabilities = list[float]):
        """
        NOTE: This method preserves your interval calculation step (using probabilities).
        Here we only correct renormalisation/emission and output handling.
        (You asked to focus on bit output / where to add things.)
        """
        if self.eof_reached:
            raise ValueError("Cannot encode after EOF has been reached.")

        # ---- compute new_low/new_high (you already had this) ----
        current_range = self.current_high - self.current_low + 1
        cumulative_probability = sum(probabilities[:symbol_index])
        symbol_probability = probabilities[symbol_index]
        # WARNING: using float probabilities here is brittle — leave for now as you requested.
        new_low = self.current_low + int(current_range * cumulative_probability)
        new_high = self.current_low + int(current_range * (cumulative_probability + symbol_probability)) - 1

        # ---- RENORMALISATION & EMBIT EMISSION (corrected) ----
        while True:
            # E1: MSB = 0 for both low and high
            if new_high < self.HALF:
                self._bit_plus_follow(0)
                # shift out MSB
                new_low = new_low * 2
                new_high = new_high * 2 + 1

            # E2: MSB = 1 for both low and high
            elif new_low >= self.HALF:
                self._bit_plus_follow(1)
                # subtract half and shift
                new_low = (new_low - self.HALF) * 2
                new_high = (new_high - self.HALF) * 2 + 1

            # E3: underflow zone: postpone bit
            elif new_low >= self.FIRST_QTR and new_high < self.THREE_QTR:
                self.bits_to_follow += 1
                new_low = (new_low - self.FIRST_QTR) * 2
                new_high = (new_high - self.FIRST_QTR) * 2 + 1

            else:
                break

            # Defensive: keep values inside precision (masking)
            new_low &= self.TOP_VALUE
            new_high &= self.TOP_VALUE

        # update current interval
        self.current_low = new_low
        self.current_high = new_high

        # if this was EOF symbol, finalise
        if symbol == self.eof_symbol:
            self.eof_reached = True
            # finalization: output one decisive bit then flush follow
            self.bits_to_follow += 1
            if self.current_low < self.FIRST_QTR:
                self._bit_plus_follow(0)
            else:
                self._bit_plus_follow(1)
            # flush partial byte
            self._flush_output()

    # -------------------------
    # accessors for output
    # -------------------------
    def get_encoded_bytes(self) -> bytes:
        """Return the emitted bytes so far (make sure finalize was called if you used EOF)."""
        return bytes(self.output_bytes)

    def get_encoded_bit_length(self) -> int:
        """Return number of meaningful bits emitted (not necessarily multiple of 8)."""
        return self.total_bits_emitted
