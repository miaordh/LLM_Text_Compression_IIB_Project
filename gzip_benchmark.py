# read a file
with open("my_corpus/sonnet.txt", "r") as f:
    text = f.read()

# compress it using gzip
import gzip
import time
start = time.time()
compressed_bytes = gzip.compress(text.encode("utf-8"))
compress_seconds = time.time() - start
print(f"Original size: {len(text.encode('utf-8'))} bytes")
print(f"Compressed size: {len(compressed_bytes)} bytes")
print(f"Compression time: {compress_seconds:.10f} seconds")
print(f"Compression ratio: {len(compressed_bytes) * 8 / len(text.encode('utf-8')):.5f}")

# decompress it using gzip
start = time.time()
decompressed_bytes = gzip.decompress(compressed_bytes)
decompress_seconds = time.time() - start
print(f"Decompressed size: {len(decompressed_bytes)} bytes")
print(f"Decompression time: {decompress_seconds:.10f} seconds")