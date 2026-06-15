import pyppmd
import time
with open('cantrbry/asyoulik.txt', 'r') as f:
    text = f.read() # original size in bytes using windows-1252 encoding
original_bytes = len(text.encode('utf-8'))
print(f'Original size: {original_bytes} bytes')
# compress using ppmd
start = time.time()
compressed = pyppmd.compress(text)
end = time.time()
print(f'Compressed size: {len(compressed)} bytes')
print(f'Compression time: {end - start:.3f} seconds')
# decompress using ppmd
start = time.time()
decompressed = pyppmd.decompress(compressed)
end = time.time()
print(f'Decompression time: {end - start:.3f} seconds')

# compression ratio in encoded bits/original byte
compression_ratio = (len(compressed) * 8) / original_bytes
print(f'Compression ratio: {compression_ratio:.3f} bits/byte')