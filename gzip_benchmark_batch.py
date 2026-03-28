# batch load files in a folder
import gzip
import time
import os
folder_path = "my_corpus"
filenames = []
compression_times = []
decompression_times = []
compression_ratios = []
for filename in os.listdir(folder_path):
    try:
        with open(os.path.join(folder_path, filename), "r") as f:
            text = f.read()
        start = time.time()
        compressed_bytes = gzip.compress(text.encode("utf-8"))
        compress_seconds = time.time() - start
        print(f"File: {filename}")
        print(f"Original size: {len(text.encode('utf-8'))} bytes")
        print(f"Compressed size: {len(compressed_bytes)} bytes")
        print(f"Compression time: {compress_seconds:.10f} seconds")
        print(f"Compression ratio: {len(compressed_bytes) * 8 / len(text.encode('utf-8')):.5f}")
        start = time.time()
        decompressed_bytes = gzip.decompress(compressed_bytes)
        decompress_seconds = time.time() - start
        print(f"Decompressed size: {len(decompressed_bytes)} bytes")
        print(f"Decompression time: {decompress_seconds:.10f} seconds")
        filenames.append(filename)
        compression_times.append(compress_seconds)
        decompression_times.append(decompress_seconds)
        compression_ratios.append(len(compressed_bytes) * 8 / len(text.encode('utf-8')))
    except Exception as e:
        print(f"Error processing file {filename}: {e}")

# make it a pandas dataframe
import pandas as pd
df = pd.DataFrame({
    "filename": filenames,
    "compression_time": compression_times,
    "decompression_time": decompression_times,
    "compression_ratio": compression_ratios
})
print(df)