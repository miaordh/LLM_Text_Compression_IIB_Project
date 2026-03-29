# batch load files in a folder
import gzip
import time
import os
folder_path = "cantrbry"
filenames = []
compression_times = []
decompression_times = []
compression_ratios = []
encode_override = {"cp.html": "windows-1252", "kennedy.xls": None, "sum": None, "ptt5": None}
for filename in os.listdir(folder_path):
    try:
        file_path = os.path.join(folder_path, filename)
        if filename in encode_override:
            if encode_override[filename] is None:
                with open(file_path, "rb") as f:
                    text_bytes = f.read()
            else:
                with open(file_path, "r", encoding=encode_override[filename]) as f:
                    text = f.read()
                text_bytes = text.encode(encode_override[filename])
        else:
            with open(file_path, "r", encoding="utf-8") as f:
                text = f.read()
            text_bytes = text.encode("utf-8")
        start = time.time()
        compressed_bytes = gzip.compress(text_bytes)
        compress_seconds = time.time() - start
        print(f"File: {filename}")
        print(f"Original size: {len(text_bytes)} bytes")
        print(f"Compressed size: {len(compressed_bytes)} bytes")
        print(f"Compression time: {compress_seconds:.10f} seconds")
        print(f"Compression ratio: {len(compressed_bytes) * 8 / len(text_bytes):.5f}")
        start = time.time()
        decompressed_bytes = gzip.decompress(compressed_bytes)
        decompress_seconds = time.time() - start
        print(f"Decompressed size: {len(decompressed_bytes)} bytes")
        print(f"Decompression time: {decompress_seconds:.10f} seconds")
        filenames.append(filename)
        compression_times.append(compress_seconds)
        decompression_times.append(decompress_seconds)
        compression_ratios.append(len(compressed_bytes) * 8 / len(text_bytes))
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