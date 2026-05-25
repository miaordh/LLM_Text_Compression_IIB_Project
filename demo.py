from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_codec import CodecConfig, LLMCodec

text = "This is a roundtrip demonstration of the DeterministicLLMCodec. The codec should be able to compress and decompress this text without any loss of information."

model_id = "deepseek-ai/deepseek-coder-1.3b-base"

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, dtype="auto")

config = CodecConfig(
    model_id=model_id,
    inference_backend="huggingface",
    determinism_mode=None,
    quant=False,
    precision=32,
    slots=1 << 24,
    context_window=100,
    margin=16,
    strategy="rolling",
)

codec = LLMCodec(
    tokenizer=tokenizer,
    model=model,
    device="cpu",  # or "cpu", "mps", "auto"
    config=config,
)

compressed_bytes = codec.encode(text, show_progress=False)
decompressed_text = codec.decode(compressed_bytes, show_progress=False)

print("Original text:", text)
print("Decompressed text:", decompressed_text)
print("Compression ratio:", len(compressed_bytes) * 8 / len(text.encode("utf-8")))
print("Roundtrip successful:", decompressed_text == text)