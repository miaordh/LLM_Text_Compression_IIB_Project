import argparse

from transformers import AutoModelForCausalLM, AutoTokenizer

from llm_codec_deterministic import DeterministicCodecConfig, DeterministicLLMCodec


def main():
    parser = argparse.ArgumentParser(description="Cross-device deterministic codec harness")
    parser.add_argument("--model", required=True, help="HuggingFace model id or local path")
    parser.add_argument("--text", required=True, help="Text to encode/decode")
    parser.add_argument("--encode-device", default="cpu", help="Device used for encoding")
    parser.add_argument("--decode-device", default="mps", help="Device used for decoding")
    parser.add_argument(
        "--determinism-mode",
        default="strict_cpu",
        choices=["strict_cpu", "gpu_best_effort"],
        help="strict_cpu = strongest determinism; gpu_best_effort = faster, weaker cross-platform guarantee",
    )
    parser.add_argument("--precision", type=int, default=32)
    parser.add_argument("--slots", type=int, default=(1 << 24))
    args = parser.parse_args()

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    model = AutoModelForCausalLM.from_pretrained(args.model)

    cfg = DeterministicCodecConfig(
        determinism_mode=args.determinism_mode,
        precision=args.precision,
        slots=args.slots,
        use_legacy_counts=True,
        use_kv_cache=False,
        patch_linear=True,
        patch_rmsnorm=True,
        patch_attention=True,
    )

    encoder = DeterministicLLMCodec(tokenizer, model, device=args.encode_device, config=cfg)
    encoded = encoder.encode(args.text)

    decoder = DeterministicLLMCodec(tokenizer, model, device=args.decode_device, config=cfg)
    decoded = decoder.decode(encoded, max_decode_tokens=4096)

    ok = decoded == args.text
    print(f"roundtrip_match={ok}")
    if not ok:
        print("decoded_text:")
        print(decoded)


if __name__ == "__main__":
    main()
