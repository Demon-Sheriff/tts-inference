"""
Analyze Audio Token Generation

Compares the token generation and SNAC decoding between:
1. TensorRT-LLM pipeline
2. Reference vLLM pipeline

This helps debug any issues with audio quality.

Usage:
    modal run analyze_tokens.py
"""

import modal

app = modal.App("orpheus-token-analysis")

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt-llm/release:0.21.0",
        secret=modal.Secret.from_name("nvcr-credentials"),
    )
    .pip_install(
        "snac",
        "numpy",
        "transformers",
    )
)

engine_vol = modal.Volume.from_name("orpheus-trtllm-engine", create_if_missing=True)
cache_vol = modal.Volume.from_name("orpheus-cache", create_if_missing=True)

ENGINE_DIR = "/engine/trt_engine"
MODEL_DIR = "/cache/orpheus-3b"

# Special token IDs (from Orpheus)
SOS_TOKEN = 128257  # Start of speech
EOS_TOKEN = 128258  # End of speech
START_TOKEN = 128259  # Start of turn
END_TOKENS = [128009, 128260]  # End of text, end of turn
TOKEN_BASE = 128266  # First audio token

# Audio constants
FRAME_SIZE = 7
SAMPLE_RATE = 24000
POSITION_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]


def analyze_token_distribution(tokens: list[int], name: str):
    """Analyze the distribution of generated tokens."""
    import numpy as np

    print(f"\n{'='*60}")
    print(f"Token Analysis: {name}")
    print(f"{'='*60}")

    print(f"Total tokens: {len(tokens)}")

    if not tokens:
        print("No tokens to analyze")
        return

    # Check for special tokens
    special_tokens = {
        128257: "SOS",
        128258: "EOS",
        128259: "START",
        128009: "END_TEXT",
        128260: "END_TURN",
        128261: "DELIMITER",
    }

    special_count = {}
    for t in tokens:
        if t in special_tokens:
            name_t = special_tokens[t]
            special_count[name_t] = special_count.get(name_t, 0) + 1

    print(f"\nSpecial tokens found:")
    for name_t, count in special_count.items():
        print(f"  {name_t} ({special_tokens.get(t, t)}): {count}")

    # Analyze audio tokens (after subtracting TOKEN_BASE)
    audio_tokens = [t for t in tokens if t >= TOKEN_BASE]
    print(f"\nAudio tokens (>= {TOKEN_BASE}): {len(audio_tokens)}")

    if audio_tokens:
        # Convert to raw codes
        raw_codes = [t - TOKEN_BASE for t in audio_tokens]

        print(f"\nRaw code statistics:")
        print(f"  Min: {min(raw_codes)}")
        print(f"  Max: {max(raw_codes)}")
        print(f"  Mean: {np.mean(raw_codes):.1f}")
        print(f"  Std: {np.std(raw_codes):.1f}")

        # Check code validity per position
        num_frames = len(raw_codes) // FRAME_SIZE
        print(f"\nFrames: {num_frames}")

        if num_frames > 0:
            # Analyze codes by position in frame
            position_codes = {i: [] for i in range(FRAME_SIZE)}

            for frame_idx in range(num_frames):
                base = frame_idx * FRAME_SIZE
                for pos in range(FRAME_SIZE):
                    if base + pos < len(raw_codes):
                        position_codes[pos].append(raw_codes[base + pos])

            print(f"\nCodes by position (raw, before subtracting offset):")
            for pos in range(FRAME_SIZE):
                codes = position_codes[pos]
                if codes:
                    expected_offset = POSITION_OFFSETS[pos]
                    # After subtracting offset, should be 0-4095
                    adjusted = [c - expected_offset for c in codes]
                    valid = sum(1 for c in adjusted if 0 <= c <= 4095)
                    print(f"  Position {pos} (offset {expected_offset}): "
                          f"raw range [{min(codes)}-{max(codes)}], "
                          f"adjusted range [{min(adjusted)}-{max(adjusted)}], "
                          f"valid: {valid}/{len(codes)}")

            # Show first few frames in detail
            print(f"\nFirst 3 frames (raw codes):")
            for frame_idx in range(min(3, num_frames)):
                base = frame_idx * FRAME_SIZE
                frame_codes = raw_codes[base:base + FRAME_SIZE]
                print(f"  Frame {frame_idx}: {frame_codes}")

            print(f"\nFirst 3 frames (after subtracting position offsets):")
            for frame_idx in range(min(3, num_frames)):
                base = frame_idx * FRAME_SIZE
                frame_codes = raw_codes[base:base + FRAME_SIZE]
                adjusted = [frame_codes[i] - POSITION_OFFSETS[i] for i in range(len(frame_codes))]
                print(f"  Frame {frame_idx}: {adjusted}")


def redistribute_and_analyze(audio_tokens: list[int]):
    """Redistribute audio tokens into SNAC layers and analyze."""
    import numpy as np

    print(f"\n{'='*60}")
    print("SNAC Layer Redistribution Analysis")
    print(f"{'='*60}")

    # Convert to raw codes
    codes = [t - TOKEN_BASE for t in audio_tokens]
    num_frames = len(codes) // FRAME_SIZE
    codes = codes[:num_frames * FRAME_SIZE]

    print(f"Total codes: {len(codes)}")
    print(f"Frames: {num_frames}")

    # Redistribute into layers
    layer0 = []  # 1 per frame (position 0)
    layer1 = []  # 2 per frame (positions 1, 4)
    layer2 = []  # 4 per frame (positions 2, 3, 5, 6)

    invalid_codes = []

    for frame_idx in range(num_frames):
        base = frame_idx * FRAME_SIZE

        def get_code(pos):
            raw = codes[base + pos]
            adjusted = raw - POSITION_OFFSETS[pos]
            clamped = max(0, min(4095, adjusted))
            if adjusted < 0 or adjusted > 4095:
                invalid_codes.append({
                    'frame': frame_idx,
                    'position': pos,
                    'raw': raw,
                    'adjusted': adjusted,
                    'clamped': clamped
                })
            return clamped

        # Layer 0: position 0
        layer0.append(get_code(0))

        # Layer 1: positions 1, 4
        layer1.append(get_code(1))
        layer1.append(get_code(4))

        # Layer 2: positions 2, 3, 5, 6
        layer2.append(get_code(2))
        layer2.append(get_code(3))
        layer2.append(get_code(5))
        layer2.append(get_code(6))

    print(f"\nLayer sizes:")
    print(f"  Layer 0: {len(layer0)} codes (1 per frame)")
    print(f"  Layer 1: {len(layer1)} codes (2 per frame)")
    print(f"  Layer 2: {len(layer2)} codes (4 per frame)")

    print(f"\nLayer statistics:")
    for name, layer in [("Layer 0", layer0), ("Layer 1", layer1), ("Layer 2", layer2)]:
        if layer:
            print(f"  {name}: min={min(layer)}, max={max(layer)}, "
                  f"mean={np.mean(layer):.1f}, std={np.std(layer):.1f}")

    print(f"\nInvalid codes (outside 0-4095 after offset subtraction): {len(invalid_codes)}")
    if invalid_codes and len(invalid_codes) <= 20:
        for ic in invalid_codes[:20]:
            print(f"  Frame {ic['frame']}, Pos {ic['position']}: "
                  f"raw={ic['raw']}, adjusted={ic['adjusted']}, clamped={ic['clamped']}")
    elif invalid_codes:
        print(f"  (showing first 20 of {len(invalid_codes)})")
        for ic in invalid_codes[:20]:
            print(f"  Frame {ic['frame']}, Pos {ic['position']}: "
                  f"raw={ic['raw']}, adjusted={ic['adjusted']}, clamped={ic['clamped']}")

    return layer0, layer1, layer2


@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=600,
)
def analyze_generation(text: str = "Hello, this is a test.", voice: str = "tara"):
    """Generate audio and analyze the token distribution."""
    import os
    import time
    import torch
    import numpy as np
    from tensorrt_llm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from snac import SNAC

    print("="*60)
    print(f"Analyzing: '{text}' with voice '{voice}'")
    print("="*60)

    # Load models
    print("\nLoading tokenizer...")
    tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)

    print("Loading TRT-LLM engine...")
    llm = LLM(model=ENGINE_DIR)

    print("Loading SNAC...")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

    # Format prompt
    prompt_text = f"{voice}: {text}"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_ids = [START_TOKEN] + prompt_tokens + END_TOKENS

    print(f"\nPrompt: {prompt_text}")
    print(f"Prompt token IDs: {prompt_ids}")
    print(f"Prompt tokens decoded: {tokenizer.decode(prompt_tokens)}")

    # Generate
    print("\nGenerating...")
    sampling_params = SamplingParams(
        max_tokens=2000,
        stop_token_ids=[EOS_TOKEN],
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    t0 = time.perf_counter()
    outputs = llm.generate([prompt_ids], sampling_params=sampling_params)
    gen_time = time.perf_counter() - t0

    output_ids = outputs[0].outputs[0].token_ids
    print(f"Generation time: {gen_time:.2f}s")
    print(f"Generated {len(output_ids)} tokens")

    # Analyze all generated tokens
    analyze_token_distribution(output_ids, "All Generated Tokens")

    # Extract audio tokens (between SOS and EOS)
    audio_tokens = []
    found_sos = False

    for tid in output_ids:
        if not found_sos:
            if tid == SOS_TOKEN:
                found_sos = True
            continue
        if tid == EOS_TOKEN:
            break
        audio_tokens.append(tid)

    print(f"\nAudio tokens (between SOS and EOS): {len(audio_tokens)}")

    if not audio_tokens:
        print("ERROR: No audio tokens found!")
        return {"status": "error", "message": "No audio tokens"}

    # Analyze audio tokens
    analyze_token_distribution(audio_tokens, "Audio Tokens Only")

    # Redistribute and analyze
    layer0, layer1, layer2 = redistribute_and_analyze(audio_tokens)

    # Decode with SNAC
    print(f"\n{'='*60}")
    print("SNAC Decoding")
    print(f"{'='*60}")

    codes_l0 = torch.tensor(layer0, dtype=torch.int32, device="cuda").unsqueeze(0)
    codes_l1 = torch.tensor(layer1, dtype=torch.int32, device="cuda").unsqueeze(0)
    codes_l2 = torch.tensor(layer2, dtype=torch.int32, device="cuda").unsqueeze(0)

    print(f"Tensor shapes: L0={codes_l0.shape}, L1={codes_l1.shape}, L2={codes_l2.shape}")

    with torch.inference_mode():
        audio = snac.decode([codes_l0, codes_l1, codes_l2])

    audio_np = audio.squeeze().cpu().numpy()

    print(f"\nAudio output:")
    print(f"  Shape: {audio_np.shape}")
    print(f"  Duration: {len(audio_np) / SAMPLE_RATE:.2f}s")
    print(f"  Range: [{audio_np.min():.4f}, {audio_np.max():.4f}]")
    print(f"  Mean: {audio_np.mean():.6f}")
    print(f"  Std: {audio_np.std():.4f}")

    # Check for silence or clipping
    if audio_np.std() < 0.01:
        print("  WARNING: Audio appears to be mostly silence!")
    if abs(audio_np.max()) > 0.99 or abs(audio_np.min()) > 0.99:
        print("  WARNING: Audio may be clipping!")

    # Convert to int16
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    print(f"\nInt16 audio:")
    print(f"  Range: [{audio_int16.min()}, {audio_int16.max()}]")

    return {
        "status": "success",
        "text": text,
        "voice": voice,
        "gen_tokens": len(output_ids),
        "audio_tokens": len(audio_tokens),
        "frames": len(layer0),
        "duration_s": len(audio_np) / SAMPLE_RATE,
        "audio_range": [int(audio_int16.min()), int(audio_int16.max())],
    }


@app.local_entrypoint()
def main(
    text: str = "Hello, this is a test of the audio generation system.",
    voice: str = "tara",
):
    """Run token analysis."""
    result = analyze_generation.remote(text=text, voice=voice)

    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    for k, v in result.items():
        print(f"  {k}: {v}")
