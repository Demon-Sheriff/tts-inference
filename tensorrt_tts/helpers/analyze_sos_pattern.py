"""
Analyze SOS Token Pattern in Generated Output

This script investigates whether the model generates multiple SOS tokens
and if using tokens after the LAST SOS produces better audio.

Usage:
    modal run analyze_sos_pattern.py --text "Your text here"
"""

import modal

app = modal.App("orpheus-sos-analysis")

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

# Token constants
SOS_TOKEN = 128257      # Start of speech
EOS_TOKEN = 128258      # End of speech
START_TOKEN = 128259    # Start of human/text
END_OF_TEXT = 128009    # End of text
END_OF_TURN = 128260    # End of turn
DELIMITER = 128261      # Delimiter
TOKEN_BASE = 128266
FRAME_SIZE = 7
SAMPLE_RATE = 24000


def redistribute_codes(codes: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Redistribute codes into SNAC layers (Orpheus notebook method)."""
    layer_1, layer_2, layer_3 = [], [], []

    for i in range((len(codes) + 1) // 7):
        layer_1.append(codes[7 * i])
        layer_2.append(codes[7 * i + 1] - 4096)
        layer_3.append(codes[7 * i + 2] - (2 * 4096))
        layer_3.append(codes[7 * i + 3] - (3 * 4096))
        layer_2.append(codes[7 * i + 4] - (4 * 4096))
        layer_3.append(codes[7 * i + 5] - (5 * 4096))
        layer_3.append(codes[7 * i + 6] - (6 * 4096))

    return layer_1, layer_2, layer_3


def decode_audio(audio_tokens: list[int], snac, device: str):
    """Decode audio tokens to waveform."""
    import torch
    import numpy as np

    if len(audio_tokens) < FRAME_SIZE:
        return None, {}

    codes = [t - TOKEN_BASE for t in audio_tokens]
    num_frames = len(codes) // FRAME_SIZE
    codes = codes[:num_frames * FRAME_SIZE]

    layer_1, layer_2, layer_3 = redistribute_codes(codes)

    codes_tensor = [
        torch.tensor(layer_1, dtype=torch.int32, device=device).unsqueeze(0),
        torch.tensor(layer_2, dtype=torch.int32, device=device).unsqueeze(0),
        torch.tensor(layer_3, dtype=torch.int32, device=device).unsqueeze(0),
    ]

    with torch.inference_mode():
        audio = snac.decode(codes_tensor)

    audio_np = audio.squeeze().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    return audio_int16, {
        "frames": num_frames,
        "duration_s": len(audio_int16) / SAMPLE_RATE,
        "min": int(audio_int16.min()),
        "max": int(audio_int16.max()),
        "std": float(audio_int16.std()),
    }


@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=600,
)
def analyze_sos_pattern(text: str, voice: str = "tara", max_tokens: int = 4000):
    """Analyze SOS token pattern in generated output."""
    import torch
    import numpy as np
    from tensorrt_llm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from snac import SNAC

    print("=" * 70)
    print("SOS TOKEN PATTERN ANALYSIS")
    print("=" * 70)
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print(f"Max tokens: {max_tokens}")

    # Load models
    print("\nLoading models...")
    tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)
    llm = LLM(model=ENGINE_DIR)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

    # Format prompt
    prompt_text = f"{voice}: {text}"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_ids = [START_TOKEN] + prompt_tokens + [END_OF_TEXT, END_OF_TURN]

    print(f"\nPrompt: {prompt_text}")
    print(f"Prompt tokens: {len(prompt_ids)}")

    # Generate
    print("\nGenerating...")
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        stop_token_ids=[EOS_TOKEN],
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    outputs = llm.generate([prompt_ids], sampling_params=sampling_params)
    output_ids = list(outputs[0].outputs[0].token_ids)

    print(f"Generated {len(output_ids)} tokens")

    # Analyze SOS occurrences
    print("\n" + "=" * 70)
    print("SOS TOKEN ANALYSIS")
    print("=" * 70)

    sos_positions = [i for i, t in enumerate(output_ids) if t == SOS_TOKEN]
    eos_positions = [i for i, t in enumerate(output_ids) if t == EOS_TOKEN]
    delimiter_positions = [i for i, t in enumerate(output_ids) if t == DELIMITER]

    print(f"\nSOS token (128257) found at positions: {sos_positions}")
    print(f"EOS token (128258) found at positions: {eos_positions}")
    print(f"DELIMITER token (128261) found at positions: {delimiter_positions}")
    print(f"Total SOS occurrences: {len(sos_positions)}")

    # Show token sequence around each SOS
    print("\n" + "=" * 70)
    print("TOKEN CONTEXT AROUND EACH SOS")
    print("=" * 70)

    for i, pos in enumerate(sos_positions):
        start = max(0, pos - 3)
        end = min(len(output_ids), pos + 10)
        context = output_ids[start:end]
        print(f"\nSOS #{i+1} at position {pos}:")
        print(f"  Context [{start}:{end}]: {context}")

        # Check what comes after this SOS
        after_sos = output_ids[pos+1:pos+15] if pos+1 < len(output_ids) else []
        print(f"  Next 14 tokens after SOS: {after_sos}")

        # Check if these look like valid audio tokens
        valid_audio = all(t >= TOKEN_BASE and t < TOKEN_BASE + 28672 for t in after_sos if t != EOS_TOKEN)
        print(f"  Valid audio tokens after SOS: {valid_audio}")

    # Compare extraction methods
    print("\n" + "=" * 70)
    print("COMPARING EXTRACTION METHODS")
    print("=" * 70)

    results = {}

    # Method 1: First SOS (current method)
    if sos_positions:
        first_sos_pos = sos_positions[0]
        tokens_after_first = []
        for t in output_ids[first_sos_pos + 1:]:
            if t == EOS_TOKEN or t == SOS_TOKEN:
                break
            tokens_after_first.append(t)

        print(f"\nMethod 1 - After FIRST SOS (position {first_sos_pos}):")
        print(f"  Audio tokens: {len(tokens_after_first)}")

        if tokens_after_first:
            audio, stats = decode_audio(tokens_after_first, snac, "cuda")
            if audio is not None:
                print(f"  Duration: {stats['duration_s']:.2f}s")
                print(f"  Audio range: [{stats['min']}, {stats['max']}]")
                print(f"  Audio std: {stats['std']:.2f}")
                results["first_sos"] = {"audio": audio.tobytes(), "stats": stats, "tokens": len(tokens_after_first)}

    # Method 2: Last SOS
    if sos_positions:
        last_sos_pos = sos_positions[-1]
        tokens_after_last = []
        for t in output_ids[last_sos_pos + 1:]:
            if t == EOS_TOKEN:
                break
            tokens_after_last.append(t)

        print(f"\nMethod 2 - After LAST SOS (position {last_sos_pos}):")
        print(f"  Audio tokens: {len(tokens_after_last)}")

        if tokens_after_last:
            audio, stats = decode_audio(tokens_after_last, snac, "cuda")
            if audio is not None:
                print(f"  Duration: {stats['duration_s']:.2f}s")
                print(f"  Audio range: [{stats['min']}, {stats['max']}]")
                print(f"  Audio std: {stats['std']:.2f}")
                results["last_sos"] = {"audio": audio.tobytes(), "stats": stats, "tokens": len(tokens_after_last)}

    # Method 3: All tokens between first SOS and EOS (ignoring intermediate SOS)
    if sos_positions:
        first_sos_pos = sos_positions[0]
        all_tokens_after_first = []
        for t in output_ids[first_sos_pos + 1:]:
            if t == EOS_TOKEN:
                break
            if t == SOS_TOKEN:
                continue  # Skip intermediate SOS tokens
            all_tokens_after_first.append(t)

        print(f"\nMethod 3 - ALL tokens after first SOS (skipping intermediate SOS):")
        print(f"  Audio tokens: {len(all_tokens_after_first)}")

        if all_tokens_after_first:
            audio, stats = decode_audio(all_tokens_after_first, snac, "cuda")
            if audio is not None:
                print(f"  Duration: {stats['duration_s']:.2f}s")
                print(f"  Audio range: [{stats['min']}, {stats['max']}]")
                print(f"  Audio std: {stats['std']:.2f}")
                results["all_after_first"] = {"audio": audio.tobytes(), "stats": stats, "tokens": len(all_tokens_after_first)}

    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total generated tokens: {len(output_ids)}")
    print(f"Number of SOS tokens: {len(sos_positions)}")

    if len(sos_positions) > 1:
        print("\n*** MULTIPLE SOS TOKENS DETECTED ***")
        print("This suggests the model generates structured output with multiple speech segments.")
        print("Using tokens after the LAST SOS may give the actual speech content.")

    return {
        "total_tokens": len(output_ids),
        "sos_count": len(sos_positions),
        "sos_positions": sos_positions,
        "results": {k: {"tokens": v["tokens"], "duration": v["stats"]["duration_s"], "std": v["stats"]["std"]}
                   for k, v in results.items()},
        "audio_files": results,
    }


@app.local_entrypoint()
def main(
    text: str = "Hello, this is a test of the audio generation system. The quick brown fox jumps over the lazy dog.",
    voice: str = "tara",
    max_tokens: int = 4000,
):
    """Analyze SOS token pattern."""
    import wave

    result = analyze_sos_pattern.remote(text=text, voice=voice, max_tokens=max_tokens)

    print("\n" + "=" * 70)
    print("SAVING AUDIO FILES FOR COMPARISON")
    print("=" * 70)

    audio_files = result.get("audio_files", {})

    for method, data in audio_files.items():
        if "audio" in data:
            filename = f"sos_analysis_{method}.wav"
            with wave.open(filename, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(data["audio"])
            print(f"Saved: {filename} ({data['stats']['duration_s']:.2f}s)")

    print("\nListen to the files to compare which extraction method produces better audio!")
