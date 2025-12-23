"""
TensorRT-LLM Inference for Canopy Labs Hindi Orpheus TTS

High-performance TTS inference using TensorRT-LLM engine
for the official Canopy Labs Hindi fine-tuned model.

Model: canopylabs/3b-hi-ft-research_release

Usage:
    modal run tensorrt_tts/hindi_canopy/inference.py --text "नमस्ते, आप कैसे हैं?"
    modal run tensorrt_tts/hindi_canopy/inference.py --text "Hello world" --voice tara --max-tokens 8000
"""

import modal

app = modal.App("orpheus-hindi-canopy-trtllm-inference")

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

engine_vol = modal.Volume.from_name("orpheus-hindi-canopy-engine", create_if_missing=True)
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
    """Redistribute codes into SNAC layers."""
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


@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=900,  # 15 min for long generations
)
def generate_speech(
    text: str,
    voice: str = "tara",
    max_tokens: int = 8000,  # ~97 seconds of audio
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> dict:
    """
    Generate speech from text using Canopy Hindi TRT-LLM engine.

    Args:
        text: Input text (Hindi or English)
        voice: Voice name ("tara" recommended for Hindi)
        max_tokens: Maximum tokens to generate (8000 = ~97s audio)
        temperature: Sampling temperature
        top_p: Nucleus sampling parameter

    Returns:
        Dict with audio bytes and metadata
    """
    import torch
    import numpy as np
    import time
    from tensorrt_llm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from snac import SNAC

    print(f"Generating speech for: {text[:80]}...")
    start_time = time.perf_counter()

    # Load models
    print("Loading models...")
    tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)
    llm = LLM(model=ENGINE_DIR)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    load_time = time.perf_counter() - start_time
    print(f"Models loaded in {load_time:.2f}s")

    # Format prompt
    prompt_text = f"{voice}: {text}"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_ids = [START_TOKEN] + prompt_tokens + [END_OF_TEXT, END_OF_TURN]

    print(f"Prompt: {prompt_text}")
    print(f"Prompt tokens: {len(prompt_ids)}")
    print(f"Max output tokens: {max_tokens}")

    # Generate
    print("Generating...")
    gen_start = time.perf_counter()
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        stop_token_ids=[EOS_TOKEN],
        temperature=temperature,
        top_p=top_p,
        repetition_penalty=1.1,
    )

    outputs = llm.generate([prompt_ids], sampling_params=sampling_params)
    output_ids = list(outputs[0].outputs[0].token_ids)
    gen_time = time.perf_counter() - gen_start

    print(f"Generated {len(output_ids)} tokens in {gen_time:.2f}s")
    print(f"Tokens/sec: {len(output_ids) / gen_time:.1f}")

    # Extract audio tokens - find LAST SOS token (key insight from reference code)
    # The model may generate multiple SOS tokens, we want everything after the LAST one
    sos_indices = [i for i, tid in enumerate(output_ids) if tid == SOS_TOKEN]

    if sos_indices:
        last_sos_idx = sos_indices[-1]
        print(f"Found {len(sos_indices)} SOS token(s), using last one at position {last_sos_idx}")

        # Extract tokens after the last SOS, until EOS
        audio_tokens = []
        for tid in output_ids[last_sos_idx + 1:]:
            if tid == EOS_TOKEN:
                break
            audio_tokens.append(tid)
    else:
        print("Warning: No SOS token found in output")
        audio_tokens = []

    print(f"Audio tokens: {len(audio_tokens)}")

    # Debug token info
    if audio_tokens:
        print(f"First 10 tokens: {audio_tokens[:10]}")
        print(f"Token range: [{min(audio_tokens)}, {max(audio_tokens)}]")
        valid_count = sum(1 for t in audio_tokens if TOKEN_BASE <= t < TOKEN_BASE + 28672)
        print(f"Valid audio tokens: {valid_count}/{len(audio_tokens)}")

    if not audio_tokens:
        return {"error": "No audio tokens generated", "total_tokens": len(output_ids)}

    # Decode with SNAC
    print("Decoding with SNAC...")
    decode_start = time.perf_counter()

    codes = [t - TOKEN_BASE for t in audio_tokens]
    num_frames = len(codes) // FRAME_SIZE
    codes = codes[:num_frames * FRAME_SIZE]

    # Check for invalid codes
    invalid_codes = [(i, c) for i, c in enumerate(codes) if c < 0 or c >= 28672]
    if invalid_codes:
        print(f"Warning: {len(invalid_codes)} invalid codes (first 5): {invalid_codes[:5]}")

    layer_1, layer_2, layer_3 = redistribute_codes(codes)

    codes_tensor = [
        torch.tensor(layer_1, dtype=torch.int32, device="cuda").unsqueeze(0),
        torch.tensor(layer_2, dtype=torch.int32, device="cuda").unsqueeze(0),
        torch.tensor(layer_3, dtype=torch.int32, device="cuda").unsqueeze(0),
    ]

    # Validate and clamp codes
    for layer_idx, layer in enumerate(codes_tensor):
        if layer.min() < 0 or layer.max() >= 4096:
            print(f"Warning: Layer {layer_idx} out of range [{layer.min()}, {layer.max()}], clamping")
            codes_tensor[layer_idx] = torch.clamp(layer, 0, 4095)

    with torch.inference_mode():
        audio = snac.decode(codes_tensor)

    audio_np = audio.squeeze().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    decode_time = time.perf_counter() - decode_start

    total_time = time.perf_counter() - start_time
    duration = len(audio_int16) / SAMPLE_RATE

    print(f"Audio: {duration:.2f}s, range=[{audio_int16.min()}, {audio_int16.max()}]")
    print(f"Total time: {total_time:.2f}s, RTF: {duration/total_time:.2f}x realtime")

    return {
        "text": text,
        "voice": voice,
        "total_tokens": len(output_ids),
        "audio_tokens": len(audio_tokens),
        "frames": num_frames,
        "duration_s": duration,
        "load_time_s": load_time,
        "gen_time_s": gen_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "tokens_per_sec": len(output_ids) / gen_time,
        "rtf": duration / total_time if total_time > 0 else 0,
        "audio_min": int(audio_int16.min()),
        "audio_max": int(audio_int16.max()),
        "audio_std": float(audio_int16.std()),
        "audio": audio_int16.tobytes(),
    }


@app.local_entrypoint()
def main(
    text: str = "नमस्ते, मैं एक हिंदी टेक्स्ट टू स्पीच मॉडल हूं। आज का मौसम बहुत अच्छा है। क्या आप मेरी आवाज सुन सकते हैं?",
    voice: str = "tara",
    max_tokens: int = 8000,
    temperature: float = 0.6,
    output: str = "hindi_canopy_output.wav",
):
    """Generate speech from Hindi/English text using Canopy Labs model."""
    import wave

    print("=" * 70)
    print("CANOPY LABS HINDI TTS with TensorRT-LLM")
    print("=" * 70)
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print(f"Max tokens: {max_tokens}")
    print()

    result = generate_speech.remote(
        text=text,
        voice=voice,
        max_tokens=max_tokens,
        temperature=temperature,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Save audio
    audio_bytes = result["audio"]
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_bytes)

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"Audio saved to: {output}")
    print(f"Duration: {result['duration_s']:.2f}s")
    print(f"Total tokens: {result['total_tokens']}")
    print(f"Audio tokens: {result['audio_tokens']}")
    print(f"Frames: {result['frames']}")
    print(f"Generation time: {result['gen_time_s']:.2f}s ({result['tokens_per_sec']:.1f} tok/s)")
    print(f"Decode time: {result['decode_time_s']:.2f}s")
    print(f"Total time: {result['total_time_s']:.2f}s")
    print(f"RTF: {result['rtf']:.2f}x realtime")
    print(f"Audio range: [{result['audio_min']}, {result['audio_max']}]")
    print(f"Audio std: {result['audio_std']:.2f}")
