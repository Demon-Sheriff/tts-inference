"""
TensorRT-LLM Inference on A100 GPU

Benefits of A100 (40GB):
- Both TRT-LLM engine and SNAC decoder fit in VRAM simultaneously
- No memory management needed between generation and decoding
- Faster generation throughput
- Can decode full audio without chunking

Usage:
    modal run tensorrt_tts/hindi_canopy/inference_a100.py --text "नमस्ते"
    modal run tensorrt_tts/hindi_canopy/inference_a100.py --text "..." --max-tokens 11000
"""

import modal

app = modal.App("orpheus-hindi-canopy-trtllm-inference-a100")

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

# A100-specific engine volume
engine_vol = modal.Volume.from_name("orpheus-hindi-canopy-engine-a100", create_if_missing=True)
cache_vol = modal.Volume.from_name("orpheus-cache", create_if_missing=True)

ENGINE_DIR = "/engine/trt_engine"

# Token constants
SOS_TOKEN = 128257
EOS_TOKEN = 128258
START_TOKEN = 128259
END_OF_TEXT = 128009
END_OF_TURN = 128260
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
    gpu="A100",  # A100 40GB
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=1800,
)
def generate_speech(
    text: str,
    voice: str = "tara",
    max_tokens: int = 8000,
    temperature: float = 0.6,
    top_p: float = 0.95,
) -> dict:
    """Generate speech with A100 - no memory management needed."""
    import torch
    import numpy as np
    import time
    import os
    from tensorrt_llm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from snac import SNAC

    # Check engine exists
    if not os.path.exists(f"{ENGINE_DIR}/rank0.engine"):
        return {"error": "A100 engine not found. Run build_engine_a100.py first."}

    print(f"Generating on A100: {text[:80]}...")
    start_time = time.perf_counter()

    # Load BOTH models - A100 has 40GB VRAM
    print("Loading models (A100 has room for both)...")
    tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)
    llm = LLM(model=ENGINE_DIR)
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    load_time = time.perf_counter() - start_time
    print(f"Models loaded in {load_time:.2f}s")

    # Format prompt
    prompt_text = f"{voice}: {text}"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = [START_TOKEN] + prompt_tokens + [END_OF_TEXT, END_OF_TURN]

    print(f"Input tokens: {len(prompt_ids)}")
    print(f"Max output: {max_tokens}")

    # Generate
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

    print(f"Generated {len(output_ids)} tokens in {gen_time:.2f}s ({len(output_ids)/gen_time:.1f} tok/s)")

    # Extract audio tokens after LAST SOS
    sos_indices = [i for i, tid in enumerate(output_ids) if tid == SOS_TOKEN]

    if sos_indices:
        last_sos_idx = sos_indices[-1]
        audio_tokens = []
        for tid in output_ids[last_sos_idx + 1:]:
            if tid == EOS_TOKEN:
                break
            audio_tokens.append(tid)
    else:
        return {"error": "No SOS token found", "output_tokens": len(output_ids)}

    print(f"Audio tokens: {len(audio_tokens)}")

    if not audio_tokens:
        return {"error": "No audio tokens", "output_tokens": len(output_ids)}

    # Decode with SNAC - A100 can handle full sequence at once!
    print("Decoding with SNAC (no chunking needed on A100)...")
    decode_start = time.perf_counter()

    codes = [t - TOKEN_BASE for t in audio_tokens]
    num_frames = len(codes) // FRAME_SIZE
    codes = codes[:num_frames * FRAME_SIZE]

    layer_1, layer_2, layer_3 = redistribute_codes(codes)

    codes_tensor = [
        torch.tensor(layer_1, dtype=torch.int32, device="cuda").unsqueeze(0),
        torch.tensor(layer_2, dtype=torch.int32, device="cuda").unsqueeze(0),
        torch.tensor(layer_3, dtype=torch.int32, device="cuda").unsqueeze(0),
    ]

    # Clamp to valid range
    for i in range(3):
        codes_tensor[i] = torch.clamp(codes_tensor[i], 0, 4095)

    with torch.inference_mode():
        audio = snac.decode(codes_tensor)

    audio_np = audio.squeeze().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)
    decode_time = time.perf_counter() - decode_start

    total_time = time.perf_counter() - start_time
    duration = len(audio_int16) / SAMPLE_RATE

    print(f"Duration: {duration:.2f}s, range=[{audio_int16.min()}, {audio_int16.max()}]")
    print(f"Total: {total_time:.2f}s, RTF: {duration/total_time:.2f}x")

    return {
        "text": text,
        "voice": voice,
        "gpu": "A100",
        "output_tokens": len(output_ids),
        "audio_tokens": len(audio_tokens),
        "frames": num_frames,
        "duration_s": duration,
        "load_time_s": load_time,
        "gen_time_s": gen_time,
        "decode_time_s": decode_time,
        "total_time_s": total_time,
        "tokens_per_sec": len(output_ids) / gen_time,
        "rtf": duration / total_time,
        "audio_min": int(audio_int16.min()),
        "audio_max": int(audio_int16.max()),
        "audio_std": float(audio_int16.std()),
        "audio": audio_int16.tobytes(),
    }


@app.local_entrypoint()
def main(
    text: str = "नमस्ते, मैं एक हिंदी टेक्स्ट टू स्पीच मॉडल हूं। आज का मौसम बहुत अच्छा है।",
    voice: str = "tara",
    max_tokens: int = 8000,
    output: str = "hindi_a100_output.wav",
):
    """Generate speech on A100."""
    import wave

    print("=" * 70)
    print("A100 INFERENCE - Hindi Canopy TTS")
    print("=" * 70)
    print(f"Text: {text[:100]}...")
    print(f"Max tokens: {max_tokens}")
    print()

    result = generate_speech.remote(
        text=text,
        voice=voice,
        max_tokens=max_tokens,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Save audio
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(result["audio"])

    print("\n" + "=" * 70)
    print("RESULTS")
    print("=" * 70)
    print(f"GPU: {result['gpu']}")
    print(f"Duration: {result['duration_s']:.2f}s")
    print(f"Generation: {result['tokens_per_sec']:.1f} tok/s")
    print(f"RTF: {result['rtf']:.2f}x realtime")
    print(f"Audio range: [{result['audio_min']}, {result['audio_max']}]")
    print(f"Saved to: {output}")
