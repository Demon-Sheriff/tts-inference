"""
Compare SNAC Audio Output: HuggingFace Transformers vs TensorRT-LLM

This script runs the SAME prompt through both pipelines and compares:
1. HuggingFace Transformers model.generate() - the Orpheus reference way
2. TensorRT-LLM pipeline - our optimized version

Then decodes both with SNAC to verify audio quality differences.

Usage:
    modal run compare_snac.py
"""

import modal

app = modal.App("orpheus-snac-compare")

# TensorRT-LLM image
trtllm_image = (
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

# HuggingFace Transformers image (like Orpheus notebook uses)
hf_image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "torch",
        "transformers",
        "accelerate",
        "snac",
        "numpy",
        "huggingface_hub",
        "hf_transfer",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

engine_vol = modal.Volume.from_name("orpheus-trtllm-engine", create_if_missing=True)
cache_vol = modal.Volume.from_name("orpheus-cache", create_if_missing=True)

ENGINE_DIR = "/engine/trt_engine"
HF_MODEL_PATH = "/cache/orpheus-3b"

# Token constants (from Orpheus notebook)
# Reference: https://github.com/canopyai/Orpheus-TTS
SOS_TOKEN = 128257      # Start of speech
EOS_TOKEN = 128258      # End of speech
START_TOKEN = 128259    # Start of human/text
END_OF_TEXT = 128009    # End of text
END_OF_TURN = 128260    # End of turn
DELIMITER = 128261      # Delimiter before SOS

# Orpheus notebook uses: [128009, 128260, 128261, 128257] to trigger audio generation
# The SOS token (128257) at the end signals the model to start generating speech
END_TOKENS_WITH_SOS = [END_OF_TEXT, END_OF_TURN, DELIMITER, SOS_TOKEN]

TOKEN_BASE = 128266
FRAME_SIZE = 7
SAMPLE_RATE = 24000


def redistribute_codes(codes: list[int]) -> tuple[list[int], list[int], list[int]]:
    """
    Redistribute codes into SNAC layers.
    EXACT copy from Orpheus notebook.
    """
    layer_1 = []
    layer_2 = []
    layer_3 = []

    for i in range((len(codes) + 1) // 7):
        layer_1.append(codes[7 * i])
        layer_2.append(codes[7 * i + 1] - 4096)
        layer_3.append(codes[7 * i + 2] - (2 * 4096))
        layer_3.append(codes[7 * i + 3] - (3 * 4096))
        layer_2.append(codes[7 * i + 4] - (4 * 4096))
        layer_3.append(codes[7 * i + 5] - (5 * 4096))
        layer_3.append(codes[7 * i + 6] - (6 * 4096))

    return layer_1, layer_2, layer_3


def extract_audio_tokens_hf(output_ids: list[int], prompt_length: int) -> list[int]:
    """
    Extract audio tokens from HuggingFace output.

    Since we include SOS in the prompt, the generated tokens start immediately
    after the prompt and end at EOS.
    """
    # Get only the newly generated tokens (after prompt)
    generated = output_ids[prompt_length:]

    # Remove EOS if present
    audio_tokens = []
    for tid in generated:
        if tid == EOS_TOKEN:
            break
        audio_tokens.append(tid)

    return audio_tokens


def extract_audio_tokens_trt(output_ids: list[int]) -> list[int]:
    """
    Extract audio tokens from TensorRT-LLM output.

    TRT-LLM output starts after the prompt and includes SOS token first,
    then audio tokens, then EOS.
    """
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

    return audio_tokens


def decode_to_audio(audio_tokens: list[int], snac_model, device: str):
    """Decode audio tokens to waveform using SNAC."""
    import torch
    import numpy as np

    # Subtract TOKEN_BASE
    codes = [t - TOKEN_BASE for t in audio_tokens]

    # Trim to complete frames
    num_frames = len(codes) // FRAME_SIZE
    codes = codes[:num_frames * FRAME_SIZE]

    if num_frames == 0:
        return None, {}

    # Redistribute using Orpheus method
    layer_1, layer_2, layer_3 = redistribute_codes(codes)

    # Create tensors
    codes_tensor = [
        torch.tensor(layer_1, dtype=torch.int32, device=device).unsqueeze(0),
        torch.tensor(layer_2, dtype=torch.int32, device=device).unsqueeze(0),
        torch.tensor(layer_3, dtype=torch.int32, device=device).unsqueeze(0),
    ]
    

    # Decode
    with torch.inference_mode():
        audio = snac_model.decode(codes_tensor)

    audio_np = audio.squeeze().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    stats = {
        "frames": num_frames,
        "samples": len(audio_int16),
        "duration_s": len(audio_int16) / SAMPLE_RATE,
        "min": int(audio_int16.min()),
        "max": int(audio_int16.max()),
        "std": float(audio_int16.std()),
    }

    return audio_int16, stats


@app.function(
    image=hf_image,
    gpu="A10G",
    volumes={"/cache": cache_vol},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_with_huggingface(text: str, voice: str = "tara"):
    """
    Generate audio using HuggingFace Transformers.
    This matches exactly how the Orpheus inference notebook does it.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer
    from snac import SNAC
    import time

    print("=" * 70)
    print("HUGGINGFACE TRANSFORMERS GENERATION")
    print("=" * 70)
    print(f"Text: {text}")
    print(f"Voice: {voice}")

    result = {"pipeline": "HuggingFace", "text": text, "voice": voice}

    t0 = time.perf_counter()

    # Load model and tokenizer (exactly like Orpheus notebook)
    print("\nLoading model...")
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)

    result["load_time_s"] = time.perf_counter() - t0
    print(f"Model loaded in {result['load_time_s']:.2f}s")

    # Load SNAC
    print("Loading SNAC...")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

    # Format prompt - use [START_TOKEN] + text + [END_OF_TEXT, END_OF_TURN]
    # The model will then generate SOS + audio tokens + EOS
    prompt_text = f"{voice}: {text}"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = torch.tensor([[START_TOKEN] + prompt_tokens + [END_OF_TEXT, END_OF_TURN]], device="cuda")

    print(f"\nPrompt: {prompt_text}")
    print(f"Input tokens: {input_ids.shape[1]}")

    # Generate (matching Orpheus notebook parameters)
    print("\nGenerating...")
    t1 = time.perf_counter()

    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=2000,
            do_sample=True,
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
            eos_token_id=EOS_TOKEN,
        )

    result["gen_time_s"] = time.perf_counter() - t1
    print(f"Generation time: {result['gen_time_s']:.2f}s")

    # Extract generated tokens
    output_ids = generated_ids[0].tolist()
    prompt_length = input_ids.shape[1]
    result["total_tokens"] = len(output_ids)
    result["new_tokens"] = len(output_ids) - prompt_length
    print(f"Total tokens: {result['total_tokens']} (prompt: {prompt_length}, new: {result['new_tokens']})")

    # Extract audio tokens (find SOS, then collect until EOS)
    audio_tokens = []
    found_sos = False
    for tid in output_ids[prompt_length:]:
        if not found_sos:
            if tid == SOS_TOKEN:
                found_sos = True
            continue
        if tid == EOS_TOKEN:
            break
        audio_tokens.append(tid)
    result["audio_tokens"] = len(audio_tokens)
    print(f"Audio tokens: {result['audio_tokens']}")

    if not audio_tokens:
        result["error"] = "No audio tokens"
        return result

    # Store first 50 tokens for comparison
    result["first_50_tokens"] = audio_tokens[:50]
    result["last_20_tokens"] = audio_tokens[-20:]

    # Decode with SNAC
    print("\nDecoding with SNAC...")
    audio_int16, stats = decode_to_audio(audio_tokens, snac, "cuda")

    if audio_int16 is None:
        result["error"] = "Decode failed"
        return result

    result.update(stats)
    result["audio"] = audio_int16.tobytes()

    print(f"Audio: {stats['duration_s']:.2f}s, range=[{stats['min']}, {stats['max']}]")

    return result


@app.function(
    image=trtllm_image,
    gpu="L4",
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=600,
)
def generate_with_tensorrt(text: str, voice: str = "tara"):
    """
    Generate audio using TensorRT-LLM.
    """
    import torch
    from tensorrt_llm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from snac import SNAC
    import time

    print("=" * 70)
    print("TENSORRT-LLM GENERATION")
    print("=" * 70)
    print(f"Text: {text}")
    print(f"Voice: {voice}")

    result = {"pipeline": "TensorRT-LLM", "text": text, "voice": voice}

    t0 = time.perf_counter()

    # Load TRT-LLM engine
    print("\nLoading TRT-LLM engine...")
    tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)
    llm = LLM(model=ENGINE_DIR)

    result["load_time_s"] = time.perf_counter() - t0
    print(f"Engine loaded in {result['load_time_s']:.2f}s")

    # Load SNAC
    print("Loading SNAC...")
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

    # Format prompt - use SAME format as HuggingFace for fair comparison
    # [START_TOKEN] + text + [END_OF_TEXT, END_OF_TURN]
    # Model generates SOS + audio tokens + EOS
    prompt_text = f"{voice}: {text}"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_ids = [START_TOKEN] + prompt_tokens + [END_OF_TEXT, END_OF_TURN]

    print(f"\nPrompt: {prompt_text}")
    print(f"Input tokens: {len(prompt_ids)}")

    # Generate
    print("\nGenerating...")
    t1 = time.perf_counter()

    sampling_params = SamplingParams(
        max_tokens=2000,
        stop_token_ids=[EOS_TOKEN],
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    outputs = llm.generate([prompt_ids], sampling_params=sampling_params)
    output_ids = outputs[0].outputs[0].token_ids

    result["gen_time_s"] = time.perf_counter() - t1
    print(f"Generation time: {result['gen_time_s']:.2f}s")

    result["total_tokens"] = len(output_ids)
    print(f"Total tokens: {result['total_tokens']}")

    # Extract audio tokens (find SOS, then collect until EOS)
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
    result["audio_tokens"] = len(audio_tokens)
    print(f"Audio tokens: {result['audio_tokens']}")

    if not audio_tokens:
        result["error"] = "No audio tokens"
        return result

    # Store first 50 tokens for comparison
    result["first_50_tokens"] = audio_tokens[:50]
    result["last_20_tokens"] = audio_tokens[-20:]

    # Decode with SNAC
    print("\nDecoding with SNAC...")
    audio_int16, stats = decode_to_audio(audio_tokens, snac, "cuda")

    if audio_int16 is None:
        result["error"] = "Decode failed"
        return result

    result.update(stats)
    result["audio"] = audio_int16.tobytes()

    print(f"Audio: {stats['duration_s']:.2f}s, range=[{stats['min']}, {stats['max']}]")

    return result


@app.local_entrypoint()
def main(
    text: str = "Hello, this is a test of the audio generation system.",
    voice: str = "tara",
):
    """
    Compare HuggingFace Transformers vs TensorRT-LLM for Orpheus TTS.
    """
    import wave
    import numpy as np

    print("=" * 70)
    print("COMPARING: HuggingFace Transformers vs TensorRT-LLM")
    print("=" * 70)
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print()

    # Run both in parallel
    print("Running both pipelines in parallel...")
    hf_future = generate_with_huggingface.spawn(text=text, voice=voice)
    trt_future = generate_with_tensorrt.spawn(text=text, voice=voice)

    hf_result = hf_future.get()
    trt_result = trt_future.get()

    # Compare results
    print("\n" + "=" * 70)
    print("RESULTS COMPARISON")
    print("=" * 70)

    metrics = ["gen_time_s", "total_tokens", "audio_tokens", "frames",
               "duration_s", "min", "max", "std"]

    print(f"\n{'Metric':<20} {'HuggingFace':>15} {'TensorRT':>15} {'Diff':>15}")
    print("-" * 70)

    for metric in metrics:
        hf_val = hf_result.get(metric, "N/A")
        trt_val = trt_result.get(metric, "N/A")

        if isinstance(hf_val, float):
            hf_str = f"{hf_val:.3f}"
            trt_str = f"{trt_val:.3f}" if isinstance(trt_val, float) else str(trt_val)
            if isinstance(trt_val, (int, float)):
                diff = trt_val - hf_val
                diff_str = f"{diff:+.3f}"
            else:
                diff_str = "N/A"
        else:
            hf_str = str(hf_val)
            trt_str = str(trt_val)
            if isinstance(hf_val, int) and isinstance(trt_val, int):
                diff_str = f"{trt_val - hf_val:+d}"
            else:
                diff_str = "N/A"

        print(f"{metric:<20} {hf_str:>15} {trt_str:>15} {diff_str:>15}")

    # Compare token patterns
    print("\n" + "=" * 70)
    print("TOKEN ANALYSIS")
    print("=" * 70)

    hf_tokens = hf_result.get("first_50_tokens", [])
    trt_tokens = trt_result.get("first_50_tokens", [])

    print(f"\nFirst 20 audio tokens:")
    print(f"  HF:  {hf_tokens[:20]}")
    print(f"  TRT: {trt_tokens[:20]}")

    # Check token distribution
    if hf_tokens and trt_tokens:
        hf_codes = [t - TOKEN_BASE for t in hf_tokens]
        trt_codes = [t - TOKEN_BASE for t in trt_tokens]

        print(f"\nFirst 20 raw codes (after subtracting TOKEN_BASE):")
        print(f"  HF:  {hf_codes[:20]}")
        print(f"  TRT: {trt_codes[:20]}")

        # Check if tokens are in valid range
        hf_valid = all(0 <= c < 28672 for c in hf_codes)  # 7 * 4096
        trt_valid = all(0 <= c < 28672 for c in trt_codes)
        print(f"\nTokens in valid range (0-28671):")
        print(f"  HF:  {'YES' if hf_valid else 'NO'}")
        print(f"  TRT: {'YES' if trt_valid else 'NO'}")

    # Save audio files
    print("\n" + "=" * 70)
    print("SAVING AUDIO FILES")
    print("=" * 70)

    for name, result in [("huggingface", hf_result), ("tensorrt", trt_result)]:
        if "audio" in result and result["audio"]:
            filename = f"compare_{name}.wav"
            with wave.open(filename, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(result["audio"])
            print(f"Saved: {filename} ({result.get('duration_s', 0):.2f}s)")

    # Audio similarity analysis
    print("\n" + "=" * 70)
    print("AUDIO QUALITY NOTES")
    print("=" * 70)

    print("""
Since both pipelines use different random sampling, the actual tokens
will be DIFFERENT. This is expected! What matters is:

1. Token counts should be similar (both generate reasonable speech)
2. Audio duration should be similar for same text
3. Audio quality (std, range) should be comparable
4. Both should produce intelligible speech

If TensorRT audio sounds worse despite similar metrics, it could be:
- TensorRT quantization affecting model behavior
- Different sampling implementation
- Engine build parameters

Listen to both compare_huggingface.wav and compare_tensorrt.wav to judge.
""")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)

    hf_ok = "error" not in hf_result
    trt_ok = "error" not in trt_result

    print(f"HuggingFace: {'SUCCESS' if hf_ok else 'FAILED - ' + hf_result.get('error', '')}")
    print(f"TensorRT:    {'SUCCESS' if trt_ok else 'FAILED - ' + trt_result.get('error', '')}")

    if hf_ok and trt_ok:
        hf_dur = hf_result.get("duration_s", 0)
        trt_dur = trt_result.get("duration_s", 0)
        print(f"\nDuration difference: {abs(trt_dur - hf_dur):.2f}s")
        print(f"Token count difference: {abs(trt_result.get('audio_tokens', 0) - hf_result.get('audio_tokens', 0))}")
