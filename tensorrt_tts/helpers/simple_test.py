"""
Simple TensorRT-LLM TTS Test

Tests the TRT-LLM pipeline without the full Modal deployment.
Useful for debugging and verifying the conversion process.

Usage:
    modal run simple_test.py
"""

import modal
app = modal.App("orpheus-trtllm-test")

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


@app.function(
    image=image,
    gpu="L4",  # Match build GPU
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=300,
)
def test_engine():
    """Test if the TRT-LLM engine is properly built and can load."""
    import os

    print("=" * 60)
    print("Testing TRT-LLM Engine")
    print("=" * 60)

    # Check if engine exists
    if not os.path.exists(ENGINE_DIR):
        print(f"ERROR: Engine not found at {ENGINE_DIR}")
        print("Run 'modal run build_engine.py' first")
        return {"status": "error", "message": "Engine not found"}

    print("\nEngine files:")
    for f in os.listdir(ENGINE_DIR):
        size = os.path.getsize(f"{ENGINE_DIR}/{f}") / 1e6
        print(f"  {f}: {size:.1f} MB")

    # Try to load using the executor/runtime API for pre-built engines
    try:
        import tensorrt_llm
        from tensorrt_llm.executor import GenerationExecutor

        print("\nLoading engine with GenerationExecutor...")
        executor = GenerationExecutor.create(ENGINE_DIR)
        print(f"SUCCESS: Engine loaded with GenerationExecutor")

        return {
            "status": "success",
            "engine_dir": ENGINE_DIR,
        }

    except Exception as e:
        import traceback
        print(f"ERROR: Failed to load engine: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.function(
    image=image,
    gpu="L4",  # Match build GPU
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=300,
)
def test_inference():
    """Test a simple inference with the TRT-LLM engine."""
    import os
    import torch
    import numpy as np
    import time

    print("=" * 60)
    print("Testing TRT-LLM Inference")
    print("=" * 60)

    if not os.path.exists(ENGINE_DIR):
        return {"status": "error", "message": "Engine not found"}

    # Constants
    SOS_TOKEN = 128257
    EOS_TOKEN = 128258
    START_TOKEN = 128259
    END_TOKENS = [128009, 128260]
    TOKEN_BASE = 128266
    FRAME_SIZE = 7
    SAMPLE_RATE = 24000
    POSITION_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]

    try:
        from tensorrt_llm import LLM, SamplingParams
        from transformers import AutoTokenizer
        from snac import SNAC

        # Load tokenizer from the engine directory
        print("Loading tokenizer...")
        tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)

        # Load engine using LLM API (can load pre-built engines)
        print("Loading TRT-LLM engine...")
        t0 = time.perf_counter()
        llm = LLM(model=ENGINE_DIR)
        print(f"Engine loaded in {time.perf_counter() - t0:.2f}s")

        # Load SNAC
        print("Loading SNAC...")
        snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()

        # Format prompt
        text = "Hello, this is a test."
        voice = "tara"
        prompt_text = f"{voice}: {text}"
        prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
        prompt_ids = [START_TOKEN] + prompt_tokens + END_TOKENS

        print(f"\nPrompt: {prompt_text}")
        print(f"Prompt tokens: {len(prompt_ids)}")

        # Generate
        print("\nGenerating...")
        t1 = time.perf_counter()

        # Create sampling params matching the vLLM version
        sampling_params = SamplingParams(
            max_tokens=2000,
            stop_token_ids=[EOS_TOKEN],
            temperature=0.6,
            top_p=0.95,
            repetition_penalty=1.1,
        )
        print(f"SamplingParams: max_tokens=2000, stop_token_ids=[{EOS_TOKEN}], repetition_penalty=1.1")

        # Generate using LLM API
        outputs = llm.generate([prompt_ids], sampling_params=sampling_params)

        gen_time = time.perf_counter() - t1
        print(f"Generation time: {gen_time:.2f}s")

        # Extract audio tokens - LLM API returns list of RequestOutput
        # Each output has .outputs which is a list of CompletionOutput
        output = outputs[0]
        print(f"Output type: {type(output)}")

        # Get token ids from the output
        # The .outputs[0].token_ids contains only generated tokens (not prompt)
        # We need just the generated part for audio
        if hasattr(output, 'outputs') and output.outputs:
            completion = output.outputs[0]
            output_ids = completion.token_ids
            print(f"Finish reason: {completion.finish_reason}")
            print(f"Stop reason: {completion.stop_reason}")
            print(f"Generated tokens: {len(output_ids)}")
            print(f"First 30 tokens: {output_ids[:30]}")
            print(f"Last 10 tokens: {output_ids[-10:] if len(output_ids) > 10 else output_ids}")
        elif hasattr(output, 'token_ids'):
            output_ids = output.token_ids
        else:
            print(f"Output attrs: {[a for a in dir(output) if not a.startswith('_')]}")
            raise ValueError(f"Cannot find token_ids in output: {type(output)}")

        print(f"Total output tokens: {len(output_ids)}")

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

        print(f"Audio tokens: {len(audio_tokens)}")

        if not audio_tokens:
            return {"status": "error", "message": "No audio tokens generated"}

        # Convert to SNAC codes
        codes = [t - TOKEN_BASE for t in audio_tokens]
        num_frames = len(codes) // FRAME_SIZE
        codes = codes[:num_frames * FRAME_SIZE]

        # Redistribute
        layer0, layer1, layer2 = [], [], []
        for i in range(num_frames):
            base = i * FRAME_SIZE

            def get_code(pos):
                raw = codes[base + pos] - POSITION_OFFSETS[pos]
                return max(0, min(4095, raw))

            layer0.append(get_code(0))
            layer1.extend([get_code(1), get_code(4)])
            layer2.extend([get_code(2), get_code(3), get_code(5), get_code(6)])

        print(f"Frames: {num_frames}")

        # Decode
        print("\nDecoding with SNAC...")
        t2 = time.perf_counter()

        codes_l0 = torch.tensor(layer0, dtype=torch.int32, device="cuda").unsqueeze(0)
        codes_l1 = torch.tensor(layer1, dtype=torch.int32, device="cuda").unsqueeze(0)
        codes_l2 = torch.tensor(layer2, dtype=torch.int32, device="cuda").unsqueeze(0)

        with torch.inference_mode():
            audio = snac.decode([codes_l0, codes_l1, codes_l2])

        decode_time = time.perf_counter() - t2
        print(f"Decode time: {decode_time:.3f}s")

        # Stats
        audio_np = audio.squeeze().cpu().numpy()
        audio_np = np.clip(audio_np, -1.0, 1.0)
        audio_int16 = (audio_np * 32767).astype(np.int16)

        duration = len(audio_int16) / SAMPLE_RATE

        print(f"\n" + "=" * 60)
        print("SUCCESS!")
        print("=" * 60)
        print(f"Duration: {duration:.2f}s")
        print(f"Audio range: [{audio_int16.min()}, {audio_int16.max()}]")
        print(f"Total time: {time.perf_counter() - t0:.2f}s")

        return {
            "status": "success",
            "duration_s": duration,
            "audio_tokens": len(audio_tokens),
            "gen_time_s": gen_time,
            "decode_time_s": decode_time,
        }

    except Exception as e:
        import traceback
        print(f"ERROR: {e}")
        traceback.print_exc()
        return {"status": "error", "message": str(e)}


@app.local_entrypoint()
def main(test_type: str = "all"):
    """Run tests.

    Args:
        test_type: "engine", "inference", or "all"
    """
    if test_type in ["engine", "all"]:
        print("\n" + "=" * 60)
        print("TEST 1: Engine Loading")
        print("=" * 60)
        result = test_engine.remote()
        print(f"Result: {result}")

    if test_type in ["inference", "all"]:
        print("\n" + "=" * 60)
        print("TEST 2: Inference")
        print("=" * 60)
        result = test_inference.remote()
        print(f"Result: {result}")
