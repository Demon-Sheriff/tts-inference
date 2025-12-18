"""
Debug Token Generation: Compare HuggingFace vs TensorRT-LLM token-by-token

This script uses greedy decoding (temperature=0) to eliminate randomness
and compare the exact token outputs from both pipelines.

Usage:
    modal run debug_tokens.py
"""

import modal

app = modal.App("orpheus-debug-tokens")

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

# HuggingFace Transformers image
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

# Token constants
SOS_TOKEN = 128257
EOS_TOKEN = 128258
START_TOKEN = 128259
END_OF_TEXT = 128009
END_OF_TURN = 128260
TOKEN_BASE = 128266


@app.function(
    image=hf_image,
    gpu="A10G",
    volumes={"/cache": cache_vol},
    timeout=600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def generate_hf_greedy(text: str, voice: str = "tara", max_tokens: int = 100):
    """Generate with HuggingFace using greedy decoding."""
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer

    print("Loading HuggingFace model...")
    model = AutoModelForCausalLM.from_pretrained(
        HF_MODEL_PATH,
        torch_dtype=torch.float16,
        device_map="auto",
        trust_remote_code=True,
    )
    tokenizer = AutoTokenizer.from_pretrained(HF_MODEL_PATH)

    # Format prompt - same as TensorRT
    prompt_text = f"{voice}: {text}"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    input_ids = torch.tensor([[START_TOKEN] + prompt_tokens + [END_OF_TEXT, END_OF_TURN]], device="cuda")

    print(f"Prompt: {prompt_text}")
    print(f"Input IDs: {input_ids[0].tolist()}")

    # Generate with GREEDY decoding (no randomness)
    with torch.no_grad():
        generated_ids = model.generate(
            input_ids=input_ids,
            max_new_tokens=max_tokens,
            do_sample=False,  # GREEDY - deterministic
            eos_token_id=EOS_TOKEN,
            pad_token_id=EOS_TOKEN,
        )

    output_ids = generated_ids[0].tolist()
    prompt_length = input_ids.shape[1]
    new_tokens = output_ids[prompt_length:]

    print(f"Generated {len(new_tokens)} new tokens")

    return {
        "prompt_ids": input_ids[0].tolist(),
        "output_ids": output_ids,
        "new_tokens": new_tokens,
    }


@app.function(
    image=trtllm_image,
    gpu="L4",
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=600,
)
def generate_trt_greedy(text: str, voice: str = "tara", max_tokens: int = 100):
    """Generate with TensorRT-LLM using greedy decoding."""
    from tensorrt_llm import LLM, SamplingParams
    from transformers import AutoTokenizer

    print("Loading TensorRT-LLM engine...")
    tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)
    llm = LLM(model=ENGINE_DIR)

    # Format prompt - same as HuggingFace
    prompt_text = f"{voice}: {text}"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
    prompt_ids = [START_TOKEN] + prompt_tokens + [END_OF_TEXT, END_OF_TURN]

    print(f"Prompt: {prompt_text}")
    print(f"Input IDs: {prompt_ids}")

    # Generate with GREEDY decoding (temperature=0 equivalent)
    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        stop_token_ids=[EOS_TOKEN],
        temperature=0.0,  # GREEDY - deterministic
        top_p=1.0,
        top_k=1,  # Only pick top token
    )

    outputs = llm.generate([prompt_ids], sampling_params=sampling_params)
    output_ids = outputs[0].outputs[0].token_ids

    print(f"Generated {len(output_ids)} tokens")

    return {
        "prompt_ids": prompt_ids,
        "output_ids": list(output_ids),
        "new_tokens": list(output_ids),
    }


@app.local_entrypoint()
def main(
    text: str = "Hello.",
    voice: str = "tara",
    max_tokens: int = 50,
):
    """Compare greedy token generation."""
    print("=" * 70)
    print("GREEDY DECODING COMPARISON")
    print("=" * 70)
    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print(f"Max tokens: {max_tokens}")
    print()

    # Run both
    print("Running both pipelines...")
    hf_future = generate_hf_greedy.spawn(text=text, voice=voice, max_tokens=max_tokens)
    trt_future = generate_trt_greedy.spawn(text=text, voice=voice, max_tokens=max_tokens)

    hf_result = hf_future.get()
    trt_result = trt_future.get()

    print("\n" + "=" * 70)
    print("PROMPT COMPARISON")
    print("=" * 70)
    print(f"HF prompt:  {hf_result['prompt_ids']}")
    print(f"TRT prompt: {trt_result['prompt_ids']}")
    print(f"Prompts match: {hf_result['prompt_ids'] == trt_result['prompt_ids']}")

    print("\n" + "=" * 70)
    print("TOKEN-BY-TOKEN COMPARISON")
    print("=" * 70)

    hf_tokens = hf_result['new_tokens']
    trt_tokens = trt_result['new_tokens']

    print(f"HF generated:  {len(hf_tokens)} tokens")
    print(f"TRT generated: {len(trt_tokens)} tokens")

    # Compare token by token
    max_len = max(len(hf_tokens), len(trt_tokens))
    mismatches = []

    print(f"\n{'Pos':<5} {'HF Token':<12} {'TRT Token':<12} {'Match':<8} {'HF-BASE':<10} {'TRT-BASE':<10}")
    print("-" * 70)

    for i in range(min(max_len, 50)):  # Show first 50
        hf_tok = hf_tokens[i] if i < len(hf_tokens) else None
        trt_tok = trt_tokens[i] if i < len(trt_tokens) else None

        match = "✓" if hf_tok == trt_tok else "✗"
        if hf_tok != trt_tok:
            mismatches.append(i)

        hf_base = hf_tok - TOKEN_BASE if hf_tok and hf_tok >= TOKEN_BASE else hf_tok
        trt_base = trt_tok - TOKEN_BASE if trt_tok and trt_tok >= TOKEN_BASE else trt_tok

        hf_str = str(hf_tok) if hf_tok else "-"
        trt_str = str(trt_tok) if trt_tok else "-"
        hf_base_str = str(hf_base) if hf_base else "-"
        trt_base_str = str(trt_base) if trt_base else "-"

        print(f"{i:<5} {hf_str:<12} {trt_str:<12} {match:<8} {hf_base_str:<10} {trt_base_str:<10}")

    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    print(f"Total HF tokens: {len(hf_tokens)}")
    print(f"Total TRT tokens: {len(trt_tokens)}")
    print(f"Mismatches: {len(mismatches)}")

    if mismatches:
        print(f"First mismatch at position: {mismatches[0]}")

        # Check if it's a special token issue
        if hf_tokens and trt_tokens:
            hf_first = hf_tokens[0]
            trt_first = trt_tokens[0]
            print(f"\nFirst token analysis:")
            print(f"  HF first token: {hf_first} (is SOS: {hf_first == SOS_TOKEN})")
            print(f"  TRT first token: {trt_first} (is SOS: {trt_first == SOS_TOKEN})")
    else:
        print("All tokens match! ✓")
