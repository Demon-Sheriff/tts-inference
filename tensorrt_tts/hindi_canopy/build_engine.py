"""
TensorRT-LLM Engine Builder for Canopy Labs Hindi Orpheus TTS

This script converts the official Canopy Labs Hindi fine-tuned model
to TensorRT-LLM engine format for optimized inference.

Model: canopylabs/3b-hi-ft-research_release
- Official Hindi fine-tuned Orpheus 3B model
- No custom LoRA adapter, pure Canopy Labs release

Usage:
    modal run tensorrt_tts/hindi_canopy/build_engine.py
    modal run tensorrt_tts/hindi_canopy/build_engine.py --max-output-len 12000
"""

import modal
import os

app = modal.App("orpheus-hindi-canopy-trtllm-build")

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt-llm/release:0.21.0",
        secret=modal.Secret.from_name("nvcr-credentials"),
    )
    .apt_install("git-lfs")
    .pip_install(
        "huggingface_hub",
        "hf_transfer",
        "safetensors",
        "transformers",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
    })
)

# Separate volume for this engine
engine_vol = modal.Volume.from_name("orpheus-hindi-canopy-engine", create_if_missing=True)
cache_vol = modal.Volume.from_name("orpheus-cache", create_if_missing=True)

# Official Canopy Labs Hindi model (pretrain version as per reference code)
MODEL_ID = "canopylabs/3b-hi-pretrain-research_release"
MODEL_DIR = "/cache/canopy-hindi-3b"
ENGINE_DIR = "/engine/trt_engine"


@app.function(
    image=image,
    gpu="L4",
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=3600,
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def build_engine(
    quantization: str = "fp16",
    max_input_len: int = 512,
    max_output_len: int = 12000,  # ~145 seconds of audio
    max_batch_size: int = 1,
):
    """
    Build TensorRT-LLM engine from Canopy Labs Hindi model.

    Args:
        quantization: Quantization type - "fp16", "bf16"
        max_input_len: Maximum input sequence length
        max_output_len: Maximum output sequence length (~145s of audio with 12000)
        max_batch_size: Maximum batch size
    """
    import shutil
    import json
    from huggingface_hub import snapshot_download

    print("=" * 60)
    print("STEP 1: Download Canopy Labs Hindi Model")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")

    # Download model if not cached
    if not os.path.exists(f"{MODEL_DIR}/config.json"):
        print(f"Downloading {MODEL_ID}...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
        )
        cache_vol.commit()
        print("Model downloaded successfully")
    else:
        print("Model already cached")

    # List model files
    print("\nModel files:")
    for f in os.listdir(MODEL_DIR):
        fpath = f"{MODEL_DIR}/{f}"
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1e9
            print(f"  {f}: {size:.2f} GB")

    print("\n" + "=" * 60)
    print("STEP 2: Build TRT-LLM Engine")
    print("=" * 60)

    # Clean previous engine
    if os.path.exists(ENGINE_DIR):
        shutil.rmtree(ENGINE_DIR)
    os.makedirs(ENGINE_DIR, exist_ok=True)

    # Determine dtype
    if quantization == "fp16":
        dtype = "float16"
    elif quantization == "bf16":
        dtype = "bfloat16"
    else:
        dtype = "float16"

    print(f"Building with dtype={dtype}")
    print(f"Max input: {max_input_len}, Max output: {max_output_len}")
    print(f"Max seq len: {max_input_len + max_output_len}")

    from tensorrt_llm import LLM, BuildConfig

    # Build config - larger sequence length for longer audio
    # 7 tokens = 1 frame = ~85ms
    # 12000 tokens = ~1714 frames = ~145 seconds of audio
    build_config = BuildConfig(
        max_input_len=max_input_len,
        max_seq_len=max_input_len + max_output_len,
        max_batch_size=max_batch_size,
        max_num_tokens=max_input_len + max_output_len,
    )

    # Try loading directly from HuggingFace if local fails
    print(f"\nLoading model from {MODEL_ID} via HuggingFace...")
    llm = LLM(
        model=MODEL_ID,
        dtype=dtype,
        build_config=build_config,
    )

    print(f"Saving engine to {ENGINE_DIR}...")
    llm.save(ENGINE_DIR)

    # Save build info
    build_info = {
        "model_id": MODEL_ID,
        "builder_config": {
            "precision": dtype,
            "max_batch_size": max_batch_size,
            "max_input_len": max_input_len,
            "max_output_len": max_output_len,
            "max_seq_len": max_input_len + max_output_len,
        },
        "audio_capacity": {
            "max_tokens": max_output_len,
            "max_frames": max_output_len // 7,
            "max_duration_s": (max_output_len // 7) * 0.085,
        }
    }
    with open(f"{ENGINE_DIR}/build_info.json", "w") as f:
        json.dump(build_info, f, indent=2)

    print("\n" + "=" * 60)
    print("Engine build complete!")
    print("=" * 60)

    engine_vol.commit()

    # List engine files
    print("\nEngine files:")
    total_size = 0
    for f in os.listdir(ENGINE_DIR):
        fpath = f"{ENGINE_DIR}/{f}"
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            print(f"  {f}: {size / 1e9:.2f} GB")
    print(f"\nTotal engine size: {total_size / 1e9:.2f} GB")
    print(f"Max audio duration: ~{build_info['audio_capacity']['max_duration_s']:.0f} seconds")

    return {
        "status": "success",
        "model_id": MODEL_ID,
        "engine_dir": ENGINE_DIR,
        "total_size_gb": total_size / 1e9,
        "max_seq_len": max_input_len + max_output_len,
        "max_audio_duration_s": build_info['audio_capacity']['max_duration_s'],
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
def verify_engine():
    """Verify the built engine."""
    import json

    print("Checking TRT-LLM engine...")

    if not os.path.exists(ENGINE_DIR):
        return {"status": "error", "message": "Engine not found"}

    # List files
    print("Engine files:")
    for f in os.listdir(ENGINE_DIR):
        fpath = f"{ENGINE_DIR}/{f}"
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1e6
            print(f"  {f}: {size:.1f} MB")

    # Check build info
    build_info_path = f"{ENGINE_DIR}/build_info.json"
    if os.path.exists(build_info_path):
        with open(build_info_path) as f:
            info = json.load(f)
        print(f"\nModel: {info['model_id']}")
        print(f"Max seq len: {info['builder_config']['max_seq_len']}")
        print(f"Max audio: ~{info['audio_capacity']['max_duration_s']:.0f}s")

    engine_path = f"{ENGINE_DIR}/rank0.engine"
    if os.path.exists(engine_path):
        size = os.path.getsize(engine_path) / 1e9
        return {"status": "success", "engine_size_gb": size}
    else:
        return {"status": "error", "message": "Engine file not found"}


@app.local_entrypoint()
def main(
    quantization: str = "fp16",
    max_input_len: int = 512,
    max_output_len: int = 12000,  # ~145 seconds of audio
    verify: bool = True,
):
    """Build TRT-LLM engine for Canopy Labs Hindi model."""
    print(f"Building Canopy Hindi TRT-LLM engine...")
    print(f"Model: {MODEL_ID}")
    print(f"Max input: {max_input_len}, Max output: {max_output_len}")
    print(f"Max seq len: {max_input_len + max_output_len}")

    result = build_engine.remote(
        quantization=quantization,
        max_input_len=max_input_len,
        max_output_len=max_output_len,
    )

    print("\nBuild result:")
    for k, v in result.items():
        print(f"  {k}: {v}")

    if verify and result["status"] == "success":
        print("\nVerifying engine...")
        verify_result = verify_engine.remote()
        print(f"Verification: {verify_result['status']}")
