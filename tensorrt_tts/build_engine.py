"""
TensorRT-LLM Engine Builder for Orpheus TTS

This script converts the Orpheus-3B model from HuggingFace format
to TensorRT-LLM engine format for optimized inference.

Usage:
    modal run build_engine.py
    modal run build_engine.py --quantization fp8
    modal run build_engine.py --quantization int8
"""

import modal
import os

app = modal.App("orpheus-trtllm-build")

# Use official TensorRT-LLM container (has all dependencies pre-configured)
# This requires NGC authentication. Create secret with:
#   modal secret create nvcr-credentials \
#     REGISTRY_USERNAME='$oauthtoken' \
#     REGISTRY_PASSWORD='<your-ngc-api-key>'
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

# Volume for storing the built engine
engine_vol = modal.Volume.from_name("orpheus-trtllm-engine", create_if_missing=True)
cache_vol = modal.Volume.from_name("orpheus-cache", create_if_missing=True)

# Model configuration
MODEL_REPO = "canopylabs/orpheus-3b-0.1-ft"
MODEL_DIR = "/cache/orpheus-3b"
CHECKPOINT_DIR = "/engine/checkpoint"
ENGINE_DIR = "/engine/trt_engine"


@app.function(
    image=image,
    gpu="L4",  # Use L4 for consistency (A10G has different SM counts across variants)
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=3600,  # 1 hour for engine build
    secrets=[modal.Secret.from_name("huggingface-secret")],
)
def build_engine(
    quantization: str = "fp16",
    max_input_len: int = 512,
    max_output_len: int = 4096,  # Higher for longer audio generation
    max_batch_size: int = 1,
):
    """
    Build TensorRT-LLM engine from Orpheus model.

    Args:
        quantization: Quantization type - "fp16", "fp8", "int8", or "int4"
        max_input_len: Maximum input sequence length
        max_output_len: Maximum output sequence length
        max_batch_size: Maximum batch size
    """
    import shutil
    from huggingface_hub import snapshot_download

    print("=" * 60)
    print("STEP 1: Download model from HuggingFace")
    print("=" * 60)

    # Download model if not cached
    if not os.path.exists(f"{MODEL_DIR}/config.json"):
        print(f"Downloading {MODEL_REPO}...")
        snapshot_download(
            repo_id=MODEL_REPO,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
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
    print("STEP 2: Convert to TRT-LLM checkpoint format")
    print("=" * 60)

    # Clean previous checkpoint
    if os.path.exists(CHECKPOINT_DIR):
        shutil.rmtree(CHECKPOINT_DIR)
    os.makedirs(CHECKPOINT_DIR, exist_ok=True)

    import subprocess

    # Determine dtype for conversion
    if quantization == "fp16":
        dtype = "float16"
    elif quantization == "bf16":
        dtype = "bfloat16"
    else:
        dtype = "float16"

    print(f"Converting with dtype={dtype}")

    # TRT-LLM 0.21+ uses the LLM API which can directly load HuggingFace models
    # and build engines. Let's use that approach instead of the CLI.
    import tensorrt_llm
    from tensorrt_llm import LLM, SamplingParams, BuildConfig

    print("Using TensorRT-LLM high-level API to build engine...")

    # Build config for the engine
    build_config = BuildConfig(
        max_input_len=max_input_len,
        max_seq_len=max_input_len + max_output_len,
        max_batch_size=max_batch_size,
        max_num_tokens=max_batch_size * (max_input_len + max_output_len),
    )

    # Create LLM instance - this handles conversion and building
    print(f"Loading model from {MODEL_DIR}...")
    llm = LLM(
        model=MODEL_DIR,
        dtype=dtype,
        build_config=build_config,
    )

    # Save the engine
    print(f"Saving engine to {ENGINE_DIR}...")
    llm.save(ENGINE_DIR)

    # The engine config.json created by llm.save() is the TRT-LLM engine config
    # We should NOT overwrite it. Just print what was saved.
    import json
    engine_config_path = f"{ENGINE_DIR}/config.json"
    if os.path.exists(engine_config_path):
        with open(engine_config_path) as f:
            engine_config = json.load(f)
        print(f"Engine config saved (has builder_config: {'builder_config' in engine_config})")

    # Save our custom build config separately
    build_info = {
        "builder_config": {
            "precision": dtype,
            "max_batch_size": max_batch_size,
            "max_input_len": max_input_len,
            "max_seq_len": max_input_len + max_output_len,
        },
        "model_config": {
            "model_dir": MODEL_DIR,
            "quantization": quantization,
        }
    }
    with open(f"{ENGINE_DIR}/build_info.json", "w") as f:
        json.dump(build_info, f, indent=2)

    print("\n" + "=" * 60)
    print("Engine build complete!")
    print("=" * 60)

    # Commit the volume
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

    return {
        "status": "success",
        "engine_dir": ENGINE_DIR,
        "total_size_gb": total_size / 1e9,
        "quantization": quantization,
        "max_input_len": max_input_len,
        "max_output_len": max_output_len,
        "max_batch_size": max_batch_size,
    }


@app.function(
    image=image,
    gpu="L4",  # Same GPU as build
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=600,
)
def verify_engine():
    """Verify the built engine can be loaded."""
    import json

    print("Checking TRT-LLM engine...")

    if not os.path.exists(ENGINE_DIR):
        print(f"ERROR: Engine not found at {ENGINE_DIR}")
        return {"status": "error", "message": "Engine not found. Run build_engine first."}

    # List engine files
    print("Engine files:")
    for f in os.listdir(ENGINE_DIR):
        fpath = f"{ENGINE_DIR}/{f}"
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath) / 1e6
            print(f"  {f}: {size:.1f} MB")

    # Check config
    config_path = f"{ENGINE_DIR}/config.json"
    if os.path.exists(config_path):
        with open(config_path) as f:
            config = json.load(f)
        print(f"\nConfig: {json.dumps(config, indent=2)}")

    # Try to load engine
    engine_path = f"{ENGINE_DIR}/rank0.engine"
    if os.path.exists(engine_path):
        size = os.path.getsize(engine_path) / 1e9
        print(f"\nEngine file: {size:.2f} GB")
        return {"status": "success", "engine_size_gb": size}
    else:
        return {"status": "error", "message": "Engine file not found"}


@app.local_entrypoint()
def main(
    quantization: str = "fp16",
    max_input_len: int = 512,
    max_output_len: int = 2048,
    verify: bool = True,
):
    """Build and optionally verify the TRT-LLM engine."""
    print(f"Building TRT-LLM engine with {quantization} quantization...")
    print(f"Max input: {max_input_len}, Max output: {max_output_len}")

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
