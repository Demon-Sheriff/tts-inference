"""
TensorRT-LLM Engine Builder for A100 GPU

Builds on A100 (40GB/80GB) for:
- More VRAM to hold both TRT-LLM + SNAC simultaneously
- No need for memory management between generation and decoding
- Faster generation throughput

Usage:
    modal run tensorrt_tts/hindi_canopy/build_engine_a100.py
"""

import modal
import os

app = modal.App("orpheus-hindi-canopy-trtllm-build-a100")

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

# Separate volume for A100 engine (different GPU architecture)
engine_vol = modal.Volume.from_name("orpheus-hindi-canopy-engine-a100", create_if_missing=True)
cache_vol = modal.Volume.from_name("orpheus-cache", create_if_missing=True)

MODEL_ID = "canopylabs/3b-hi-pretrain-research_release"
MODEL_DIR = "/cache/canopy-hindi-3b"
ENGINE_DIR = "/engine/trt_engine"


@app.function(
    image=image,
    gpu="A100",  # A100 40GB - much more VRAM
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
    max_output_len: int = 12000,
    max_batch_size: int = 1,
):
    """Build TensorRT-LLM engine on A100."""
    import shutil
    import json
    from huggingface_hub import snapshot_download

    print("=" * 60)
    print("BUILDING A100 ENGINE")
    print("=" * 60)
    print(f"Model: {MODEL_ID}")
    print(f"GPU: A100 (40GB)")

    # Download model
    if not os.path.exists(f"{MODEL_DIR}/config.json"):
        print(f"Downloading {MODEL_ID}...")
        snapshot_download(
            repo_id=MODEL_ID,
            local_dir=MODEL_DIR,
            local_dir_use_symlinks=False,
            token=os.environ.get("HF_TOKEN"),
        )
        cache_vol.commit()
    else:
        print("Model already cached")

    print("\n" + "=" * 60)
    print("Building TRT-LLM Engine for A100")
    print("=" * 60)

    if os.path.exists(ENGINE_DIR):
        shutil.rmtree(ENGINE_DIR)
    os.makedirs(ENGINE_DIR, exist_ok=True)

    dtype = "float16" if quantization == "fp16" else "bfloat16"
    print(f"dtype={dtype}, max_seq_len={max_input_len + max_output_len}")

    from tensorrt_llm import LLM, BuildConfig

    build_config = BuildConfig(
        max_input_len=max_input_len,
        max_seq_len=max_input_len + max_output_len,
        max_batch_size=max_batch_size,
        max_num_tokens=max_input_len + max_output_len,
    )

    print(f"\nLoading model from {MODEL_ID}...")
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
        "gpu": "A100",
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
    print("A100 Engine build complete!")
    print("=" * 60)

    engine_vol.commit()

    total_size = 0
    print("\nEngine files:")
    for f in os.listdir(ENGINE_DIR):
        fpath = f"{ENGINE_DIR}/{f}"
        if os.path.isfile(fpath):
            size = os.path.getsize(fpath)
            total_size += size
            print(f"  {f}: {size / 1e9:.2f} GB")

    print(f"\nTotal: {total_size / 1e9:.2f} GB")
    print(f"Max audio: ~{build_info['audio_capacity']['max_duration_s']:.0f}s")

    return {
        "status": "success",
        "gpu": "A100",
        "engine_dir": ENGINE_DIR,
        "total_size_gb": total_size / 1e9,
        "max_audio_s": build_info['audio_capacity']['max_duration_s'],
    }


@app.local_entrypoint()
def main():
    """Build A100 TRT-LLM engine."""
    print("Building A100 TRT-LLM engine for Hindi Canopy model...")
    result = build_engine.remote()
    print("\nResult:")
    for k, v in result.items():
        print(f"  {k}: {v}")
