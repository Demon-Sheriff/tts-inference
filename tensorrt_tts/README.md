# Orpheus TTS with TensorRT-LLM

High-performance Text-to-Speech inference using TensorRT-LLM for the Orpheus model.

## Overview

This pipeline converts the Orpheus-3B model (based on Llama 3.2-3B) to TensorRT-LLM format
for optimized inference on NVIDIA GPUs.

## Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                     Orpheus TRT-LLM Pipeline                    │
├─────────────────────────────────────────────────────────────────┤
│                                                                 │
│  1. Model Conversion (one-time)                                 │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│     │  HuggingFace │ -> │  TRT-LLM     │ -> │  TRT Engine  │    │
│     │  Checkpoint  │    │  Checkpoint  │    │  (.engine)   │    │
│     └──────────────┘    └──────────────┘    └──────────────┘    │
│                                                                 │
│  2. Inference                                                   │
│     ┌──────────────┐    ┌──────────────┐    ┌──────────────┐    │
│     │   Text +     │ -> │  TRT-LLM     │ -> │    SNAC      │    │
│     │   Voice      │    │  Generate    │    │   Decode     │    │
│     └──────────────┘    └──────────────┘    └──────────────┘    │
│                                    │                │           │
│                              Audio Tokens      Audio PCM        │
│                                                                 │
└─────────────────────────────────────────────────────────────────┘
```

## Files

- `build_engine.py` - Modal script to convert HF model -> TRT-LLM engine
- `inference.py` - Main inference pipeline with SNAC decoding
- `simple_test.py` - Simple test script to verify the pipeline

## Usage

### Step 1: Build the TensorRT Engine (one-time)

```bash
modal run build_engine.py
```

This will:
1. Download the Orpheus-3B model from HuggingFace
2. Convert to TRT-LLM checkpoint format
3. Build the TensorRT engine
4. Save to Modal volume for reuse

### Step 2: Run Inference

```bash
modal run inference.py --text "Hello, this is a test." --voice "tara"
```

Or deploy as a service:

```bash
modal serve inference.py
```

## Performance

Expected improvements over vLLM:
- 2-4x faster token generation
- Lower latency for first token
- Better GPU memory efficiency

## Requirements

- NVIDIA GPU with compute capability >= 8.0 (A10G, A100, H100)
- Modal account with GPU access
- ~20GB disk space for engine files

## Model Details

- **Base Model**: canopylabs/orpheus-3b-0.1-ft (English)
- **Architecture**: Llama 3.2-3B with custom audio tokens
- **Vocab Size**: 128261 (standard) + special audio tokens
- **Audio Codec**: SNAC 24kHz

## Special Tokens

| Token | ID | Description |
|-------|-----|-------------|
| SOS | 128257 | Start of speech |
| EOS | 128258 | End of speech |
| START | 128259 | Start of turn |
| END | 128260 | End of turn |

## References

- [TensorRT-LLM Documentation](https://nvidia.github.io/TensorRT-LLM/)
- [Orpheus TTS GitHub](https://github.com/canopyai/Orpheus-TTS)
- [Baseten TRT-LLM Workshop](https://github.com/basetenlabs/Workshop-TRT-LLM)
