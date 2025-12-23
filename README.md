# Orpheus Hindi Tag-Aware Text Generation

Fine-tuned CanopyLabs 3B Hindi model for generating text with Orpheus-style emotional/prosody tags.

## 🎯 Project Overview

**Model**: `canopylabs/3b-hi-ft-research_release` + LoRA adapter (`Andy004/canopy-3b-hi-elise-finetune`)  
**Dataset**: `rumik-ai/hi-elise` (805 samples, 31 unique tags)  
**Deployment**: Modal (vLLM on A100)

### Model Capabilities

✅ **Fluent Hindi Generation**: Coherent text continuation  
✅ **Tag Recognition**: Respects tags like `<laugh>`, `<sigh>`, `<gasps>` in prompts  
✅ **Fast Inference**: ~100ms TTFT, ~46 TPS  
✅ **WebSocket Streaming**: Real-time token streaming

❌ **Limitation**: Cannot autonomously insert tags (requires tags in input prompt)

## 📊 Performance Metrics

| Metric | Value |
|--------|-------|
| TTFT (Average) | 100ms |
| TPS (Highest) | 46 tokens/s |
| Coherent Output Rate | 78% |
| Deployment | A100 GPU |

*View benchmarks*: [WandB Dashboard](https://wandb.ai/andy404-bits-pilani/orpheus-benchmark)

## 🚀 Quick Start

### Prerequisites

```bash
pip install modal
modal setup
modal secret create huggingface-secret HF_TOKEN=<your-token>
modal secret create api-keys WANDB_API_KEY=<your-key>
```

### 1. Deploy vLLM Server

```bash
# Merge LoRA adapter (one-time setup)
modal run hindi_finetuned/modal_merge.py

# Start vLLM inference server
modal serve hindi_finetuned/modal_inference_vllm.py
```

### 2. Test Inference

```python
from hindi_finetuned.modal_inference_vllm import ModelInference

prompts = [
    "नमस्ते <laugh> कैसे हो?",
    "मुझे विश्वास नहीं हो रहा। <sigh>",
]

results = ModelInference().generate.remote(prompts)
for p, r in zip(prompts, results):
    print(f"{p} → {r}")
```

### 3. WebSocket Streaming

**Start Server:**
```bash
modal serve hindi_finetuned/modal_inference_stream.py
```

**Test Client:**
```python
import asyncio
import websockets

async def test_stream():
    uri = "ws://localhost:8000/ws"  # Replace with Modal URL
    async with websockets.connect(uri) as websocket:
        await websocket.send("नमस्ते <laugh> कैसे हो?")
        
        async for message in websocket:
            print(message, end='', flush=True)

asyncio.run(test_stream())
```

Or use the provided test script:
```bash
python hindi_finetuned/test_stream.py
```

### 4. Measure TTFB (Time-To-First-Byte)

Test end-to-end latency including network overhead:

```bash
# Start WebSocket server first
modal serve hindi_finetuned/modal_inference_stream.py

# In another terminal, run TTFB test
python hindi_finetuned/test_ttfb.py
```

This measures:
- **TTFB**: Network latency + time to first token
- **Total Response Time**: Complete generation duration
- **TPS**: Tokens per second over WebSocket

## 📁 Project Structure

```
hindi_finetuned/
├── modal_finetune.py           # Training script (LoRA + 4-bit)
├── modal_merge.py              # Merge adapter for vLLM
├── modal_inference_vllm.py     # High-performance vLLM server
├── modal_inference_stream.py   # WebSocket streaming API
├── modal_benchmark_vllm.py     # Performance benchmarks
├── modal_test_vllm_final.py   # Final validation test
└── test_stream.py              # WebSocket client example

pretrained_base/
└── modal_finetune_base.py     # Comparative experiment (poor results)
```

## 🔬 Research Findings

### Base Model Comparison

We tested two base models:

1. **✅ Research-FT** (`3b-hi-ft-research_release`): 
   - Coherent Hindi generation
   - Proper grammar and context understanding
   - **Selected for deployment**

2. **❌ Pretrained** (`3b-hi-pretrain-research_release`):
   - Garbled/incoherent output after fine-tuning
   - Poor language modeling
   - **Conclusion**: Instruction-tuned base is crucial

### Tag Insertion Experiments

**Attempt**: Instruction fine-tuning to teach autonomous tag insertion  
**Result**: Failed - model generated corrupted Unicode instead of coherent Hindi  
**Why**: The model learned tag *continuation*, not tag *prediction*  
**Implication**: Tags must be provided in input prompts

## 🛠️ Advanced Usage

### Custom Generation Parameters

```python
from vllm import SamplingParams

sampling_params = SamplingParams(
    temperature=0.7,
    max_tokens=150,
    top_p=0.9,
    repetition_penalty=1.2
)

results = llm.generate(prompts, sampling_params)
```

### Available Tags

`<laugh>`, `<sigh>`, `<gasps>`, `<giggles>`, `<chuckles>`, `<whispers>`, `<romantic music playing>`, `<smooches>`, `<crying>`, and 22 more.

See full list: [Dataset Analysis](./inspect_dataset.py)

## 📈 Benchmarking

Run comprehensive tests:
```bash
# vLLM performance test
modal run hindi_finetuned/modal_test_vllm_final.py

# Compare 3 model variants
modal run hindi_finetuned/modal_benchmark.py
```

## 🐛 Troubleshooting

**Issue**: vLLM fails to start on T4  
**Solution**: Use A100 GPU (set in Modal function decorator)

**Issue**: Tags not appearing in output  
**Solution**: Ensure tags are present in input prompt (model doesn't insert autonomously)

**Issue**: Embedding size mismatch  
**Solution**: Run `model.resize_token_embeddings(len(tokenizer))` before loading adapter

## 📝 Citation

```bibtex
@misc{canopylabs-orpheus-hindi,
  title={Fine-tuned Hindi Text Generation with Orpheus Tags},
  author={Andy004},
  year={2025},
  publisher={Hugging Face},
  url={https://huggingface.co/Andy004/canopy-3b-hi-elise-finetune}
}
```

## 📄 License

Model weights: Follow [CanopyLabs licensing](https://huggingface.co/canopylabs)  
Code: MIT License
