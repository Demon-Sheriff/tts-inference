# Orpheus Hindi TTS: Technical Spec

## Stack
- **Inference**: TensorRT-LLM on Modal A100-80GB
- **Fallback**: vLLM (sync mode only - async engine unstable in v0.13)
- **Audio Codec**: SNAC 24kHz
- **Deployment**: Modal with warm containers (`@modal.cls` + `@modal.enter()`)

---

## The Problem: SNAC Backward-Looking Context

SNAC decoder uses **backward-looking context** - early audio samples change when more frames are added later:

```
decode(10_frames)[:N] ≠ decode(all_frames)[:N]
```

This breaks naive streaming approaches:
- **Chunked decode**: Clicks/pops at boundaries
- **Crossfade overlap**: Still audible artifacts
- **Root cause**: Prefix samples are unstable until future context exists

---

## The Breakthrough: LookaheadStreamingDecoder

**Solution**: Only emit samples that have enough "future" context to be stable.

### Algorithm
1. Buffer ALL audio tokens received
2. When N new complete frames arrive:
   - Decode ALL frames from frame 0 (context-preserving)
   - Only emit samples with >= 5 frames of future context
   - Track `samples_emitted` to avoid re-emission
3. On EOS: emit all remaining samples (now stable)

### Lookahead Sweep Results
| Lookahead | MSE | Correlation | Notes |
|-----------|-----|-------------|-------|
| 0 frames | 3.66e-04 | 0.977 | Artifacts |
| **5 frames** | **1.62e-05** | **0.999** | Sweet spot |
| 10 frames | 1.58e-05 | 0.999 | Diminishing returns |

**Trade-off**: ~430ms additional latency for 99.9% quality match with batch decode.

---

## Performance Metrics (A100-80GB)

| Metric | TensorRT-LLM | vLLM |
|--------|--------------|------|
| Tokens/sec | 129-160 | 50-100 |
| TTFA (warm) | **1.5s** | 4-30s |
| RTF (streaming) | **1.4-1.5x** | 0.6-0.8x |
| Cold start | 45-82s | 120-150s |
| Quality | 0.998+ corr | Same |

### Benchmark Summary (22-prompt sweep)
- Mean TTFA: **1525ms**
- Mean RTF: **1.42x realtime**
- Mean chunk gap: **254ms**
- Max chunk gap: **714ms** (worst case)
- Cold vs warm ratio: **1.005x** (negligible difference after warmup)

---

## Why vLLM Failed

vLLM v0.13 introduced V1 architecture with `AsyncLLMEngine`:
- Engine core process dies unexpectedly mid-generation
- No graceful recovery - connections hang
- Async streaming unreliable for production

**Workaround**: Sync generation with `ThreadPoolExecutor` works but limits throughput.

---

## Current Architecture

```
WebSocket Client
       │
       ▼
Modal Container (warm, pre-loaded)
       │
       ├── TensorRT-LLM Engine
       │       └── Streaming token generation
       │
       └── LookaheadStreamingDecoder
               └── SNAC decode with 5-frame lookahead
                       │
                       ▼
               Binary PCM chunks (int16 @ 24kHz)
```

### Warm Container Strategy
```python
@app.cls(gpu="A100", scaledown_window=300)
class TTSEngine:
    @modal.enter()
    def load_models(self):
        self.llm = load_trt_llm()   # Load once
        self.snac = load_snac()
```

Result: **20x latency reduction** (52s → 2.5s)

---

## Token Constants
```python
SOS_TOKEN = 128257   # Start of Speech
EOS_TOKEN = 128258   # End of Speech
TOKEN_BASE = 128266  # Audio token offset
FRAME_SIZE = 7       # Tokens per SNAC frame
```

---

## Plots

See `tensorrt_tts/hindi_finetuned/plot_metrics/output/`:
- `sweep_analysis.png` - TTFA & RTF vs text length
- `gap_analysis.png` - Chunk gap distribution
- `per_prompt_breakdown.png` - Per-prompt metrics

W&B: https://wandb.ai/andy404-bits-pilani/orpheus-tts-benchmark

---

## Current Limits
- Concurrent users: ~5 (single container)
- Text: 2000 chars max
- Audio: 120s max duration

## Next Steps
- Scale to ~100 concurrent users
- Queue management with position visibility
- API key auth + rate limiting
