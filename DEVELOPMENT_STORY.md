# Orpheus Hindi TTS Pipeline: Development Story

## The Journey from Batch Inference to Production Streaming

---

## Chapter 1: The Beginning - vLLM Batch Inference

### Starting Point
- **Goal**: Build a Hindi Text-to-Speech system using Orpheus 3B model
- **Initial approach**: vLLM for LLM inference + SNAC for audio decoding
- **Model**: `canopylabs/3b-hi-pretrain-research_release` (Hindi pretrained)

### First Implementation (Batch Mode)
```
Text → Tokenize → vLLM Generate (wait for ALL tokens) → SNAC Decode → WAV
```

- Generated all tokens in one batch call
- Decoded entire audio sequence at once
- **Problem**: User waits 5-30 seconds before hearing anything

### Key Discovery: Token Format
- Orpheus uses special tokens for audio generation:
  - `SOS_TOKEN (128257)` - Start of Speech, audio begins after this
  - `EOS_TOKEN (128258)` - End of Speech, stop generation
  - `START_TOKEN (128259)` - Start of human turn
  - `TOKEN_BASE (128266)` - Offset for audio codes
- Audio tokens come in 7-token frames (interleaved SNAC layers)

---

## Chapter 2: First Streaming Attempts with vLLM

### The Vision
- Stream audio chunks to user as tokens are generated
- Reduce perceived latency from seconds to milliseconds

### Attempt 1: Naive Chunked Decode
```python
# Every N frames, decode and send
if len(buffer) >= 7 * FRAMES_PER_CHUNK:
    audio = snac.decode(buffer)
    yield audio
    buffer = []  # Reset buffer
```

**Result**: Audio had clicks/pops at chunk boundaries

### Attempt 2: Crossfade Between Chunks
```python
# Overlap chunks with crossfade
fade_out = np.linspace(1, 0, crossfade_samples)
fade_in = np.linspace(0, 1, crossfade_samples)
overlap = prev_chunk[-N:] * fade_out + new_chunk[:N] * fade_in
```

**Result**: Better, but still audible artifacts

### The SNAC Problem Discovered
- SNAC uses **backward-looking context**
- Early samples CHANGE when you add more frames later
- `decode(10_frames)[:N] ≠ decode(all_frames)[:N]`
- The prefix is unstable until future context exists

### vLLM Async Engine Issues
- vLLM v0.13 introduced new V1 architecture
- `AsyncLLMEngine` became unstable - engine core process dies unexpectedly
- Needed alternative approach

---

## Chapter 3: Pivot to TensorRT-LLM

### Why TensorRT-LLM?
- Native streaming support via `generate_async(streaming=True)`
- Higher throughput (140-160 TPS vs 50-100 TPS on vLLM)
- Better GPU utilization on A100
- More mature streaming API

### Engine Build Process
- Converted Orpheus model to TensorRT-LLM format
- Built optimized engine for A100-80GB
- Configuration: FP16, max_seq_len=12512, paged KV cache

### Initial TensorRT Results
| Metric | Value |
|--------|-------|
| TPS | 140-160 |
| Cold Start | 45-82s |
| SNAC Decode RTF | 100-285x realtime |

---

## Chapter 4: Solving the SNAC Streaming Problem

### The Breakthrough: Lookahead Buffering

**Key Insight**: Only emit samples that have enough "future" context to be stable

### Experimentation
Tested different lookahead values:

| Lookahead | MSE | Correlation | Notes |
|-----------|-----|-------------|-------|
| 0 frames | 3.66e-04 | 0.977 | Noticeable artifacts |
| 5 frames | 1.62e-05 | 0.999 | **Sweet spot** |
| 10 frames | 1.58e-05 | 0.999 | Diminishing returns |
| 20 frames | 1.32e-05 | 0.999 | Too much latency |

### The LookaheadStreamingDecoder Algorithm
```
1. Generate tokens via TRT-LLM streaming
2. Buffer ALL audio tokens received
3. When N new complete frames arrive:
   a. Decode ALL frames from the beginning (context-preserving)
   b. Only emit samples with >= 5 frames of future context
   c. Track samples_emitted to avoid re-emitting
4. On EOS: emit all remaining samples (now stable with full context)
```

### Why This Works
- SNAC's backward context affects ~5 frames (~430ms)
- By keeping 5 frames of "lookahead", we only emit stable audio
- Trade-off: ~430ms additional latency for 99.9% quality match

### Quality Verification
| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| MSE | 1.59e-05 | < 1e-03 | ✓ |
| Max diff | 0.086 | < 0.5 | ✓ |
| Correlation | 0.9987 | > 0.998 | ✓ |
| Std ratio | 0.998 | > 0.95 | ✓ |

---

## Chapter 5: Warm Containers - Eliminating Cold Start

### The Problem
- Cold start: 45-82 seconds (model loading)
- Unacceptable for real-time applications

### Solution: Modal `@modal.cls` with `@modal.enter()`
```python
@app.cls(gpu="A100", scaledown_window=300)
class TTSEngine:
    @modal.enter()
    def load_models(self):
        # Load once at container start
        self.llm = load_trt_llm()
        self.snac = load_snac()

    @modal.method()
    def generate(self, text):
        # Models already loaded!
        return self._generate(text)
```

### Results
| Request | Time | Notes |
|---------|------|-------|
| First (cold) | ~52s | Includes model loading |
| Second (warm) | ~2.5s | **20x faster** |
| Third (warm) | ~2.1s | Consistent |

---

## Chapter 6: WebSocket Server for Production

### Architecture
```
Browser/Client
     │
     │ WebSocket (wss://)
     ▼
Modal WebSocket Server
     │
     ├── Warm Container (pre-loaded models)
     ├── TRT-LLM Streaming Generation
     ├── LookaheadStreamingDecoder
     └── Binary Audio Chunks (int16 PCM @ 24kHz)
```

### Protocol Design
```
1. Client connects to WebSocket
2. Client sends JSON: {"text": "...", "voice": "tara"}
3. Server sends: {"status": "started", "sample_rate": 24000}
4. Server streams binary audio chunks
5. Server sends: {"done": true, "duration_s": X, "chunks": N}
```

### Features Implemented
- Built-in HTML test client at `/test`
- Benchmark mode with detailed server metrics
- Voice selection (tara, leah, jess, leo, dan, etc.)
- Temperature/top_p controls

---

## Chapter 7: Benchmarking & Metrics

### Metrics We Optimized For

1. **TTFA (Time to First Audio)** - Most critical for UX
   - Target: < 3 seconds
   - Achieved: 2.5-3.2s (TensorRT)

2. **RTF (Real-Time Factor)** - Must be > 1.0 for streaming
   - RTF = audio_duration / generation_time
   - Achieved: 1.0-1.3x (streaming), 1.5-1.7x (batch)

3. **TPS (Tokens Per Second)** - Throughput metric
   - Achieved: 140-160 TPS (TensorRT), 50-100 TPS (vLLM)

4. **Chunk Jitter** - Streaming smoothness
   - Target: < 200ms stddev
   - Achieved: ~100-150ms

5. **Audio Quality** - Correlation with batch decode
   - Target: > 0.998 correlation
   - Achieved: 0.9987

### Benchmark Tools Created
- `benchmark_websocket.py` - End-to-end WebSocket benchmarking
- `benchmark_with_wandb.py` - Metrics visualization + W&B logging
- `comprehensive_sweep.py` - 22 prompt length sweep with cold-start analysis

### W&B Dashboard
- Project: `orpheus-tts-benchmark`
- Logged: TTFA, RTF, chunk gaps, per-prompt breakdown
- Run: https://wandb.ai/andy404-bits-pilani/orpheus-tts-benchmark

---

## Chapter 8: Returning to vLLM (Fixed)

### Why Revisit vLLM?
- Not everyone has TensorRT-LLM setup
- More portable/easier deployment
- Development/testing flexibility

### The Fix: Sync vLLM with ThreadPoolExecutor
```python
# Instead of broken AsyncLLMEngine:
self.executor = ThreadPoolExecutor(max_workers=2)

# Generate in thread pool
token_ids = await loop.run_in_executor(
    self.executor,
    self.generate_tokens_sync,
    prompt_ids,
)
```

### Key Corrections Applied
1. **Prompt format**: Must match TensorRT exactly
   ```python
   [START_TOKEN] + tokenize("voice: text") + [END_OF_TEXT, END_OF_TURN]
   ```

2. **Same LookaheadStreamingDecoder** for SNAC handling

3. **Stable sync generation** instead of flaky async

### vLLM Results
| Metric | Value |
|--------|-------|
| TPS | 50-100 |
| Cold Start | 120-150s |
| RTF | 0.6-0.8x |
| Stability | Good (sync mode) |

---

## Chapter 9: Final Architecture

### TensorRT Pipeline (Production)
```
tensorrt_tts/hindi_finetuned/
├── websocket_server.py         # PRODUCTION SERVER
├── inference_a100_optimized.py # CLI with warm+streaming
├── build_engine_a100.py        # Engine builder
└── plot_metrics/               # Benchmarking tools
```

### vLLM Pipeline (Alternative)
```
vllm_inference/
├── websocket_server_sync.py    # STABLE sync server
├── simple_tts.py               # Batch TTS test
├── test_websocket_client.py    # Test client
└── out/                        # Audio outputs
```

### Performance Comparison

| Metric | TensorRT-LLM | vLLM |
|--------|--------------|------|
| Token Generation | 140-160 TPS | 50-100 TPS |
| TTFA (Streaming) | 2.5-3.2s | 4-30s |
| RTF (Streaming) | 1.0-1.3x | 0.6-0.8x |
| Cold Start | 45-82s | 120-150s |
| Audio Quality | 0.998+ | Same |

---

## Chapter 10: Key Learnings & References

### Critical Discoveries

1. **SNAC Backward Context**: The codec looks backward, requiring lookahead buffering for stable streaming

2. **Prompt Format Matters**: Wrong special tokens = no audio or garbage

3. **Warm Containers Essential**: 20x latency improvement for subsequent requests

4. **vLLM v0.13 Breaking Changes**: Async engine unstable, sync+threadpool works

5. **5-Frame Lookahead Sweet Spot**: Balances latency vs quality perfectly

### References Used

- **Orpheus Model**: canopylabs/3b-hi-pretrain-research_release
- **SNAC Codec**: hubertsiuzdak/snac_24khz
- **TensorRT-LLM**: v0.21.0
- **vLLM**: v0.13.0
- **Modal**: Cloud GPU infrastructure

### Token Constants (Critical Reference)
```python
SOS_TOKEN = 128257   # Start of Speech
EOS_TOKEN = 128258   # End of Speech
START_TOKEN = 128259 # Start of Human turn
END_OF_TEXT = 128009 # End of text content
END_OF_TURN = 128260 # End of turn
TOKEN_BASE = 128266  # Audio token offset
FRAME_SIZE = 7       # Tokens per SNAC frame
SAMPLE_RATE = 24000  # Audio sample rate
```

### SNAC Layer Redistribution
```python
# 7 tokens per frame, interleaved pattern:
# [L1, L2_a, L3_a, L3_b, L2_b, L3_c, L3_d]
#
# Redistribute with offsets:
layer_1.append(codes[base + 0])           # No offset
layer_2.append(codes[base + 1] - 4096)    # -4096
layer_2.append(codes[base + 4] - 4*4096)  # -16384
layer_3.append(codes[base + 2] - 2*4096)  # -8192
layer_3.append(codes[base + 3] - 3*4096)  # -12288
layer_3.append(codes[base + 5] - 5*4096)  # -20480
layer_3.append(codes[base + 6] - 6*4096)  # -24576
```

---

## Timeline

| Date | Milestone |
|------|-----------|
| Dec 12-14 | Initial vLLM batch inference |
| Dec 15-17 | First streaming attempts, discovered SNAC issues |
| Dec 18-19 | Pivot to TensorRT-LLM |
| Dec 20-21 | Token format debugging, prompt corrections |
| Dec 22 | SNAC backward-context discovery |
| Dec 23 | LookaheadStreamingDecoder implementation |
| Dec 23 | WebSocket server, warm containers |
| Dec 23 | Comprehensive benchmarking, W&B integration |
| Dec 24 | vLLM sync fix, directory reorganization |
| Dec 24 | Final documentation (PIPELINE_REPORT.md) |

---

## What We Built

### Production-Ready Components
1. **TensorRT WebSocket Server** - High-performance streaming TTS
2. **LookaheadStreamingDecoder** - SNAC-aware streaming with 99.9% quality
3. **Warm Container Architecture** - 20x latency reduction
4. **Comprehensive Benchmarking** - TTFA, RTF, jitter, W&B logging
5. **vLLM Alternative** - Portable fallback option

### Metrics Achieved
- **TTFA**: 2.5-3.2 seconds (from 5-30s batch)
- **RTF**: 1.0-1.3x realtime (sustainable streaming)
- **Quality**: 0.998+ correlation with batch
- **Warm Latency**: ~2.5s (from 45-82s cold)

### Files to Remember
```
tensorrt_tts/hindi_finetuned/websocket_server.py  # PRODUCTION
tensorrt_tts/hindi_finetuned/inference_a100_optimized.py  # CLI
vllm_inference/websocket_server_sync.py  # vLLM ALTERNATIVE
tensorrt_tts/PIPELINE_REPORT.md  # DOCUMENTATION
```

---

*Story compiled: 2024-12-24*
*Total development time: ~12 days*
*Lines of code written: ~5000+*
*Audio files generated: 100+*
*Cups of chai consumed: ∞*
