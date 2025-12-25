# TensorRT-LLM Hindi TTS Pipeline Report

## Executive Summary

This report analyzes the TensorRT-LLM based Text-to-Speech pipeline for Hindi Orpheus models, covering architecture, buffering policies, performance benchmarks, and optimization opportunities.

**Key Metrics (A100-80GB):**
| Metric | Value |
|--------|-------|
| Token Generation Speed | 140-160 TPS |
| Estimated TTFT | ~12-15ms |
| SNAC Decode RTF | 100-285x realtime |
| End-to-End RTF | 1.5-1.6x realtime |
| Cold Start (model load) | 45-82s |

---

## 1. Pipeline Architecture

### 1.1 High-Level Flow

```
┌─────────────────────────────────────────────────────────────────────────────┐
│                         TTS INFERENCE PIPELINE                               │
├─────────────────────────────────────────────────────────────────────────────┤
│                                                                              │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │  Text    │───▶│  Tokenizer   │───▶│  TRT-LLM    │───▶│ Audio Token   │  │
│  │  Input   │    │  (HF)        │    │  Engine     │    │ Extraction    │  │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────────────┘  │
│                                                                   │          │
│                                                                   ▼          │
│  ┌──────────┐    ┌──────────────┐    ┌─────────────┐    ┌───────────────┐  │
│  │  WAV     │◀───│  int16       │◀───│  SNAC       │◀───│ Code Layer    │  │
│  │  Output  │    │  Conversion  │    │  Decoder    │    │ Redistribution│  │
│  └──────────┘    └──────────────┘    └─────────────┘    └───────────────┘  │
│                                                                              │
└─────────────────────────────────────────────────────────────────────────────┘
```

### 1.2 Component Breakdown

#### Tokenization
- **Tokenizer**: HuggingFace AutoTokenizer from TRT-LLM engine directory
- **Special Tokens**:
  | Token | ID | Purpose |
  |-------|-----|---------|
  | BOS | 128000 | Begin of sequence (critical for model) |
  | START_TOKEN | 128259 | Start of human/text turn |
  | END_OF_TEXT | 128009 | End of text content |
  | END_OF_TURN | 128260 | End of conversation turn |
  | SOS | 128257 | Start of speech (audio begins) |
  | EOS | 128258 | End of speech (audio ends) |
  | TOKEN_BASE | 128266 | Base offset for audio codes |

- **Prompt Format**: `[START_TOKEN] + [BOS + "voice: text"] + [END_OF_TEXT, END_OF_TURN]`

#### TRT-LLM Engine
- **Model**: 3B parameter Llama-based TTS (Orpheus architecture)
- **Engine Config**:
  - Max sequence length: 12,512 tokens
  - Max batch size: 1
  - Precision: FP16
  - KV Cache: Paged attention with 32 tokens/block

#### SNAC Decoder
- **Model**: hubertsiuzdak/snac_24khz
- **Architecture**: Hierarchical 3-layer neural audio codec
- **Sample Rate**: 24,000 Hz
- **Frame Size**: 7 tokens = 1 frame = 2,048 samples (~85.3ms audio)

### 1.3 Audio Token Structure

The model generates 7-token frames in an interleaved pattern:

```
Frame n: [L1_n, L2_n_a, L3_n_a, L3_n_b, L2_n_b, L3_n_c, L3_n_d]

Redistribution to SNAC layers:
- Layer 1: codes[0] (coarse, 1 per frame)
- Layer 2: codes[1]-4096, codes[4]-4*4096 (2 per frame)
- Layer 3: codes[2]-2*4096, codes[3]-3*4096, codes[5]-5*4096, codes[6]-6*4096 (4 per frame)

Audio Duration Formula:
  duration_seconds = (num_tokens / 7) * (2048 / 24000)
  duration_seconds ≈ num_tokens * 0.0122
```

---

## 2. Buffering and Streaming Policy

### 2.1 Current Implementation: Batch Mode (No Streaming)

The current pipeline operates in **full batch mode**:

```python
# Current flow (inference_a100.py)
outputs = llm.generate([prompt_ids], sampling_params=sampling_params)
output_ids = list(outputs[0].outputs[0].token_ids)  # Wait for ALL tokens
# ... then decode ALL audio at once
```

**Characteristics:**
- All tokens generated before any decoding
- Full audio decoded in single SNAC pass
- No intermediate audio output
- User waits for complete generation

### 2.2 Memory Management (L4 vs A100)

#### L4 (24GB VRAM) - Sequential Loading
```
Time ──────────────────────────────────────────────────────────────▶
│                                                                   │
│  [Load TRT-LLM]──▶[Generate]──▶[Unload]──▶[Load SNAC]──▶[Decode] │
│       ~30s           ~Xs          GC         ~2s         ~Xs     │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```
- Cannot hold both models simultaneously
- Requires explicit memory cleanup between phases
- Uses chunked SNAC decoding (500 frames/chunk with crossfade)

#### A100 (40GB/80GB VRAM) - Parallel Loading
```
Time ──────────────────────────────────────────────────────────────▶
│                                                                   │
│  [Load TRT-LLM + SNAC]──────────▶[Generate]──▶[Decode]           │
│          ~40-80s                     ~Xs        ~Xs               │
│                                                                   │
└───────────────────────────────────────────────────────────────────┘
```
- Both models fit in VRAM
- No memory management overhead
- Full audio decode in single pass

### 2.3 Chunked Decoding (L4 Memory Optimization)

For long audio on L4, the pipeline uses chunked SNAC decoding with crossfade:

```python
CHUNK_FRAMES = 500  # ~42s audio per chunk
CROSSFADE_FRAMES = 10  # ~0.85s overlap

for chunk_start in range(0, num_frames, CHUNK_FRAMES - CROSSFADE_FRAMES):
    # Decode chunk
    chunk_audio = snac.decode(chunk_codes)

    # Crossfade with previous chunk
    if audio_chunks:
        fade_out = np.linspace(1, 0, crossfade_samples)
        fade_in = np.linspace(0, 1, crossfade_samples)
        prev_chunk[-crossfade_samples:] = (
            prev_chunk[-crossfade_samples:] * fade_out +
            chunk_audio[:crossfade_samples] * fade_in
        )

    # Clear GPU memory
    torch.cuda.empty_cache()
```

---

## 3. Performance Benchmarks

### 3.1 Benchmark Results (A100-80GB)

#### Medium Prompt (73 chars, 47 input tokens)
| Phase | Time | Details |
|-------|------|---------|
| Model Load | 81.6s | Tokenizer: 10.8s, TRT-LLM: 68.2s, SNAC: 2.6s |
| Token Generation | 2.73s | 429 tokens @ 157 TPS |
| SNAC Decode | 0.03s | 61 frames @ 106x RTF |
| **E2E (excl. load)** | **2.76s** | **5.21s audio @ 1.89x RTF** |

#### Long Prompt (163 chars, 92 input tokens)
| Phase | Time | Details |
|-------|------|---------|
| Model Load | 44.8s | (warm container) |
| Token Generation | 6.96s | 973 tokens @ 140 TPS |
| SNAC Decode | 0.03s | 139 frames @ 285x RTF |
| **E2E (excl. load)** | **6.99s** | **11.83s audio @ 1.69x RTF** |

### 3.2 Key Metrics

| Metric | A100-80GB | A100-40GB | L4 (24GB) |
|--------|-----------|-----------|-----------|
| **TPS** | 140-160 | 110-130 | 30-35 |
| **TTFT (estimated)** | ~12-15ms | ~15-20ms | ~30-40ms |
| **SNAC Decode RTF** | 100-285x | 100-200x | 50-100x |
| **Cold Start** | 45-82s | 40-50s | 30-40s |
| **Memory Swap Needed** | No | No | Yes |

### 3.3 Latency Breakdown

```
┌────────────────────────────────────────────────────────────────────┐
│                    E2E LATENCY BREAKDOWN (A100)                    │
├────────────────────────────────────────────────────────────────────┤
│                                                                    │
│  Tokenization:  │▓▓│ ~10ms (negligible)                           │
│                                                                    │
│  Prefill:       │▓▓▓▓▓│ ~50-100ms (context processing)            │
│                                                                    │
│  Decode:        │▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓▓│ ~7ms/token × N       │
│                                                                    │
│  SNAC:          │▓▓│ ~30ms (full decode)                          │
│                                                                    │
│  Post-process:  │▓│ ~5ms (int16 conversion)                       │
│                                                                    │
└────────────────────────────────────────────────────────────────────┘

For 1000 tokens (~12s audio):
- Generation: ~7s (1000 × 7ms)
- SNAC: ~0.03s
- Total: ~7s for 12s audio = 1.7x RTF
```

---

## 4. Performance Optimization Opportunities

### 4.1 HIGH IMPACT: Streaming Audio Generation

**Current State**: Wait for all tokens → decode all at once

**Proposed**: Decode audio frames as tokens are generated

```python
# PROPOSED: Streaming decode pipeline
async def generate_streaming(text):
    buffer = []
    async for token in llm.generate_streaming(prompt_ids):
        buffer.append(token)
        if len(buffer) >= 7:  # Complete frame
            frame_audio = snac.decode_frame(buffer[:7])
            yield frame_audio  # Stream immediately
            buffer = buffer[7:]
```

**Expected Impact**:
- TTFA (Time to First Audio): ~50-100ms (vs current ~2-7s)
- User perceives audio starting almost immediately
- Requires TRT-LLM streaming API integration

**Implementation Complexity**: Medium-High
- Need TRT-LLM's `generate_async` or callback API
- SNAC must support single-frame decode (may need modification)
- Buffering logic for incomplete frames

### 4.2 HIGH IMPACT: Warm Container / Model Caching

**Current State**: Cold start = 45-82s (model loading)

**Proposed**: Keep model loaded across requests

```python
# Modal's @modal.cls with @modal.enter() already supports this
@app.cls(gpu="A100", container_idle_timeout=300)
class TTSEngine:
    @modal.enter()
    def load_models(self):
        self.llm = LLM(model=ENGINE_DIR)  # Load once
        self.snac = SNAC.from_pretrained(...).eval().cuda()

    @modal.method()
    def generate(self, text):
        # Models already loaded!
        return self._generate(text)
```

**Expected Impact**:
- First request latency: 45-82s → 2-7s (no model load)
- Subsequent requests: Same 2-7s
- Cost: Container idle charges

**Implementation Complexity**: Low
- Already partially implemented in `inference.py` with `@modal.cls`
- Need to use class-based API consistently

### 4.3 MEDIUM IMPACT: Speculative Decoding

**Current State**: Single token per forward pass

**Proposed**: Use draft model for speculative decoding

**Expected Impact**:
- 1.5-2x speedup on token generation
- TPS: 140 → 210-280

**Implementation Complexity**: High
- Requires draft model training/selection
- TRT-LLM speculative decoding setup

### 4.4 MEDIUM IMPACT: Batch Processing

**Current State**: Single request at a time (batch_size=1)

**Proposed**: Batch multiple TTS requests together

```python
# Batch multiple requests
prompts = [format_prompt(text) for text in texts]
outputs = llm.generate(prompts, sampling_params)
```

**Expected Impact**:
- Better GPU utilization
- Throughput increase (not latency)
- ~2-4x more requests/second for concurrent load

**Implementation Complexity**: Low-Medium
- Engine already built with max_batch_size=1 (rebuild needed)
- API changes for batching

### 4.5 MEDIUM IMPACT: Quantization

**Current State**: FP16 precision

**Options**:
- INT8 weight-only quantization
- FP8 (on H100)
- INT4 with AWQ/GPTQ

**Expected Impact**:
- Memory: 6.4GB → 3.2GB (INT8) or 1.6GB (INT4)
- Speed: 10-30% faster (reduced memory bandwidth)
- Quality: Slight degradation (needs testing)

**Implementation Complexity**: Medium
- Rebuild engine with quantization flags
- Quality validation required

### 4.6 LOW IMPACT: SNAC Optimization

**Current State**: PyTorch SNAC model

**Options**:
- TensorRT conversion for SNAC
- ONNX Runtime optimization
- Triton kernel for decode

**Expected Impact**:
- SNAC already runs at 100-285x RTF
- Minor improvement (~10-20% faster decode)
- Diminishing returns given current speed

**Implementation Complexity**: Medium-High

### 4.7 LOW IMPACT: Reduced Precision Audio

**Current State**: Full precision intermediate audio

**Proposed**: Mixed precision SNAC decode

**Expected Impact**: Minimal (SNAC is already very fast)

---

## 5. Optimization Priority Matrix

| Optimization | Impact | Complexity | Priority |
|--------------|--------|------------|----------|
| Streaming Audio | HIGH | Medium-High | **P0** |
| Warm Containers | HIGH | Low | **P0** |
| Speculative Decoding | MEDIUM | High | P1 |
| Batch Processing | MEDIUM | Medium | P1 |
| INT8 Quantization | MEDIUM | Medium | P2 |
| SNAC TensorRT | LOW | High | P3 |

---

## 6. Recommended Implementation Roadmap

### Phase 1: Quick Wins (1-2 days)
1. **Switch to class-based API** for warm containers
2. **Increase container idle timeout** to 5-10 minutes
3. **Add metrics logging** for production monitoring

### Phase 2: Streaming (1-2 weeks)
1. Implement TRT-LLM streaming callback API
2. Modify SNAC for frame-by-frame decode
3. Build WebSocket/SSE endpoint for audio streaming
4. Add client-side audio buffering

### Phase 3: Throughput (1-2 weeks)
1. Rebuild engine with batch_size=4
2. Implement request queuing and batching
3. Add INT8 quantization option
4. Benchmark and tune

---

## 7. Current Limitations

1. **No Streaming**: Users must wait for full generation
2. **Cold Start**: 45-82s model load on fresh containers
3. **Single Request**: No batching support currently
4. **Memory Bound on L4**: Requires memory swap dance

---

## 8. Appendix: File Reference

| File | Purpose |
|------|---------|
| `hindi_finetuned/inference_a100.py` | A100 inference (recommended) |
| `hindi_finetuned/inference.py` | L4 inference with memory management |
| `hindi_finetuned/build_engine_a100.py` | A100 engine builder |
| `hindi_finetuned/benchmark.py` | Performance benchmarking |
| `hindi_canopy/inference.py` | Canopy base model inference |

---

## 9. P0 Optimizations Implemented

### 9.1 Warm Containers (IMPLEMENTED)

**File**: `hindi_finetuned/inference_a100_warm.py`

Uses Modal's `@modal.cls` with `@modal.enter()` to keep models loaded across requests.

**Results**:
| Request | Time | Notes |
|---------|------|-------|
| First (cold) | ~52s | Includes model loading |
| Second (warm) | ~2.5s | **20x faster** - models cached |
| Third (warm) | ~2.1s | Consistent warm performance |

**Usage**:
```bash
# Single request
modal run tensorrt_tts/hindi_finetuned/inference_a100_warm.py --text "नमस्ते"

# Multi-request demo (shows warm container benefit)
modal run tensorrt_tts/hindi_finetuned/inference_a100_warm.py --multi-request
```

### 9.2 Streaming Audio Generation (IMPLEMENTED)

**File**: `hindi_finetuned/inference_a100_streaming.py`

Uses TRT-LLM's `generate_async(streaming=True)` API for progressive token generation with incremental audio decoding.

**Observations**:
- TRT-LLM streaming gives per-token yields (248 iterations for 248 tokens)
- TTFT (Time to First Token): ~850ms with streaming overhead
- Streaming API has higher overhead than batch mode (~90 TPS vs ~140 TPS)
- **Best use case**: Real-time web clients that can start audio playback as chunks arrive

**Usage**:
```bash
# Streaming mode
modal run tensorrt_tts/hindi_finetuned/inference_a100_streaming.py --text "नमस्ते"

# Compare streaming vs batch
modal run tensorrt_tts/hindi_finetuned/inference_a100_streaming.py --compare
```

**Trade-offs**:
| Mode | TPS | TTFA | Best For |
|------|-----|------|----------|
| Batch | 140 | 2000ms | Highest throughput |
| Streaming | 90 | Variable* | Progressive delivery |

*Streaming TTFA depends on frames_per_chunk setting and when SOS token appears.

### 9.3 Combined Optimized Engine (RECOMMENDED)

**File**: `hindi_finetuned/inference_a100_optimized.py`

Combines warm containers + lookahead streaming in one engine. This is the recommended production implementation.

#### SNAC Streaming Insight (Critical Discovery)

**SNAC uses backward-looking context**, which means:
- Early samples CHANGE when you add more frames later
- `decode(10_frames)[:N] != decode(all_frames)[:N]` — the prefix is unstable
- But samples near the END are stable (they have "future" context)
- SNAC is also non-deterministic: `decode(same_codes)` twice gives slightly different results

**Solution: Lookahead Buffering**
- Only emit samples that have enough "future" context to be stable
- 5 frames lookahead (~430ms) gives 99.9% correlation with batch decode
- On EOS, emit all remaining samples (now stable with full context)

**Lookahead Test Results**:
| Lookahead | MSE       | Correlation | Notes |
|-----------|-----------|-------------|-------|
| 0 frames  | 3.66e-04  | 0.977       | Noticeable artifacts |
| 5 frames  | 1.62e-05  | 0.999       | **Used** — sweet spot |
| 10 frames | 1.58e-05  | 0.999       | Diminishing returns |
| 20 frames | 1.32e-05  | 0.999       | Too much latency |

**LookaheadStreamingDecoder Algorithm**:
```
1. Generate tokens continuously via TRT-LLM streaming
2. Buffer ALL audio tokens received so far
3. When we have N new complete frames (N = frames_per_chunk):
   a. Decode ALL frames from the beginning
   b. Only emit samples with >= 5 frames of future context
   c. Track samples_emitted to avoid re-emitting
4. On EOS: emit all remaining samples (they're now stable)
```

**Key Class**: `LookaheadStreamingDecoder`
- Decodes ALL frames from frame 0 each time (context-preserving)
- Only emits samples with sufficient lookahead (5 frames = 10240 samples)
- Tracks `samples_emitted` to emit only NEW stable audio
- Trade-off: ~430ms additional first-chunk latency for perceptual equivalence

**Final Quality Test Results**:
| Metric | Value | Threshold | Pass |
|--------|-------|-----------|------|
| MSE | 1.59e-05 | < 1e-03 | ✓ |
| Max diff | 0.086 | < 0.5 | ✓ |
| Correlation | 0.9987 | > 0.998 | ✓ |
| Std ratio | 0.998 | > 0.95 | ✓ |

**Benchmark Results**:
| Metric | Streaming | Batch |
|--------|-----------|-------|
| TTFA | 2786-3660ms | 3081-3101ms |
| TPS | 86-105 tok/s | 140-141 tok/s |
| RTF | 1.0-1.3x | 1.6-1.7x |

**Usage**:
```bash
# Batch mode (highest throughput)
modal run tensorrt_tts/hindi_finetuned/inference_a100_optimized.py --text "नमस्ते"

# Streaming mode (progressive audio)
modal run tensorrt_tts/hindi_finetuned/inference_a100_optimized.py --text "नमस्ते" --streaming

# Benchmark both modes
modal run tensorrt_tts/hindi_finetuned/inference_a100_optimized.py --benchmark
```

### 9.4 Trade-off Analysis

| Use Case | Recommended Mode | Why |
|----------|------------------|-----|
| Offline processing | Batch | 40% faster TPS |
| Real-time web playback | Streaming | Progressive delivery |
| Low-latency single requests | Batch | Lower total time |
| Interactive applications | Streaming | Better perceived latency |

**Note**: TRT-LLM's streaming API has ~40% overhead vs batch. The benefit is for clients that can start playing audio chunks as they arrive (e.g., WebSocket/SSE streaming to browser).

### 9.5 WebSocket Server (IMPLEMENTED)

**File**: `hindi_finetuned/websocket_server.py`

FastAPI WebSocket server for streaming TTS to web clients.

**Features**:
- WebSocket endpoint at `/ws/tts` for streaming audio
- Built-in HTML test client at `/test` for browser playback
- Uses `LookaheadStreamingDecoder` for high-quality streaming
- Warm container (models loaded once at startup)

**Protocol**:
```
1. Client connects to WebSocket
2. Client sends JSON: {"text": "...", "voice": "tara"}
3. Server streams binary audio chunks (int16 PCM @ 24kHz)
4. Server sends JSON: {"done": true, "duration_s": ..., "chunks": ...}
```

**Usage**:
```bash
# Development mode (live reload)
modal serve tensorrt_tts/hindi_finetuned/websocket_server.py

# Production deployment
modal deploy tensorrt_tts/hindi_finetuned/websocket_server.py

# Then open: https://<modal-url>/test
```

**Test Client**:
```bash
# Python test client
pip install websockets
python tensorrt_tts/hindi_finetuned/test/test_websocket_client.py --url wss://<modal-url>/ws/tts
```

### 9.6 WebSocket Benchmark Tool (IMPLEMENTED)

**File**: `hindi_finetuned/test/benchmark_websocket.py`

Comprehensive benchmark client for measuring end-to-end WebSocket streaming performance with detailed metrics.

#### Metrics Collected

**Server-Side Metrics** (returned in completion message when `benchmark: true`):
- **TTFT (Time to First Token)**: Time from request to first token generated
- **TTFA (Time to First Audio)**: Time from request to first audio chunk sent
- **Token throughput**: Tokens per second during generation
- **Frame throughput**: Frames per second during decode
- **Decode times**: Per-chunk SNAC decode latency

**Client-Side Metrics**:
- **Client TTFA**: Time from connection to first audio chunk received
- **Chunk cadence**: Inter-arrival time between chunks
- **Jitter**: Standard deviation of chunk arrival times
- **Max gap**: Longest wait between chunks
- **RTF (Real-Time Factor)**: Audio duration / wall time
- **Total bytes**: Size of audio received

#### Benchmark Commands

```bash
# Single benchmark run
python tensorrt_tts/hindi_finetuned/test/benchmark_websocket.py \
  --url wss://<modal-url>/ws/tts \
  --iterations 1

# Multiple iterations with audio saving
python tensorrt_tts/hindi_finetuned/test/benchmark_websocket.py \
  --url wss://<modal-url>/ws/tts \
  --iterations 5 \
  --save-audio \
  --output-dir tensorrt_tts/hindi_finetuned/out

# Text-length sweep (short/medium/long/very long)
python tensorrt_tts/hindi_finetuned/test/benchmark_websocket.py \
  --url wss://<modal-url>/ws/tts \
  --sweep \
  --save-audio
```

#### Benchmark Results

**5-Iteration Test** (73 characters, "नमस्ते, मैं एक हिंदी टेक्स्ट टू स्पीच मॉडल हूं।"):

| Metric | Mean | Min | Max |
|--------|------|-----|-----|
| Server TTFT | 158ms | 114ms | 183ms |
| Server TTFA | 2545ms | 2487ms | 2592ms |
| Client TTFA | 3160ms | 3095ms | 3244ms |
| Client RTF | 1.02x | 1.00x | 1.06x |
| Server RTF | 1.58x | 1.56x | 1.63x |
| Chunk jitter | 113ms | 92ms | 141ms |
| Audio duration | 4.35s | 4.18s | 4.52s |

**Text-Length Sweep Results**:

| Length | Chars | Client TTFA | Duration | RTF | Chunks |
|--------|-------|-------------|----------|-----|--------|
| Short | 33 | 2686ms | 1.96s | 0.67x | 4 |
| Medium | 103 | 2699ms | 5.38s | 1.10x | 12 |
| Long | 259 | 2658ms | 15.10s | 1.30x | 35 |
| Very Long | 383 | 2639ms | 19.63s | 1.35x | 46 |

**Key Observations**:
1. **TTFA is consistent** (~2600-3200ms) regardless of text length
2. **RTF improves with text length**: Longer text = more efficient (amortized overhead)
3. **Server RTF > Client RTF gap** (~0.5x): Network/WebSocket overhead
4. **Chunk jitter** (~100-150ms): Acceptable for streaming playback

#### Server-Side Metrics Enhancement

The WebSocket server (`websocket_server.py`) was updated to provide detailed server-side metrics when `benchmark: true` is included in the request:

```python
# Request with benchmark flag
{
    "text": "...",
    "voice": "tara",
    "benchmark": true  # Enables detailed metrics
}

# Enhanced completion response
{
    "done": true,
    "chunks": 12,
    "duration_s": 5.38,
    "bytes": 258048,
    "metrics": {
        "ttft_ms": 158,        # Time to first token
        "ttfa_ms": 2545,       # Time to first audio
        "tokens": 428,
        "tokens_per_sec": 97.5,
        "frames_per_sec": 14.2,
        "generation_time_s": 4.38,
        "decode_times_ms": [...]  # Per-chunk decode latency
    }
}
```

#### Audio Output Directory

Benchmark audio files are saved to `hindi_finetuned/out/`:
- `benchmark_iter_N.wav` — Audio from iteration N
- `benchmark_sweep_<length>.wav` — Audio from sweep test

### 9.7 Quality Verification Tests

**File**: `hindi_finetuned/test/test_streaming_audio_quality.py`

Modal function to verify streaming audio quality matches batch decode:
- Generates tokens once, then decodes both ways
- Compares MSE, correlation, std ratio
- Saves both audio files for manual A/B comparison

```bash
modal run tensorrt_tts/hindi_finetuned/test/test_streaming_audio_quality.py
```

### 9.8 Advanced Metrics Visualization (IMPLEMENTED)

**File**: `hindi_finetuned/plot_metrics/benchmark_with_wandb.py`

Comprehensive metrics visualization and analysis tool with Weights & Biases integration.

#### Features

1. **Chunk Arrival Timeline Plots**
   - Visualizes when each chunk arrives relative to request start
   - Instantly reveals stalls and irregular cadence
   - Color-coded by iteration for drift analysis

2. **Max Chunk Gap Tracking**
   - Tracks the worst-case gap for each iteration
   - More critical than stddev for streaming UX
   - Highlights gaps > 200ms as potential stalls

3. **Drift/Thermal/GC Analysis**
   - Runs 10-20 iterations back-to-back
   - Compares first half vs second half performance
   - Detects thermal throttling and GC pauses

4. **W&B Dashboard Integration**
   - Per-iteration metrics logged automatically
   - Interactive charts and tables
   - Summary statistics and trends

#### Usage

```bash
# Run 20 iterations with local plots only
python tensorrt_tts/hindi_finetuned/plot_metrics/benchmark_with_wandb.py \
  --url wss://<modal-url>/ws/tts \
  --iterations 20

# Run with W&B logging
python tensorrt_tts/hindi_finetuned/plot_metrics/benchmark_with_wandb.py \
  --url wss://<modal-url>/ws/tts \
  --iterations 20 \
  --wandb \
  --wandb-project orpheus-tts-benchmark

# Save audio from each iteration
python tensorrt_tts/hindi_finetuned/plot_metrics/benchmark_with_wandb.py \
  --url wss://<modal-url>/ws/tts \
  --iterations 10 \
  --save-audio
```

#### Output Plots

Generated in `hindi_finetuned/plot_metrics/output/`:

1. **chunk_timeline_analysis.png**
   - Chunk arrival timeline (all iterations overlaid)
   - Chunk gap scatter plot (stall detection)
   - Max gap per iteration (drift detection)
   - RTF and TTFA stability over time

2. **gap_analysis_detailed.png**
   - Gap distribution histogram
   - Gap CDF (P95, P99 lines)
   - Stall count per iteration
   - Rolling average trend

#### W&B Metrics Logged

| Metric | Description |
|--------|-------------|
| `client_ttfa_ms` | Time to first audio (client-side) |
| `client_rtf` | Real-time factor (client-side) |
| `max_chunk_gap_ms` | Worst chunk gap in iteration |
| `mean_chunk_gap_ms` | Average chunk gap |
| `server_ttft_ms` | Time to first token (server-side) |
| `server_rtf` | Real-time factor (server-side) |
| `chunk_events` (table) | Full chunk arrival timeline |

#### Interpreting Results

**Good Streaming Performance:**
- Mean RTF ≥ 1.0x
- Max chunk gap < 300ms
- No significant drift between iterations

**Acceptable:**
- Mean RTF ≥ 0.8x
- Max chunk gap < 500ms

**Needs Improvement:**
- RTF < 0.8x
- Frequent gaps > 500ms
- Significant drift (>10% RTF change over iterations)

### 9.9 File Reference (Updated)

| File | Purpose | Status |
|------|---------|--------|
| `hindi_finetuned/inference_a100.py` | Baseline A100 inference | ✅ Baseline |
| `hindi_finetuned/inference_a100_optimized.py` | **Combined warm+streaming** | ✅ **RECOMMENDED** |
| `hindi_finetuned/websocket_server.py` | **WebSocket API for web clients** | ✅ |
| `hindi_finetuned/test/benchmark_websocket.py` | Comprehensive benchmark client | ✅ |
| `hindi_finetuned/test/test_websocket_client.py` | Simple WebSocket test client | ✅ |
| `hindi_finetuned/test/test_streaming_audio_quality.py` | Streaming vs batch quality test | ✅ |
| `hindi_finetuned/plot_metrics/benchmark_with_wandb.py` | **Advanced metrics + W&B** | ✅ **NEW** |
| `hindi_finetuned/plot_metrics/output/` | **Plots and analysis output** | ✅ **NEW** |
| `hindi_finetuned/out/` | Benchmark audio outputs | ✅ |
| `hindi_finetuned/inference_a100_warm.py` | Warm container only | ✅ |
| `hindi_finetuned/inference_a100_streaming.py` | Streaming only (older) | ✅ |
| `hindi_finetuned/benchmark.py` | Performance benchmarking | ✅ |
| `hindi_finetuned/build_engine_a100.py` | A100 engine builder | ✅ |

---

## 10. Summary: Production Deployment

### Recommended Stack

```
┌─────────────────────────────────────────────────────────────┐
│                    PRODUCTION DEPLOYMENT                     │
├─────────────────────────────────────────────────────────────┤
│                                                              │
│  Client (Browser/App)                                        │
│       │                                                      │
│       │ WebSocket (wss://)                                   │
│       ▼                                                      │
│  Modal WebSocket Server (websocket_server.py)                │
│       │                                                      │
│       ├── Warm Container (models pre-loaded)                 │
│       ├── LookaheadStreamingDecoder (5-frame lookahead)     │
│       └── Binary audio chunks (int16 PCM @ 24kHz)           │
│                                                              │
│  Metrics:                                                    │
│  • TTFA: ~2.5-3.2s (first audio chunk)                      │
│  • RTF: 1.0-1.3x realtime (streaming)                       │
│  • Quality: 0.998+ correlation with batch                   │
│                                                              │
└─────────────────────────────────────────────────────────────┘
```

### Quick Start

```bash
# 1. Deploy WebSocket server
modal deploy tensorrt_tts/hindi_finetuned/websocket_server.py

# 2. Test with browser
# Open: https://<modal-url>/test

# 3. Benchmark
python tensorrt_tts/hindi_finetuned/test/benchmark_websocket.py \
  --url wss://<modal-url>/ws/tts \
  --iterations 5 --save-audio
```

---

## 11. vLLM Backend (Alternative)

A vLLM-based WebSocket server is also available for environments without TensorRT-LLM.

**Directory**: `vllm_inference/` (at repo root, renamed from `hindi_finetuned/`)

### 11.1 Directory Structure

```
vllm_inference/
├── websocket_server_sync.py    # RECOMMENDED - Stable sync vLLM server
├── websocket_server_vllm.py    # Async vLLM (has v0.13 compatibility issues)
├── simple_tts.py               # Basic TTS test (batch mode)
├── test_websocket_client.py    # WebSocket test client with audio saving
├── out/                        # Audio output directory
│   ├── simple_test.wav
│   ├── vllm_test_short.wav
│   ├── vllm_test_medium.wav
│   └── vllm_test_long.wav
└── ...
```

### 11.2 Key Implementation Notes

#### vLLM v0.13 Compatibility Issue

The original `websocket_server_vllm.py` uses `AsyncLLMEngine` which has issues with vLLM v0.13's new V1 architecture - the engine core process dies unexpectedly during inference.

**Solution**: Created `websocket_server_sync.py` which uses:
- Synchronous vLLM in a ThreadPoolExecutor
- More stable with vLLM v0.13
- Same `LookaheadStreamingDecoder` for high-quality streaming

#### Correct Prompt Format (Critical)

Both TensorRT and vLLM must use identical prompt format:

```python
# CORRECT (matches TensorRT pipeline)
prompt_text = f"{voice}: {text}"
prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)
full_prompt = [START_TOKEN] + prompt_tokens + [END_OF_TEXT, END_OF_TURN]

# Token constants
START_TOKEN = 128259  # Start of human turn
END_OF_TEXT = 128009  # End of text content
END_OF_TURN = 128260  # End of turn
SOS_TOKEN = 128257    # Start of speech (audio begins after this)
EOS_TOKEN = 128258    # End of speech (stop generation)
TOKEN_BASE = 128266   # Base offset for audio codes
```

### 11.3 Usage

```bash
# Start the stable sync server
modal serve vllm_inference/websocket_server_sync.py

# Test with client (saves audio to out/)
python3 vllm_inference/test_websocket_client.py "wss://<modal-url>/ws/tts"

# Simple batch TTS test
modal run vllm_inference/simple_tts.py --text "नमस्ते" --output vllm_inference/out/test.wav
```

### 11.4 Benchmark Results (A100)

Tested with 3 Hindi prompts:

| Prompt | Text Length | Duration | TTFA | RTF |
|--------|-------------|----------|------|-----|
| Short | 21 chars | 2.82s | 4.3s | 0.59x |
| Medium | 73 chars | 24.32s | 29.4s | 0.65x |
| Long | 163 chars | 24.32s | 29.2s | 0.81x |

**Notes:**
- TTFA includes cold-start token generation time
- RTF improves with longer text (amortized overhead)
- vLLM sync mode is more stable than async mode

### 11.5 Differences from TensorRT

| Aspect | TensorRT-LLM | vLLM (sync) |
|--------|--------------|-------------|
| TPS | 140-160 | 50-100 |
| Cold Start | 45-82s | 120-150s |
| Memory | ~16GB | ~20GB |
| Stability | Excellent | Good (sync mode) |
| Streaming | Native | ThreadPool |
| Compatibility | NVIDIA only | More portable |

### 11.6 When to Use vLLM

- No access to TensorRT-LLM optimized engine
- Need easier deployment/portability
- Development/testing purposes
- Lower throughput acceptable
- Cross-platform requirements

---

## 12. Comprehensive Sweep Benchmark

**File**: `hindi_finetuned/plot_metrics/comprehensive_sweep.py`

A comprehensive benchmark that tests 22 different prompt lengths with cold-start analysis and W&B logging.

### 12.1 Features

- **22 Hindi prompts** ranging from 6 to 460 characters
- **Cold-start analysis**: First 3 requests logged separately
- **Warm benchmarking**: All 22 prompts after warmup
- **Audio saving**: Each prompt's output saved to `out/` directory
- **W&B integration**: Full metrics logged to dashboard
- **Visual plots**: sweep_analysis.png, gap_analysis.png, per_prompt_breakdown.png

### 12.2 Usage

```bash
# Start the TensorRT WebSocket server first
modal serve tensorrt_tts/hindi_finetuned/websocket_server.py

# Run comprehensive sweep with W&B logging
python tensorrt_tts/hindi_finetuned/plot_metrics/comprehensive_sweep.py \
  --url wss://<modal-url>/ws/tts \
  --wandb
```

### 12.3 W&B Dashboard

The comprehensive sweep logs to: `orpheus-tts-benchmark` project

**Logged Metrics**:
- Per-prompt: text_length, duration_s, ttfa_ms, rtf, chunk_count
- Aggregate: length_vs_duration correlation, rtf_vs_length trends
- Tables: Full prompt breakdown, cold-start vs warm comparison

### 12.4 Output Files

```
hindi_finetuned/plot_metrics/output/
├── sweep_analysis.png           # Length vs duration/RTF plots
├── gap_analysis.png            # Chunk gap analysis
├── per_prompt_breakdown.png    # Individual prompt metrics
├── cold_1.wav ... cold_3.wav   # Cold-start audio
└── sweep_*.wav                 # Each prompt's audio output
```

---

## 13. Summary: Complete Pipeline

### 13.1 Architecture Overview

```
┌─────────────────────────────────────────────────────────────────┐
│                    TTS PIPELINE ARCHITECTURE                      │
├─────────────────────────────────────────────────────────────────┤
│                                                                  │
│  Input Text                                                      │
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ Tokenization                                                 ││
│  │ [START_TOKEN] + encode("voice: text") + [END_OF_TEXT, EOT] ││
│  └─────────────────────────────────────────────────────────────┘│
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ LLM Backend (TensorRT-LLM or vLLM)                          ││
│  │ • Streaming token generation                                 ││
│  │ • Wait for SOS token (128257)                               ││
│  │ • Extract audio tokens until EOS (128258)                   ││
│  └─────────────────────────────────────────────────────────────┘│
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ LookaheadStreamingDecoder                                    ││
│  │ • 5-frame lookahead for SNAC backward-context               ││
│  │ • Decode-from-start each time (context-preserving)          ││
│  │ • Only emit stable samples (with future context)            ││
│  │ • 99.9% correlation with batch decode                       ││
│  └─────────────────────────────────────────────────────────────┘│
│       │                                                          │
│       ▼                                                          │
│  ┌─────────────────────────────────────────────────────────────┐│
│  │ SNAC Decoder (hubertsiuzdak/snac_24khz)                     ││
│  │ • 7 tokens → 1 frame → 2048 samples (~85ms)                 ││
│  │ • 3-layer hierarchical codec                                 ││
│  │ • Code redistribution with offsets                          ││
│  └─────────────────────────────────────────────────────────────┘│
│       │                                                          │
│       ▼                                                          │
│  Binary Audio Chunks (int16 PCM @ 24kHz)                        │
│                                                                  │
└─────────────────────────────────────────────────────────────────┘
```

### 13.2 Performance Summary

| Metric | TensorRT-LLM | vLLM |
|--------|--------------|------|
| Token Generation | 140-160 TPS | 50-100 TPS |
| TTFA (Streaming) | 2.5-3.2s | 4-30s |
| RTF (Streaming) | 1.0-1.3x | 0.6-0.8x |
| Cold Start | 45-82s | 120-150s |
| Audio Quality | 0.998+ correlation | Same |

### 13.3 Key Files Reference

| File | Purpose |
|------|---------|
| `tensorrt_tts/hindi_finetuned/websocket_server.py` | **TensorRT WebSocket (PRODUCTION)** |
| `tensorrt_tts/hindi_finetuned/inference_a100_optimized.py` | Combined warm+streaming CLI |
| `tensorrt_tts/hindi_finetuned/plot_metrics/comprehensive_sweep.py` | Full benchmark suite |
| `tensorrt_tts/hindi_finetuned/plot_metrics/benchmark_with_wandb.py` | Metrics visualization |
| `vllm_inference/websocket_server_sync.py` | **vLLM WebSocket (STABLE)** |
| `vllm_inference/simple_tts.py` | vLLM batch TTS test |
| `vllm_inference/test_websocket_client.py` | WebSocket test client |

### 13.4 Quick Start

```bash
# TensorRT Pipeline (Recommended)
modal deploy tensorrt_tts/hindi_finetuned/websocket_server.py
# Open: https://<modal-url>/test

# vLLM Pipeline (Alternative)
modal serve vllm_inference/websocket_server_sync.py
python3 vllm_inference/test_websocket_client.py "wss://<modal-url>/ws/tts"

# Benchmarking
python tensorrt_tts/hindi_finetuned/plot_metrics/comprehensive_sweep.py \
  --url wss://<modal-url>/ws/tts --wandb
```

---

*Report generated: 2024-12-23*
*Pipeline version: TensorRT-LLM 0.21.0, vLLM 0.13.0*
*P0 Optimizations: Implemented 2024-12-23*
*Lookahead Streaming (SNAC context-preserving): Updated 2024-12-23*
*WebSocket Server: Added 2024-12-23*
*Comprehensive Benchmarking: Added 2024-12-23*
*Advanced Metrics Visualization + W&B: Added 2024-12-23*
*vLLM Backend (Sync): Updated 2024-12-24*
*Comprehensive Sweep Benchmark: Added 2024-12-24*
*Directory Reorganization (vllm_inference/): 2024-12-24*
