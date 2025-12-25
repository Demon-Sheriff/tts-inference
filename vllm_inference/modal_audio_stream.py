"""
FastAPI WebSocket Server - vLLM + SNAC Audio Streaming (Hindi)

Full TTS pipeline:
1. Text input via WebSocket
2. vLLM generates audio tokens (custom_token_N format)
3. SNAC decoder converts tokens to audio
4. Stream PCM audio chunks back to client

Audio format: 24kHz, 16-bit mono PCM

IMPORTANT: Prompt Format for Audio Token Generation
===================================================

Orpheus/Canopy Speech LLMs require SPECIAL TOKEN WRAPPING to generate
audio tokens instead of text. The prompt must be wrapped with:

    [128259] + tokenize("voice: text") + [128009, 128260, 128261, 128257]

Without this wrapping, the model outputs text. With it, outputs <custom_token_N>.

Supported Models (all use same prompt format):
- canopylabs/3b-hi-pretrain-research_release (Hindi pretrained)
- canopylabs/3b-hi-ft-research_release (Hindi fine-tuned)
- canopylabs/orpheus-3b-0.1-ft (English, GATED)
- canopylabs/orpheus-3b-0.1-pretrained (English)

Voice options for Hindi: "tara" (female)
See: https://github.com/canopyai/Orpheus-TTS for other voices per language

Architecture Reference:
- https://github.com/taresh18/orpheus-streaming
- https://github.com/canopyai/Orpheus-TTS
- Colab: https://colab.research.google.com/drive/1KhXT56UePPUHhqitJNUxq63k-pQomz3N
"""

import modal
import time
import uuid
import base64
import asyncio
from contextlib import asynccontextmanager
from fastapi import FastAPI, WebSocket, WebSocketDisconnect
from fastapi.responses import HTMLResponse, FileResponse, JSONResponse
from pydantic import BaseModel

app = modal.App("orpheus-audio-stream")

image = (
    modal.Image.debian_slim(python_version="3.11")
    .apt_install("libsndfile1")  # Required for audio processing
    .pip_install(
        "vllm",
        "fastapi",
        "uvicorn",
        "huggingface_hub",
        "hf_transfer",
        "snac",  # SNAC audio codec
        "torch",
        "numpy",
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("orpheus-cache")

# Hindi Speech LLM - outputs audio tokens when using correct prompt format
# The key is using special token IDs (128259, 128009, 128260, 128261, 128257)
MODEL_PATH = "canopylabs/3b-hi-pretrain-research_release"

# Alternatives:
# MODEL_PATH = "canopylabs/3b-hi-ft-research_release"  # Hindi fine-tuned
# MODEL_PATH = "/cache/orpheus-merged-vllm"  # Local merged Hindi model
# MODEL_PATH = "canopylabs/orpheus-3b-0.1-ft"  # English Orpheus (GATED)

# Global references
engine = None
tokenizer = None
snac_model = None
snac_device = None

# Single-stream guard: only one generation at a time
generation_lock = asyncio.Lock()

# SNAC decoder constants
# 7 tokens = 1 audio frame (~2048 samples at 24kHz)
# First chunk emitted after 7 tokens for low latency
# Subsequent chunks use sliding window approach:
#   - Buffer last 28 tokens (4 frames)
#   - Decode all 28, extract middle 2048 samples to avoid edge artifacts
MIN_FRAMES_FIRST = 7
MIN_FRAMES_SUBSEQ = 28
PROCESS_EVERY = 7
AUDIO_SLICE_START = 2048  # Start index for audio extraction from decoded output
AUDIO_SLICE_END = 4096    # End index (2048 samples per chunk)

# Special token IDs for Orpheus/Canopy models
TOKEN_SOH = 128259   # Start of Human
TOKEN_EOT = 128009   # End of Text
TOKEN_EOH = 128260   # End of Human
TOKEN_SOS = 128257   # Start of Speech
TOKEN_EOS = 128258   # End of Speech (EOS)
TOKEN_AUDIO_BASE = 128266  # Audio token offset (code = token_id - 128266)


def init_snac():
    """Initialize SNAC model for audio decoding."""
    global snac_model, snac_device
    import torch
    from snac import SNAC

    print("Loading SNAC model...")
    snac_model = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
    snac_device = "cuda" if torch.cuda.is_available() else "cpu"
    snac_model = snac_model.to(snac_device)

    if snac_device == "cuda":
        torch.backends.cudnn.benchmark = True

        # Warmup
        dummy_codes = [
            torch.randint(0, 4096, (1, 1), dtype=torch.int32, device=snac_device),
            torch.randint(0, 4096, (1, 2), dtype=torch.int32, device=snac_device),
            torch.randint(0, 4096, (1, 4), dtype=torch.int32, device=snac_device)
        ]
        with torch.inference_mode():
            _ = snac_model.decode(dummy_codes)

    print(f"SNAC model loaded on {snac_device}")


def convert_to_audio(code_list: list, extract_slice: bool = False) -> bytes:
    """
    Convert token codes to audio using SNAC decoder.

    Based on orpheus-streaming decoder.py and working notebook:
    - Token IDs are already adjusted (token_id - 128266)
    - Redistribution splits 7 tokens into 3 layers with different offsets:
      - Layer 1: positions [0, 7, 14, ...] - 1 per frame
      - Layer 2: positions [1, 4, 8, 11, ...] with -4096 offset - 2 per frame
      - Layer 3: positions [2, 3, 5, 6, ...] with varying offsets - 4 per frame

    Args:
        code_list: List of SNAC codes (token_id - 128266)
        extract_slice: If True, extract only middle 2048 samples (for streaming chunks)
                      If False, return full audio (for first/final chunks)

    Returns PCM audio bytes (16-bit, 24kHz mono).
    """
    import torch
    import numpy as np

    if len(code_list) < 7:
        return None

    # Ensure length is multiple of 7
    num_frames = len(code_list) // 7
    code_list = code_list[:num_frames * 7]

    # Reorganize into 3 layers (matching orpheus-streaming decoder.py)
    codes_0 = []  # 1 per frame
    codes_1 = []  # 2 per frame
    codes_2 = []  # 4 per frame

    for i in range(num_frames):
        base_idx = i * 7
        codes_0.append(code_list[base_idx])
        codes_1.append(code_list[base_idx + 1] - 4096)
        codes_2.append(code_list[base_idx + 2] - (2 * 4096))
        codes_2.append(code_list[base_idx + 3] - (3 * 4096))
        codes_1.append(code_list[base_idx + 4] - (4 * 4096))
        codes_2.append(code_list[base_idx + 5] - (5 * 4096))
        codes_2.append(code_list[base_idx + 6] - (6 * 4096))

    # Create tensors on SNAC device
    codes = [
        torch.tensor(codes_0, dtype=torch.int32, device=snac_device).unsqueeze(0),
        torch.tensor(codes_1, dtype=torch.int32, device=snac_device).unsqueeze(0),
        torch.tensor(codes_2, dtype=torch.int32, device=snac_device).unsqueeze(0),
    ]

    # Validate codes are in valid SNAC range (0-4095)
    for layer_idx, layer_codes in enumerate(codes):
        min_val = layer_codes.min().item()
        max_val = layer_codes.max().item()
        if min_val < 0 or max_val >= 4096:
            # Clamp to valid range - this can happen with Hindi model tokens
            codes[layer_idx] = torch.clamp(layer_codes, 0, 4095)

    with torch.inference_mode():
        audio_hat = snac_model.decode(codes)

        # Extract middle portion for streaming (avoids edge artifacts)
        # or return full audio for first/final chunks
        if extract_slice and audio_hat.shape[-1] > AUDIO_SLICE_END:
            audio_slice = audio_hat[:, :, AUDIO_SLICE_START:AUDIO_SLICE_END]
        else:
            audio_slice = audio_hat

        # Convert to int16 PCM
        audio_int16 = (audio_slice * 32767.0).clamp(-32768, 32767).to(torch.int16)
        return audio_int16.flatten().cpu().numpy().tobytes()


@asynccontextmanager
async def lifespan(app: FastAPI):
    """Initialize vLLM engine and SNAC on startup."""
    global engine, tokenizer

    from vllm.engine.async_llm_engine import AsyncLLMEngine
    from vllm.engine.arg_utils import AsyncEngineArgs

    # Initialize SNAC first
    init_snac()

    # Initialize vLLM
    engine_args = AsyncEngineArgs(
        model=MODEL_PATH,
        tokenizer=MODEL_PATH,
        # tokenizer=MODEL_PATH,
        # tokenizer_kwargs={
        #     "pad_token_id": 128263,
        # },
        trust_remote_code=True,
        dtype="float16",
        tensor_parallel_size=1,
        gpu_memory_utilization=0.85,
        enable_prefix_caching=True,
    )

    print("Loading vLLM engine...")
    engine = AsyncLLMEngine.from_engine_args(engine_args)
    tokenizer = engine.tokenizer
    print("vLLM engine ready.")

    yield

    print("Shutting down...")


web_app = FastAPI(lifespan=lifespan)


def format_prompt(text: str, voice: str = "tara") -> list[int]:
    """
    Format prompt for Orpheus model with special audio tokens.

    Based on working notebook (test-tts-inference.ipynb):
    - Start token: 128259 (Start of Human)
    - End tokens: 128009, 128260 (End of Text, End of Human)

    The prompt is formatted as "voice: text" and wrapped with special tokens.

    Returns token IDs (not text) for direct vLLM input.
    """
    # Format: "voice: text"
    prompt_text = f"{voice}: {text}"

    # Tokenize the prompt
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=False)

    # Wrap with special audio tokens (from working notebook)
    # Start: TOKEN_SOH (Start of Human)
    # End: TOKEN_EOT (End of Text), TOKEN_EOH (End of Human)
    start_token = [TOKEN_SOH]
    end_tokens = [TOKEN_EOT, TOKEN_EOH]

    full_tokens = start_token + prompt_tokens + end_tokens
    return full_tokens


async def generate_audio_tokens(prompt: str, voice: str = "tara", max_tokens: int = 1200):
    """
    Generate audio tokens using vLLM.

    Uses special token wrapping to trigger audio token generation mode.
    Based on working notebook (test-tts-inference.ipynb).

    Yields raw token IDs (integers). Audio tokens are those >= 128266.
    Special tokens:
    - 128257: Start of speech tokens (skip this and everything before)
    - 128258: EOS token (stop generation)
    """
    from vllm.sampling_params import SamplingParams
    from vllm.inputs import TokensPrompt

    # Get token IDs with special audio tokens
    prompt_token_ids = format_prompt(prompt, voice)

    sampling_params = SamplingParams(
        temperature=0.6,
        top_p=0.95,
        max_tokens=max_tokens,
        repetition_penalty=1.1,
        stop_token_ids=[TOKEN_EOS],  # Stop at EOS (128258)
    )

    request_id = str(uuid.uuid4())
    start_time = time.perf_counter()
    first_token_time = None
    found_speech_start = False

    # Use TokensPrompt to pass token IDs directly
    tokens_prompt = TokensPrompt(prompt_token_ids=prompt_token_ids)

    try:
        async for out in engine.generate(
            tokens_prompt,
            sampling_params,
            request_id=request_id,
        ):
            for o in out.outputs:
                token_id = o.token_ids[-1]

                if first_token_time is None:
                    first_token_time = time.perf_counter()
                    ttft = (first_token_time - start_time) * 1000
                    print(f"TTFT: {ttft:.2f}ms")

                # Skip until we find TOKEN_SOS (start of speech)
                if not found_speech_start:
                    if token_id == TOKEN_SOS:
                        found_speech_start = True
                        print(f"Found speech start token ({TOKEN_SOS})")
                    continue

                # Stop at EOS
                if token_id == TOKEN_EOS:
                    print(f"Found EOS token ({TOKEN_EOS})")
                    break

                # Yield audio token ID (>= 128266)
                yield token_id

    finally:
        await engine.abort(request_id)



async def stream_audio(prompt: str, voice: str = "tara"):
    """
    Correct SNAC streaming implementation.

    Strategy:
    - Accumulate SNAC tokens
    - Decode in NON-overlapping chunks
    - Never re-decode past tokens
    - Never overlap frames
    - Matches offline Colab behavior exactly
    """

    FRAME_TOKENS = 7
    CHUNK_TOKENS = 28  # 4 frames

    buffer: list[int] = []   # SNAC codes (token_id - TOKEN_AUDIO_BASE)

    start_time = time.perf_counter()
    first_audio_time = None

    total_audio_bytes = 0
    total_decode_time = 0.0
    chunks_sent = 0

    async for token_id in generate_audio_tokens(prompt, voice):
        # Convert token_id → SNAC code
        snac_code = token_id - TOKEN_AUDIO_BASE
        buffer.append(snac_code)

        # When enough tokens are available, decode a chunk
        if len(buffer) >= CHUNK_TOKENS:
            chunk = buffer[:CHUNK_TOKENS]
            buffer = buffer[CHUNK_TOKENS:]

            decode_start = time.perf_counter()
            audio = convert_to_audio(chunk, extract_slice=False)
            decode_time = time.perf_counter() - decode_start

            if audio is not None:
                if first_audio_time is None:
                    first_audio_time = time.perf_counter()
                    ttfa_ms = (first_audio_time - start_time) * 1000
                    print(f"[TTFA] {ttfa_ms:.2f} ms")

                total_decode_time += decode_time
                total_audio_bytes += len(audio)
                chunks_sent += 1

                yield audio

    # ---- Flush remainder (trim to full frames) ----
    remaining_frames = len(buffer) // FRAME_TOKENS
    if remaining_frames > 0:
        final_tokens = buffer[:remaining_frames * FRAME_TOKENS]

        decode_start = time.perf_counter()
        audio = convert_to_audio(final_tokens, extract_slice=False)
        decode_time = time.perf_counter() - decode_start

        if audio is not None:
            if first_audio_time is None:
                first_audio_time = time.perf_counter()
                ttfa_ms = (first_audio_time - start_time) * 1000
                print(f"[TTFA] {ttfa_ms:.2f} ms")

            total_decode_time += decode_time
            total_audio_bytes += len(audio)
            chunks_sent += 1

            yield audio

    # ---- Metrics ----
    total_time = time.perf_counter() - start_time
    audio_samples = total_audio_bytes // 2  # int16
    audio_duration = audio_samples / 24000.0
    rtf = audio_duration / total_time if total_time > 0 else 0.0

    print(
        f"[DONE] chunks={chunks_sent}, "
        f"audio={audio_duration:.2f}s, "
        f"wall={total_time:.2f}s, "
        f"RTF={rtf:.2f}x, "
        f"decode_time={total_decode_time:.2f}s"
    )


@web_app.get("/")
async def home():
    """Simple test page."""
    return HTMLResponse("""
    <html>
    <head><title>Orpheus TTS Audio Streaming</title></head>
    <body>
        <h1>Orpheus TTS Audio Streaming</h1>
        <p>WebSocket endpoint: <code>/ws/audio</code></p>
        <p>Text endpoint: <code>/ws</code> (for text token streaming)</p>
        <h3>Audio Format:</h3>
        <ul>
            <li>Sample Rate: 24000 Hz</li>
            <li>Bit Depth: 16-bit</li>
            <li>Channels: Mono</li>
            <li>Format: Raw PCM (base64 encoded in JSON)</li>
        </ul>
    </body>
    </html>
    """)


@web_app.websocket("/ws/audio")
async def websocket_audio_stream(ws: WebSocket):
    """
    WebSocket endpoint for audio streaming.

    Client sends: {"text": "...", "voice": "tara"}
    Server streams: {"audio": "<base64 PCM>", "chunk_index": N}
    Final message: {"event": "EOS", "total_chunks": N}

    Uses global generation_lock for single-stream semantics.
    """
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()
            text = data.get("text", "")
            voice = data.get("voice", "tara")

            if not text:
                await ws.send_json({"error": "No text provided"})
                continue

            # Single-stream guard: only one generation at a time
            if generation_lock.locked():
                await ws.send_json({"error": "Generation in progress, try again later"})
                continue

            async with generation_lock:
                print(f"Generating audio for: {text[:50]}...")

                chunk_index = 0
                async for audio_chunk in stream_audio(text, voice):
                    chunk_index += 1
                    # Encode audio as base64 for JSON transport
                    audio_b64 = base64.b64encode(audio_chunk).decode("ascii")
                    await ws.send_json({
                        "audio": audio_b64,
                        "chunk_index": chunk_index,
                    })

                await ws.send_json({
                    "event": "EOS",
                    "total_chunks": chunk_index,
                })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await ws.close()


class TTSRequest(BaseModel):
    text: str
    voice: str = "tara"


@web_app.post("/generate-batch")
async def generate_audio_batch(request: TTSRequest):
    """
    Non-streaming audio generation for debugging.

    Collects ALL tokens first, then decodes in one batch.
    This bypasses all streaming logic to test if basic SNAC decode works.
    """
    from fastapi.responses import Response
    import io
    import wave

    text = request.text
    voice = request.voice

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    if generation_lock.locked():
        return JSONResponse({"error": "Generation in progress"}, status_code=503)

    async with generation_lock:
        print(f"[/generate-batch] Collecting all tokens for: {text[:50]}...")

        start_time = time.perf_counter()

        # Collect ALL tokens first (no streaming)
        all_snac_codes = []
        async for token_id in generate_audio_tokens(text, voice):
            snac_code = token_id - TOKEN_AUDIO_BASE
            all_snac_codes.append(snac_code)

        token_time = time.perf_counter()
        print(f"[/generate-batch] Collected {len(all_snac_codes)} tokens in {(token_time - start_time)*1000:.0f}ms")

        if len(all_snac_codes) < 7:
            return JSONResponse({"error": f"Not enough tokens: {len(all_snac_codes)}"}, status_code=500)

        # Decode ALL at once (no streaming, no slicing)
        audio_bytes = convert_to_audio(all_snac_codes, extract_slice=False)

        decode_time = time.perf_counter()
        print(f"[/generate-batch] Decoded in {(decode_time - token_time)*1000:.0f}ms")

        if audio_bytes is None:
            return JSONResponse({"error": "SNAC decode failed"}, status_code=500)

        # Calculate metrics
        total_samples = len(audio_bytes) // 2
        audio_duration_ms = (total_samples / 24000) * 1000
        total_time_ms = (decode_time - start_time) * 1000

        print(f"[/generate-batch] Audio: {audio_duration_ms:.0f}ms ({len(all_snac_codes)//7} frames)")

        # Create WAV
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)
            wav.setframerate(24000)
            wav.writeframes(audio_bytes)

        return Response(
            content=wav_buffer.getvalue(),
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=batch_output.wav",
                "X-Audio-Duration-Ms": str(round(audio_duration_ms, 2)),
                "X-Total-Time-Ms": str(round(total_time_ms, 2)),
                "X-Tokens": str(len(all_snac_codes)),
                "X-Frames": str(len(all_snac_codes) // 7),
            }
        )


@web_app.post("/generate")
async def generate_audio_endpoint(request: TTSRequest):
    """
    HTTP POST endpoint for audio generation - returns WAV file directly.

    Request body: {"text": "...", "voice": "tara"}
    Response: WAV audio file (24kHz, 16-bit mono)

    Headers in response include generation metrics (separated by layer):
    - X-TTFT-Ms: time to first token (vLLM latency)
    - X-TTFA-Ms: time to first audio (vLLM + SNAC decode)
    - X-Audio-Duration-Ms: audio duration in milliseconds
    - X-Total-Time-Ms: generation time in milliseconds
    - X-Decode-Time-Ms: total SNAC decode time
    - X-Real-Time-Factor: audio_duration / generation_time
    - X-Chunks: number of audio chunks generated

    Uses global generation_lock for single-stream semantics.
    """
    from fastapi.responses import Response
    import io
    import wave

    text = request.text
    voice = request.voice

    if not text:
        return JSONResponse({"error": "No text provided"}, status_code=400)

    # Single-stream guard
    if generation_lock.locked():
        return JSONResponse({"error": "Generation in progress, try again later"}, status_code=503)

    async with generation_lock:
        print(f"[/generate] Generating audio for: {text[:50]}...")

        start_time = time.perf_counter()
        audio_chunks = []
        first_audio_time = None

        # Collect all audio chunks
        async for audio_chunk in stream_audio(text, voice):
            if first_audio_time is None:
                first_audio_time = time.perf_counter()
            audio_chunks.append(audio_chunk)

        end_time = time.perf_counter()

        if not audio_chunks:
            return JSONResponse({
                "success": False,
                "error": "No audio generated",
                "text": text[:100],
            }, status_code=500)

        # Combine all chunks
        all_audio = b"".join(audio_chunks)

        # Calculate metrics
        total_bytes = len(all_audio)
        total_samples = total_bytes // 2  # 16-bit = 2 bytes per sample
        audio_duration_ms = (total_samples / 24000) * 1000
        total_time_ms = (end_time - start_time) * 1000
        ttfa_ms = (first_audio_time - start_time) * 1000 if first_audio_time else 0
        rtf = audio_duration_ms / total_time_ms if total_time_ms > 0 else 0

        print(f"[/generate] Done: {len(audio_chunks)} chunks, {audio_duration_ms:.0f}ms audio, RTF={rtf:.2f}x")

        # Create WAV file in memory
        wav_buffer = io.BytesIO()
        with wave.open(wav_buffer, "wb") as wav:
            wav.setnchannels(1)
            wav.setsampwidth(2)  # 16-bit
            wav.setframerate(24000)
            wav.writeframes(all_audio)

        wav_data = wav_buffer.getvalue()

        # Return WAV file with metrics in headers
        return Response(
            content=wav_data,
            media_type="audio/wav",
            headers={
                "Content-Disposition": "attachment; filename=output.wav",
                "X-Audio-Duration-Ms": str(round(audio_duration_ms, 2)),
                "X-Audio-Duration-S": str(round(audio_duration_ms / 1000, 2)),
                "X-Total-Time-Ms": str(round(total_time_ms, 2)),
                "X-TTFA-Ms": str(round(ttfa_ms, 2)),
                "X-Real-Time-Factor": str(round(rtf, 2)),
                "X-Chunks": str(len(audio_chunks)),
            }
        )


@web_app.websocket("/ws")
async def websocket_text_stream(ws: WebSocket):
    """
    WebSocket endpoint for raw token streaming (no audio).

    For debugging/testing the token generation.
    Shows raw token IDs and their SNAC codes.

    Uses global generation_lock for single-stream semantics.
    """
    await ws.accept()

    try:
        while True:
            data = await ws.receive_json()
            text = data.get("text", "")
            voice = data.get("voice", "tara")

            if not text:
                await ws.send_json({"error": "No text provided"})
                continue

            # Single-stream guard
            if generation_lock.locked():
                await ws.send_json({"error": "Generation in progress, try again later"})
                continue

            async with generation_lock:
                token_count = 0
                async for token_id in generate_audio_tokens(text, voice):
                    token_count += 1
                    snac_code = token_id - TOKEN_AUDIO_BASE
                    await ws.send_json({
                        "token_id": token_id,
                        "snac_code": snac_code,
                        "index": token_count,
                    })

                await ws.send_json({
                    "event": "EOS",
                    "total_tokens": token_count,
                })

    except WebSocketDisconnect:
        print("Client disconnected")
    except Exception as e:
        print(f"WebSocket error: {e}")
        await ws.close()

@web_app.post("/dump-tokens")
async def dump_tokens(request: TTSRequest):
    tokens = []
    async for tid in generate_audio_tokens(request.text, request.voice):
        tokens.append(tid)
    return {"tokens": tokens}

@app.function(
    image=image,
    gpu="A100",
    volumes={"/cache": vol},
    secrets=[modal.Secret.from_name("huggingface-secret")],
    timeout=600,
)
@modal.asgi_app()
def fastapi_app():
    return web_app
