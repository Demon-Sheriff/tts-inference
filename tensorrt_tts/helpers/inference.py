"""
TensorRT-LLM Inference Pipeline for Orpheus TTS

High-performance TTS inference using TensorRT-LLM with SNAC audio decoding.

Usage:
    # Run single inference
    modal run inference.py --text "Hello world" --voice "tara"

    # Deploy as service
    modal serve inference.py
"""

import modal
import time

app = modal.App("orpheus-trtllm-inference")

# TensorRT-LLM image with SNAC
image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt-llm/release:0.21.0",
        secret=modal.Secret.from_name("nvcr-credentials"),
    )
    .pip_install(
        "snac",
        "numpy",
        "fastapi",
        "uvicorn",
        "transformers",
    )
)

# Volumes
engine_vol = modal.Volume.from_name("orpheus-trtllm-engine", create_if_missing=True)
cache_vol = modal.Volume.from_name("orpheus-cache", create_if_missing=True)

# Paths
ENGINE_DIR = "/engine/trt_engine"
MODEL_DIR = "/cache/orpheus-3b"

# Special token IDs (same as Orpheus)
SOS_TOKEN = 128257  # Start of speech
EOS_TOKEN = 128258  # End of speech
START_TOKEN = 128259  # Start of turn
END_TOKENS = [128009, 128260]  # End of text, end of turn

# Audio constants
FRAME_SIZE = 7
SAMPLE_RATE = 24000
POSITION_OFFSETS = [0, 4096, 8192, 12288, 16384, 20480, 24576]


def redistribute_codes(codes: list[int]) -> tuple[list[int], list[int], list[int]]:
    """
    Redistribute flat audio codes into SNAC layers.

    Frame structure: [c0, c1, c2, c3, c4, c5, c6]
    - Layer 0: position 0 (1 per frame)
    - Layer 1: positions 1, 4 (2 per frame)
    - Layer 2: positions 2, 3, 5, 6 (4 per frame)

    The model adds offset = 4096 * position to each code.
    We subtract to get valid 0-4095 codes.
    """
    num_frames = len(codes) // FRAME_SIZE
    codes = codes[:num_frames * FRAME_SIZE]

    layer0 = []
    layer1 = []
    layer2 = []

    for frame_idx in range(num_frames):
        base = frame_idx * FRAME_SIZE

        def get_code(pos):
            raw = codes[base + pos] - POSITION_OFFSETS[pos]
            return max(0, min(4095, raw))

        # Layer 0: position 0
        layer0.append(get_code(0))

        # Layer 1: positions 1, 4
        layer1.append(get_code(1))
        layer1.append(get_code(4))

        # Layer 2: positions 2, 3, 5, 6
        layer2.append(get_code(2))
        layer2.append(get_code(3))
        layer2.append(get_code(5))
        layer2.append(get_code(6))

    return layer0, layer1, layer2


def decode_snac(layer0: list[int], layer1: list[int], layer2: list[int], snac_model, device: str) -> bytes:
    """Decode SNAC codes to audio bytes."""
    import torch
    import numpy as np

    codes_l0 = torch.tensor(layer0, dtype=torch.int32, device=device).unsqueeze(0)
    codes_l1 = torch.tensor(layer1, dtype=torch.int32, device=device).unsqueeze(0)
    codes_l2 = torch.tensor(layer2, dtype=torch.int32, device=device).unsqueeze(0)

    with torch.inference_mode():
        audio = snac_model.decode([codes_l0, codes_l1, codes_l2])

    audio_np = audio.squeeze().cpu().numpy()
    audio_np = np.clip(audio_np, -1.0, 1.0)
    audio_int16 = (audio_np * 32767).astype(np.int16)

    return audio_int16.tobytes()


@app.cls(
    image=image,
    gpu="L4",  # Match build GPU
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=300,
    scaledown_window=120,
)
class OrpheusTRTLLM:
    """TensorRT-LLM based Orpheus TTS inference."""

    @modal.enter()
    def load_models(self):
        """Load TRT-LLM engine and SNAC model on container start."""
        import torch
        from snac import SNAC
        from tensorrt_llm import LLM
        from transformers import AutoTokenizer

        print("=" * 60)
        print("Loading models...")
        print("=" * 60)

        t0 = time.perf_counter()

        # Load tokenizer separately (TRT-LLM tokenizer has issues with __len__)
        print(f"Loading tokenizer from {ENGINE_DIR}...")
        self.tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)
        print(f"Tokenizer loaded, vocab size: {self.tokenizer.vocab_size}")

        # Load TRT-LLM engine
        print(f"Loading TRT-LLM engine from {ENGINE_DIR}...")
        self.llm = LLM(model=ENGINE_DIR)
        print("Engine loaded")

        # Load SNAC
        print("Loading SNAC...")
        self.snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval()
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.snac = self.snac.to(self.device)

        # Warmup SNAC
        dummy_codes = [
            torch.randint(0, 4096, (1, 1), dtype=torch.int32, device=self.device),
            torch.randint(0, 4096, (1, 2), dtype=torch.int32, device=self.device),
            torch.randint(0, 4096, (1, 4), dtype=torch.int32, device=self.device)
        ]
        with torch.inference_mode():
            _ = self.snac.decode(dummy_codes)

        load_time = time.perf_counter() - t0
        print(f"All models loaded in {load_time:.2f}s")

    def format_prompt(self, text: str, voice: str = "tara") -> list[int]:
        """Format text into prompt token IDs."""
        prompt_text = f"{voice}: {text}"
        prompt_tokens = self.tokenizer.encode(prompt_text, add_special_tokens=False)
        return [START_TOKEN] + prompt_tokens + END_TOKENS

    @modal.method()
    def generate(
        self,
        text: str,
        voice: str = "tara",
        max_tokens: int = 4000,  # Higher for longer audio generation
        temperature: float = 0.6,
        top_p: float = 0.95,
    ) -> dict:
        """
        Generate speech from text.

        Returns:
            dict with audio bytes, duration, and timing info
        """
        import torch

        result = {
            "text": text,
            "voice": voice,
            "timings": {},
        }

        from tensorrt_llm import SamplingParams

        t0 = time.perf_counter()

        # Format prompt
        prompt_ids = self.format_prompt(text, voice)
        print(f"Prompt tokens: {len(prompt_ids)}")

        t1 = time.perf_counter()

        # Generate with TRT-LLM high-level API
        # NOTE: repetition_penalty=1.1 is critical for proper audio generation
        sampling_params = SamplingParams(
            max_tokens=max_tokens,
            stop_token_ids=[EOS_TOKEN],
            temperature=temperature,
            top_p=top_p,
            repetition_penalty=1.1,
        )

        outputs = self.llm.generate(
            [prompt_ids],
            sampling_params=sampling_params,
        )

        result["timings"]["token_gen_ms"] = (time.perf_counter() - t1) * 1000

        # Extract generated tokens - LLM API returns list of RequestOutput
        # Each output has .outputs which is a list of CompletionOutput
        output_ids = outputs[0].outputs[0].token_ids
        print(f"Generated {len(output_ids)} tokens")

        # Find audio tokens (between SOS and EOS)
        audio_tokens = []
        found_sos = False

        for tid in output_ids:
            if not found_sos:
                if tid == SOS_TOKEN:
                    found_sos = True
                continue
            if tid == EOS_TOKEN:
                break
            audio_tokens.append(tid)

        print(f"Audio tokens: {len(audio_tokens)}")
        result["audio_tokens"] = len(audio_tokens)

        if not audio_tokens:
            result["error"] = "No audio tokens generated"
            return result

        t2 = time.perf_counter()

        # Convert to SNAC codes (subtract token base)
        TOKEN_BASE = 128266
        codes = [t - TOKEN_BASE for t in audio_tokens]

        # Redistribute into layers
        layer0, layer1, layer2 = redistribute_codes(codes)
        print(f"Frames: {len(layer0)}")

        # Decode with SNAC
        audio_bytes = decode_snac(layer0, layer1, layer2, self.snac, self.device)

        result["timings"]["decode_ms"] = (time.perf_counter() - t2) * 1000
        result["timings"]["total_ms"] = (time.perf_counter() - t0) * 1000

        # Audio stats
        import numpy as np
        audio_np = np.frombuffer(audio_bytes, dtype=np.int16)
        result["samples"] = len(audio_np)
        result["duration_s"] = len(audio_np) / SAMPLE_RATE
        result["audio"] = audio_bytes

        print(f"Audio: {result['duration_s']:.2f}s, {len(audio_bytes)} bytes")
        print(f"Total time: {result['timings']['total_ms']:.0f}ms")

        return result

    # Note: Streaming generation requires different API - implement if needed
    # The high-level LLM API supports generate_async for streaming


# FastAPI web app for HTTP endpoint
from fastapi import FastAPI
from fastapi.responses import Response
from pydantic import BaseModel

web_app = FastAPI()


class TTSRequest(BaseModel):
    text: str
    voice: str = "tara"
    max_tokens: int = 2000


@web_app.post("/tts")
async def tts(request: TTSRequest):
    """Generate TTS audio from text."""
    tts_model = OrpheusTRTLLM()
    result = tts_model.generate.remote(
        text=request.text,
        voice=request.voice,
        max_tokens=request.max_tokens,
    )

    if "error" in result:
        return {"error": result["error"]}

    # Return WAV audio
    import io
    import wave

    audio_bytes = result["audio"]
    wav_buffer = io.BytesIO()

    with wave.open(wav_buffer, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_bytes)

    return Response(
        content=wav_buffer.getvalue(),
        media_type="audio/wav",
        headers={
            "X-Duration": str(result["duration_s"]),
            "X-Tokens": str(result["audio_tokens"]),
        }
    )


@app.function(image=image)
@modal.asgi_app()
def serve():
    """Serve the TTS API."""
    return web_app


@app.local_entrypoint()
def main(
    text: str = "Hello, this is a test of the TensorRT LLM text to speech system.",
    voice: str = "tara",
    output: str = "trtllm_output.wav",
):
    """Run TTS inference and save to file."""
    import wave

    print(f"Text: {text}")
    print(f"Voice: {voice}")
    print()

    tts = OrpheusTRTLLM()
    result = tts.generate.remote(text=text, voice=voice)

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Save audio
    audio_bytes = result["audio"]
    with wave.open(output, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(audio_bytes)

    print(f"\nSaved to: {output}")
    print(f"Duration: {result['duration_s']:.2f}s")
    print(f"Audio tokens: {result['audio_tokens']}")

    print("\nTimings:")
    for k, v in result["timings"].items():
        print(f"  {k}: {v:.0f}ms")
