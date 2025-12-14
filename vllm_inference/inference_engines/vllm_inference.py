"""
vLLM Async Inference Engine for Orpheus Hindi Model
Streaming-first design using AsyncLLMEngine
"""

import modal
import time

app = modal.App("orpheus-vllm")

# vLLM Image
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install(
        "vllm",
        "huggingface_hub",
        "hf_transfer"
    )
    .env({"HF_HUB_ENABLE_HF_TRANSFER": "1"})
)

vol = modal.Volume.from_name("orpheus-cache")

# Test prompt
TEST_PROMPT = "मुझे विश्वास नहीं हो रहा कि तुमने ऐसा किया। <sigh> खैर, अब जो हुआ सो हुआ। <laughs nervously>"


@app.cls(
    image=image,
    gpu="A100",
    volumes={"/cache": vol},
    timeout=600,
    scaledown_window=300
)
class ModelInference:
    @modal.enter()
    def load_model(self):
        from vllm.engine.async_llm_engine import AsyncLLMEngine
        from vllm.engine.arg_utils import AsyncEngineArgs

        model_path = "/cache/orpheus-merged-vllm"
        print(f"Loading async vLLM engine from {model_path}...")

        engine_args = AsyncEngineArgs(
            model=model_path,
            tokenizer=model_path,
            trust_remote_code=True,
            dtype="float16",
            gpu_memory_utilization=0.90,
            tensor_parallel_size=1,
            enable_prefix_caching=True,
        )

        self.engine = AsyncLLMEngine.from_engine_args(engine_args)
        print("Async vLLM engine loaded.")

    @modal.method()
    async def generate_stream(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """
        Async streaming generation - yields tokens one by one with TTFT measurement.
        """
        from vllm import SamplingParams
        import uuid

        sampling_params = SamplingParams(
            temperature=temperature,
            max_tokens=max_tokens,
        )

        request_id = str(uuid.uuid4())
        start_time = time.perf_counter()
        first_token_time = None
        token_index = 0
        prev_text = ""

        async for request_output in self.engine.generate(
            prompt,
            sampling_params,
            request_id=request_id,
        ):
            for output in request_output.outputs:
                # Get only the new text (delta)
                new_text = output.text[len(prev_text):]
                prev_text = output.text

                if new_text:
                    token_index += 1

                    if first_token_time is None:
                        first_token_time = time.perf_counter()
                        ttft_ms = round((first_token_time - start_time) * 1000, 2)
                    else:
                        ttft_ms = None

                    yield {
                        "text": new_text,
                        "token_index": token_index,
                        "ttft_ms": ttft_ms,
                    }


@app.local_entrypoint()
async def main():
    """Test streaming inference"""
    print("=" * 70)
    print("vLLM STREAMING TEST")
    print("=" * 70)

    inference = ModelInference()

    print(f"\nPrompt: {TEST_PROMPT}")
    print("\nStreaming Response: ", end="", flush=True)

    ttft = None
    token_count = 0

    async for chunk in inference.generate_stream.remote_gen.aio(TEST_PROMPT):
        print(chunk["text"], end="", flush=True)
        token_count += 1
        if chunk["ttft_ms"]:
            ttft = chunk["ttft_ms"]

    print(f"\n\n--- Stats ---")
    print(f"TTFT: {ttft}ms")
    print(f"Tokens: {token_count}")
    print("=" * 70)
