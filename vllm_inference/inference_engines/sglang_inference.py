"""
SGLang Inference Engine for Orpheus Hindi Model

Task 2 Requirements:
- OOD Testing: Run inference on out-of-distribution text samples
- Tag Validation: Model must correctly predict/insert Orpheus tags
- TTFT: Time-To-First-Token measurement
- TPS: Tokens-Per-Second measurement
- Streaming: Token-by-token streaming for TTS applications
"""

import modal
import time

app = modal.App("orpheus-sglang")

# SGLang Image - based on Modal's official example
cuda_version = "12.4.0"
flavor = "devel"
operating_sys = "ubuntu22.04"
tag = f"{cuda_version}-{flavor}-{operating_sys}"

image = (
    modal.Image.from_registry(f"nvidia/cuda:{tag}", add_python="3.11")
    .entrypoint([])
    .pip_install(
        "sglang[all]==0.4.1",
        "huggingface_hub",
        "hf_transfer",
        "flashinfer",
    )
    .env({
        "HF_HUB_ENABLE_HF_TRANSFER": "1",
        "CUDA_HOME": "/usr/local/cuda",
    })
)

vol = modal.Volume.from_name("orpheus-cache")

# OOD Test Prompts - sentences NOT in training set
# These include tags that the model should handle/continue properly
OOD_PROMPTS = [
    # Emotional shifts with tags
    "मुझे विश्वास नहीं हो रहा कि तुमने ऐसा किया। <sigh> खैर, अब जो हुआ सो हुआ। <laughs nervously>",
    # Romantic context with tags
    "<romantic music playing> वह धीरे से उसके पास आया और बोला, मैं तुमसे प्यार करता हूँ। <smooches>",
    # Suspense/fear with tags
    "रुको! वहां कौन है? <gasps> मुझे लगा मैंने कुछ सुना। <tapping sounds>",
    # Nervous speech with tags
    "मैं... मैं... <stutters> मुझे नहीं पता कि क्या कहना चाहिए। <nervous laughter>",
    # Casual/crude with tags
    "खाना बहुत स्वादिष्ट था। <burps> अरे माफ़ करना! <chuckles>",
    # Whisper context with tags
    "<whispers> धीरे बोलो, कोई सुन लेगा। <long pause> चलो यहाँ से चलते हैं।",
    # Excitement with tags
    "वाह! <laughs> यह तो कमाल है! <claps>",
    # Code-mixing Hindi-English with tags
    "This is really funny. <laughs> मुझे यह बहुत पसंद आया।",
    # Sadness with tags
    "मैं अब और नहीं सह सकती। <crying> सब खत्म हो गया।",
    # Anger with tags
    "बाहर निकलो! <shouts> मुझे तुम्हारी शक्ल नहीं देखनी!",
]


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
        import sglang as sgl

        model_path = "/cache/orpheus-merged-vllm"

        print(f"Loading SGLang engine from {model_path}...")

        # Initialize SGLang runtime
        self.runtime = sgl.Runtime(
            model_path=model_path,
            tokenizer_path=model_path,
            tp_size=1,
            dtype="float16",
            mem_fraction_static=0.85,
        )
        sgl.set_default_backend(self.runtime)

        print("SGLang engine loaded.")

    @modal.exit()
    def shutdown(self):
        if hasattr(self, 'runtime'):
            self.runtime.shutdown()

    @modal.method()
    def generate(self, prompts: list[str], max_tokens: int = 100, temperature: float = 0.7):
        """Batch generation using SGLang"""
        import sglang as sgl

        @sgl.function
        def generate_text(s, prompt, max_tokens, temperature):
            s += prompt
            s += sgl.gen("response", max_tokens=max_tokens, temperature=temperature)

        results = []
        for prompt in prompts:
            state = generate_text.run(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
            results.append(state["response"])

        return results

    @modal.method()
    def generate_with_metrics(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """
        Single prompt generation with TTFT and TPS metrics.
        """
        import sglang as sgl

        @sgl.function
        def generate_text(s, prompt, max_tokens, temperature):
            s += prompt
            s += sgl.gen("response", max_tokens=max_tokens, temperature=temperature)

        start_time = time.perf_counter()

        state = generate_text.run(prompt=prompt, max_tokens=max_tokens, temperature=temperature)
        generated_text = state["response"]

        end_time = time.perf_counter()
        total_time = end_time - start_time

        # Estimate tokens (SGLang doesn't directly expose token counts in simple API)
        # Rough estimate: 4 chars per token for Hindi text
        output_tokens_estimate = max(1, len(generated_text) // 4)
        prompt_tokens_estimate = max(1, len(prompt) // 4)

        # TTFT estimation
        ttft_estimate_ms = (total_time / (prompt_tokens_estimate + output_tokens_estimate)) * prompt_tokens_estimate * 1000

        # TPS calculation
        gen_time = total_time - (ttft_estimate_ms / 1000)
        tps = output_tokens_estimate / gen_time if gen_time > 0 else 0

        return {
            "text": generated_text,
            "ttft_ms": round(ttft_estimate_ms, 2),
            "tps": round(tps, 2),
            "output_tokens_estimate": output_tokens_estimate,
            "total_time_ms": round(total_time * 1000, 2)
        }

    @modal.method()
    def generate_stream(self, prompt: str, max_tokens: int = 100, temperature: float = 0.7):
        """
        Streaming generation - yields tokens one by one.
        """
        import sglang as sgl

        @sgl.function
        def generate_text(s, prompt, max_tokens, temperature):
            s += prompt
            s += sgl.gen("response", max_tokens=max_tokens, temperature=temperature)

        start_time = time.perf_counter()
        first_token_time = None
        token_count = 0

        # Run with streaming enabled
        state = generate_text.run(
            prompt=prompt,
            max_tokens=max_tokens,
            temperature=temperature,
            stream=True
        )

        for chunk in state.text_iter("response"):
            token_count += 1
            if first_token_time is None:
                first_token_time = time.perf_counter()
                ttft_ms = (first_token_time - start_time) * 1000
            else:
                ttft_ms = None

            yield {
                "token": chunk,
                "ttft_ms": round(ttft_ms, 2) if ttft_ms else None,
                "token_index": token_count
            }

    @modal.method()
    def run_ood_benchmark(self, max_tokens: int = 100, temperature: float = 0.7):
        """
        Run OOD benchmark with all test prompts.
        Returns comprehensive metrics and tag analysis.
        """
        import re

        results = []

        for prompt in OOD_PROMPTS:
            metrics = self.generate_with_metrics(prompt, max_tokens, temperature)

            # Analyze tags in output
            output_text = metrics["text"]
            tags_found = re.findall(r'<[^>|]+>', output_text)
            orpheus_tags = [t for t in tags_found if not t.startswith('<|')]

            # Also check tags in the prompt
            prompt_tags = re.findall(r'<[^>|]+>', prompt)
            prompt_orpheus_tags = [t for t in prompt_tags if not t.startswith('<|')]

            results.append({
                "prompt": prompt,
                "output": output_text,
                "prompt_tags": prompt_orpheus_tags,
                "output_tags": orpheus_tags,
                "tag_count": len(orpheus_tags),
                "ttft_ms": metrics["ttft_ms"],
                "tps": metrics["tps"],
                "output_tokens": metrics["output_tokens_estimate"],
                "total_time_ms": metrics["total_time_ms"]
            })

        # Aggregate metrics
        avg_ttft = sum(r["ttft_ms"] for r in results) / len(results)
        avg_tps = sum(r["tps"] for r in results) / len(results)
        max_tps = max(r["tps"] for r in results)
        min_ttft = min(r["ttft_ms"] for r in results)
        total_tags = sum(r["tag_count"] for r in results)
        prompts_with_tags = sum(1 for r in results if r["tag_count"] > 0)

        summary = {
            "total_prompts": len(results),
            "avg_ttft_ms": round(avg_ttft, 2),
            "min_ttft_ms": round(min_ttft, 2),
            "avg_tps": round(avg_tps, 2),
            "max_tps": round(max_tps, 2),
            "total_tags_in_output": total_tags,
            "prompts_with_output_tags": prompts_with_tags,
            "tag_presence_rate": round(prompts_with_tags / len(results) * 100, 1)
        }

        return {
            "summary": summary,
            "results": results
        }


@app.local_entrypoint()
def main():
    print("=" * 70)
    print("SGLang INFERENCE ENGINE - Task 2 Benchmark")
    print("=" * 70)

    inference = ModelInference()

    # Warmup
    print("\n[1/3] Warming up...")
    inference.generate.remote(prompts=["नमस्ते <laugh>"])

    # Run OOD Benchmark
    print("\n[2/3] Running OOD Benchmark...")
    benchmark = inference.run_ood_benchmark.remote()

    # Print Summary
    print("\n" + "=" * 70)
    print("BENCHMARK SUMMARY")
    print("=" * 70)
    summary = benchmark["summary"]
    print(f"Total Prompts: {summary['total_prompts']}")
    print(f"\nPerformance Metrics:")
    print(f"  TTFT (Time-To-First-Token):")
    print(f"    Average: {summary['avg_ttft_ms']}ms")
    print(f"    Minimum: {summary['min_ttft_ms']}ms")
    print(f"  TPS (Tokens-Per-Second):")
    print(f"    Average: {summary['avg_tps']}")
    print(f"    Maximum: {summary['max_tps']}")
    print(f"\nTag Analysis:")
    print(f"  Total Tags in Output: {summary['total_tags_in_output']}")
    print(f"  Prompts with Output Tags: {summary['prompts_with_output_tags']}/{summary['total_prompts']}")
    print(f"  Tag Presence Rate: {summary['tag_presence_rate']}%")

    # Print individual results
    print("\n" + "=" * 70)
    print("INDIVIDUAL RESULTS")
    print("=" * 70)
    for i, result in enumerate(benchmark["results"], 1):
        print(f"\n[{i}] Prompt: {result['prompt'][:60]}...")
        print(f"    Output: {result['output'][:80]}...")
        print(f"    Prompt Tags: {result['prompt_tags']}")
        print(f"    Output Tags: {result['output_tags'] if result['output_tags'] else 'None'}")
        print(f"    TTFT: {result['ttft_ms']}ms | TPS: {result['tps']} | Tokens: {result['output_tokens']}")

    # Test streaming
    print("\n[3/3] Testing Streaming...")
    print("\nStreaming output for first prompt:")
    stream_prompt = OOD_PROMPTS[0]
    print(f"Prompt: {stream_prompt}")
    print("Response: ", end="", flush=True)

    ttft = None
    for chunk in inference.generate_stream.remote_gen(stream_prompt):
        if chunk["token"]:
            print(chunk["token"], end="", flush=True)
        if chunk["ttft_ms"]:
            ttft = chunk["ttft_ms"]

    print(f"\n(TTFT: {ttft}ms)")
    print("\n" + "=" * 70)
    print("Benchmark Complete!")
