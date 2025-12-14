"""
TensorRT-LLM TTS Benchmark Script

Measures key performance metrics:
- TTFT (Time to First Token): Time from request to first generated token
- TPS (Tokens Per Second): Token generation throughput
- SNAC Decode Time: Audio decoding latency
- End-to-End Latency: Total time from text to audio

Usage:
    modal run tensorrt_tts/hindi_finetuned/benchmark.py
    modal run tensorrt_tts/hindi_finetuned/benchmark.py --runs 5
"""

import modal

app = modal.App("orpheus-hindi-trtllm-benchmark")

image = (
    modal.Image.from_registry(
        "nvcr.io/nvidia/tensorrt-llm/release:0.21.0",
        secret=modal.Secret.from_name("nvcr-credentials"),
    )
    .pip_install(
        "snac",
        "numpy",
        "transformers",
    )
)

engine_vol = modal.Volume.from_name("orpheus-hindi-trtllm-engine-a100", create_if_missing=True)
cache_vol = modal.Volume.from_name("orpheus-cache", create_if_missing=True)

ENGINE_DIR = "/engine/trt_engine"

# Token constants
SOS_TOKEN = 128257
EOS_TOKEN = 128258
START_TOKEN = 128259
END_OF_TEXT = 128009
END_OF_TURN = 128260
TOKEN_BASE = 128266
FRAME_SIZE = 7
SAMPLE_RATE = 24000

# Test prompts of varying lengths
TEST_PROMPTS = {
    "tiny": "नमस्ते",  # 6 chars
    "short": "नमस्ते, आप कैसे हैं?",  # 20 chars
    "medium": "नमस्ते, मैं एक हिंदी टेक्स्ट टू स्पीच मॉडल हूं। आज का मौसम बहुत अच्छा है।",  # 70 chars
    "long": "नमस्ते, मेरा नाम एलिस है। मैं एक हिंदी टेक्स्ट टू स्पीच मॉडल हूं जो आपकी भाषा में बातचीत कर सकती हूं। आज का मौसम बहुत खूबसूरत है और मुझे उम्मीद है कि आप अच्छे हैं।",  # 160 chars
}


def redistribute_codes(codes: list[int]) -> tuple[list[int], list[int], list[int]]:
    """Redistribute codes into SNAC layers."""
    layer_1, layer_2, layer_3 = [], [], []
    for i in range((len(codes) + 1) // 7):
        layer_1.append(codes[7 * i])
        layer_2.append(codes[7 * i + 1] - 4096)
        layer_3.append(codes[7 * i + 2] - (2 * 4096))
        layer_3.append(codes[7 * i + 3] - (3 * 4096))
        layer_2.append(codes[7 * i + 4] - (4 * 4096))
        layer_3.append(codes[7 * i + 5] - (5 * 4096))
        layer_3.append(codes[7 * i + 6] - (6 * 4096))
    return layer_1, layer_2, layer_3


@app.function(
    image=image,
    gpu="A100",
    volumes={
        "/engine": engine_vol,
        "/cache": cache_vol,
    },
    timeout=1800,
)
def run_benchmark(
    prompt_key: str = "medium",
    num_runs: int = 3,
    warmup_runs: int = 1,
    max_tokens: int = 2000,
) -> dict:
    """
    Run comprehensive benchmark measuring TTFT, TPS, and decode times.

    Returns detailed metrics for each run and aggregated statistics.
    """
    import torch
    import numpy as np
    import time
    import os
    from tensorrt_llm import LLM, SamplingParams
    from transformers import AutoTokenizer
    from snac import SNAC

    if not os.path.exists(f"{ENGINE_DIR}/rank0.engine"):
        return {"error": "Engine not found"}

    text = TEST_PROMPTS.get(prompt_key, TEST_PROMPTS["medium"])

    print("=" * 70)
    print("TensorRT-LLM TTS BENCHMARK")
    print("=" * 70)
    print(f"Prompt: {prompt_key} ({len(text)} chars)")
    print(f"Runs: {num_runs} (+ {warmup_runs} warmup)")
    print(f"Max tokens: {max_tokens}")
    print()

    # =========================================================================
    # PHASE 1: Model Loading (cold start measurement)
    # =========================================================================
    print("Phase 1: Model Loading...")

    load_start = time.perf_counter()

    tokenizer_start = time.perf_counter()
    tokenizer = AutoTokenizer.from_pretrained(ENGINE_DIR)
    tokenizer_time = time.perf_counter() - tokenizer_start

    llm_start = time.perf_counter()
    llm = LLM(model=ENGINE_DIR)
    llm_load_time = time.perf_counter() - llm_start

    snac_start = time.perf_counter()
    snac = SNAC.from_pretrained("hubertsiuzdak/snac_24khz").eval().cuda()
    snac_load_time = time.perf_counter() - snac_start

    total_load_time = time.perf_counter() - load_start

    print(f"  Tokenizer: {tokenizer_time:.3f}s")
    print(f"  TRT-LLM Engine: {llm_load_time:.3f}s")
    print(f"  SNAC Decoder: {snac_load_time:.3f}s")
    print(f"  Total: {total_load_time:.3f}s")
    print()

    # =========================================================================
    # PHASE 2: Prepare prompt
    # =========================================================================
    prompt_text = f"tara: {text}"
    prompt_tokens = tokenizer.encode(prompt_text, add_special_tokens=True)
    prompt_ids = [START_TOKEN] + prompt_tokens + [END_OF_TEXT, END_OF_TURN]

    print(f"Input tokens: {len(prompt_ids)}")
    print()

    # =========================================================================
    # PHASE 3: Warmup runs
    # =========================================================================
    print(f"Phase 2: Warmup ({warmup_runs} runs)...")

    sampling_params = SamplingParams(
        max_tokens=max_tokens,
        stop_token_ids=[EOS_TOKEN],
        temperature=0.6,
        top_p=0.95,
        repetition_penalty=1.1,
    )

    for i in range(warmup_runs):
        outputs = llm.generate([prompt_ids], sampling_params=sampling_params)
        output_ids = list(outputs[0].outputs[0].token_ids)
        print(f"  Warmup {i+1}: {len(output_ids)} tokens")
    print()

    # =========================================================================
    # PHASE 4: Benchmark runs
    # =========================================================================
    print(f"Phase 3: Benchmark ({num_runs} runs)...")

    results = []

    for run_idx in range(num_runs):
        run_result = {}

        # Clear CUDA cache
        torch.cuda.synchronize()
        torch.cuda.empty_cache()

        # ----- Token Generation -----
        gen_start = time.perf_counter()

        # Note: TRT-LLM's high-level API doesn't expose per-token timing
        # TTFT would require using the lower-level streaming API
        # For now, we measure total generation time
        outputs = llm.generate([prompt_ids], sampling_params=sampling_params)

        torch.cuda.synchronize()
        gen_end = time.perf_counter()

        output_ids = list(outputs[0].outputs[0].token_ids)
        gen_time = gen_end - gen_start

        # Extract audio tokens
        sos_indices = [i for i, tid in enumerate(output_ids) if tid == SOS_TOKEN]
        if sos_indices:
            last_sos_idx = sos_indices[-1]
            audio_tokens = []
            for tid in output_ids[last_sos_idx + 1:]:
                if tid == EOS_TOKEN:
                    break
                audio_tokens.append(tid)
        else:
            audio_tokens = []

        run_result["output_tokens"] = len(output_ids)
        run_result["audio_tokens"] = len(audio_tokens)
        run_result["gen_time_s"] = gen_time
        run_result["tps"] = len(output_ids) / gen_time if gen_time > 0 else 0

        # ----- SNAC Decoding -----
        if audio_tokens:
            codes = [t - TOKEN_BASE for t in audio_tokens]
            num_frames = len(codes) // FRAME_SIZE
            codes = codes[:num_frames * FRAME_SIZE]

            layer_1, layer_2, layer_3 = redistribute_codes(codes)

            torch.cuda.synchronize()
            decode_start = time.perf_counter()

            codes_tensor = [
                torch.tensor(layer_1, dtype=torch.int32, device="cuda").unsqueeze(0),
                torch.tensor(layer_2, dtype=torch.int32, device="cuda").unsqueeze(0),
                torch.tensor(layer_3, dtype=torch.int32, device="cuda").unsqueeze(0),
            ]

            for i in range(3):
                codes_tensor[i] = torch.clamp(codes_tensor[i], 0, 4095)

            with torch.inference_mode():
                audio = snac.decode(codes_tensor)

            torch.cuda.synchronize()
            decode_end = time.perf_counter()

            audio_np = audio.squeeze().cpu().numpy()
            audio_np = np.clip(audio_np, -1.0, 1.0)

            decode_time = decode_end - decode_start
            duration = len(audio_np) / SAMPLE_RATE

            run_result["decode_time_s"] = decode_time
            run_result["duration_s"] = duration
            run_result["frames"] = num_frames
            run_result["samples"] = len(audio_np)
            run_result["decode_rtf"] = duration / decode_time if decode_time > 0 else 0
        else:
            run_result["decode_time_s"] = 0
            run_result["duration_s"] = 0
            run_result["frames"] = 0
            run_result["decode_rtf"] = 0

        # Total end-to-end time (excluding model load)
        run_result["e2e_time_s"] = gen_time + run_result["decode_time_s"]
        run_result["e2e_rtf"] = run_result["duration_s"] / run_result["e2e_time_s"] if run_result["e2e_time_s"] > 0 else 0

        results.append(run_result)

        print(f"  Run {run_idx+1}: {run_result['output_tokens']} tokens, "
              f"gen={run_result['gen_time_s']:.3f}s ({run_result['tps']:.1f} TPS), "
              f"decode={run_result['decode_time_s']:.3f}s, "
              f"audio={run_result['duration_s']:.2f}s")

    print()

    # =========================================================================
    # PHASE 5: Aggregate statistics
    # =========================================================================
    print("=" * 70)
    print("RESULTS SUMMARY")
    print("=" * 70)

    # Calculate aggregates
    def calc_stats(values):
        arr = np.array(values)
        return {
            "mean": float(arr.mean()),
            "std": float(arr.std()),
            "min": float(arr.min()),
            "max": float(arr.max()),
            "p50": float(np.percentile(arr, 50)),
            "p95": float(np.percentile(arr, 95)),
        }

    summary = {
        "prompt_key": prompt_key,
        "prompt_chars": len(text),
        "input_tokens": len(prompt_ids),
        "num_runs": num_runs,
        "gpu": "A100",
        "model_load": {
            "tokenizer_s": tokenizer_time,
            "trt_llm_s": llm_load_time,
            "snac_s": snac_load_time,
            "total_s": total_load_time,
        },
        "generation": {
            "output_tokens": calc_stats([r["output_tokens"] for r in results]),
            "audio_tokens": calc_stats([r["audio_tokens"] for r in results]),
            "time_s": calc_stats([r["gen_time_s"] for r in results]),
            "tps": calc_stats([r["tps"] for r in results]),
        },
        "decode": {
            "time_s": calc_stats([r["decode_time_s"] for r in results]),
            "frames": calc_stats([r["frames"] for r in results]),
            "duration_s": calc_stats([r["duration_s"] for r in results]),
            "rtf": calc_stats([r["decode_rtf"] for r in results]),
        },
        "e2e": {
            "time_s": calc_stats([r["e2e_time_s"] for r in results]),
            "rtf": calc_stats([r["e2e_rtf"] for r in results]),
        },
        "raw_results": results,
    }

    print(f"Prompt: {prompt_key} ({len(text)} chars, {len(prompt_ids)} input tokens)")
    print()
    print("Model Loading (cold start):")
    print(f"  Tokenizer:      {tokenizer_time:.3f}s")
    print(f"  TRT-LLM Engine: {llm_load_time:.3f}s")
    print(f"  SNAC Decoder:   {snac_load_time:.3f}s")
    print(f"  Total:          {total_load_time:.3f}s")
    print()
    print("Token Generation:")
    print(f"  Tokens:    {summary['generation']['output_tokens']['mean']:.0f} ± {summary['generation']['output_tokens']['std']:.0f}")
    print(f"  Time:      {summary['generation']['time_s']['mean']:.3f}s ± {summary['generation']['time_s']['std']:.3f}s")
    print(f"  TPS:       {summary['generation']['tps']['mean']:.1f} ± {summary['generation']['tps']['std']:.1f} tok/s")
    print()
    print("SNAC Decoding:")
    print(f"  Frames:    {summary['decode']['frames']['mean']:.0f}")
    print(f"  Time:      {summary['decode']['time_s']['mean']:.3f}s ± {summary['decode']['time_s']['std']:.3f}s")
    print(f"  RTF:       {summary['decode']['rtf']['mean']:.1f}x realtime")
    print()
    print("End-to-End (excl. model load):")
    print(f"  Time:      {summary['e2e']['time_s']['mean']:.3f}s ± {summary['e2e']['time_s']['std']:.3f}s")
    print(f"  RTF:       {summary['e2e']['rtf']['mean']:.2f}x realtime")
    print(f"  Audio:     {summary['decode']['duration_s']['mean']:.2f}s")
    print()

    # TTFT estimation
    # TRT-LLM batch API doesn't give per-token timing, but we can estimate:
    # TTFT ≈ prefill_time + first_decode_step
    # For this model: prefill is fast (~10-50ms), decode is ~8-10ms/token
    avg_tokens = summary['generation']['output_tokens']['mean']
    avg_gen_time = summary['generation']['time_s']['mean']
    estimated_decode_time_per_token = avg_gen_time / avg_tokens if avg_tokens > 0 else 0
    estimated_ttft = estimated_decode_time_per_token * 2  # Rough estimate: prefill + 1 decode

    print("Estimated TTFT (Time to First Token):")
    print(f"  ~{estimated_ttft*1000:.1f}ms (estimated from avg decode time)")
    print(f"  Decode time/token: {estimated_decode_time_per_token*1000:.2f}ms")
    print()

    summary["estimated_ttft_ms"] = estimated_ttft * 1000
    summary["estimated_decode_time_per_token_ms"] = estimated_decode_time_per_token * 1000

    return summary


@app.local_entrypoint()
def main(
    prompt: str = "medium",
    runs: int = 3,
    warmup: int = 1,
    max_tokens: int = 2000,
):
    """Run TTS benchmark."""
    import json

    result = run_benchmark.remote(
        prompt_key=prompt,
        num_runs=runs,
        warmup_runs=warmup,
        max_tokens=max_tokens,
    )

    if "error" in result:
        print(f"Error: {result['error']}")
        return

    # Save results
    with open("benchmark_results.json", "w") as f:
        # Remove raw audio data for JSON serialization
        result_clean = {k: v for k, v in result.items()}
        json.dump(result_clean, f, indent=2)

    print(f"\nResults saved to benchmark_results.json")
