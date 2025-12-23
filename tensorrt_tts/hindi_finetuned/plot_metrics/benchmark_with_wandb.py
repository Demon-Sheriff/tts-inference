"""
Comprehensive Benchmark with Metrics Visualization and W&B Logging

Features:
- Chunk arrival timeline plots (reveals stalls)
- Max chunk gap tracking (critical metric)
- 10-20 iteration runs (drift/thermal/GC analysis)
- W&B dashboard integration

Usage:
    # Run 20 iterations with W&B logging
    python tensorrt_tts/hindi_finetuned/plot_metrics/benchmark_with_wandb.py \
        --url wss://<modal-url>/ws/tts \
        --iterations 20 \
        --wandb

    # Local plots only (no W&B)
    python tensorrt_tts/hindi_finetuned/plot_metrics/benchmark_with_wandb.py \
        --url wss://<modal-url>/ws/tts \
        --iterations 10
"""

import asyncio
import json
import time
import wave
import argparse
import statistics
import os
from dataclasses import dataclass, field
from typing import Optional


SAMPLE_RATE = 24000


@dataclass
class ChunkEvent:
    """Single chunk arrival event."""
    chunk_id: int
    arrival_time_ms: float  # Relative to connection start
    size_bytes: int
    duration_ms: float  # Audio duration in this chunk
    gap_from_prev_ms: float = 0.0  # Time since previous chunk


@dataclass
class IterationResult:
    """Full metrics for one benchmark iteration."""
    iteration: int
    text_length: int

    # Timing
    client_ttfa_ms: float = 0.0
    client_wall_time_ms: float = 0.0
    audio_duration_ms: float = 0.0

    # Server metrics (if available)
    server_ttft_ms: Optional[float] = None
    server_ttfa_ms: Optional[float] = None
    server_rtf: Optional[float] = None

    # Chunk analysis
    total_chunks: int = 0
    chunk_events: list = field(default_factory=list)
    max_chunk_gap_ms: float = 0.0
    mean_chunk_gap_ms: float = 0.0
    stddev_chunk_gap_ms: float = 0.0

    # RTF
    client_rtf: float = 0.0

    # Audio data (for saving)
    audio_bytes: bytes = b""


async def run_single_iteration(
    url: str,
    text: str,
    voice: str = "tara",
    iteration: int = 1,
    frames_per_chunk: int = 5,
) -> IterationResult:
    """Run a single benchmark iteration with detailed chunk tracking."""
    try:
        import websockets
    except ImportError:
        raise ImportError("Please install websockets: pip install websockets")

    result = IterationResult(
        iteration=iteration,
        text_length=len(text),
    )

    audio_chunks = []
    chunk_events = []

    connect_start = time.perf_counter()

    try:
        async with websockets.connect(url, ping_interval=30, ping_timeout=60) as ws:
            # Send request with benchmark flag
            request = {
                "text": text,
                "voice": voice,
                "temperature": 0.6,
                "top_p": 0.95,
                "frames_per_chunk": frames_per_chunk,
                "benchmark": True,
            }
            await ws.send(json.dumps(request))
            request_time = time.perf_counter()

            last_chunk_time = None
            chunk_id = 0

            while True:
                message = await ws.recv()
                recv_time = time.perf_counter()

                if isinstance(message, bytes):
                    # Binary audio chunk
                    chunk_id += 1
                    arrival_ms = (recv_time - request_time) * 1000

                    if result.client_ttfa_ms == 0:
                        result.client_ttfa_ms = arrival_ms

                    # Calculate gap from previous chunk
                    gap_ms = 0.0
                    if last_chunk_time is not None:
                        gap_ms = (recv_time - last_chunk_time) * 1000
                    last_chunk_time = recv_time

                    # Audio duration of this chunk
                    samples = len(message) // 2  # int16 = 2 bytes
                    chunk_duration_ms = (samples / SAMPLE_RATE) * 1000

                    chunk_events.append(ChunkEvent(
                        chunk_id=chunk_id,
                        arrival_time_ms=arrival_ms,
                        size_bytes=len(message),
                        duration_ms=chunk_duration_ms,
                        gap_from_prev_ms=gap_ms,
                    ))

                    audio_chunks.append(message)

                else:
                    # JSON message
                    data = json.loads(message)

                    if data.get("done"):
                        result.client_wall_time_ms = (recv_time - request_time) * 1000
                        result.total_chunks = data.get("chunks", chunk_id)

                        # Extract server metrics
                        server_metrics = data.get("server_metrics", {})
                        if server_metrics:
                            result.server_ttft_ms = server_metrics.get("server_ttft_ms")
                            result.server_ttfa_ms = server_metrics.get("server_ttfa_ms")
                            result.server_rtf = server_metrics.get("server_rtf")

                        break

                    elif data.get("error"):
                        raise Exception(f"Server error: {data['error']}")

    except Exception as e:
        print(f"  Error in iteration {iteration}: {e}")
        return result

    # Calculate audio duration
    total_bytes = sum(len(c) for c in audio_chunks)
    total_samples = total_bytes // 2
    result.audio_duration_ms = (total_samples / SAMPLE_RATE) * 1000
    result.audio_bytes = b"".join(audio_chunks)

    # Analyze chunk gaps
    result.chunk_events = chunk_events
    gaps = [e.gap_from_prev_ms for e in chunk_events if e.gap_from_prev_ms > 0]

    if gaps:
        result.max_chunk_gap_ms = max(gaps)
        result.mean_chunk_gap_ms = statistics.mean(gaps)
        if len(gaps) > 1:
            result.stddev_chunk_gap_ms = statistics.stdev(gaps)

    # Calculate RTF
    if result.client_wall_time_ms > 0:
        result.client_rtf = result.audio_duration_ms / result.client_wall_time_ms

    return result


def create_chunk_timeline_plot(results: list[IterationResult], output_dir: str):
    """Create chunk arrival timeline visualization."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping local plots")
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Plot 1: Chunk arrival timeline for each iteration
    ax1 = axes[0, 0]
    colors = plt.cm.viridis(np.linspace(0, 1, len(results)))

    for i, result in enumerate(results):
        if result.chunk_events:
            times = [e.arrival_time_ms / 1000 for e in result.chunk_events]
            chunk_ids = [e.chunk_id for e in result.chunk_events]
            ax1.plot(times, chunk_ids, marker='.', markersize=3,
                    color=colors[i], alpha=0.7, label=f"Iter {result.iteration}")

    ax1.set_xlabel("Time (seconds)")
    ax1.set_ylabel("Chunk ID")
    ax1.set_title("Chunk Arrival Timeline")
    ax1.grid(True, alpha=0.3)
    if len(results) <= 10:
        ax1.legend(fontsize=8)

    # Plot 2: Chunk gaps over time (reveals stalls)
    ax2 = axes[0, 1]

    for i, result in enumerate(results):
        if result.chunk_events:
            times = [e.arrival_time_ms / 1000 for e in result.chunk_events[1:]]
            gaps = [e.gap_from_prev_ms for e in result.chunk_events[1:]]
            ax2.scatter(times, gaps, s=10, color=colors[i], alpha=0.6)

    ax2.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='200ms threshold')
    ax2.axhline(y=500, color='darkred', linestyle='--', alpha=0.5, label='500ms threshold')
    ax2.set_xlabel("Time (seconds)")
    ax2.set_ylabel("Gap from previous chunk (ms)")
    ax2.set_title("Chunk Gap Analysis (Stall Detection)")
    ax2.grid(True, alpha=0.3)
    ax2.legend()

    # Plot 3: Max chunk gap per iteration (drift/thermal analysis)
    ax3 = axes[1, 0]
    iterations = [r.iteration for r in results]
    max_gaps = [r.max_chunk_gap_ms for r in results]
    mean_gaps = [r.mean_chunk_gap_ms for r in results]

    ax3.bar(iterations, max_gaps, alpha=0.7, label='Max gap', color='coral')
    ax3.plot(iterations, mean_gaps, marker='o', color='blue', label='Mean gap')
    ax3.axhline(y=200, color='red', linestyle='--', alpha=0.5)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel("Chunk gap (ms)")
    ax3.set_title("Max Chunk Gap per Iteration (Drift Detection)")
    ax3.grid(True, alpha=0.3)
    ax3.legend()

    # Plot 4: RTF and TTFA over iterations
    ax4 = axes[1, 1]
    rtfs = [r.client_rtf for r in results]
    ttfas = [r.client_ttfa_ms for r in results]

    ax4_twin = ax4.twinx()

    line1, = ax4.plot(iterations, rtfs, marker='o', color='green', label='Client RTF')
    ax4.axhline(y=1.0, color='green', linestyle='--', alpha=0.3)
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("RTF (x realtime)", color='green')
    ax4.tick_params(axis='y', labelcolor='green')

    line2, = ax4_twin.plot(iterations, ttfas, marker='s', color='purple', label='TTFA')
    ax4_twin.set_ylabel("TTFA (ms)", color='purple')
    ax4_twin.tick_params(axis='y', labelcolor='purple')

    ax4.set_title("RTF and TTFA Stability Over Time")
    ax4.grid(True, alpha=0.3)
    ax4.legend([line1, line2], ['Client RTF', 'TTFA'], loc='upper right')

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "chunk_timeline_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")


def create_detailed_gap_analysis(results: list[IterationResult], output_dir: str):
    """Create detailed gap analysis plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        return

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect all gaps across all iterations
    all_gaps = []
    for r in results:
        gaps = [e.gap_from_prev_ms for e in r.chunk_events if e.gap_from_prev_ms > 0]
        all_gaps.extend(gaps)

    if not all_gaps:
        return

    # Plot 1: Gap distribution histogram
    ax1 = axes[0, 0]
    ax1.hist(all_gaps, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
    ax1.axvline(x=np.mean(all_gaps), color='red', linestyle='--', label=f'Mean: {np.mean(all_gaps):.0f}ms')
    ax1.axvline(x=np.median(all_gaps), color='green', linestyle='--', label=f'Median: {np.median(all_gaps):.0f}ms')
    ax1.axvline(x=np.percentile(all_gaps, 95), color='orange', linestyle='--', label=f'P95: {np.percentile(all_gaps, 95):.0f}ms')
    ax1.set_xlabel("Chunk gap (ms)")
    ax1.set_ylabel("Frequency")
    ax1.set_title("Chunk Gap Distribution")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Plot 2: Gap CDF
    ax2 = axes[0, 1]
    sorted_gaps = np.sort(all_gaps)
    cdf = np.arange(1, len(sorted_gaps) + 1) / len(sorted_gaps)
    ax2.plot(sorted_gaps, cdf, color='steelblue', linewidth=2)
    ax2.axhline(y=0.95, color='orange', linestyle='--', alpha=0.5, label='P95')
    ax2.axhline(y=0.99, color='red', linestyle='--', alpha=0.5, label='P99')
    ax2.set_xlabel("Chunk gap (ms)")
    ax2.set_ylabel("Cumulative probability")
    ax2.set_title("Chunk Gap CDF")
    ax2.legend()
    ax2.grid(True, alpha=0.3)

    # Plot 3: Stall detection (gaps > 200ms)
    ax3 = axes[1, 0]
    stall_threshold = 200
    stalls_per_iter = []
    for r in results:
        gaps = [e.gap_from_prev_ms for e in r.chunk_events if e.gap_from_prev_ms > stall_threshold]
        stalls_per_iter.append(len(gaps))

    iterations = [r.iteration for r in results]
    ax3.bar(iterations, stalls_per_iter, color='coral', alpha=0.7)
    ax3.set_xlabel("Iteration")
    ax3.set_ylabel(f"Stalls (gaps > {stall_threshold}ms)")
    ax3.set_title("Stall Count per Iteration")
    ax3.grid(True, alpha=0.3)

    # Plot 4: Rolling statistics (drift detection)
    ax4 = axes[1, 1]
    window_size = max(3, len(results) // 5)

    max_gaps = [r.max_chunk_gap_ms for r in results]
    rtfs = [r.client_rtf for r in results]

    # Rolling average of max gaps
    if len(max_gaps) >= window_size:
        rolling_max = [np.mean(max_gaps[max(0, i-window_size):i+1]) for i in range(len(max_gaps))]
        ax4.plot(iterations, rolling_max, color='coral', linewidth=2, label=f'Rolling avg max gap (w={window_size})')

    ax4.plot(iterations, max_gaps, color='coral', alpha=0.3, marker='.', linestyle='--')
    ax4.set_xlabel("Iteration")
    ax4.set_ylabel("Max chunk gap (ms)")
    ax4.set_title("Drift Detection (Rolling Max Gap)")
    ax4.legend()
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()

    plot_path = os.path.join(output_dir, "gap_analysis_detailed.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    print(f"Saved: {plot_path}")


def log_to_wandb(results: list[IterationResult], project_name: str = "tts-benchmark"):
    """Log all metrics to Weights & Biases."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping W&B logging")
        print("Install with: pip install wandb")
        return

    # Initialize W&B run
    run = wandb.init(
        project=project_name,
        name=f"benchmark-{len(results)}iter-{int(time.time())}",
        config={
            "iterations": len(results),
            "text_length": results[0].text_length if results else 0,
        }
    )

    # Log per-iteration metrics
    for r in results:
        wandb.log({
            "iteration": r.iteration,
            "client_ttfa_ms": r.client_ttfa_ms,
            "client_wall_time_ms": r.client_wall_time_ms,
            "audio_duration_ms": r.audio_duration_ms,
            "client_rtf": r.client_rtf,
            "server_ttft_ms": r.server_ttft_ms,
            "server_ttfa_ms": r.server_ttfa_ms,
            "server_rtf": r.server_rtf,
            "total_chunks": r.total_chunks,
            "max_chunk_gap_ms": r.max_chunk_gap_ms,
            "mean_chunk_gap_ms": r.mean_chunk_gap_ms,
            "stddev_chunk_gap_ms": r.stddev_chunk_gap_ms,
        })

    # Log summary statistics
    client_ttfas = [r.client_ttfa_ms for r in results]
    client_rtfs = [r.client_rtf for r in results]
    max_gaps = [r.max_chunk_gap_ms for r in results]

    wandb.run.summary["mean_ttfa_ms"] = statistics.mean(client_ttfas)
    wandb.run.summary["std_ttfa_ms"] = statistics.stdev(client_ttfas) if len(client_ttfas) > 1 else 0
    wandb.run.summary["mean_rtf"] = statistics.mean(client_rtfs)
    wandb.run.summary["std_rtf"] = statistics.stdev(client_rtfs) if len(client_rtfs) > 1 else 0
    wandb.run.summary["mean_max_gap_ms"] = statistics.mean(max_gaps)
    wandb.run.summary["max_max_gap_ms"] = max(max_gaps)
    wandb.run.summary["min_max_gap_ms"] = min(max_gaps)

    # Create and log chunk timeline table
    chunk_data = []
    for r in results:
        for e in r.chunk_events:
            chunk_data.append({
                "iteration": r.iteration,
                "chunk_id": e.chunk_id,
                "arrival_time_ms": e.arrival_time_ms,
                "gap_from_prev_ms": e.gap_from_prev_ms,
                "size_bytes": e.size_bytes,
                "duration_ms": e.duration_ms,
            })

    if chunk_data:
        chunk_table = wandb.Table(
            columns=["iteration", "chunk_id", "arrival_time_ms", "gap_from_prev_ms", "size_bytes", "duration_ms"],
            data=[[d[k] for k in ["iteration", "chunk_id", "arrival_time_ms", "gap_from_prev_ms", "size_bytes", "duration_ms"]] for d in chunk_data]
        )
        wandb.log({"chunk_events": chunk_table})

    # Create custom charts
    try:
        import matplotlib.pyplot as plt
        import numpy as np

        # Chart 1: TTFA stability
        fig, ax = plt.subplots(figsize=(10, 5))
        iterations = [r.iteration for r in results]
        ax.plot(iterations, client_ttfas, marker='o', color='purple')
        ax.axhline(y=statistics.mean(client_ttfas), color='red', linestyle='--', alpha=0.5)
        ax.fill_between(iterations,
                        [statistics.mean(client_ttfas) - statistics.stdev(client_ttfas) if len(client_ttfas) > 1 else 0] * len(iterations),
                        [statistics.mean(client_ttfas) + statistics.stdev(client_ttfas) if len(client_ttfas) > 1 else 0] * len(iterations),
                        alpha=0.2, color='red')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("TTFA (ms)")
        ax.set_title("Time to First Audio Stability")
        ax.grid(True, alpha=0.3)
        wandb.log({"ttfa_stability": wandb.Image(fig)})
        plt.close()

        # Chart 2: Max gap trend
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.bar(iterations, max_gaps, color='coral', alpha=0.7)
        ax.axhline(y=200, color='red', linestyle='--', label='200ms threshold')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("Max chunk gap (ms)")
        ax.set_title("Maximum Chunk Gap per Iteration")
        ax.legend()
        ax.grid(True, alpha=0.3)
        wandb.log({"max_gap_trend": wandb.Image(fig)})
        plt.close()

        # Chart 3: RTF over time
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(iterations, client_rtfs, marker='o', color='green')
        ax.axhline(y=1.0, color='green', linestyle='--', alpha=0.5, label='1x realtime')
        ax.set_xlabel("Iteration")
        ax.set_ylabel("RTF")
        ax.set_title("Real-Time Factor Stability")
        ax.legend()
        ax.grid(True, alpha=0.3)
        wandb.log({"rtf_stability": wandb.Image(fig)})
        plt.close()

    except Exception as e:
        print(f"Could not create W&B charts: {e}")

    wandb.finish()
    print(f"W&B run logged: {run.url}")


def print_summary(results: list[IterationResult]):
    """Print summary statistics."""
    print()
    print("=" * 80)
    print(f"  BENCHMARK SUMMARY ({len(results)} iterations)")
    print("=" * 80)

    client_ttfas = [r.client_ttfa_ms for r in results]
    client_rtfs = [r.client_rtf for r in results]
    max_gaps = [r.max_chunk_gap_ms for r in results]
    mean_gaps = [r.mean_chunk_gap_ms for r in results]
    durations = [r.audio_duration_ms / 1000 for r in results]

    print()
    print("LATENCY METRICS:")
    print(f"  TTFA (Time to First Audio):")
    print(f"    Mean:   {statistics.mean(client_ttfas):.0f}ms")
    print(f"    Min:    {min(client_ttfas):.0f}ms")
    print(f"    Max:    {max(client_ttfas):.0f}ms")
    if len(client_ttfas) > 1:
        print(f"    Stddev: {statistics.stdev(client_ttfas):.0f}ms")

    print()
    print("RTF (Real-Time Factor):")
    print(f"  Mean:   {statistics.mean(client_rtfs):.2f}x")
    print(f"  Min:    {min(client_rtfs):.2f}x")
    print(f"  Max:    {max(client_rtfs):.2f}x")
    if len(client_rtfs) > 1:
        print(f"  Stddev: {statistics.stdev(client_rtfs):.3f}")

    print()
    print("CHUNK GAP ANALYSIS (Critical for Streaming):")
    print(f"  Max chunk gap (worst case):")
    print(f"    Mean across iterations: {statistics.mean(max_gaps):.0f}ms")
    print(f"    Worst ever:             {max(max_gaps):.0f}ms")
    print(f"    Best:                   {min(max_gaps):.0f}ms")
    print(f"  Mean chunk gap:")
    print(f"    Average:                {statistics.mean(mean_gaps):.0f}ms")

    # Stall analysis
    stall_threshold = 200
    total_stalls = sum(
        1 for r in results
        for e in r.chunk_events
        if e.gap_from_prev_ms > stall_threshold
    )
    total_chunks = sum(r.total_chunks for r in results)
    stall_rate = (total_stalls / total_chunks * 100) if total_chunks > 0 else 0

    print()
    print(f"STALL DETECTION (gaps > {stall_threshold}ms):")
    print(f"  Total stalls: {total_stalls} / {total_chunks} chunks ({stall_rate:.1f}%)")

    # Drift analysis
    print()
    print("DRIFT ANALYSIS:")
    first_half = results[:len(results)//2]
    second_half = results[len(results)//2:]

    if first_half and second_half:
        first_rtf = statistics.mean([r.client_rtf for r in first_half])
        second_rtf = statistics.mean([r.client_rtf for r in second_half])
        rtf_drift = ((second_rtf - first_rtf) / first_rtf * 100) if first_rtf > 0 else 0

        first_gap = statistics.mean([r.max_chunk_gap_ms for r in first_half])
        second_gap = statistics.mean([r.max_chunk_gap_ms for r in second_half])
        gap_drift = ((second_gap - first_gap) / first_gap * 100) if first_gap > 0 else 0

        print(f"  RTF drift (1st half vs 2nd half): {rtf_drift:+.1f}%")
        print(f"  Max gap drift:                    {gap_drift:+.1f}%")

        if abs(rtf_drift) > 10:
            print(f"  WARNING: Significant RTF drift detected!")
        if abs(gap_drift) > 20:
            print(f"  WARNING: Significant gap drift detected (possible thermal throttling)")

    print()
    print("AUDIO OUTPUT:")
    print(f"  Total iterations: {len(results)}")
    print(f"  Mean duration:    {statistics.mean(durations):.2f}s")

    # Verdict
    print()
    print("=" * 80)
    mean_rtf = statistics.mean(client_rtfs)
    worst_gap = max(max_gaps)

    if mean_rtf >= 1.0 and worst_gap < 300:
        print("  VERDICT: STREAMING PERFORMANCE IS GOOD")
    elif mean_rtf >= 0.8 and worst_gap < 500:
        print("  VERDICT: STREAMING PERFORMANCE IS ACCEPTABLE")
    else:
        print("  VERDICT: STREAMING PERFORMANCE NEEDS IMPROVEMENT")
    print("=" * 80)


async def run_benchmark(
    url: str,
    text: str,
    voice: str = "tara",
    iterations: int = 10,
    frames_per_chunk: int = 5,
    save_audio: bool = False,
    output_dir: str = ".",
    use_wandb: bool = False,
    wandb_project: str = "tts-benchmark",
):
    """Run full benchmark with specified iterations."""
    print()
    print("=" * 80)
    print(f"  TTS STREAMING BENCHMARK ({iterations} iterations)")
    print("=" * 80)
    print(f"URL: {url}")
    print(f"Text: {text[:60]}... ({len(text)} chars)")
    print(f"Voice: {voice}")
    print()

    results = []

    for i in range(1, iterations + 1):
        print(f"Iteration {i}/{iterations}...", end=" ", flush=True)

        result = await run_single_iteration(
            url=url,
            text=text,
            voice=voice,
            iteration=i,
            frames_per_chunk=frames_per_chunk,
        )
        results.append(result)

        print(f"TTFA={result.client_ttfa_ms:.0f}ms, RTF={result.client_rtf:.2f}x, MaxGap={result.max_chunk_gap_ms:.0f}ms")

        # Save audio if requested
        if save_audio and result.audio_bytes:
            os.makedirs(output_dir, exist_ok=True)
            filepath = os.path.join(output_dir, f"iteration_{i:02d}.wav")
            with wave.open(filepath, "wb") as wav:
                wav.setnchannels(1)
                wav.setsampwidth(2)
                wav.setframerate(SAMPLE_RATE)
                wav.writeframes(result.audio_bytes)

        # Small delay between iterations
        if i < iterations:
            await asyncio.sleep(0.5)

    # Print summary
    print_summary(results)

    # Create local plots
    os.makedirs(output_dir, exist_ok=True)
    create_chunk_timeline_plot(results, output_dir)
    create_detailed_gap_analysis(results, output_dir)

    # Log to W&B
    if use_wandb:
        log_to_wandb(results, project_name=wandb_project)

    return results


def main():
    parser = argparse.ArgumentParser(description="Comprehensive TTS Streaming Benchmark with W&B")
    parser.add_argument(
        "--url",
        default="wss://localhost:8000/ws/tts",
        help="WebSocket URL",
    )
    parser.add_argument(
        "--text",
        default="नमस्ते, मैं एक हिंदी टेक्स्ट टू स्पीच मॉडल हूं। आज का मौसम बहुत अच्छा है।",
        help="Text to synthesize",
    )
    parser.add_argument(
        "--voice",
        default="tara",
        help="Voice to use",
    )
    parser.add_argument(
        "--iterations",
        type=int,
        default=10,
        help="Number of iterations (default: 10)",
    )
    parser.add_argument(
        "--frames-per-chunk",
        type=int,
        default=5,
        help="Frames per audio chunk",
    )
    parser.add_argument(
        "--save-audio",
        action="store_true",
        help="Save audio from each iteration",
    )
    parser.add_argument(
        "--output-dir",
        default="tensorrt_tts/hindi_finetuned/plot_metrics/output",
        help="Output directory for plots and audio",
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        help="Log metrics to Weights & Biases",
    )
    parser.add_argument(
        "--wandb-project",
        default="orpheus-tts-benchmark",
        help="W&B project name",
    )
    args = parser.parse_args()

    asyncio.run(run_benchmark(
        url=args.url,
        text=args.text,
        voice=args.voice,
        iterations=args.iterations,
        frames_per_chunk=args.frames_per_chunk,
        save_audio=args.save_audio,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
    ))


if __name__ == "__main__":
    main()
