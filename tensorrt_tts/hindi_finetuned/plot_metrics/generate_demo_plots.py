"""
Generate demo plots with synthetic benchmark data.

This creates sample visualizations without requiring a live WebSocket server.
Useful for testing plot generation and understanding the output format.

Usage:
    python tensorrt_tts/hindi_finetuned/plot_metrics/generate_demo_plots.py
"""

import os
import random
import statistics
from dataclasses import dataclass, field
from typing import Optional

# Import the plotting functions from the main benchmark module
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

SAMPLE_RATE = 24000
OUTPUT_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), "output")


@dataclass
class ChunkEvent:
    """Single chunk arrival event."""
    chunk_id: int
    arrival_time_ms: float
    size_bytes: int
    duration_ms: float
    gap_from_prev_ms: float = 0.0


@dataclass
class IterationResult:
    """Full metrics for one benchmark iteration."""
    iteration: int
    text_length: int
    client_ttfa_ms: float = 0.0
    client_wall_time_ms: float = 0.0
    audio_duration_ms: float = 0.0
    server_ttft_ms: Optional[float] = None
    server_ttfa_ms: Optional[float] = None
    server_rtf: Optional[float] = None
    total_chunks: int = 0
    chunk_events: list = field(default_factory=list)
    max_chunk_gap_ms: float = 0.0
    mean_chunk_gap_ms: float = 0.0
    stddev_chunk_gap_ms: float = 0.0
    client_rtf: float = 0.0
    audio_bytes: bytes = b""


def generate_synthetic_iteration(iteration: int, base_ttfa: float = 2800.0) -> IterationResult:
    """Generate synthetic benchmark data for one iteration."""
    # Add some realistic variation
    ttfa_jitter = random.gauss(0, 150)
    rtf_base = random.uniform(0.95, 1.25)

    # Simulate slight thermal drift (performance degrades slightly over time)
    thermal_factor = 1.0 + (iteration - 1) * 0.005  # 0.5% degradation per iteration

    num_chunks = random.randint(8, 15)
    chunk_duration_ms = 426.7  # ~5 frames * 2048 samples / 24000 * 1000

    # Generate chunk events
    chunk_events = []
    current_time = base_ttfa + ttfa_jitter

    for i in range(num_chunks):
        # Base gap is chunk duration / RTF
        base_gap = chunk_duration_ms / rtf_base * thermal_factor

        # Add jitter
        gap_jitter = random.gauss(0, 30)
        gap = max(50, base_gap + gap_jitter)

        # Occasionally add a "stall" (larger gap)
        if random.random() < 0.1:  # 10% chance
            gap += random.uniform(100, 300)

        if i == 0:
            gap = 0  # First chunk has no gap

        chunk_events.append(ChunkEvent(
            chunk_id=i + 1,
            arrival_time_ms=current_time,
            size_bytes=int(chunk_duration_ms / 1000 * SAMPLE_RATE * 2),  # int16
            duration_ms=chunk_duration_ms,
            gap_from_prev_ms=gap,
        ))

        current_time += gap if i > 0 else chunk_duration_ms / rtf_base

    # Calculate metrics
    gaps = [e.gap_from_prev_ms for e in chunk_events if e.gap_from_prev_ms > 0]
    audio_duration_ms = num_chunks * chunk_duration_ms
    wall_time_ms = chunk_events[-1].arrival_time_ms - chunk_events[0].arrival_time_ms + chunk_duration_ms / rtf_base

    result = IterationResult(
        iteration=iteration,
        text_length=73,
        client_ttfa_ms=base_ttfa + ttfa_jitter,
        client_wall_time_ms=wall_time_ms,
        audio_duration_ms=audio_duration_ms,
        server_ttft_ms=random.uniform(100, 200),
        server_ttfa_ms=base_ttfa + ttfa_jitter - random.uniform(400, 600),
        server_rtf=rtf_base * 1.4,  # Server RTF typically higher
        total_chunks=num_chunks,
        chunk_events=chunk_events,
        max_chunk_gap_ms=max(gaps) if gaps else 0,
        mean_chunk_gap_ms=statistics.mean(gaps) if gaps else 0,
        stddev_chunk_gap_ms=statistics.stdev(gaps) if len(gaps) > 1 else 0,
        client_rtf=audio_duration_ms / wall_time_ms if wall_time_ms > 0 else 0,
    )

    return result


def create_chunk_timeline_plot(results: list[IterationResult], output_dir: str):
    """Create chunk arrival timeline visualization."""
    import matplotlib.pyplot as plt
    import numpy as np

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
    import matplotlib.pyplot as plt
    import numpy as np

    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Collect all gaps across all iterations
    all_gaps = []
    for r in results:
        gaps = [e.gap_from_prev_ms for e in r.chunk_events if e.gap_from_prev_ms > 0]
        all_gaps.extend(gaps)

    if not all_gaps:
        print("No gaps to analyze")
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


def print_summary(results: list[IterationResult]):
    """Print summary statistics."""
    print()
    print("=" * 80)
    print(f"  BENCHMARK SUMMARY ({len(results)} iterations) [DEMO DATA]")
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


def main():
    """Generate demo plots with synthetic data."""
    print()
    print("=" * 80)
    print("  GENERATING DEMO PLOTS (Synthetic Data)")
    print("=" * 80)
    print()
    print("This generates sample visualizations using synthetic benchmark data.")
    print("For real benchmarks, use benchmark_with_wandb.py with a live server.")
    print()

    # Create output directory
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    # Generate synthetic results for 20 iterations
    num_iterations = 20
    print(f"Generating {num_iterations} synthetic iterations...")

    results = []
    for i in range(1, num_iterations + 1):
        result = generate_synthetic_iteration(i)
        results.append(result)
        print(f"  Iter {i:2d}: TTFA={result.client_ttfa_ms:.0f}ms, RTF={result.client_rtf:.2f}x, MaxGap={result.max_chunk_gap_ms:.0f}ms")

    # Print summary
    print_summary(results)

    # Generate plots
    print()
    print("Generating plots...")
    create_chunk_timeline_plot(results, OUTPUT_DIR)
    create_detailed_gap_analysis(results, OUTPUT_DIR)

    print()
    print("=" * 80)
    print(f"  PLOTS SAVED TO: {OUTPUT_DIR}")
    print("=" * 80)
    print()
    print("Files generated:")
    print(f"  - {OUTPUT_DIR}/chunk_timeline_analysis.png")
    print(f"  - {OUTPUT_DIR}/gap_analysis_detailed.png")
    print()


if __name__ == "__main__":
    main()
