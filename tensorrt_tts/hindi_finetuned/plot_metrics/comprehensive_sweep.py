"""
Comprehensive TTS Benchmark with Cold-Start Analysis and Multi-Length Sweep

Features:
- Cold-start analysis (first 2-3 requests)
- 20+ different text lengths sweep
- W&B logging with plots
- Audio output for each prompt

Usage:
    python tensorrt_tts/hindi_finetuned/plot_metrics/comprehensive_sweep.py \
        --url wss://<modal-url>/ws/tts \
        --wandb
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
from datetime import datetime


SAMPLE_RATE = 24000
SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DEFAULT_OUTPUT_DIR = os.path.join(SCRIPT_DIR, "output")


# Hindi test prompts of varying lengths
TEST_PROMPTS = [
    # Very Short (10-30 chars)
    ("very_short_1", "नमस्ते"),  # Hello
    ("very_short_2", "धन्यवाद"),  # Thank you
    ("very_short_3", "शुभ प्रभात"),  # Good morning

    # Short (30-60 chars)
    ("short_1", "आज का मौसम बहुत अच्छा है।"),  # Today's weather is very nice
    ("short_2", "मुझे हिंदी बोलना पसंद है।"),  # I like speaking Hindi
    ("short_3", "यह एक परीक्षण वाक्य है।"),  # This is a test sentence

    # Medium-Short (60-100 chars)
    ("medium_short_1", "भारत एक विविधताओं से भरा देश है जहाँ अनेक भाषाएँ बोली जाती हैं।"),
    ("medium_short_2", "मैं एक हिंदी टेक्स्ट टू स्पीच मॉडल हूं और मैं आपकी मदद कर सकता हूं।"),
    ("medium_short_3", "कृपया अपना नाम और पता बताएं ताकि हम आपसे संपर्क कर सकें।"),

    # Medium (100-150 chars)
    ("medium_1", "आज के इस डिजिटल युग में कृत्रिम बुद्धिमत्ता ने हमारे जीवन को काफी बदल दिया है। यह तकनीक हर क्षेत्र में उपयोग हो रही है।"),
    ("medium_2", "हिंदी भारत की राजभाषा है और विश्व की चौथी सबसे ज्यादा बोली जाने वाली भाषा है। इसे करोड़ों लोग बोलते हैं।"),
    ("medium_3", "मशीन लर्निंग और डीप लर्निंग ने भाषा प्रसंस्करण में क्रांति ला दी है। अब कंप्यूटर मानव जैसी आवाज में बोल सकते हैं।"),

    # Medium-Long (150-200 chars)
    ("medium_long_1", "टेक्स्ट टू स्पीच तकनीक का उपयोग कई क्षेत्रों में हो रहा है जैसे कि ऑडियोबुक, नेविगेशन सिस्टम, और विकलांग लोगों की सहायता के लिए। यह तकनीक दिन प्रतिदिन बेहतर होती जा रही है।"),
    ("medium_long_2", "भारतीय संस्कृति विश्व की सबसे प्राचीन संस्कृतियों में से एक है। यहाँ की परंपराएं, त्योहार, और खान-पान अद्वितीय हैं। हर राज्य की अपनी विशेषता है जो इसे और भी रोचक बनाती है।"),
    ("medium_long_3", "न्यूरल नेटवर्क आधारित स्पीच सिंथेसिस में हाल के वर्षों में काफी प्रगति हुई है। अब सिस्टम प्राकृतिक और भावनात्मक आवाज उत्पन्न कर सकते हैं जो मानव आवाज से मिलती-जुलती है।"),

    # Long (200-300 chars)
    ("long_1", "आर्टिफिशियल इंटेलिजेंस यानी कृत्रिम बुद्धिमत्ता आज के समय की सबसे महत्वपूर्ण तकनीकों में से एक है। यह हमारे दैनिक जीवन के कई पहलुओं को प्रभावित कर रही है। स्मार्टफोन से लेकर स्वचालित वाहनों तक, एआई हर जगह मौजूद है। इसकी क्षमता लगातार बढ़ रही है।"),
    ("long_2", "हिंदी साहित्य का इतिहास बहुत समृद्ध है। प्रेमचंद, महादेवी वर्मा, और हरिवंश राय बच्चन जैसे महान लेखकों ने इसे अमर बना दिया है। उनकी रचनाएं आज भी प्रासंगिक हैं और पाठकों को प्रेरित करती हैं। हिंदी कविता और गद्य दोनों में अनमोल खजाना है।"),
    ("long_3", "वॉयस असिस्टेंट जैसे एलेक्सा, सिरी, और गूगल असिस्टेंट ने लोगों के डिवाइस के साथ इंटरैक्ट करने का तरीका बदल दिया है। अब लोग बोलकर कमांड दे सकते हैं, संगीत सुन सकते हैं, और जानकारी प्राप्त कर सकते हैं। यह तकनीक विशेष रूप से वृद्ध और विकलांग लोगों के लिए उपयोगी है।"),

    # Very Long (300-400 chars)
    ("very_long_1", "भारत में डिजिटल क्रांति ने पिछले कुछ वर्षों में अभूतपूर्व गति पकड़ी है। इंटरनेट की पहुंच गांवों तक हो गई है और स्मार्टफोन की कीमतें कम होने से आम लोग भी डिजिटल दुनिया से जुड़ गए हैं। यूपीआई पेमेंट ने लेनदेन को आसान बना दिया है। ऑनलाइन शिक्षा ने दूरदराज के छात्रों को भी गुणवत्तापूर्ण शिक्षा का अवसर दिया है। यह बदलाव भारत को एक डिजिटल महाशक्ति बनने की ओर ले जा रहा है।"),
    ("very_long_2", "स्पीच टेक्नोलॉजी में टेंसर आरटी और अन्य ऑप्टिमाइजेशन फ्रेमवर्क ने इनफरेंस स्पीड को कई गुना बढ़ा दिया है। पहले जहां एक वाक्य को संश्लेषित करने में सेकंड लगते थे, अब यह मिलीसेकंड में हो जाता है। स्ट्रीमिंग आर्किटेक्चर ने रियल-टाइम एप्लिकेशन को संभव बना दिया है। वेबसॉकेट प्रोटोकॉल का उपयोग करके ऑडियो चंक्स को तुरंत क्लाइंट को भेजा जा सकता है, जिससे उपयोगकर्ता को तुरंत प्रतिक्रिया मिलती है।"),

    # Extra Long (400-500 chars)
    ("extra_long_1", "भारतीय अंतरिक्ष अनुसंधान संगठन यानी इसरो ने विश्व में अपनी एक अलग पहचान बनाई है। चंद्रयान और मंगलयान जैसे मिशनों ने भारत की अंतरिक्ष क्षमता को साबित किया है। इसरो की सबसे बड़ी विशेषता यह है कि यह कम बजट में बड़े-बड़े मिशन सफलतापूर्वक पूरा करता है। आने वाले समय में गगनयान मिशन भारतीय अंतरिक्ष यात्रियों को अंतरिक्ष में भेजेगा। इससे भारत अमेरिका, रूस और चीन के बाद चौथा देश बन जाएगा जिसने अपने नागरिकों को अंतरिक्ष में भेजा है। यह भारत के लिए गर्व का क्षण होगा।"),
    ("extra_long_2", "कृत्रिम बुद्धिमत्ता आधारित वॉयस क्लोनिंग तकनीक ने मनोरंजन उद्योग में नई संभावनाएं खोली हैं। अब किसी भी व्यक्ति की आवाज को कुछ ही सेकंड की रिकॉर्डिंग से क्लोन किया जा सकता है। इस तकनीक का उपयोग फिल्मों में डबिंग, गेम्स में कैरेक्टर वॉयस, और पॉडकास्ट निर्माण में हो रहा है। हालांकि, इसके दुरुपयोग की संभावना भी है जैसे फर्जी ऑडियो बनाना। इसलिए इस तकनीक के नैतिक उपयोग के लिए दिशानिर्देश और नियम बनाए जा रहे हैं। भविष्य में यह तकनीक और भी उन्नत होगी।"),
]


@dataclass
class ChunkEvent:
    """Single chunk arrival event."""
    chunk_id: int
    arrival_time_ms: float
    size_bytes: int
    duration_ms: float
    gap_from_prev_ms: float = 0.0


@dataclass
class BenchmarkResult:
    """Full metrics for one benchmark run."""
    prompt_id: str
    prompt_text: str
    text_length: int
    is_cold_start: bool = False

    # Timing
    client_ttfa_ms: float = 0.0
    client_wall_time_ms: float = 0.0
    audio_duration_ms: float = 0.0

    # Server metrics
    server_ttft_ms: Optional[float] = None
    server_ttfa_ms: Optional[float] = None
    server_rtf: Optional[float] = None
    tokens_per_sec: Optional[float] = None

    # Chunk analysis
    total_chunks: int = 0
    chunk_events: list = field(default_factory=list)
    max_chunk_gap_ms: float = 0.0
    mean_chunk_gap_ms: float = 0.0
    stddev_chunk_gap_ms: float = 0.0

    # RTF
    client_rtf: float = 0.0

    # Audio
    audio_bytes: bytes = b""


async def run_single_benchmark(
    url: str,
    prompt_id: str,
    text: str,
    voice: str = "tara",
    is_cold_start: bool = False,
) -> BenchmarkResult:
    """Run a single benchmark."""
    try:
        import websockets
    except ImportError:
        raise ImportError("pip install websockets")

    result = BenchmarkResult(
        prompt_id=prompt_id,
        prompt_text=text,
        text_length=len(text),
        is_cold_start=is_cold_start,
    )

    audio_chunks = []
    chunk_events = []

    try:
        async with websockets.connect(url, ping_interval=30, ping_timeout=120) as ws:
            request = {
                "text": text,
                "voice": voice,
                "temperature": 0.6,
                "top_p": 0.95,
                "frames_per_chunk": 5,
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
                    chunk_id += 1
                    arrival_ms = (recv_time - request_time) * 1000

                    if result.client_ttfa_ms == 0:
                        result.client_ttfa_ms = arrival_ms

                    gap_ms = 0.0
                    if last_chunk_time is not None:
                        gap_ms = (recv_time - last_chunk_time) * 1000
                    last_chunk_time = recv_time

                    samples = len(message) // 2
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
                    data = json.loads(message)

                    if data.get("done"):
                        result.client_wall_time_ms = (recv_time - request_time) * 1000
                        result.total_chunks = data.get("chunks", chunk_id)

                        server_metrics = data.get("server_metrics", {})
                        if server_metrics:
                            result.server_ttft_ms = server_metrics.get("server_ttft_ms")
                            result.server_ttfa_ms = server_metrics.get("server_ttfa_ms")
                            result.server_rtf = server_metrics.get("server_rtf")
                            result.tokens_per_sec = server_metrics.get("tokens_per_sec")
                        break
                    elif data.get("error"):
                        raise Exception(f"Server error: {data['error']}")

    except Exception as e:
        print(f"  Error: {e}")
        return result

    # Calculate metrics
    total_bytes = sum(len(c) for c in audio_chunks)
    total_samples = total_bytes // 2
    result.audio_duration_ms = (total_samples / SAMPLE_RATE) * 1000
    result.audio_bytes = b"".join(audio_chunks)

    result.chunk_events = chunk_events
    gaps = [e.gap_from_prev_ms for e in chunk_events if e.gap_from_prev_ms > 0]

    if gaps:
        result.max_chunk_gap_ms = max(gaps)
        result.mean_chunk_gap_ms = statistics.mean(gaps)
        if len(gaps) > 1:
            result.stddev_chunk_gap_ms = statistics.stdev(gaps)

    if result.client_wall_time_ms > 0:
        result.client_rtf = result.audio_duration_ms / result.client_wall_time_ms

    return result


def save_audio(result: BenchmarkResult, output_dir: str, prefix: str = ""):
    """Save audio to WAV file."""
    if not result.audio_bytes:
        return None

    os.makedirs(output_dir, exist_ok=True)

    filename = f"{prefix}{result.prompt_id}.wav"
    filepath = os.path.join(output_dir, filename)

    with wave.open(filepath, "wb") as wav:
        wav.setnchannels(1)
        wav.setsampwidth(2)
        wav.setframerate(SAMPLE_RATE)
        wav.writeframes(result.audio_bytes)

    return filepath


def create_plots(cold_results: list, sweep_results: list, output_dir: str):
    """Create visualization plots."""
    try:
        import matplotlib.pyplot as plt
        import numpy as np
    except ImportError:
        print("matplotlib not installed, skipping plots")
        return []

    plot_files = []
    os.makedirs(output_dir, exist_ok=True)

    # Plot 1: RTF vs Text Length
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    # Sort by text length
    sweep_sorted = sorted(sweep_results, key=lambda r: r.text_length)
    lengths = [r.text_length for r in sweep_sorted]
    rtfs = [r.client_rtf for r in sweep_sorted]
    ttfas = [r.client_ttfa_ms for r in sweep_sorted]
    durations = [r.audio_duration_ms / 1000 for r in sweep_sorted]

    # RTF vs Length
    ax1 = axes[0, 0]
    ax1.scatter(lengths, rtfs, c='green', s=50, alpha=0.7)
    ax1.axhline(y=1.0, color='red', linestyle='--', alpha=0.5, label='1x realtime')
    z = np.polyfit(lengths, rtfs, 1)
    p = np.poly1d(z)
    ax1.plot(lengths, p(lengths), 'g--', alpha=0.5, label=f'Trend')
    ax1.set_xlabel("Text Length (chars)")
    ax1.set_ylabel("RTF (x realtime)")
    ax1.set_title("Real-Time Factor vs Text Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # TTFA vs Length
    ax2 = axes[0, 1]
    ax2.scatter(lengths, ttfas, c='purple', s=50, alpha=0.7)
    ax2.set_xlabel("Text Length (chars)")
    ax2.set_ylabel("TTFA (ms)")
    ax2.set_title("Time to First Audio vs Text Length")
    ax2.grid(True, alpha=0.3)

    # Audio Duration vs Length
    ax3 = axes[1, 0]
    ax3.scatter(lengths, durations, c='blue', s=50, alpha=0.7)
    z = np.polyfit(lengths, durations, 1)
    p = np.poly1d(z)
    ax3.plot(lengths, p(lengths), 'b--', alpha=0.5)
    ax3.set_xlabel("Text Length (chars)")
    ax3.set_ylabel("Audio Duration (s)")
    ax3.set_title("Audio Duration vs Text Length")
    ax3.grid(True, alpha=0.3)

    # Cold Start Comparison
    ax4 = axes[1, 1]
    if cold_results:
        cold_ttfas = [r.client_ttfa_ms for r in cold_results]
        warm_ttfa_mean = statistics.mean(ttfas)

        x = list(range(1, len(cold_ttfas) + 1))
        ax4.bar(x, cold_ttfas, color='coral', alpha=0.7, label='Cold Start Requests')
        ax4.axhline(y=warm_ttfa_mean, color='green', linestyle='--',
                   label=f'Warm Mean: {warm_ttfa_mean:.0f}ms')
        ax4.set_xlabel("Request Number")
        ax4.set_ylabel("TTFA (ms)")
        ax4.set_title("Cold Start Analysis")
        ax4.legend()
        ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "sweep_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    plot_files.append(plot_path)

    # Plot 2: Chunk Gap Analysis
    fig, axes = plt.subplots(2, 2, figsize=(14, 10))

    max_gaps = [r.max_chunk_gap_ms for r in sweep_sorted]
    mean_gaps = [r.mean_chunk_gap_ms for r in sweep_sorted]

    # Max Gap vs Length
    ax1 = axes[0, 0]
    ax1.scatter(lengths, max_gaps, c='coral', s=50, alpha=0.7)
    ax1.axhline(y=200, color='red', linestyle='--', alpha=0.5, label='200ms threshold')
    ax1.axhline(y=300, color='darkred', linestyle='--', alpha=0.5, label='300ms threshold')
    ax1.set_xlabel("Text Length (chars)")
    ax1.set_ylabel("Max Chunk Gap (ms)")
    ax1.set_title("Maximum Chunk Gap vs Text Length")
    ax1.legend()
    ax1.grid(True, alpha=0.3)

    # Mean Gap vs Length
    ax2 = axes[0, 1]
    ax2.scatter(lengths, mean_gaps, c='steelblue', s=50, alpha=0.7)
    ax2.set_xlabel("Text Length (chars)")
    ax2.set_ylabel("Mean Chunk Gap (ms)")
    ax2.set_title("Mean Chunk Gap vs Text Length")
    ax2.grid(True, alpha=0.3)

    # All gaps histogram
    ax3 = axes[1, 0]
    all_gaps = []
    for r in sweep_results:
        gaps = [e.gap_from_prev_ms for e in r.chunk_events if e.gap_from_prev_ms > 0]
        all_gaps.extend(gaps)

    if all_gaps:
        ax3.hist(all_gaps, bins=50, alpha=0.7, color='steelblue', edgecolor='black')
        ax3.axvline(x=np.mean(all_gaps), color='red', linestyle='--',
                   label=f'Mean: {np.mean(all_gaps):.0f}ms')
        ax3.axvline(x=np.percentile(all_gaps, 95), color='orange', linestyle='--',
                   label=f'P95: {np.percentile(all_gaps, 95):.0f}ms')
        ax3.set_xlabel("Chunk Gap (ms)")
        ax3.set_ylabel("Frequency")
        ax3.set_title("Chunk Gap Distribution (All Prompts)")
        ax3.legend()
        ax3.grid(True, alpha=0.3)

    # Chunks vs Length
    ax4 = axes[1, 1]
    chunks = [r.total_chunks for r in sweep_sorted]
    ax4.scatter(lengths, chunks, c='green', s=50, alpha=0.7)
    ax4.set_xlabel("Text Length (chars)")
    ax4.set_ylabel("Number of Chunks")
    ax4.set_title("Chunk Count vs Text Length")
    ax4.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "gap_analysis.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    plot_files.append(plot_path)

    # Plot 3: Per-prompt breakdown
    fig, ax = plt.subplots(figsize=(16, 8))

    prompt_ids = [r.prompt_id for r in sweep_sorted]
    x = np.arange(len(prompt_ids))
    width = 0.35

    ax.bar(x - width/2, rtfs, width, label='RTF', color='green', alpha=0.7)
    ax2 = ax.twinx()
    ax2.bar(x + width/2, [r.audio_duration_ms/1000 for r in sweep_sorted],
            width, label='Duration (s)', color='blue', alpha=0.7)

    ax.set_xlabel("Prompt")
    ax.set_ylabel("RTF", color='green')
    ax2.set_ylabel("Duration (s)", color='blue')
    ax.set_title("RTF and Duration per Prompt")
    ax.set_xticks(x)
    ax.set_xticklabels(prompt_ids, rotation=45, ha='right')
    ax.axhline(y=1.0, color='red', linestyle='--', alpha=0.5)
    ax.legend(loc='upper left')
    ax2.legend(loc='upper right')
    ax.grid(True, alpha=0.3)

    plt.tight_layout()
    plot_path = os.path.join(output_dir, "per_prompt_breakdown.png")
    plt.savefig(plot_path, dpi=150)
    plt.close()
    plot_files.append(plot_path)

    return plot_files


def log_to_wandb(cold_results: list, sweep_results: list, plot_files: list,
                 output_dir: str, project_name: str = "orpheus-tts-benchmark"):
    """Log all results to W&B."""
    try:
        import wandb
    except ImportError:
        print("wandb not installed, skipping W&B logging")
        print("Install with: pip install wandb")
        return None

    # Initialize run
    run = wandb.init(
        project=project_name,
        name=f"comprehensive-sweep-{datetime.now().strftime('%Y%m%d-%H%M%S')}",
        config={
            "num_prompts": len(TEST_PROMPTS),
            "cold_start_requests": len(cold_results),
            "sweep_prompts": len(sweep_results),
        }
    )

    # Log cold start metrics
    for i, r in enumerate(cold_results):
        wandb.log({
            "cold_start/request": i + 1,
            "cold_start/ttfa_ms": r.client_ttfa_ms,
            "cold_start/rtf": r.client_rtf,
            "cold_start/duration_ms": r.audio_duration_ms,
        })

    # Log sweep metrics
    for r in sweep_results:
        wandb.log({
            "sweep/prompt_id": r.prompt_id,
            "sweep/text_length": r.text_length,
            "sweep/client_ttfa_ms": r.client_ttfa_ms,
            "sweep/client_rtf": r.client_rtf,
            "sweep/audio_duration_ms": r.audio_duration_ms,
            "sweep/max_chunk_gap_ms": r.max_chunk_gap_ms,
            "sweep/mean_chunk_gap_ms": r.mean_chunk_gap_ms,
            "sweep/total_chunks": r.total_chunks,
            "sweep/server_rtf": r.server_rtf,
            "sweep/tokens_per_sec": r.tokens_per_sec,
        })

    # Log summary statistics
    sweep_rtfs = [r.client_rtf for r in sweep_results]
    sweep_ttfas = [r.client_ttfa_ms for r in sweep_results]
    sweep_max_gaps = [r.max_chunk_gap_ms for r in sweep_results]

    wandb.run.summary["sweep_mean_rtf"] = statistics.mean(sweep_rtfs)
    wandb.run.summary["sweep_mean_ttfa_ms"] = statistics.mean(sweep_ttfas)
    wandb.run.summary["sweep_mean_max_gap_ms"] = statistics.mean(sweep_max_gaps)
    wandb.run.summary["sweep_worst_max_gap_ms"] = max(sweep_max_gaps)

    if cold_results:
        cold_ttfas = [r.client_ttfa_ms for r in cold_results]
        wandb.run.summary["cold_first_ttfa_ms"] = cold_ttfas[0]
        wandb.run.summary["cold_mean_ttfa_ms"] = statistics.mean(cold_ttfas)
        wandb.run.summary["cold_vs_warm_ratio"] = cold_ttfas[0] / statistics.mean(sweep_ttfas)

    # Log plots
    for plot_path in plot_files:
        wandb.log({os.path.basename(plot_path): wandb.Image(plot_path)})

    # Create results table
    table_data = []
    for r in sweep_results:
        table_data.append([
            r.prompt_id,
            r.text_length,
            f"{r.client_ttfa_ms:.0f}",
            f"{r.client_rtf:.2f}",
            f"{r.audio_duration_ms/1000:.2f}",
            f"{r.max_chunk_gap_ms:.0f}",
            r.total_chunks,
        ])

    results_table = wandb.Table(
        columns=["Prompt ID", "Length", "TTFA (ms)", "RTF", "Duration (s)", "Max Gap (ms)", "Chunks"],
        data=table_data
    )
    wandb.log({"sweep_results": results_table})

    wandb.finish()
    return run.url


def print_summary(cold_results: list, sweep_results: list):
    """Print comprehensive summary."""
    print()
    print("=" * 90)
    print("  COMPREHENSIVE BENCHMARK SUMMARY")
    print("=" * 90)

    # Cold start analysis
    if cold_results:
        print()
        print("COLD START ANALYSIS:")
        print("-" * 40)
        for i, r in enumerate(cold_results):
            print(f"  Request {i+1}: TTFA={r.client_ttfa_ms:.0f}ms, RTF={r.client_rtf:.2f}x")

    # Sweep summary
    print()
    print("SWEEP RESULTS (sorted by text length):")
    print("-" * 90)
    print(f"{'Prompt ID':<20} {'Len':>5} {'TTFA':>8} {'RTF':>6} {'Dur':>7} {'MaxGap':>8} {'Chunks':>7}")
    print("-" * 90)

    sweep_sorted = sorted(sweep_results, key=lambda r: r.text_length)
    for r in sweep_sorted:
        print(f"{r.prompt_id:<20} {r.text_length:>5} {r.client_ttfa_ms:>7.0f}ms {r.client_rtf:>5.2f}x "
              f"{r.audio_duration_ms/1000:>6.2f}s {r.max_chunk_gap_ms:>7.0f}ms {r.total_chunks:>7}")

    # Statistics
    print()
    print("AGGREGATE STATISTICS:")
    print("-" * 40)

    rtfs = [r.client_rtf for r in sweep_results]
    ttfas = [r.client_ttfa_ms for r in sweep_results]
    max_gaps = [r.max_chunk_gap_ms for r in sweep_results]

    print(f"  RTF:      Mean={statistics.mean(rtfs):.2f}x, Min={min(rtfs):.2f}x, Max={max(rtfs):.2f}x")
    print(f"  TTFA:     Mean={statistics.mean(ttfas):.0f}ms, Min={min(ttfas):.0f}ms, Max={max(ttfas):.0f}ms")
    print(f"  Max Gap:  Mean={statistics.mean(max_gaps):.0f}ms, Worst={max(max_gaps):.0f}ms")

    if cold_results:
        cold_ttfa = cold_results[0].client_ttfa_ms
        warm_ttfa = statistics.mean(ttfas)
        print(f"  Cold/Warm TTFA Ratio: {cold_ttfa/warm_ttfa:.1f}x")


async def run_comprehensive_benchmark(
    url: str,
    output_dir: str,
    use_wandb: bool = False,
    wandb_project: str = "orpheus-tts-benchmark",
    cold_start_count: int = 3,
):
    """Run the full comprehensive benchmark."""
    print()
    print("=" * 90)
    print("  COMPREHENSIVE TTS BENCHMARK")
    print("=" * 90)
    print(f"URL: {url}")
    print(f"Output: {output_dir}")
    print(f"Prompts: {len(TEST_PROMPTS)}")
    print(f"Cold Start Requests: {cold_start_count}")
    print(f"W&B Logging: {use_wandb}")
    print()

    cold_results = []
    sweep_results = []
    audio_files = []

    # Phase 1: Cold Start Analysis
    print("=" * 50)
    print("  PHASE 1: COLD START ANALYSIS")
    print("=" * 50)

    # Use a medium prompt for cold start
    cold_prompt = TEST_PROMPTS[10]  # medium_1

    for i in range(cold_start_count):
        print(f"\nCold start request {i+1}/{cold_start_count}...", end=" ", flush=True)

        result = await run_single_benchmark(
            url=url,
            prompt_id=f"cold_start_{i+1}",
            text=cold_prompt[1],
            is_cold_start=True,
        )
        cold_results.append(result)

        print(f"TTFA={result.client_ttfa_ms:.0f}ms, RTF={result.client_rtf:.2f}x")

        # Save audio
        audio_path = save_audio(result, output_dir, prefix="cold_")
        if audio_path:
            audio_files.append(audio_path)

        # Small delay
        if i < cold_start_count - 1:
            await asyncio.sleep(0.5)

    # Phase 2: Warm Server Sweep
    print()
    print("=" * 50)
    print("  PHASE 2: WARM SERVER - LENGTH SWEEP")
    print("=" * 50)

    for i, (prompt_id, prompt_text) in enumerate(TEST_PROMPTS):
        print(f"\n[{i+1}/{len(TEST_PROMPTS)}] {prompt_id} ({len(prompt_text)} chars)...", end=" ", flush=True)

        result = await run_single_benchmark(
            url=url,
            prompt_id=prompt_id,
            text=prompt_text,
            is_cold_start=False,
        )
        sweep_results.append(result)

        print(f"TTFA={result.client_ttfa_ms:.0f}ms, RTF={result.client_rtf:.2f}x, "
              f"Dur={result.audio_duration_ms/1000:.1f}s")

        # Save audio
        audio_path = save_audio(result, output_dir, prefix="sweep_")
        if audio_path:
            audio_files.append(audio_path)

        # Small delay between requests
        await asyncio.sleep(0.3)

    # Print summary
    print_summary(cold_results, sweep_results)

    # Create plots
    print()
    print("Generating plots...")
    plot_files = create_plots(cold_results, sweep_results, output_dir)
    for pf in plot_files:
        print(f"  Saved: {pf}")

    # Log to W&B
    wandb_url = None
    if use_wandb:
        print()
        print("Logging to Weights & Biases...")
        wandb_url = log_to_wandb(cold_results, sweep_results, plot_files, output_dir, wandb_project)
        if wandb_url:
            print(f"  W&B Run: {wandb_url}")

    return {
        "cold_results": cold_results,
        "sweep_results": sweep_results,
        "audio_files": audio_files,
        "plot_files": plot_files,
        "wandb_url": wandb_url,
        "output_dir": output_dir,
    }


def main():
    parser = argparse.ArgumentParser(description="Comprehensive TTS Benchmark")
    parser.add_argument("--url", required=True, help="WebSocket URL")
    parser.add_argument("--output-dir", default=DEFAULT_OUTPUT_DIR, help="Output directory")
    parser.add_argument("--wandb", action="store_true", help="Log to W&B")
    parser.add_argument("--wandb-project", default="orpheus-tts-benchmark", help="W&B project")
    parser.add_argument("--cold-start-count", type=int, default=3, help="Cold start requests")
    args = parser.parse_args()

    results = asyncio.run(run_comprehensive_benchmark(
        url=args.url,
        output_dir=args.output_dir,
        use_wandb=args.wandb,
        wandb_project=args.wandb_project,
        cold_start_count=args.cold_start_count,
    ))

    # Print final artifact locations
    print()
    print("=" * 90)
    print("  ARTIFACTS GENERATED")
    print("=" * 90)
    print()
    print(f"OUTPUT DIRECTORY: {results['output_dir']}")
    print()
    print("AUDIO FILES:")
    for af in results['audio_files']:
        print(f"  - {af}")
    print()
    print("PLOT FILES:")
    for pf in results['plot_files']:
        print(f"  - {pf}")
    if results['wandb_url']:
        print()
        print(f"W&B DASHBOARD: {results['wandb_url']}")
    print()
    print("=" * 90)


if __name__ == "__main__":
    main()
