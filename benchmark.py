"""Performance benchmarking for BirdNET v3.0 India.

Measures:
- Model load time & initialization
- Audio preprocessing time
- Warmup inference time
- Multi-run inference latency statistics (Mean, P50, P90, P99, Min, Max)
- End-to-end classification throughput
"""

import argparse
from pathlib import Path
import sys
import time
import numpy as np
import onnxruntime as ort

from audio.preprocess import AudioPreprocessor
from config.settings import (
    MODEL_PATH,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    get_hardware_info,
    select_execution_providers,
)
from inference.model import BirdNETClassifier, get_model


def run_benchmark(
    audio_path: Path | None = None,
    num_runs: int = 20,
    warmup_runs: int = 5,
    force_cpu: bool = False,
) -> None:
    print("=" * 65)
    print(" AvianAI India — Performance Benchmark")
    print(" Developed & Engineered by Mrutyunjay Joshi")
    print("=" * 65)

    # 1. Model Loading
    t0 = time.perf_counter()
    classifier = BirdNETClassifier(force_cpu=force_cpu)
    load_time_ms = (time.perf_counter() - t0) * 1000.0

    hw = classifier.hardware_info
    print(f"Device:           {hw.get('hardware_name', 'Unknown')}")
    print(f"ONNX Provider:    {hw.get('primary_provider', 'Unknown')}")
    print(f"ORT Version:      {ort.__version__}")
    print(f"Model File:       {classifier.model_path.name}")
    print(f"Classes:          {classifier.num_classes:,}")
    print(f"Benchmark Runs:   {num_runs} (Warmup: {warmup_runs})")
    print("-" * 65)
    print(f"[1] Model Load Time:        {load_time_ms:.2f} ms")

    # 2. Audio Preprocessing
    preprocessor = AudioPreprocessor()
    if audio_path and Path(audio_path).exists():
        print(f"[2] Audio Source:           {Path(audio_path).name}")
        batch, dur, num_w, prep_ms = preprocessor.process_file(audio_path)
    else:
        print(f"[2] Audio Source:           5.0s Synthetic Sine/Noise Waveform")
        t = np.linspace(0, 5.0, int(SAMPLE_RATE * 5.0), endpoint=False, dtype=np.float32)
        signal = 0.3 * np.sin(2 * np.pi * 1000 * t) + 0.05 * np.random.randn(len(t)).astype(np.float32)
        batch, dur, num_w, prep_ms = preprocessor.process_waveform(signal)

    print(f"    Preprocessing Time:     {prep_ms:.2f} ms ({dur:.2f}s audio, {num_w} windows)")

    # 3. Warmup
    print(f"[3] Running {warmup_runs} Warmup Iterations...")
    for _ in range(warmup_runs):
        _ = classifier._run_forward_chunks(batch)

    # 4. Latency Measurements
    print(f"[4] Executing {num_runs} Timed Runs...")
    latencies: list[float] = []
    for _ in range(num_runs):
        t_start = time.perf_counter()
        _ = classifier._run_forward_chunks(batch)
        latencies.append((time.perf_counter() - t_start) * 1000.0)

    lat_arr = np.array(latencies)
    mean_lat = float(np.mean(lat_arr))
    p50 = float(np.percentile(lat_arr, 50))
    p90 = float(np.percentile(lat_arr, 90))
    p99 = float(np.percentile(lat_arr, 99))
    min_lat = float(np.min(lat_arr))
    max_lat = float(np.max(lat_arr))
    std_lat = float(np.std(lat_arr))

    print("-" * 65)
    print(" 📊 Benchmark Latency Results (per forward pass):")
    print(f"    Mean Latency:           {mean_lat:.2f} ms ± {std_lat:.2f} ms")
    print(f"    Median (P50):           {p50:.2f} ms")
    print(f"    P90 Latency:            {p90:.2f} ms")
    print(f"    P99 Latency:            {p99:.2f} ms")
    print(f"    Min Latency:            {min_lat:.2f} ms")
    print(f"    Max Latency:            {max_lat:.2f} ms")
    print(f"    Per-Window Latency:     {mean_lat / max(1, num_w):.2f} ms / 5.0s window")
    print(f"    Throughput:             {1000.0 / mean_lat:.1f} audio passes / sec")
    print("=" * 65)


def main() -> None:
    parser = argparse.ArgumentParser(description="BirdNET v3.0 Performance Benchmark")
    parser.add_argument("--audio", type=str, default=None, help="Path to audio file for benchmark")
    parser.add_argument("--runs", type=int, default=20, help="Number of benchmark iterations")
    parser.add_argument("--warmup", type=int, default=5, help="Number of warmup iterations")
    parser.add_argument("--cpu", action="store_true", help="Force CPU inference")
    args = parser.parse_args()

    run_benchmark(
        audio_path=Path(args.audio) if args.audio else None,
        num_runs=args.runs,
        warmup_runs=args.warmup,
        force_cpu=args.cpu,
    )


if __name__ == "__main__":
    main()
