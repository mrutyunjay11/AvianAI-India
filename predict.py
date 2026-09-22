"""CLI prediction tool for AvianAI India.

Supports:
- Audio file inference: python predict.py --audio samples/indian_birds/01_indian_peafowl.wav
- Microphone recording: python predict.py --mic --duration 5.0
- Configurable coordinates & regional filtering for India
- Detailed benchmark latency reporting
"""

from pathlib import Path
import argparse
import json
import sys

from config.settings import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_REGION,
    DEFAULT_TOP_K,
    MODEL_PATH,
)
from inference.model import BirdNETClassifier, get_model
from audio.microphone import is_microphone_available, record_audio, save_audio


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="AvianAI India - Bioacoustic Sound Identification CLI",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    group = parser.add_mutually_exclusive_group(required=True)
    group.add_argument(
        "--audio", "-a",
        type=str,
        help="Path to input audio file (WAV, OGG, MP3, FLAC)",
    )
    group.add_argument(
        "--mic", "-m",
        action="store_true",
        help="Record live audio from default microphone",
    )

    parser.add_argument(
        "--duration", "-d",
        type=float,
        default=5.0,
        help="Microphone recording duration in seconds (5.0 to 10.0s recommended)",
    )
    parser.add_argument(
        "--lat",
        type=float,
        default=DEFAULT_LATITUDE,
        help="Latitude for geographic species filtering",
    )
    parser.add_argument(
        "--lon",
        type=float,
        default=DEFAULT_LONGITUDE,
        help="Longitude for geographic species filtering",
    )
    parser.add_argument(
        "--region", "-r",
        type=str,
        default=DEFAULT_REGION,
        choices=["pan-india", "south-asia-peninsular", "indo-gangetic", "himalaya", "all"],
        help="Geographic region preset",
    )
    parser.add_argument(
        "--threshold", "-t",
        type=float,
        default=CONFIDENCE_THRESHOLD,
        help="Confidence threshold for positive detection",
    )
    parser.add_argument(
        "--top-k", "-k",
        type=int,
        default=DEFAULT_TOP_K,
        help="Number of top candidates to display",
    )
    parser.add_argument(
        "--force-cpu",
        action="store_true",
        help="Force CPU execution provider instead of GPU/accelerator",
    )
    parser.add_argument(
        "--json",
        action="store_true",
        help="Output results as JSON string",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    # Initialize model
    classifier = get_model(force_cpu=args.force_cpu)

    # Process input
    if args.mic:
        if not is_microphone_available():
            print("[ERROR] No active microphone input device detected.")
            return 1
        waveform, sr = record_audio(duration=args.duration)
        # Optionally save recording
        saved_path = save_audio(waveform, sample_rate=sr)
        print(f"[MIC] Saved recording to: {saved_path}")

        result = classifier.predict_waveform(
            waveform,
            sample_rate=sr,
            lat=args.lat,
            lon=args.lon,
            region=args.region,
            threshold=args.threshold,
            top_k=args.top_k,
        )
    else:
        audio_path = Path(args.audio)
        if not audio_path.exists():
            print(f"[ERROR] Audio file not found: {audio_path}")
            return 1

        result = classifier.predict(
            audio_path,
            lat=args.lat,
            lon=args.lon,
            region=args.region,
            threshold=args.threshold,
            top_k=args.top_k,
        )

    # Output formatting
    if args.json:
        out_dict = {
            "is_confident": result.is_confident,
            "detected_bird": result.top_species.common_name if result.is_confident else None,
            "scientific_name": result.top_species.scientific_name if result.is_confident else None,
            "confidence": round(result.top_confidence, 4),
            "confidence_percent": result.display_confidence_percent,
            "status": "DETECTED" if result.is_confident else "NO_CONFIDENT_BIRD_DETECTED",
            "top_candidates": [
                {
                    "rank": i + 1,
                    "common_name": sp.common_name,
                    "scientific_name": sp.scientific_name,
                    "confidence": round(p, 4),
                    "confidence_percent": f"{p * 100:.1f}%",
                }
                for i, (sp, p) in enumerate(result.top_k)
            ],
            "benchmark": {
                "audio_duration_s": round(result.duration_seconds, 2),
                "num_windows": result.num_windows,
                "preprocess_time_ms": round(result.preprocess_time_ms, 2),
                "inference_time_ms": round(result.inference_time_ms, 2),
                "total_time_ms": round(result.total_time_ms, 2),
                "compute_device": result.hardware_name,
                "provider": result.provider_name,
                "geo_region": result.region_applied,
            },
        }
        print(json.dumps(out_dict, indent=2))
    else:
        print("\n" + result.format_summary() + "\n")

    return 0


if __name__ == "__main__":
    sys.exit(main())
