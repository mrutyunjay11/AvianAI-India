"""Microphone recording module for BirdNET v3.0 India.

Captures real-time audio from local microphone input using sounddevice,
with fallback options for headless or containerized environments.
"""

from pathlib import Path
import time
from typing import Optional, Tuple
import numpy as np
import soundfile as sf

from config.settings import SAMPLE_RATE, SAMPLES_DIR


def is_microphone_available() -> bool:
    """Checks if an audio input device is available."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        input_devs = [d for d in devices if d.get("max_input_channels", 0) > 0]
        return len(input_devs) > 0
    except Exception:
        return False


def get_input_devices() -> list[dict]:
    """Lists available audio recording input devices."""
    try:
        import sounddevice as sd
        devices = sd.query_devices()
        return [d for d in devices if d.get("max_input_channels", 0) > 0]
    except Exception:
        return []


def record_audio(
    duration: float = 5.0,
    sample_rate: int = SAMPLE_RATE,
    device_index: Optional[int] = None,
) -> Tuple[np.ndarray, int]:
    """Records audio from default or specified microphone for given duration.

    Args:
        duration: Recording duration in seconds (recommended: 5.0 to 10.0s).
        sample_rate: Target sample rate in Hz (default: 32000 for BirdNET v3.0).
        device_index: Optional specific input device index.

    Returns:
        Tuple[np.ndarray, int]: (waveform 1D float32 array, sample_rate)
    """
    try:
        import sounddevice as sd
    except ImportError:
        raise RuntimeError("sounddevice is required for microphone recording. Run: pip install sounddevice")

    num_samples = int(duration * sample_rate)
    print(f"[MIC] Recording {duration:.1f}s of audio @ {sample_rate} Hz...")

    recording = sd.rec(
        frames=num_samples,
        samplerate=sample_rate,
        channels=1,
        dtype="float32",
        device=device_index,
    )
    sd.wait()
    waveform = recording.flatten()
    print("[MIC] Recording completed.")

    return waveform, sample_rate


def save_audio(
    waveform: np.ndarray,
    output_path: Optional[Path] = None,
    sample_rate: int = SAMPLE_RATE,
) -> Path:
    """Saves recorded waveform to WAV file."""
    if output_path is None:
        SAMPLES_DIR.mkdir(parents=True, exist_ok=True)
        timestamp = int(time.time())
        output_path = SAMPLES_DIR / f"mic_recording_{timestamp}.wav"

    output_path.parent.mkdir(parents=True, exist_ok=True)
    sf.write(str(output_path), waveform, sample_rate, subtype="PCM_16")
    return output_path
