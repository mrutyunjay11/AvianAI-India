"""Audio preprocessing pipeline for AvianAI India.

Loads audio files (WAV, OGG, MP3, FLAC), converts to mono, resamples to 32,000 Hz,
segments into 5.0-second (160,000 samples) windows with stride, and formats for ONNX inference.
"""

from pathlib import Path
import time
from typing import List, Optional, Tuple, Union
import numpy as np
import soundfile as sf
import librosa

from config.settings import (
    SAMPLE_RATE,
    WINDOW_SAMPLES,
    WINDOW_SECONDS,
    WINDOW_STRIDE_SAMPLES,
    WINDOW_STRIDE_SECONDS,
)


def load_audio(
    audio_path_or_bytes: Union[str, Path, bytes],
    target_sr: int = SAMPLE_RATE,
) -> Tuple[np.ndarray, int, float]:
    """Loads an audio file or bytes buffer into a 1D float32 numpy array resampled to target_sr.

    Supports WAV, OGG, MP3, FLAC.

    Returns:
        Tuple[np.ndarray, int, float]: (waveform, sample_rate, duration_seconds)
    """
    path_str = str(audio_path_or_bytes) if isinstance(audio_path_or_bytes, (str, Path)) else None

    waveform: Optional[np.ndarray] = None
    orig_sr: int = target_sr

    # Attempt soundfile loading first (fastest for uncompressed/WAV/OGG/FLAC)
    if path_str is not None:
        try:
            data, sr = sf.read(path_str, dtype="float32", always_2d=False)
            orig_sr = sr
            if data.ndim > 1:
                # Convert multi-channel to mono
                waveform = data.mean(axis=1)
            else:
                waveform = data
        except Exception:
            waveform = None

    # Fallback to librosa (handles MP3, varied codecs via audioread/ffmpeg)
    if waveform is None:
        try:
            waveform, orig_sr = librosa.load(
                audio_path_or_bytes,
                sr=None,  # keep original rate initially to control resampling cleanly
                mono=True,
                dtype=np.float32,
            )
        except Exception as e:
            raise ValueError(f"Failed to load audio from {audio_path_or_bytes}: {e}")

    # Resample if needed
    if orig_sr != target_sr:
        waveform = librosa.resample(
            waveform,
            orig_sr=orig_sr,
            target_sr=target_sr,
            res_type="soxr_hq" if hasattr(librosa.resample, "soxr") else "kaiser_fast",
        )

    # Ensure float32 and finite values
    waveform = np.nan_to_num(waveform, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)

    # Peak normalization if signal is non-empty
    max_val = np.max(np.abs(waveform))
    if max_val > 1.0:
        waveform = waveform / max_val

    duration = float(len(waveform) / target_sr)
    return waveform, target_sr, duration


def chunk_audio(
    waveform: np.ndarray,
    chunk_samples: int = WINDOW_SAMPLES,
    stride_samples: int = WINDOW_STRIDE_SAMPLES,
) -> List[np.ndarray]:
    """Slices a 1D audio waveform into 5-second (chunk_samples) segments with overlap.

    If the audio is shorter than chunk_samples, pads with zeros at the end to chunk_samples.
    """
    total_samples = len(waveform)

    if total_samples <= chunk_samples:
        # Pad short audio to exact chunk size
        padded = np.zeros(chunk_samples, dtype=np.float32)
        padded[:total_samples] = waveform
        return [padded]

    chunks: List[np.ndarray] = []
    start = 0

    while start < total_samples:
        end = start + chunk_samples
        if end <= total_samples:
            chunks.append(waveform[start:end])
        else:
            # Last partial chunk: zero-pad to full chunk length
            remaining = waveform[start:]
            padded = np.zeros(chunk_samples, dtype=np.float32)
            padded[: len(remaining)] = remaining
            chunks.append(padded)
            break

        start += stride_samples

    return chunks


def prepare_input_tensor(
    chunks: List[np.ndarray],
) -> np.ndarray:
    """Stacks audio chunks into an ONNX-ready batch tensor of shape [batch, 160000]."""
    if not chunks:
        # Fallback for empty audio: single 5s silence chunk
        return np.zeros((1, WINDOW_SAMPLES), dtype=np.float32)

    batch_array = np.stack(chunks, axis=0).astype(np.float32)
    return batch_array


class AudioPreprocessor:
    """Preprocessor class encapsulating loading, chunking, and batching."""

    def __init__(
        self,
        sample_rate: int = SAMPLE_RATE,
        window_samples: int = WINDOW_SAMPLES,
        stride_samples: int = WINDOW_STRIDE_SAMPLES,
    ) -> None:
        self.sample_rate = sample_rate
        self.window_samples = window_samples
        self.stride_samples = stride_samples

    def process_file(
        self,
        audio_path: Union[str, Path],
    ) -> Tuple[np.ndarray, float, int, float]:
        """Loads and processes an audio file.

        Returns:
            Tuple[np.ndarray, float, int, float]:
                (batch_tensor [N, 160000], duration_seconds, num_chunks, preprocess_time_ms)
        """
        t0 = time.perf_counter()
        waveform, sr, duration = load_audio(audio_path, target_sr=self.sample_rate)
        chunks = chunk_audio(waveform, chunk_samples=self.window_samples, stride_samples=self.stride_samples)
        tensor = prepare_input_tensor(chunks)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        return tensor, duration, len(chunks), preprocess_ms

    def process_waveform(
        self,
        waveform: np.ndarray,
        orig_sr: int = SAMPLE_RATE,
    ) -> Tuple[np.ndarray, float, int, float]:
        """Processes an in-memory numpy waveform (e.g. from microphone stream).

        Returns:
            Tuple[np.ndarray, float, int, float]:
                (batch_tensor [N, 160000], duration_seconds, num_chunks, preprocess_time_ms)
        """
        t0 = time.perf_counter()
        
        # Convert multi-channel if needed
        if waveform.ndim > 1:
            waveform = waveform.mean(axis=-1)

        # Convert to float32
        if waveform.dtype != np.float32:
            if np.issubdtype(waveform.dtype, np.integer):
                max_int = np.iinfo(waveform.dtype).max
                waveform = waveform.astype(np.float32) / max_int
            else:
                waveform = waveform.astype(np.float32)

        # Resample if needed
        if orig_sr != self.sample_rate:
            waveform = librosa.resample(waveform, orig_sr=orig_sr, target_sr=self.sample_rate)

        # Peak normalization
        max_val = np.max(np.abs(waveform))
        if max_val > 1.0:
            waveform = waveform / max_val

        duration = float(len(waveform) / self.sample_rate)
        chunks = chunk_audio(waveform, chunk_samples=self.window_samples, stride_samples=self.stride_samples)
        tensor = prepare_input_tensor(chunks)
        preprocess_ms = (time.perf_counter() - t0) * 1000.0
        return tensor, duration, len(chunks), preprocess_ms
