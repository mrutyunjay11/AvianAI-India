"""Unit tests for BirdNET v3.0 audio preprocessing."""

from pathlib import Path
import numpy as np
import pytest

from audio.preprocess import (
    AudioPreprocessor,
    chunk_audio,
    load_audio,
    prepare_input_tensor,
)
from config.settings import SAMPLE_RATE, WINDOW_SAMPLES, WINDOW_STRIDE_SAMPLES


def test_chunk_audio_exact():
    """Test chunking with audio matching 10s (320,000 samples)."""
    duration = 10.0
    waveform = np.zeros(int(SAMPLE_RATE * duration), dtype=np.float32)
    chunks = chunk_audio(waveform, chunk_samples=WINDOW_SAMPLES, stride_samples=WINDOW_STRIDE_SAMPLES)

    # 10s audio with 5s window and 2.5s stride produces 4 chunks
    assert len(chunks) == 4
    for chunk in chunks:
        assert len(chunk) == WINDOW_SAMPLES
        assert chunk.dtype == np.float32


def test_chunk_audio_padding():
    """Test padding when audio is shorter than 5.0s (e.g. 2.0s)."""
    duration = 2.0
    num_samples = int(SAMPLE_RATE * duration)
    waveform = np.ones(num_samples, dtype=np.float32)
    chunks = chunk_audio(waveform, chunk_samples=WINDOW_SAMPLES)

    assert len(chunks) == 1
    assert len(chunks[0]) == WINDOW_SAMPLES
    # First 2s should be 1.0, remainder 0.0
    assert np.all(chunks[0][:num_samples] == 1.0)
    assert np.all(chunks[0][num_samples:] == 0.0)


def test_prepare_input_tensor_shape():
    """Test stacking chunks into ONNX batch tensor [N, 160000]."""
    chunks = [np.random.randn(WINDOW_SAMPLES).astype(np.float32) for _ in range(4)]
    tensor = prepare_input_tensor(chunks)

    assert tensor.shape == (4, WINDOW_SAMPLES)
    assert tensor.dtype == np.float32
    assert not np.isnan(tensor).any()


def test_preprocessor_waveform():
    """Test AudioPreprocessor on in-memory waveform."""
    preprocessor = AudioPreprocessor()
    waveform = np.random.randn(int(SAMPLE_RATE * 6.0)).astype(np.float32)

    tensor, duration, num_chunks, preprocess_ms = preprocessor.process_waveform(waveform, orig_sr=SAMPLE_RATE)
    assert tensor.ndim == 2
    assert tensor.shape[1] == WINDOW_SAMPLES
    assert duration == pytest.approx(6.0, rel=1e-2)
    assert num_chunks == tensor.shape[0]
    assert preprocess_ms >= 0.0
