"""Unit tests for BirdNET v3.0 ONNX model loading and inference."""

from pathlib import Path
import numpy as np
import pytest

from config.settings import (
    MODEL_PATH,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from inference.model import BirdNETClassifier, PredictionResult, get_model


def test_model_initialization():
    """Tests loading the BirdNET v3.0 ONNX classifier."""
    classifier = get_model()
    assert classifier.session is not None
    assert classifier.hardware_info is not None
    assert "hardware_name" in classifier.hardware_info
    assert "primary_provider" in classifier.hardware_info


def test_dummy_forward_shape():
    """Tests running dummy 5s audio tensor through model."""
    classifier = get_model()
    dummy = np.random.randn(2, WINDOW_SAMPLES).astype(np.float32)
    probs, latency_ms = classifier._run_forward_chunks(dummy)

    assert probs.shape == (11560,)
    assert latency_ms > 0.0
    assert not np.isnan(probs).any()
    assert not np.isinf(probs).any()
    assert np.all(probs >= 0.0)
    assert np.all(probs <= 1.0)


def test_predict_waveform_structure():
    """Tests prediction return type and fields on in-memory waveform."""
    classifier = get_model()
    waveform = np.random.randn(int(SAMPLE_RATE * 5.0)).astype(np.float32)

    result = classifier.predict_waveform(
        waveform,
        sample_rate=SAMPLE_RATE,
        region="pan-india",
        threshold=0.25,
        top_k=5,
    )

    assert isinstance(result, PredictionResult)
    assert result.top_species is not None
    assert 0.0 <= result.top_confidence <= 1.0
    assert len(result.top_k) <= 5
    assert result.duration_seconds == pytest.approx(5.0, rel=1e-2)
    assert result.num_windows >= 1
    assert result.inference_time_ms > 0.0
    assert result.total_time_ms > 0.0
    assert result.hardware_name
    assert result.provider_name
