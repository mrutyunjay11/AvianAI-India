"""Integration tests for predict.py CLI and Indian bird sample identification."""

import json
from pathlib import Path
import subprocess
import sys
import pytest

from config.settings import INDIAN_SAMPLES_DIR


def test_cli_predict_sample_peafowl():
    """Test running predict.py CLI on Indian Peafowl sample."""
    sample = INDIAN_SAMPLES_DIR / "01_indian_peafowl.wav"
    assert sample.exists(), f"Sample {sample} does not exist"

    cmd = [sys.executable, "predict.py", "--audio", str(sample)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "Indian Peafowl" in res.stdout
    assert "Pavo cristatus" in res.stdout
    assert "CONFIDENCE:" in res.stdout


def test_cli_predict_json_output():
    """Test running predict.py with --json flag and verifying JSON schema."""
    sample = INDIAN_SAMPLES_DIR / "02_indian_roller.wav"
    assert sample.exists(), f"Sample {sample} does not exist"

    cmd = [sys.executable, "predict.py", "--audio", str(sample), "--json"]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    # Find JSON block in stdout
    lines = res.stdout.strip().split("\n")
    json_start = -1
    for i, line in enumerate(lines):
        if line.strip().startswith("{"):
            json_start = i
            break
    assert json_start != -1, "No JSON found in stdout"
    json_str = "\n".join(lines[json_start:])
    data = json.loads(json_str)

    assert data["is_confident"] is True
    assert data["detected_bird"] == "Indian Roller"
    assert data["scientific_name"] == "Coracias benghalensis"
    assert data["confidence"] > 0.5
    assert "top_candidates" in data
    assert len(data["top_candidates"]) > 0
    assert "benchmark" in data
    assert "inference_time_ms" in data["benchmark"]


def test_cli_predict_noise_low_confidence():
    """Test running predict.py on noise/silence sample triggers low confidence."""
    noise_path = Path("samples/test_noise.wav")
    if not noise_path.exists():
        import numpy as np, soundfile as sf
        noise = np.random.randn(32000 * 5).astype(np.float32) * 0.001
        sf.write(str(noise_path), noise, 32000)

    cmd = [sys.executable, "predict.py", "--audio", str(noise_path)]
    res = subprocess.run(cmd, capture_output=True, text=True)

    assert res.returncode == 0
    assert "No confident bird detected" in res.stdout
