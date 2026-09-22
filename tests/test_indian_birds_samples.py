"""Integration tests for all Indian bird sample files."""

from pathlib import Path
import pytest
from config.settings import INDIAN_SAMPLES_DIR
from inference.model import get_model


@pytest.mark.parametrize(
    "filename, expected_common, expected_sci, min_confidence",
    [
        ("01_indian_peafowl.wav", "Indian Peafowl", "Pavo cristatus", 0.70),
        ("02_indian_roller.wav", "Indian Roller", "Coracias benghalensis", 0.60),
        ("03_common_kingfisher.wav", "Common Kingfisher", "Alcedo atthis", 0.70),
        ("04_red_vented_bulbul.wav", "Red-vented Bulbul", "Pycnonotus cafer", 0.70),
        ("05_oriental_magpie_robin.wav", "Oriental Magpie-Robin", "Copsychus saularis", 0.50),
    ],
)
def test_sample_species_identification(filename, expected_common, expected_sci, min_confidence):
    """Verifies that each sample audio file correctly predicts the expected Indian bird."""
    sample_path = INDIAN_SAMPLES_DIR / filename
    assert sample_path.exists(), f"Sample file not found: {sample_path}"

    classifier = get_model()
    result = classifier.predict(sample_path, region="pan-india")

    assert result.is_confident is True
    assert result.top_species.common_name == expected_common
    assert result.top_species.scientific_name == expected_sci
    assert result.top_confidence >= min_confidence
    assert result.total_time_ms > 0
