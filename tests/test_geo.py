"""Unit tests for geographic species filtering."""

import numpy as np
import pytest

from config.geo import GeoFilter, get_geo_filter


def test_geo_filter_initialization():
    geo = get_geo_filter()
    regions = geo.get_region_names()
    assert "pan-india" in regions
    assert "south-asia-peninsular" in regions
    assert "indo-gangetic" in regions
    assert "himalaya" in regions
    assert "all" in regions


def test_coordinates_matching():
    geo = get_geo_filter()
    # Bengaluru (South India): 12.97° N, 77.59° E -> south-asia-peninsular
    reg_south = geo.match_region_by_coordinates(12.97, 77.59)
    assert reg_south == "south-asia-peninsular"

    # Delhi / Agra (Indo-Gangetic): 28.61° N, 77.20° E -> indo-gangetic or himalaya
    reg_north = geo.match_region_by_coordinates(26.0, 80.0)
    assert reg_north == "indo-gangetic"

    # Leh / Ladakh / Himalaya: 34.15° N, 77.57° E -> himalaya
    reg_himalaya = geo.match_region_by_coordinates(34.15, 77.57)
    assert reg_himalaya == "himalaya"


def test_species_mask_shape():
    geo = get_geo_filter()
    mask = geo.create_species_mask(total_classes=11560, region="pan-india")
    assert mask.shape == (11560,)
    assert mask.dtype == bool
    # Pan India should allow a realistic subset (e.g. > 500 species, < 11560)
    allowed_count = int(mask.sum())
    assert 500 <= allowed_count < 11560


def test_filter_probabilities():
    geo = get_geo_filter()
    probs = np.ones(11560, dtype=np.float32)
    filtered = geo.filter_probabilities(probs, region="south-asia-peninsular")

    mask = geo.create_species_mask(total_classes=11560, region="south-asia-peninsular")
    assert np.all(filtered[~mask] == 0.0)
    assert np.all(filtered[mask] == 1.0)
