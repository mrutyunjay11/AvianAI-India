"""Unit tests for BirdNET v3.0 label catalog and parsing."""

import pytest
from config.labels import BirdSpecies, LabelCatalog, get_label_catalog
from config.settings import LABELS_PATH


def test_label_catalog_loading():
    catalog = get_label_catalog()
    catalog.ensure_loaded()
    assert len(catalog) == 11560


def test_species_resolution_by_index():
    catalog = get_label_catalog()
    sp = catalog.get_by_index(0)
    assert isinstance(sp, BirdSpecies)
    assert sp.index == 0
    assert sp.raw_label
    assert sp.scientific_name
    assert sp.common_name


def test_indian_birds_lookup():
    catalog = get_label_catalog()
    # Test lookups for iconic Indian birds
    peafowl_idx = catalog.find_index("Indian Peafowl")
    assert peafowl_idx is not None
    peafowl = catalog.get_by_index(peafowl_idx)
    assert "Peafowl" in peafowl.common_name
    assert "Pavo" in peafowl.scientific_name

    roller_idx = catalog.find_index("Indian Roller")
    assert roller_idx is not None
    roller = catalog.get_by_index(roller_idx)
    assert "Roller" in roller.common_name

    koel_idx = catalog.find_index("Asian Koel")
    assert koel_idx is not None
    koel = catalog.get_by_index(koel_idx)
    assert "Koel" in koel.common_name
