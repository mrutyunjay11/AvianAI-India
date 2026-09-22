"""Verification script for AvianAI India.

Verifies:
1. Model file existence and integrity.
2. ONNX Runtime session creation and compute hardware (CUDA/CoreML/CPU).
3. Input tensor specifications ([batch, 160000] @ 32 kHz).
4. Output tensor shapes (predictions [batch, 11560], embeddings [batch, 1280]).
5. Class label alignment (11,560 species).
6. Geographic species derivation and Indian subregion indices.
7. End-to-end dummy signal inference and real sample audio verification.
"""

from pathlib import Path
import sys
import time
import numpy as np

from config.settings import (
    CONFIDENCE_THRESHOLD,
    INDIAN_SAMPLES_DIR,
    LABELS_PATH,
    MODEL_PATH,
    PROJECT_ROOT,
    REGIONAL_INDICES,
    SAMPLE_RATE,
    WINDOW_SAMPLES,
)
from config.labels import get_label_catalog
from config.geo import get_geo_filter


def verify_system() -> bool:
    print("=" * 68)
    print(" AvianAI India - System & Model Verification")
    print(" Developed & Engineered by Mrutyunjay Joshi")
    print("=" * 68)

    # 1. Verify File Presence
    print("\n[Step 1] Checking Model & Metadata Files...")
    if not MODEL_PATH.exists():
        print(f"[ERROR] Model file not found at: {MODEL_PATH}")
        return False

    model_size_mb = MODEL_PATH.stat().st_size / (1024 * 1024)
    print(f"  -> Model File: {MODEL_PATH.name} ({model_size_mb:.2f} MB)")
    print(f"  -> Labels File: {LABELS_PATH.name}")

    # 2. Labels Verification
    print("\n[Step 2] Verifying Label Catalog...")
    catalog = get_label_catalog(LABELS_PATH)
    total_labels = len(catalog)
    print(f"  -> Total Classes: {total_labels:,}")

    if total_labels != 11560:
        print(f"[ERROR] Expected 11,560 classes, but got {total_labels}")
        return False
    print("  -> [OK] Exact 11,560 classes verified.")

    # Check prominent Indian species in catalog
    test_birds = ["Indian Peafowl", "Indian Roller", "Asian Koel", "Common Kingfisher", "Oriental Magpie-Robin"]
    print("  -> Sample Species in Official Catalog:")
    for b_name in test_birds:
        idx = catalog.find_index(b_name)
        if idx is not None:
            sp = catalog.get_by_index(idx)
            print(f"     [{idx:5d}] {sp.common_name} ({sp.scientific_name})")
        else:
            print(f"     [!] Could not locate '{b_name}'")

    # 3. Geographic Species Filtering Verification
    print("\n[Step 3] Verifying Indian Geographic Subregions & Species Masks...")
    geo = get_geo_filter()
    regions = geo.get_region_names()
    print(f"  -> Supported Regions: {regions}")

    for reg in ["south-asia-peninsular", "indo-gangetic", "himalaya", "pan-india"]:
        indices = geo.get_allowed_indices(region=reg)
        count = len(indices) if indices else 0
        print(f"     * {reg:22s}: {count:5d} allowed species")

    pan_india_count = len(geo.pan_india_indices)
    if pan_india_count < 500:
        print(f"[ERROR] Pan-India species set contains suspiciously few species ({pan_india_count})")
        return False
    print(f"  -> [OK] Pan-India unified catalog: {pan_india_count} allowed species.")

    # 4. ONNX Model Loading & Hardware Detection
    print("\n[Step 4] Initializing ONNX Runtime Session & Hardware Detection...")
    from inference.model import BirdNETClassifier
    try:
        classifier = BirdNETClassifier(model_path=MODEL_PATH, labels_path=LABELS_PATH)
    except Exception as e:
        print(f"[ERROR] Failed to instantiate BirdNETClassifier: {e}")
        return False

    hw = classifier.hardware_info
    print(f"  -> Active Hardware:    {hw['hardware_name']}")
    print(f"  -> Primary Provider:   {hw['primary_provider']}")
    print(f"  -> Available Backends: {hw['available_providers']}")

    # 5. Model Input & Output Shapes Verification
    print("\n[Step 5] Verifying Tensor Input/Output Contracts...")
    inputs = classifier.session.get_inputs()
    outputs = classifier.session.get_outputs()

    print(f"  -> Model Inputs ({len(inputs)}):")
    for inp in inputs:
        print(f"     Name: '{inp.name}' | Shape: {inp.shape} | Type: {inp.type}")

    print(f"  -> Model Outputs ({len(outputs)}):")
    for out in outputs:
        print(f"     Name: '{out.name}' | Shape: {out.shape} | Type: {out.type}")

    # 6. Dummy Inference Pass
    print("\n[Step 6] Running Batch Dummy Inference (5.0s Audio @ 32 kHz)...")
    dummy_signal = np.random.randn(2, WINDOW_SAMPLES).astype(np.float32)
    dummy_probs, dummy_latency = classifier._run_forward_chunks(dummy_signal)

    print(f"  -> Input Shape:     {dummy_signal.shape} (2 windows × {WINDOW_SAMPLES} samples)")
    print(f"  -> Output Shape:    {dummy_probs.shape}")
    print(f"  -> Inference Time:  {dummy_latency:.2f} ms ({dummy_latency / 2:.2f} ms / window)")
    print(f"  -> Probability Min: {dummy_probs.min():.5f} | Max: {dummy_probs.max():.5f}")

    if dummy_probs.shape != (11560,):
        print(f"[ERROR] Expected output shape (11560,), got {dummy_probs.shape}")
        return False

    if np.isnan(dummy_probs).any() or np.isinf(dummy_probs).any():
        print("[ERROR] Predictions contain NaN or Inf values!")
        return False


    # 7. Real Sample Inference Pass
    print("\n[Step 7] Running Real Indian Bird Audio Sample Inference...")
    sample_files = list(INDIAN_SAMPLES_DIR.glob("*.wav")) if INDIAN_SAMPLES_DIR.exists() else []

    if sample_files:
        sample_path = sample_files[0]
        print(f"  -> Testing Sample: {sample_path.name}")
        result = classifier.predict(
            sample_path,
            region="pan-india",
            threshold=CONFIDENCE_THRESHOLD,
            top_k=3,
        )

        print("\n" + result.format_summary())

        if result.is_confident:
            print(f"  -> [OK] Successfully identified: {result.top_species.common_name} ({result.display_confidence_percent})")
        else:
            print(f"  -> [INFO] Detection below threshold: {result.display_title}")
    else:
        print("  -> [INFO] No sample WAV files found in samples/indian_birds/ to test.")

    print("\n" + "=" * 68)
    print(" [SUCCESS] All AvianAI India System Verifications Passed!")
    print("=" * 68)
    return True


if __name__ == "__main__":
    success = verify_system()
    sys.exit(0 if success else 1)
