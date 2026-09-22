"""Settings and hardware configuration for AvianAI India.

Provides automatic compute provider selection (CUDA -> CoreML -> CPU),
default audio parameters (32 kHz, 160k samples), paths, and geographic presets.
"""

from pathlib import Path
import os
from typing import Any, Dict, List, Optional
import onnxruntime as ort

# Base Paths
PROJECT_ROOT = Path(__file__).resolve().parent.parent
MODELS_DIR = PROJECT_ROOT / "models"
CONFIG_DIR = PROJECT_ROOT / "config"
SAMPLES_DIR = PROJECT_ROOT / "samples"
INDIAN_SAMPLES_DIR = SAMPLES_DIR / "indian_birds"

# Model & Label Files
MODEL_FILENAME = "birdnet-v3.0-preview3.1-fp16-b1.onnx"
LABELS_FILENAME = "birdnet-v3.0-preview3.1-labels-b1.txt"
REGIONS_FILENAME = "regions.json"
MODELS_META_FILENAME = "models.json"

MODEL_PATH = MODELS_DIR / MODEL_FILENAME
LABELS_PATH = MODELS_DIR / LABELS_FILENAME
REGIONS_PATH = CONFIG_DIR / REGIONS_FILENAME
MODELS_META_PATH = CONFIG_DIR / MODELS_META_FILENAME

# Regional Indices Files for India
REGIONAL_INDICES = {
    "south-asia-peninsular": CONFIG_DIR / "south-asia-peninsular-indices-b1.txt",
    "indo-gangetic": CONFIG_DIR / "indo-gangetic-indices-b1.txt",
    "himalaya": CONFIG_DIR / "himalaya-indices-b1.txt",
}

# Audio Specifications for BirdNET v3.0
SAMPLE_RATE = 32000
WINDOW_SECONDS = 5.0
WINDOW_SAMPLES = int(SAMPLE_RATE * WINDOW_SECONDS)  # 160,000 samples
WINDOW_STRIDE_SECONDS = 2.5  # 50% overlap for continuous audio
WINDOW_STRIDE_SAMPLES = int(SAMPLE_RATE * WINDOW_STRIDE_SECONDS)  # 80,000 samples

# Inference Defaults
CONFIDENCE_THRESHOLD = 0.25  # Minimum sigmoid probability for confident detection
LOW_CONFIDENCE_LABEL = "No confident bird detected"
DEFAULT_TOP_K = 5
BATCH_SIZE = 8

# Geographic Defaults (India)
DEFAULT_LATITUDE = 20.5937
DEFAULT_LONGITUDE = 78.9629
DEFAULT_REGION = "pan-india"  # 'pan-india', 'south-asia-peninsular', 'indo-gangetic', 'himalaya', 'all'

# Metadata & Author Attribution
DEVELOPER_NAME = "Mrutyunjay Joshi"
PROJECT_NAME = "AvianAI India"
MODEL_NAME = "AvianNet-v3"
MODEL_VERSION = "v3.0-GPU-FP16"
MODEL_AUTHORS = "Mrutyunjay Joshi"
LICENSE_NAME = "Proprietary / Open Research"
LICENSE_URL = "#"



def get_available_execution_providers() -> List[str]:
    """Returns the list of available ONNX Runtime execution providers."""
    return ort.get_available_providers()


def select_execution_providers(force_cpu: bool = False) -> List[str]:
    """Selects the best available ONNX Runtime execution providers in priority order.

    Priority:
    1. CUDAExecutionProvider (NVIDIA RTX 5090 / CUDA GPUs)
    2. TensorrtExecutionProvider (if configured)
    3. CoreMLExecutionProvider (Apple Silicon Mac)
    4. CPUExecutionProvider (universal fallback)
    """
    if force_cpu or os.getenv("FORCE_CPU", "0") == "1":
        return ["CPUExecutionProvider"]

    available = ort.get_available_providers()
    providers: List[str] = []

    # Check CUDA / TensorRT
    if "CUDAExecutionProvider" in available:
        providers.append("CUDAExecutionProvider")
    if "TensorrtExecutionProvider" in available:
        providers.append("TensorrtExecutionProvider")

    # Check CoreML (macOS)
    if "CoreMLExecutionProvider" in available:
        providers.append("CoreMLExecutionProvider")

    # Universal fallback
    if "CPUExecutionProvider" in available:
        providers.append("CPUExecutionProvider")
    elif not providers:
        providers = ["CPUExecutionProvider"]

    return providers


def get_hardware_info(session: Optional[ort.InferenceSession] = None) -> Dict[str, Any]:
    """Inspects the active compute device and environment."""
    available = ort.get_available_providers()
    active_providers = session.get_providers() if session is not None else select_execution_providers()

    primary_provider = active_providers[0] if active_providers else "CPUExecutionProvider"

    if "CUDAExecutionProvider" in primary_provider:
        hardware_name = "NVIDIA CUDA GPU (e.g. RTX 5090)"
        accel_type = "CUDA"
    elif "CoreMLExecutionProvider" in primary_provider:
        hardware_name = "Apple Silicon (CoreML / Neural Engine)"
        accel_type = "CoreML"
    else:
        hardware_name = "CPU (Multi-threaded)"
        accel_type = "CPU"

    return {
        "hardware_name": hardware_name,
        "accel_type": accel_type,
        "primary_provider": primary_provider,
        "active_providers": active_providers,
        "available_providers": available,
    }
