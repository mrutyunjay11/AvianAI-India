"""Inference engine for BirdNET v3.0 India.

Loads the GPU-native FP16 ONNX model, manages hardware acceleration (CUDA/CoreML/CPU),
applies official geographic filtering for India, and generates structured predictions
with sub-millisecond benchmarking.
"""

from dataclasses import dataclass
from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple, Union
import numpy as np
import onnxruntime as ort

from audio.preprocess import AudioPreprocessor
from config.geo import GeoFilter, get_geo_filter
from config.labels import BirdSpecies, LabelCatalog, get_label_catalog
from config.settings import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_REGION,
    DEFAULT_TOP_K,
    LABELS_PATH,
    LOW_CONFIDENCE_LABEL,
    MODEL_NAME,
    MODEL_PATH,
    MODELS_DIR,
    get_hardware_info,
    select_execution_providers,
)


@dataclass
class PredictionResult:
    """Encapsulates the structured prediction result for a bird audio sample."""

    top_species: BirdSpecies
    top_confidence: float
    is_confident: bool
    top_k: List[Tuple[BirdSpecies, float]]
    raw_probabilities: np.ndarray
    duration_seconds: float
    num_windows: int
    preprocess_time_ms: float
    inference_time_ms: float
    total_time_ms: float
    hardware_name: str
    provider_name: str
    region_applied: str

    @property
    def display_confidence_percent(self) -> str:
        return f"{self.top_confidence * 100:.1f}%"

    @property
    def display_title(self) -> str:
        if self.is_confident:
            return self.top_species.common_name or self.top_species.scientific_name
        return LOW_CONFIDENCE_LABEL

    @property
    def display_subtitle(self) -> str:
        if self.is_confident and self.top_species.scientific_name:
            return self.top_species.scientific_name
        return ""

    def format_summary(self) -> str:
        """Returns clean text summary for CLI or logs."""
        lines = []
        lines.append("=" * 55)
        lines.append(" AvianAI India — Bioacoustic Detection Summary")
        lines.append("=" * 55)

        if self.is_confident:
            lines.append(f"DETECTED BIRD: {self.top_species.common_name}")
            if self.top_species.scientific_name:
                lines.append(f"SCIENTIFIC:    {self.top_species.scientific_name}")
            lines.append(f"CONFIDENCE:    {self.display_confidence_percent}")
        else:
            lines.append(f"STATUS:        {LOW_CONFIDENCE_LABEL}")
            lines.append(f"HIGHEST MATCH: {self.top_species.common_name} ({self.display_confidence_percent})")

        lines.append(f"\nTop {len(self.top_k)} Candidate Species:")
        for rank, (sp, prob) in enumerate(self.top_k, 1):
            sci = f" ({sp.scientific_name})" if sp.scientific_name else ""
            lines.append(f"  {rank}. {sp.common_name}{sci}: {prob * 100:.1f}%")

        lines.append(f"\nBenchmark & Hardware:")
        lines.append(f"  Audio Duration:  {self.duration_seconds:.2f}s ({self.num_windows} × 5.0s windows)")
        lines.append(f"  Preprocessing:   {self.preprocess_time_ms:.1f} ms")
        lines.append(f"  Inference:       {self.inference_time_ms:.1f} ms")
        lines.append(f"  Total Latency:   {self.total_time_ms:.1f} ms")
        lines.append(f"  Compute Device:  {self.hardware_name}")
        lines.append(f"  Provider:        {self.provider_name}")
        lines.append(f"  Geo Filter:      {self.region_applied}")
        lines.append("=" * 55)
        return "\n".join(lines)


class BirdNETClassifier:
    """High-performance classifier using the BirdNET v3.0 FP16 ONNX model."""

    def __init__(
        self,
        model_path: Optional[Union[str, Path]] = None,
        labels_path: Union[str, Path] = LABELS_PATH,
        force_cpu: bool = False,
        confidence_threshold: float = CONFIDENCE_THRESHOLD,
    ) -> None:
        self.model_path = self._resolve_model_path(model_path)
        self.labels_path = Path(labels_path)
        self.force_cpu = force_cpu
        self.confidence_threshold = confidence_threshold

        self.catalog = get_label_catalog(self.labels_path)
        self.geo_filter = get_geo_filter()
        self.preprocessor = AudioPreprocessor()

        self.session: Optional[ort.InferenceSession] = None
        self.input_name: str = "input"
        self.output_names: List[str] = ["output"]
        self.hardware_info: Dict[str, Any] = {}
        self.num_classes: int = 11560
        self.regional_mapping: Optional[List[int]] = None

        self._load_session()

    def _resolve_model_path(self, explicit_path: Optional[Union[str, Path]]) -> Path:
        """Finds the best available BirdNET v3.0 model file."""
        if explicit_path is not None:
            return Path(explicit_path)

        # Primary preference: Full global FP16 model
        if MODEL_PATH.exists() and MODEL_PATH.stat().st_size > 10 * 1024 * 1024:
            return MODEL_PATH

        # Secondary: Regional South Asia model if present
        regional_south = MODELS_DIR / "birdnet-v3.0-preview3.1-south-asia-peninsular-fp16-b1.onnx"
        if regional_south.exists() and regional_south.stat().st_size > 10 * 1024 * 1024:
            return regional_south

        return MODEL_PATH

    def _load_session(self) -> None:
        """Initializes the ONNX Runtime session."""
        if not self.model_path.exists() or self.model_path.stat().st_size < 10 * 1024 * 1024:
            raise FileNotFoundError(f"Required model file not found at: {self.model_path}")

        self.catalog.ensure_loaded()

        sess_options = ort.SessionOptions()
        sess_options.graph_optimization_level = ort.GraphOptimizationLevel.ORT_ENABLE_ALL
        sess_options.intra_op_num_threads = 4

        providers = select_execution_providers(force_cpu=self.force_cpu)
        print(f"[ONNX] Initializing BirdNET v3.0 session with providers: {providers}")

        try:
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=providers,
            )
        except Exception as e:
            print(f"[WARNING] Provider initialization with {providers} failed: {e}. Falling back to CPUExecutionProvider.")
            self.session = ort.InferenceSession(
                str(self.model_path),
                sess_options=sess_options,
                providers=["CPUExecutionProvider"],
            )

        # Inspect model I/O
        inputs = self.session.get_inputs()
        outputs = self.session.get_outputs()
        self.input_name = inputs[0].name if inputs else "input"
        self.output_names = [o.name for o in outputs]
        self.hardware_info = get_hardware_info(self.session)

        # Detect class dimension
        out_shape = outputs[0].shape
        if len(out_shape) > 1 and isinstance(out_shape[1], int):
            self.num_classes = out_shape[1]
        else:
            self.num_classes = 11560

        if self.num_classes < 11560:
            # Regional slice model
            if "south-asia-peninsular" in self.model_path.name:
                self.regional_mapping = self.geo_filter.get_allowed_indices("south-asia-peninsular")
            elif "indo-gangetic" in self.model_path.name:
                self.regional_mapping = self.geo_filter.get_allowed_indices("indo-gangetic")
            elif "himalaya" in self.model_path.name:
                self.regional_mapping = self.geo_filter.get_allowed_indices("himalaya")

        print(f"[ONNX] Active Hardware: {self.hardware_info['hardware_name']} ({self.hardware_info['primary_provider']})")
        print(f"[ONNX] Model: '{self.model_path.name}' | Output classes: {self.num_classes:,}")

    def _run_forward_window(self, window_1d: np.ndarray) -> np.ndarray:
        """Runs forward inference on a single window [1, 160000]."""
        inp_2d = window_1d.reshape(1, -1).astype(np.float32)
        outputs = self.session.run(
            [self.output_names[0]],
            {self.input_name: inp_2d},
        )
        raw_preds = outputs[0][0]  # shape [num_classes]

        # Apply sigmoid if raw logits exceed [0, 1]
        if np.any(raw_preds < 0.0) or np.any(raw_preds > 1.0):
            probs = 1.0 / (1.0 + np.exp(-np.clip(raw_preds, -20.0, 20.0)))
        else:
            probs = raw_preds

        return probs

    def _run_forward_chunks(self, batch_tensor: np.ndarray) -> Tuple[np.ndarray, float]:
        """Executes forward pass across all 5s chunks with NaN safety and pooling."""
        if self.session is None:
            raise RuntimeError("ONNX session not initialized.")

        t0 = time.perf_counter()
        valid_probs: List[np.ndarray] = []

        for i in range(len(batch_tensor)):
            chunk_probs = self._run_forward_window(batch_tensor[i])
            if not np.isnan(chunk_probs).any():
                valid_probs.append(chunk_probs)

        inference_ms = (time.perf_counter() - t0) * 1000.0

        if not valid_probs:
            # Fallback for all-nan (e.g. extreme silence)
            aggregated = np.zeros(self.num_classes, dtype=np.float32)
        elif len(valid_probs) == 1:
            aggregated = valid_probs[0]
        else:
            # Max pooling across valid temporal windows
            aggregated = np.max(np.stack(valid_probs, axis=0), axis=0)

        return aggregated, inference_ms

    def predict(
        self,
        audio_path_or_bytes: Union[str, Path, bytes],
        lat: Optional[float] = DEFAULT_LATITUDE,
        lon: Optional[float] = DEFAULT_LONGITUDE,
        region: Optional[str] = DEFAULT_REGION,
        threshold: Optional[float] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> PredictionResult:
        """Runs full prediction pipeline on an audio file."""
        t_start = time.perf_counter()
        active_threshold = threshold if threshold is not None else self.confidence_threshold

        # 1. Preprocessing
        batch_tensor, duration, num_windows, preprocess_ms = self.preprocessor.process_file(audio_path_or_bytes)

        # 2. ONNX Inference with window aggregation
        aggregated_probs, inference_ms = self._run_forward_chunks(batch_tensor)

        # 3. Map to global 11,560 dimension if regional slice is loaded
        if self.regional_mapping is not None and len(aggregated_probs) == len(self.regional_mapping):
            global_probs = np.zeros(11560, dtype=np.float32)
            global_probs[self.regional_mapping] = aggregated_probs
        else:
            global_probs = aggregated_probs

        # 4. Geographic Species Filtering
        filtered_probs = self.geo_filter.filter_probabilities(
            global_probs,
            region=region,
            lat=lat,
            lon=lon,
        )

        # 5. Extract Top Predictions
        sorted_indices = np.argsort(filtered_probs)[::-1]
        top_indices = sorted_indices[:top_k]

        top_k_list: List[Tuple[BirdSpecies, float]] = []
        for idx in top_indices:
            species_info = self.catalog.get_by_index(int(idx))
            conf = float(filtered_probs[idx])
            top_k_list.append((species_info, conf))

        primary_idx = int(sorted_indices[0])
        primary_species = self.catalog.get_by_index(primary_idx)
        primary_confidence = float(filtered_probs[primary_idx])

        is_confident = primary_confidence >= active_threshold
        total_ms = (time.perf_counter() - t_start) * 1000.0

        applied_region_str = region if region else f"GPS ({lat:.2f}, {lon:.2f})"

        return PredictionResult(
            top_species=primary_species,
            top_confidence=primary_confidence,
            is_confident=is_confident,
            top_k=top_k_list,
            raw_probabilities=filtered_probs,
            duration_seconds=duration,
            num_windows=num_windows,
            preprocess_time_ms=preprocess_ms,
            inference_time_ms=inference_ms,
            total_time_ms=total_ms,
            hardware_name=str(self.hardware_info.get("hardware_name", "CPU")),
            provider_name=str(self.hardware_info.get("primary_provider", "CPUExecutionProvider")),
            region_applied=applied_region_str,
        )

    def predict_waveform(
        self,
        waveform: np.ndarray,
        sample_rate: int = 32000,
        lat: Optional[float] = DEFAULT_LATITUDE,
        lon: Optional[float] = DEFAULT_LONGITUDE,
        region: Optional[str] = DEFAULT_REGION,
        threshold: Optional[float] = None,
        top_k: int = DEFAULT_TOP_K,
    ) -> PredictionResult:
        """Runs prediction on in-memory numpy waveform."""
        t_start = time.perf_counter()
        active_threshold = threshold if threshold is not None else self.confidence_threshold

        # 1. Preprocessing
        batch_tensor, duration, num_windows, preprocess_ms = self.preprocessor.process_waveform(
            waveform, orig_sr=sample_rate
        )

        # 2. ONNX Inference with window aggregation
        aggregated_probs, inference_ms = self._run_forward_chunks(batch_tensor)

        # 3. Map to global 11,560 dimension if regional slice is loaded
        if self.regional_mapping is not None and len(aggregated_probs) == len(self.regional_mapping):
            global_probs = np.zeros(11560, dtype=np.float32)
            global_probs[self.regional_mapping] = aggregated_probs
        else:
            global_probs = aggregated_probs

        # 4. Geographic Filtering
        filtered_probs = self.geo_filter.filter_probabilities(
            global_probs,
            region=region,
            lat=lat,
            lon=lon,
        )

        # 5. Extract Top Predictions
        sorted_indices = np.argsort(filtered_probs)[::-1]
        top_indices = sorted_indices[:top_k]

        top_k_list: List[Tuple[BirdSpecies, float]] = []
        for idx in top_indices:
            species_info = self.catalog.get_by_index(int(idx))
            conf = float(filtered_probs[idx])
            top_k_list.append((species_info, conf))

        primary_idx = int(sorted_indices[0])
        primary_species = self.catalog.get_by_index(primary_idx)
        primary_confidence = float(filtered_probs[primary_idx])
        is_confident = primary_confidence >= active_threshold

        total_ms = (time.perf_counter() - t_start) * 1000.0
        applied_region_str = region if region else f"GPS ({lat:.2f}, {lon:.2f})"

        return PredictionResult(
            top_species=primary_species,
            top_confidence=primary_confidence,
            is_confident=is_confident,
            top_k=top_k_list,
            raw_probabilities=filtered_probs,
            duration_seconds=duration,
            num_windows=num_windows,
            preprocess_time_ms=preprocess_ms,
            inference_time_ms=inference_ms,
            total_time_ms=total_ms,
            hardware_name=str(self.hardware_info.get("hardware_name", "CPU")),
            provider_name=str(self.hardware_info.get("primary_provider", "CPUExecutionProvider")),
            region_applied=applied_region_str,
        )


_MODEL_INSTANCE: Optional[BirdNETClassifier] = None


def get_model(
    model_path: Optional[Union[str, Path]] = None,
    labels_path: Union[str, Path] = LABELS_PATH,
    force_cpu: bool = False,
) -> BirdNETClassifier:
    """Returns singleton instance of BirdNETClassifier."""
    global _MODEL_INSTANCE
    if _MODEL_INSTANCE is None:
        _MODEL_INSTANCE = BirdNETClassifier(
            model_path=model_path,
            labels_path=labels_path,
            force_cpu=force_cpu,
        )
    return _MODEL_INSTANCE
