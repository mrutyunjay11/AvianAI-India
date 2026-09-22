"""AvianAI India — Bioacoustic Bird Identification Web Application.

Real-time bird sound identification system for Indian avifauna,
engineered by Mrutyunjay Joshi.
"""

from pathlib import Path
import time
from typing import Any, Dict, List, Optional, Tuple
import gradio as gr
import numpy as np

from audio.microphone import is_microphone_available, record_audio, save_audio
from config.geo import get_geo_filter
from config.labels import get_label_catalog
from config.settings import (
    CONFIDENCE_THRESHOLD,
    DEFAULT_LATITUDE,
    DEFAULT_LONGITUDE,
    DEFAULT_REGION,
    DEVELOPER_NAME,
    INDIAN_SAMPLES_DIR,
    LICENSE_NAME,
    LICENSE_URL,
    LOW_CONFIDENCE_LABEL,
    MODEL_AUTHORS,
    MODEL_NAME,
    MODEL_VERSION,
    PROJECT_NAME,
    SAMPLE_RATE,
)
from inference.model import BirdNETClassifier, PredictionResult, get_model


# ==============================================================================
# Emerald Dark & Glassmorphism Theme CSS
# ==============================================================================
CUSTOM_CSS = """
@import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=JetBrains+Mono:wght@400;600;700&display=swap');

:root {
    --bg-main: #060b09;
    --card-bg: rgba(13, 23, 19, 0.75);
    --card-border: rgba(52, 211, 153, 0.18);
    --primary: #10b981;
    --primary-glow: rgba(16, 185, 129, 0.35);
    --accent-emerald: #34d399;
    --accent-gold: #fbbf24;
    --accent-cyan: #38bdf8;
    --text-main: #f8fafc;
    --text-muted: #94a3b8;
}

body, .gradio-container {
    font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif !important;
    background: radial-gradient(circle at 50% -10%, #064e3b 0%, #071e16 35%, #030806 100%) !important;
    color: var(--text-main) !important;
    min-height: 100vh;
}

/* App Header */
.app-header-container {
    text-align: center;
    padding: 28px 20px 16px 20px;
    margin-bottom: 12px;
}

.brand-badge {
    display: inline-flex;
    align-items: center;
    gap: 8px;
    background: rgba(16, 185, 129, 0.12);
    border: 1px solid rgba(52, 211, 153, 0.3);
    border-radius: 9999px;
    padding: 6px 16px;
    font-size: 0.85rem;
    font-weight: 700;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    color: #34d399;
    margin-bottom: 12px;
    box-shadow: 0 0 16px rgba(16, 185, 129, 0.15);
}

.brand-title {
    font-size: 3rem !important;
    font-weight: 800 !important;
    letter-spacing: -0.03em;
    background: linear-gradient(135deg, #ffffff 0%, #a7f3d0 40%, #34d399 75%, #fef08a 100%);
    -webkit-background-clip: text;
    -webkit-text-fill-color: transparent;
    line-height: 1.15;
    margin-bottom: 8px;
}

.brand-subtitle {
    font-size: 1.1rem;
    color: #cbd5e1;
    max-width: 680px;
    margin: 0 auto;
    line-height: 1.5;
    font-weight: 400;
}

/* Hero Result Card */
.hero-detection-card {
    background: linear-gradient(135deg, rgba(6, 78, 59, 0.35) 0%, rgba(13, 23, 19, 0.85) 100%);
    border: 2px solid rgba(52, 211, 153, 0.4);
    border-radius: 24px;
    padding: 32px 24px;
    text-align: center;
    box-shadow: 0 20px 50px rgba(0, 0, 0, 0.5), 0 0 30px rgba(16, 185, 129, 0.15);
    position: relative;
    overflow: hidden;
    transition: all 0.3s cubic-bezier(0.4, 0, 0.2, 1);
}

.hero-detection-card::before {
    content: '';
    position: absolute;
    top: 0;
    left: -100%;
    width: 200%;
    height: 100%;
    background: linear-gradient(90deg, transparent, rgba(255,255,255,0.05), transparent);
    animation: shimmer 4s infinite;
}

@keyframes shimmer {
    0% { transform: translateX(-100%); }
    100% { transform: translateX(100%); }
}

.hero-status-pill {
    display: inline-flex;
    align-items: center;
    gap: 6px;
    background: rgba(52, 211, 153, 0.15);
    border: 1px solid rgba(52, 211, 153, 0.4);
    border-radius: 30px;
    padding: 4px 14px;
    font-size: 0.8rem;
    font-weight: 700;
    letter-spacing: 0.1em;
    text-transform: uppercase;
    color: #6ee7b7;
    margin-bottom: 12px;
}

.hero-bird-title {
    font-size: 2.7rem;
    font-weight: 800;
    color: #ffffff;
    letter-spacing: -0.02em;
    line-height: 1.15;
    margin: 6px 0;
    text-shadow: 0 2px 10px rgba(0,0,0,0.5);
}

.hero-bird-scientific {
    font-size: 1.25rem;
    font-style: italic;
    color: #94a3b8;
    margin-bottom: 20px;
    font-weight: 400;
}

.hero-score-badge {
    display: inline-flex;
    flex-direction: column;
    align-items: center;
    background: radial-gradient(circle, rgba(16, 185, 129, 0.25) 0%, rgba(0, 0, 0, 0.5) 100%);
    border: 1px solid rgba(52, 211, 153, 0.35);
    border-radius: 20px;
    padding: 12px 32px;
    box-shadow: 0 8px 24px rgba(0, 0, 0, 0.3);
}

.hero-score-num {
    font-family: 'JetBrains Mono', monospace;
    font-size: 3rem;
    font-weight: 800;
    color: #34d399;
    line-height: 1;
}

.hero-score-label {
    font-size: 0.85rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    color: #cbd5e1;
    margin-top: 4px;
}

.hero-empty-state {
    border: 2px dashed rgba(255, 255, 255, 0.15);
    background: rgba(13, 23, 19, 0.4);
    border-radius: 24px;
    padding: 48px 24px;
    text-align: center;
}

.hero-unreliable-card {
    background: linear-gradient(135deg, rgba(153, 27, 27, 0.25) 0%, rgba(13, 23, 19, 0.85) 100%);
    border: 2px solid rgba(248, 113, 113, 0.35);
    border-radius: 24px;
    padding: 32px 24px;
    text-align: center;
}

/* Benchmark Grid */
.bench-grid {
    display: grid;
    grid-template-columns: repeat(4, 1fr);
    gap: 12px;
    margin-top: 10px;
}

@media (max-width: 768px) {
    .bench-grid { grid-template-columns: repeat(2, 1fr); }
    .brand-title { font-size: 2.2rem !important; }
    .hero-bird-title { font-size: 2.1rem; }
}

.bench-tile {
    background: rgba(15, 23, 42, 0.6);
    border: 1px solid rgba(255, 255, 255, 0.08);
    border-radius: 14px;
    padding: 14px;
    text-align: center;
}

.bench-tile-label {
    font-size: 0.75rem;
    font-weight: 700;
    text-transform: uppercase;
    letter-spacing: 0.06em;
    color: #94a3b8;
    margin-bottom: 4px;
}

.bench-tile-val {
    font-family: 'JetBrains Mono', monospace;
    font-size: 1.35rem;
    font-weight: 700;
    color: #f8fafc;
}

/* Top Candidates */
.candidate-item {
    display: flex;
    align-items: center;
    justify-content: space-between;
    background: rgba(255, 255, 255, 0.03);
    border: 1px solid rgba(255, 255, 255, 0.06);
    border-radius: 12px;
    padding: 12px 16px;
    margin-bottom: 8px;
    transition: all 0.2s ease;
}

.candidate-item:hover {
    background: rgba(255, 255, 255, 0.06);
    border-color: rgba(52, 211, 153, 0.25);
    transform: translateX(3px);
}

.candidate-rank {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    color: #64748b;
    width: 24px;
}

.candidate-info {
    flex: 1;
    margin-left: 8px;
}

.candidate-name {
    font-weight: 700;
    color: #f1f5f9;
    font-size: 1rem;
}

.candidate-sci {
    font-size: 0.85rem;
    color: #94a3b8;
    font-style: italic;
    margin-left: 6px;
}

.candidate-bar-container {
    width: 140px;
    margin: 0 16px;
    background: rgba(255, 255, 255, 0.08);
    height: 8px;
    border-radius: 4px;
    overflow: hidden;
}

.candidate-bar-fill {
    height: 100%;
    border-radius: 4px;
    background: linear-gradient(90deg, #10b981, #34d399);
}

.candidate-score {
    font-family: 'JetBrains Mono', monospace;
    font-weight: 700;
    font-size: 1.05rem;
    color: #34d399;
    width: 60px;
    text-align: right;
}

/* Primary Action Buttons */
.btn-record-mic {
    background: linear-gradient(135deg, #ef4444 0%, #dc2626 100%) !important;
    color: #ffffff !important;
    font-weight: 800 !important;
    font-size: 1.15rem !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(239, 68, 68, 0.35) !important;
    padding: 16px 24px !important;
    transition: all 0.2s ease !important;
}

.btn-record-mic:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(239, 68, 68, 0.5) !important;
}

.btn-identify {
    background: linear-gradient(135deg, #10b981 0%, #059669 100%) !important;
    color: #ffffff !important;
    font-weight: 700 !important;
    font-size: 1.1rem !important;
    border-radius: 14px !important;
    border: none !important;
    box-shadow: 0 8px 24px rgba(16, 185, 129, 0.35) !important;
    padding: 14px 24px !important;
    transition: all 0.2s ease !important;
}

.btn-identify:hover {
    transform: translateY(-2px) !important;
    box-shadow: 0 12px 30px rgba(16, 185, 129, 0.5) !important;
}
"""


def render_hero_html(result: Optional[PredictionResult]) -> str:
    """Renders prominent HTML card showing ONLY detected bird & confidence."""
    if result is None:
        return """
        <div class="hero-empty-state">
            <div style="font-size: 1.3rem; font-weight: 700; color: #e2e8f0;">Ready to Identify Indian Birds</div>
            <div style="font-size: 0.95rem; color: #94a3b8; margin-top: 6px;">
                Click <strong>"Record from Microphone Now"</strong> or upload/select an Indian bird audio file.
            </div>
        </div>
        """

    if not result.is_confident:
        return f"""
        <div class="hero-unreliable-card">
            <div class="hero-status-pill" style="background: rgba(239,68,68,0.2); border-color: rgba(239,68,68,0.4); color: #fca5a5;">
                Detection Status
            </div>
            <div style="font-size: 2.2rem; font-weight: 800; color: #fecaca; margin: 8px 0;">
                {LOW_CONFIDENCE_LABEL}
            </div>
            <div style="color: #cbd5e1; font-size: 1rem; margin-top: 8px;">
                Highest Candidate: <strong style="color:#ffffff;">{result.top_species.common_name}</strong> ({result.top_confidence * 100:.1f}%)
            </div>
            <div style="color: #94a3b8; font-size: 0.88rem; margin-top: 6px;">
                Tip: Ensure the bird call is audible and minimize background interference.
            </div>
        </div>
        """

    sp = result.top_species
    return f"""
    <div class="hero-detection-card">
        <div class="hero-status-pill">
            Verified Match | {result.region_applied}
        </div>
        <div class="hero-bird-title">
            {sp.common_name}
        </div>
        <div class="hero-bird-scientific">
            {sp.scientific_name}
        </div>
        <div class="hero-score-badge">
            <div class="hero-score-num">{result.display_confidence_percent}</div>
            <div class="hero-score-label">Confidence Score</div>
        </div>
    </div>
    """


def render_topk_html(result: Optional[PredictionResult]) -> str:
    """Renders the top candidate species breakdown."""
    if result is None or not result.top_k:
        return "<p style='color:#64748b; font-style:italic; text-align:center; padding: 12px;'>No predictions to display yet.</p>"

    rows = []
    for rank, (sp, conf) in enumerate(result.top_k[:5], 1):
        pct_str = f"{conf * 100:.1f}%"
        bar_width = max(2, min(100, int(conf * 100)))

        row = f"""
        <div class="candidate-item">
            <div class="candidate-rank">#{rank}</div>
            <div class="candidate-info">
                <span class="candidate-name">{sp.common_name}</span>
                <span class="candidate-sci">({sp.scientific_name})</span>
            </div>
            <div class="candidate-bar-container">
                <div class="candidate-bar-fill" style="width: {bar_width}%;"></div>
            </div>
            <div class="candidate-score">{pct_str}</div>
        </div>
        """
        rows.append(row)

    return "".join(rows)


def render_benchmark_html(result: Optional[PredictionResult]) -> str:
    """Renders benchmark latency diagnostics."""
    if result is None:
        hw_info = get_model().hardware_info
        return f"""
        <div class="bench-grid">
            <div class="bench-tile">
                <div class="bench-tile-label">Compute Device</div>
                <div class="bench-tile-val" style="color: #34d399; font-size: 1.05rem;">{hw_info.get('hardware_name', 'CPU')}</div>
            </div>
            <div class="bench-tile">
                <div class="bench-tile-label">Provider</div>
                <div class="bench-tile-val" style="font-size: 0.95rem; color: #cbd5e1;">{hw_info.get('primary_provider', 'CPU')}</div>
            </div>
            <div class="bench-tile">
                <div class="bench-tile-label">Model</div>
                <div class="bench-tile-val" style="font-size: 1.05rem; color: #38bdf8;">AvianNet-v3</div>
            </div>
            <div class="bench-tile">
                <div class="bench-tile-label">Precision</div>
                <div class="bench-tile-val" style="color: #fcd34d;">FP16 ONNX</div>
            </div>
        </div>
        """

    return f"""
    <div class="bench-grid">
        <div class="bench-tile">
            <div class="bench-tile-label">Preprocessing</div>
            <div class="bench-tile-val" style="color: #38bdf8;">{result.preprocess_time_ms:.1f} ms</div>
        </div>
        <div class="bench-tile">
            <div class="bench-tile-label">ONNX Inference</div>
            <div class="bench-tile-val" style="color: #34d399;">{result.inference_time_ms:.1f} ms</div>
        </div>
        <div class="bench-tile">
            <div class="bench-tile-label">Total Latency</div>
            <div class="bench-tile-val" style="color: #fcd34d;">{result.total_time_ms:.1f} ms</div>
        </div>
        <div class="bench-tile">
            <div class="bench-tile-label">Hardware Device</div>
            <div class="bench-tile-val" style="font-size: 0.95rem; color: #f8fafc;">{result.hardware_name}</div>
        </div>
    </div>
    """


def predict_audio_handler(
    audio_path: Optional[str],
    region_choice: str,
    lat_val: float,
    lon_val: float,
    threshold_val: float,
) -> Tuple[str, str, str]:
    """Runs prediction on audio input and formats output HTML components."""
    if audio_path is None or not str(audio_path).strip():
        return render_hero_html(None), render_topk_html(None), render_benchmark_html(None)

    classifier = get_model()

    region_map = {
        "Pan-India (All Indian Subregions - 1,074 Species)": "pan-india",
        "South India & Sri Lanka (Peninsular - 644 Species)": "south-asia-peninsular",
        "Indo-Gangetic Plain (Northern Plains - 814 Species)": "indo-gangetic",
        "Himalaya & Mountain Ecozone (812 Species)": "himalaya",
        "Global (All 11,560 Species - Unfiltered)": "all",
    }
    target_region = region_map.get(region_choice, "pan-india")

    result = classifier.predict(
        audio_path,
        lat=lat_val,
        lon=lon_val,
        region=target_region,
        threshold=threshold_val,
        top_k=5,
    )

    return render_hero_html(result), render_topk_html(result), render_benchmark_html(result)


def record_live_mic_handler(
    duration_s: float,
    region_choice: str,
    lat_val: float,
    lon_val: float,
    threshold_val: float,
) -> Tuple[str, str, str, str]:
    """Records real-time audio directly from hardware microphone, saves WAV, and predicts."""
    duration = max(1.0, min(30.0, float(duration_s)))
    print(f"[MIC HANDLER] Recording {duration:.1f}s from microphone...")

    waveform, sr = record_audio(duration=duration, sample_rate=SAMPLE_RATE)
    saved_file = save_audio(waveform, sample_rate=sr)

    hero_html, topk_html, bench_html = predict_audio_handler(
        str(saved_file), region_choice, lat_val, lon_val, threshold_val
    )

    return str(saved_file), hero_html, topk_html, bench_html


def load_and_predict_sample(
    sample_key: str,
    region_choice: str,
    lat_val: float,
    lon_val: float,
    threshold_val: float,
) -> Tuple[Optional[str], str, str, str]:
    """Loads sample bird sound and immediately runs prediction."""
    samples_map = {
        "peafowl": "01_indian_peafowl.wav",
        "roller": "02_indian_roller.wav",
        "kingfisher": "03_common_kingfisher.wav",
        "bulbul": "04_red_vented_bulbul.wav",
        "magpie_robin": "05_oriental_magpie_robin.wav",
    }
    filename = samples_map.get(sample_key)
    if not filename or not INDIAN_SAMPLES_DIR.exists():
        return None, render_hero_html(None), render_topk_html(None), render_benchmark_html(None)

    sample_path = INDIAN_SAMPLES_DIR / filename
    if not sample_path.exists():
        return None, render_hero_html(None), render_topk_html(None), render_benchmark_html(None)

    hero, topk, bench = predict_audio_handler(str(sample_path), region_choice, lat_val, lon_val, threshold_val)
    return str(sample_path), hero, topk, bench


def create_app() -> gr.Blocks:
    """Builds the complete Gradio UI."""
    region_options = [
        "Pan-India (All Indian Subregions - 1,074 Species)",
        "South India & Sri Lanka (Peninsular - 644 Species)",
        "Indo-Gangetic Plain (Northern Plains - 814 Species)",
        "Himalaya & Mountain Ecozone (812 Species)",
        "Global (All 11,560 Species - Unfiltered)",
    ]

    with gr.Blocks(css=CUSTOM_CSS, title="AvianAI India — Bioacoustic Bird Identification") as demo:
        # Header Section
        gr.HTML(
            f"""
            <div class="app-header-container">
                <div class="brand-badge">Developed by Mrutyunjay Joshi</div>
                <div class="brand-title">AvianAI India</div>
                <div class="brand-subtitle">
                    Real-time bioacoustic neural identification system for Indian avifauna. Engineered by <strong>Mrutyunjay Joshi</strong>.
                </div>
            </div>
            """
        )

        with gr.Row():
            # Left Column: Input Sources & Controls
            with gr.Column(scale=5):
                with gr.Tabs():
                    # TAB 1: Live Hardware Microphone
                    with gr.Tab("Live Microphone"):
                        gr.HTML("<p style='font-size:0.95rem; color:#cbd5e1; margin-bottom:12px;'>Press the button below to record live audio directly from your microphone:</p>")
                        mic_duration = gr.Slider(
                            minimum=2.0,
                            maximum=15.0,
                            value=5.0,
                            step=1.0,
                            label="Recording Duration (Seconds)",
                            info="Recommended: 5.0 to 10.0 seconds",
                        )
                        record_mic_btn = gr.Button(
                            "Record from Microphone Now (5s)",
                            variant="primary",
                            elem_classes=["btn-record-mic"],
                        )
                        mic_audio_player = gr.Audio(
                            label="Recorded Microphone Audio (Playback)",
                            type="filepath",
                            interactive=False,
                        )

                    # TAB 2: Upload / Browser Audio
                    with gr.Tab("Upload Audio File"):
                        file_audio = gr.Audio(
                            sources=["upload", "microphone"],
                            type="filepath",
                            label="Upload Audio File (WAV, OGG, MP3, FLAC)",
                        )

                    # TAB 3: Indian Bird Audio Gallery
                    with gr.Tab("Indian Bird Audio Gallery"):
                        gr.HTML("<p style='font-size:0.9rem; color:#94a3b8; margin-bottom:12px;'>Click any iconic Indian bird to load audio & identify instantly:</p>")
                        with gr.Row():
                            btn_peafowl = gr.Button("Indian Peafowl\n(National Bird)", variant="secondary")
                            btn_roller = gr.Button("Indian Roller\n(State Bird)", variant="secondary")
                        with gr.Row():
                            btn_kingfisher = gr.Button("Common Kingfisher\n(Wetland Hunter)", variant="secondary")
                            btn_bulbul = gr.Button("Red-vented Bulbul\n(Garden Songbird)", variant="secondary")
                        with gr.Row():
                            btn_robin = gr.Button("Oriental Magpie-Robin\n(Melodious Songster)", variant="secondary")
                        gallery_audio_player = gr.Audio(
                            label="Loaded Sample Audio (Playback)",
                            type="filepath",
                            interactive=False,
                        )

                # Location & Filter Controls
                with gr.Accordion("Location & Geographic Species Filter", open=False):
                    region_dropdown = gr.Dropdown(
                        choices=region_options,
                        value="Pan-India (All Indian Subregions - 1,074 Species)",
                        label="Geographic Range Filter",
                        info="Restricts predictions to bird species native to the selected region",
                    )
                    with gr.Row():
                        lat_input = gr.Number(
                            value=DEFAULT_LATITUDE,
                            label="Latitude (North)",
                            info="Default: 20.59 N (India Centroid)",
                        )
                        lon_input = gr.Number(
                            value=DEFAULT_LONGITUDE,
                            label="Longitude (East)",
                            info="Default: 78.96 E (India Centroid)",
                        )

                    # Quick City Presets
                    with gr.Row():
                        btn_rajkot = gr.Button("Rajkot (Gujarat)", size="sm", variant="secondary")
                        btn_ahmedabad = gr.Button("Ahmedabad (Gujarat)", size="sm")
                        btn_delhi = gr.Button("Delhi (North)", size="sm")
                    with gr.Row():
                        btn_mum = gr.Button("Mumbai (West)", size="sm")
                        btn_blr = gr.Button("Bengaluru (South)", size="sm")
                        btn_kol = gr.Button("Kolkata (East)", size="sm")

                    thresh_slider = gr.Slider(
                        minimum=0.05,
                        maximum=0.95,
                        value=CONFIDENCE_THRESHOLD,
                        step=0.05,
                        label="Confidence Threshold",
                        info="Scores below this value display 'No confident bird detected'",
                    )

                identify_btn = gr.Button("Identify Bird Sound", variant="primary", elem_classes=["btn-identify"])

            # Right Column: Prominent Output Display & Diagnostics
            with gr.Column(scale=6):
                # Hero Output Banner
                hero_out = gr.HTML(render_hero_html(None), label="Detection Result")

                with gr.Accordion("Candidate Species Ranking", open=True):
                    topk_out = gr.HTML(render_topk_html(None))

                with gr.Accordion("Live Latency Benchmark & Diagnostics", open=True):
                    bench_out = gr.HTML(render_benchmark_html(None))

                with gr.Accordion("System Architecture & Engineering Specifications", open=False):
                    gr.HTML(
                        f"""
                        <div style="font-size: 0.88rem; color: #94a3b8; line-height: 1.8;">
                            <div><strong>Lead Developer & System Architect:</strong> <span style="color:#ffffff; font-weight:600;">{DEVELOPER_NAME}</span></div>
                            <div><strong>Acoustic Neural Model:</strong> {MODEL_NAME} ({MODEL_VERSION}) - GPU-native FP16 with in-graph Conv1d frontend</div>
                            <div><strong>Taxonomic Coverage:</strong> 11,560 classes with native Indian Subregion geographic ecozone filters</div>
                            <div><strong>Target Hardware Acceleration:</strong> NVIDIA RTX 5090 32 GB (CUDA) / Apple Silicon Mac (CoreML/Neural Engine)</div>
                            <div><strong>Audio Pipeline:</strong> 32,000 Hz Resampling -> 5.0s Multi-Window Framing -> Temporal Max-Pooling -> Geographic Masking</div>
                        </div>
                        """
                    )

        # ----------------------------------------------------------------------
        # Event Callbacks
        # ----------------------------------------------------------------------
        # Update dynamic button text when mic duration slider changes
        mic_duration.change(
            fn=lambda d: f"Record from Microphone Now ({int(d)}s)",
            inputs=[mic_duration],
            outputs=[record_mic_btn],
        )

        # Record from Mic Button Click
        record_mic_btn.click(
            fn=record_live_mic_handler,
            inputs=[mic_duration, region_dropdown, lat_input, lon_input, thresh_slider],
            outputs=[mic_audio_player, hero_out, topk_out, bench_out],
        )

        # File Upload change
        file_audio.change(
            fn=predict_audio_handler,
            inputs=[file_audio, region_dropdown, lat_input, lon_input, thresh_slider],
            outputs=[hero_out, topk_out, bench_out],
        )

        # Manual Identify Button Click
        def run_active_prediction(mic_p, file_p, gallery_p, reg, lat, lon, thresh):
            active_p = mic_p or file_p or gallery_p
            return predict_audio_handler(active_p, reg, lat, lon, thresh)

        identify_btn.click(
            fn=run_active_prediction,
            inputs=[mic_audio_player, file_audio, gallery_audio_player, region_dropdown, lat_input, lon_input, thresh_slider],
            outputs=[hero_out, topk_out, bench_out],
        )

        # Gallery Sample Loading Buttons with instant prediction
        btn_peafowl.click(
            fn=lambda reg, lat, lon, th: load_and_predict_sample("peafowl", reg, lat, lon, th),
            inputs=[region_dropdown, lat_input, lon_input, thresh_slider],
            outputs=[gallery_audio_player, hero_out, topk_out, bench_out],
        )
        btn_roller.click(
            fn=lambda reg, lat, lon, th: load_and_predict_sample("roller", reg, lat, lon, th),
            inputs=[region_dropdown, lat_input, lon_input, thresh_slider],
            outputs=[gallery_audio_player, hero_out, topk_out, bench_out],
        )
        btn_kingfisher.click(
            fn=lambda reg, lat, lon, th: load_and_predict_sample("kingfisher", reg, lat, lon, th),
            inputs=[region_dropdown, lat_input, lon_input, thresh_slider],
            outputs=[gallery_audio_player, hero_out, topk_out, bench_out],
        )
        btn_bulbul.click(
            fn=lambda reg, lat, lon, th: load_and_predict_sample("bulbul", reg, lat, lon, th),
            inputs=[region_dropdown, lat_input, lon_input, thresh_slider],
            outputs=[gallery_audio_player, hero_out, topk_out, bench_out],
        )
        btn_robin.click(
            fn=lambda reg, lat, lon, th: load_and_predict_sample("magpie_robin", reg, lat, lon, th),
            inputs=[region_dropdown, lat_input, lon_input, thresh_slider],
            outputs=[gallery_audio_player, hero_out, topk_out, bench_out],
        )

        # City Presets
        btn_rajkot.click(fn=lambda: (22.3039, 70.8022), outputs=[lat_input, lon_input])
        btn_ahmedabad.click(fn=lambda: (23.0225, 72.5714), outputs=[lat_input, lon_input])
        btn_delhi.click(fn=lambda: (28.6139, 77.2090), outputs=[lat_input, lon_input])
        btn_mum.click(fn=lambda: (19.0760, 72.8777), outputs=[lat_input, lon_input])
        btn_blr.click(fn=lambda: (12.9716, 77.5946), outputs=[lat_input, lon_input])
        btn_kol.click(fn=lambda: (22.5726, 88.3639), outputs=[lat_input, lon_input])

    return demo


# Create Gradio block app instance
app = create_app()

if __name__ == "__main__":
    try:
        app.launch(server_name="0.0.0.0", server_port=7860, share=False)
    except OSError:
        print("[INFO] Port 7860 busy, selecting next open port...")
        app.launch(server_name="0.0.0.0", share=False)