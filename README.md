# AvianAI India - Bioacoustic Bird Identification System

**Developed & Engineered by Mrutyunjay Joshi**

A high-performance, real-time bioacoustic neural identification system engineered for Indian avifauna, powered by the GPU-native **AvianNet-v3** FP16 ONNX acoustic classifier with dynamic geographic species-range filtering for the Indian Subcontinent.

---

## Overview & Architecture

```
[ Microphone (~5-10s) ]  OR  [ Audio Upload (WAV/OGG/MP3) ]
                       |
                       v
         [ 32 kHz Resampling & 5s Windowing ]
                       |
                       v
    [ AvianNet-v3 GPU-Native FP16 ONNX Inference ]
                       |
                       v
        [ India Geographic Species Filtering ]
   (Pan-India, South India, Indo-Gangetic, Himalaya)
                       |
                       v
            [ Clean Gradio Interface ]
            +------------------------+
            |     DETECTED BIRD      |
            |     Indian Peafowl     |
            |       CONFIDENCE       |
            |         97.1%          |
            +------------------------+
```

---

## Key Technical Features

1. **Acoustic Neural Architecture (AvianNet-v3)**:
   - **Model Checkpoint**: `models/aviannet-v3-gpu-fp16.onnx` (278 MB).
   - **Backbone**: EfficientNetV2-S with native in-graph `Conv1d` framing & windowed DFT log-mel frontend.
   - **Taxonomic Resolution**: 11,560 classes globally with dedicated Indian regional presence masks.
2. **Indian Geographic Ecozone Filtering**:
   - Dynamic binary masking built on regional bounding boxes and species distribution matrices:
     - `pan-india`: Complete Indian subcontinent species mask (1,074 species).
     - `south-asia-peninsular`: South India & Sri Lanka (644 species).
     - `indo-gangetic`: Indo-Gangetic Plain (814 species).
     - `himalaya`: Himalayan mountain ecozone (812 species).
   - Configurable coordinates (defaults to India centroid `20.59 N, 78.96 E`).
3. **Cross-Platform Hardware Acceleration**:
   - **Target Production Hardware**: NVIDIA RTX 5090 32 GB (`CUDAExecutionProvider`).
   - **Development Hardware**: Apple Silicon Mac (`CoreMLExecutionProvider` / `CPUExecutionProvider`).
   - Automatic execution provider hierarchy (`CUDA -> CoreML -> CPU`).
4. **Live Latency Diagnostics & Benchmarks**:
   - Real-time measurement of Preprocessing time, ONNX inference time, Total latency, and Device.
5. **Confidence & Rejection Thresholding**:
   - Configurable decision threshold (default 0.25). Unreliable sounds display `"No confident bird detected"`.

---

## Quick Start

### 1. Installation

```bash
# Clone or navigate to the repository
cd ~/Desktop/Bird

# Create & activate virtual environment (Python 3.10+)
python3 -m venv .venv
source .venv/bin/activate

# Install dependencies
pip install -r requirements.txt
```

### 2. Verify System & Model Integrity

Run `verify_model.py` to verify SHA256 integrity, test input/output shapes, check all 11,560 class labels, and run a benchmark pass:

```bash
python verify_model.py
```

### 3. Launch Web Application (Gradio)

Launch the interactive web UI:

```bash
python app.py
```
Open [http://localhost:7860](http://localhost:7860) in your web browser.

### 4. CLI Predictions

Run inference directly from the terminal:

```bash
# Predict from audio file
python predict.py --audio samples/indian_birds/01_indian_peafowl.wav

# Predict with custom region or coordinates
python predict.py --audio samples/indian_birds/02_indian_roller.wav --region south-asia-peninsular

# Record 5 seconds live from microphone
python predict.py --mic --duration 5.0

# Output as JSON
python predict.py --audio samples/indian_birds/01_indian_peafowl.wav --json
```

---

## Repository Structure

```
+-- app.py                     # Gradio web application UI
+-- verify_model.py            # Model integrity and inference verification script
+-- predict.py                 # CLI inference tool (File and Mic)
+-- benchmark.py               # Performance benchmarking utility
+-- run_all_tests.py           # Unified test runner
+-- package_for_ssd.py         # Bare minimum SSD bundle packager
+-- requirements_rtx5090.txt   # Linux CUDA dependencies for RTX 5090
+-- run_linux_rtx5090.sh       # 1-Click Launch script for Linux
+-- requirements.txt           # Python dependencies
+-- README.md                  # Project documentation
+-- SPECIES_CATALOG.md         # Recognized Indian species catalog
+-- .gitignore                 # Git ignore patterns
+-- audio/
|   +-- preprocess.py          # 32 kHz resampling, 5s windowing, padding
|   +-- microphone.py          # SoundDevice live mic capture
+-- config/
|   +-- settings.py            # Hardware provider selection, audio parameters, paths
|   +-- labels.py              # 11,560 labels parser & resolver
|   +-- geo.py                 # India geographic species filtering & subregions
|   +-- regions.json           # Geographic bounding boxes
|   +-- *-indices-b1.txt       # Indian subregion species indices
+-- inference/
|   +-- model.py               # ONNX Runtime inference engine & benchmarking
+-- models/
|   +-- aviannet-v3-*.onnx     # AvianNet-v3 FP16 ONNX model (278 MB)
|   +-- aviannet-v3-*.txt      # 11,560 classes label catalog
+-- samples/
|   +-- indian_birds/          # Test audio samples of iconic Indian species
|   +-- phone_transfer/        # Sample audio files for mobile testing
+-- tests/
    +-- test_cli.py            # CLI and JSON output tests
    +-- test_indian_birds_samples.py # Real Indian bird sample tests
    +-- test_preprocess.py     # Preprocessing and tensor shape tests
    +-- test_geo.py            # Geographic filtering & masking tests
    +-- test_labels.py         # Label parsing & species lookup tests
    +-- test_model.py          # ONNX inference & dummy pass tests
```

---

## Running Automated Tests

```bash
python run_all_tests.py
# or
pytest tests/ -v
```

---

## Author & Engineering Credits

- **Developer & Lead Engineer**: **Mrutyunjay Joshi**
- **Project**: **AvianAI India - Bioacoustic Neural Intelligence**
