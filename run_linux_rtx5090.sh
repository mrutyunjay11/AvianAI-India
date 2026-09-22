#!/usr/bin/env bash
# ==============================================================================
# AvianAI India - Launch Script for Linux (NVIDIA RTX 5090 / CUDA)
# Developed & Engineered by Mrutyunjay Joshi
# ==============================================================================

set -e

echo "=================================================================="
echo " AvianAI India - Linux RTX 5090 CUDA Setup & Runner"
echo " Developed by Mrutyunjay Joshi"
echo "=================================================================="

# Check Python 3
if ! command -v python3 &> /dev/null; then
    echo "[ERROR] python3 not found. Please install Python 3.10+ (sudo apt install python3 python3-venv python3-pip)"
    exit 1
fi

# 1. Virtual Environment Setup
if [ ! -d ".venv" ]; then
    echo "[1/4] Creating Python virtual environment (.venv)..."
    python3 -m venv .venv
fi

echo "[2/4] Activating virtual environment..."
source .venv/bin/activate

# 2. Dependencies Installation
echo "[3/4] Installing CUDA-accelerated dependencies..."
pip install --upgrade pip
pip install -r requirements_rtx5090.txt

# Check for CUDA GPU via nvidia-smi if available
if command -v nvidia-smi &> /dev/null; then
    echo "------------------------------------------------------------------"
    echo "Detected GPU Hardware:"
    nvidia-smi --query-gpu=name,memory.total,driver_version --format=csv,noheader
    echo "------------------------------------------------------------------"
fi

# 3. Model & Hardware Verification
echo "[4/4] Verifying ONNX Runtime GPU / CUDA acceleration..."
python verify_model.py

# 4. Launch Application
echo ""
echo "=================================================================="
echo " Launching AvianAI India Web Application on http://0.0.0.0:7860"
echo "=================================================================="
python app.py
