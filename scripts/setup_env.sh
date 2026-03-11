#!/bin/bash
set -e

echo "═" * 60
echo "  Setting up Environment for unsloth-candle"
echo "═" * 60

# 1. Install System Dependencies (Cmake)
if ! command -v cmake &> /dev/null; then
    echo "  ! cmake not found. Attempting to install..."
    sudo apt-get update && sudo apt-get install -y cmake
else
    echo "  ✓ cmake is already installed."
fi

# 2. Setup UV Environment
echo "  ── STEP 1: Setting up Python environment with uv ──"
uv venv
source .venv/bin/activate

# 3. Install Python Dependencies
echo "  ── STEP 2: Installing dependencies ──"
uv pip install maturin huggingface_hub transformers torch
uv pip install llama-cpp-python

# 4. Build Rust Extension
echo "  ── STEP 3: Building unsloth-candle ──"
maturin develop

echo "  ✓ Setup complete!"
