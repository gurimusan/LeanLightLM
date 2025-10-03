#!/bin/bash
# Installation script for LightLM with automatic GPU detection

set -e

echo "Installing LightLM with GPU-specific PyTorch version..."

# Install base dependencies with default PyTorch
uv sync

# Run the GPU detection and PyTorch installation script (will reinstall with correct CUDA version)
uv run python setup_pytorch.py

echo ""
echo "Installation completed!"
echo ""
echo "To verify your installation, run:"
echo "  uv run python -c 'import torch; print(f\"PyTorch: {torch.__version__}\"); print(f\"CUDA: {torch.version.cuda}\"); print(f\"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else \"No GPU\"}\")')"
