#!/usr/bin/env python3
"""
GPU detection and PyTorch version selector for RTX 4060 and RTX 5090.
RTX 5090 requires CUDA 12.8+ (PyTorch 2.8+)
RTX 4060 works with older CUDA versions
"""

import subprocess
import sys

def get_gpu_info():
    """Get GPU information using nvidia-smi"""
    try:
        result = subprocess.run(
            ['nvidia-smi', '--query-gpu=name', '--format=csv,noheader'],
            capture_output=True,
            text=True,
            check=True
        )
        gpu_names = [line.strip() for line in result.stdout.strip().split('\n') if line.strip()]
        return gpu_names
    except (subprocess.CalledProcessError, FileNotFoundError):
        return []

def requires_cuda_128(gpu_names):
    """Check if any GPU requires CUDA 12.8+"""
    rtx_50_series = ['RTX 5090', 'RTX 5080', 'RTX 5070']
    for gpu in gpu_names:
        for rtx_50 in rtx_50_series:
            if rtx_50.lower() in gpu.lower():
                return True
    return False

def install_pytorch():
    """Install appropriate PyTorch version based on GPU"""
    gpu_names = get_gpu_info()

    if not gpu_names:
        print("Warning: Could not detect GPU. Installing latest PyTorch version.")
        print("If you have an RTX 5090, this should work correctly.")
        cmd = ['uv', 'pip', 'install', 'torch>=2.8.0', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu128']
    elif requires_cuda_128(gpu_names):
        print(f"Detected GPU(s) requiring CUDA 12.8+: {', '.join(gpu_names)}")
        print("Installing PyTorch 2.8+ with CUDA 12.8...")
        cmd = ['uv', 'pip', 'install', 'torch>=2.8.0', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu128']
    else:
        print(f"Detected GPU(s): {', '.join(gpu_names)}")
        print("Installing PyTorch with CUDA 12.4...")
        cmd = ['uv', 'pip', 'install', 'torch>=2.5.0', 'torchvision', 'torchaudio', '--index-url', 'https://download.pytorch.org/whl/cu124']

    print(f"Running: {' '.join(cmd)}")
    subprocess.run(cmd, check=True)
    print("PyTorch installation completed successfully!")

if __name__ == '__main__':
    install_pytorch()
