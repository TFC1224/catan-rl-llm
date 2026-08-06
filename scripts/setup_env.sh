#!/bin/bash
set -e

echo "============================================"
echo "  Catan RL + LLM Environment Setup"
echo "  Phase 1: Installing Dependencies"
echo "============================================"

# Check Python version
echo ""
echo "[1/5] Checking Python version..."
python3 --version
echo "  Python 3.10+ confirmed."

# Check GPU
echo ""
echo "[2/5] Checking GPU availability..."
python3 -c "import torch; print(f'  CUDA available: {torch.cuda.is_available()}'); print(f'  GPU: {torch.cuda.get_device_name(0)}'); print(f'  VRAM: {torch.cuda.get_device_properties(0).total_mem / 1024**3:.1f} GB')" || echo "  WARNING: Could not check GPU"

# Install core dependencies
echo ""
echo "[3/5] Installing core ML packages..."
pip install torch>=2.1.0 --quiet
pip install transformers>=4.45.0 trl>=0.12.0 peft>=0.12.0 accelerate>=0.28.0 bitsandbytes>=0.43.0 datasets>=3.0.0 --quiet

# Install game environment
echo ""
echo "[4/5] Installing Catanatron environment..."
pip install catanatron-gym>=4.0.0 gymnasium>=0.29.0 --quiet

# Install utilities
echo ""
echo "[5/5] Installing utilities and monitoring..."
pip install wandb>=0.16.0 pyyaml>=6.0 tqdm>=4.66.0 python-dotenv>=1.0.0 --quiet
pip install matplotlib>=3.7.0 seaborn>=0.12.0 jupyter>=1.0.0 ipykernel>=6.0.0 --quiet

echo ""
echo "============================================"
echo "  Installation Complete!"
echo "============================================"
echo ""
echo "Next steps:"
echo "  1. python scripts/download_model.py   (download Qwen3-8B-Instruct)"
echo "  2. python scripts/test_imports.py      (verify all imports)"
echo "  3. Open notebooks/01_env_test.ipynb    (test environment)"
