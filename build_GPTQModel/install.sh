#!/bin/bash
set -e

source /opt/conda/etc/profile.d/conda.sh
conda activate qwen-gptq

# 安装 PyTorch
pip install torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 安装基础依赖
pip install -U pip setuptools wheel
pip install transformers datasets huggingface-hub pyyaml

# 安装 GPTQModel
cd /opt/GPTQModel
pip install --no-build-isolation -e .[vllm,ipex,auto_round]
