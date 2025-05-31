#!/bin/bash
set -euo pipefail

source /opt/conda/etc/profile.d/conda.sh
conda create -n qwen-gptq python=3.10 -y
conda activate qwen-gptq

# 安装依赖
pip install torch==2.3.0
pip install transformers datasets pyyaml

# 安装 GPTQModel（假设是纯 Python）
pip install /opt/GPTQModel
