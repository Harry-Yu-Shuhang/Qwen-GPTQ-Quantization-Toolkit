#!/bin/bash
# 带错误处理和详细日志的安装脚本
set -eEuo pipefail
trap 'echo "❌ 安装失败: 行号 $LINENO，命令: $BASH_COMMAND" >&2; exit 1' ERR

echo "=== 设置 Conda 环境 ==="
export PATH=/opt/conda/bin:$PATH
conda create -y -n qwen-compression python=3.10
conda activate qwen-compression

# 1. 安装 PyTorch（匹配 CUDA 11.8）
echo "=== 安装 PyTorch 2.3.0 ==="
pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu118

# 2. 安装常规依赖
echo "=== 安装基础依赖 ==="
pip install --no-cache-dir \
    transformers==4.40.0 \
    datasets==2.18.0 \
    huggingface-hub==0.22.2 \
    pyyaml==6.0 \
    tqdm==4.66

# 3. 安装 GPTQModel
echo "=== 安装 GPTQModel ==="
cd /opt/GPTQModel

# 检查子模块
if [ ! -d "gptqmodel" ] || [ -z "$(ls -A gptqmodel)" ]; then
    echo "⚠️ 子模块为空，尝试初始化"
    git submodule update --init --recursive || {
        echo "❌ 子模块初始化失败"; exit 1;
    }
fi

pip install --no-build-isolation -e .[vllm,ipex,auto_round]

# 4. 验证
echo "=== 验证安装 ==="
python -c "
import torch
from gptqmodel import GPTQModel, QuantizeConfig
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ CUDA available: {torch.cuda.is_available()}')
model = GPTQModel.load('gpt2', QuantizeConfig(bits=4, group_size=128))
print('✅ 模型加载成功')
"