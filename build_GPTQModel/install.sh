#!/bin/bash
# 带错误处理和详细日志的安装脚本
set -eEuo pipefail
trap 'echo "!!!!! 安装失败: 行号 $LINENO，命令: $BASH_COMMAND" >&2; exit 1' ERR

# 1. 安装 PyTorch
echo "=== 安装 PyTorch ===" >&2
pip install --no-cache-dir torch==2.3.0 torchvision==0.18.0 torchaudio==2.3.0 \
    --index-url https://download.pytorch.org/whl/cu121

# 2. 安装基础依赖
echo "=== 安装基础依赖 ===" >&2
pip install --no-cache-dir \
    transformers==4.40 \
    datasets==2.18 \
    huggingface-hub==0.22 \
    pyyaml==6.0 \
    tqdm==4.66

# 3. 安装 GPTQModel
echo "=== 安装 GPTQModel ===" >&2
cd /opt/GPTQModel

# 检查子模块状态
if [ ! -d "gptqmodel" ] || [ -z "$(ls -A gptqmodel)" ]; then
    echo "警告: GPTQModel 目录为空，尝试初始化子模块" >&2
    git submodule update --init --recursive || {
        echo "错误: 子模块初始化失败" >&2
        exit 1
    }
fi

# 安装主包
pip install --no-build-isolation -e .[vllm,ipex,auto_round]

# 4. 验证安装
echo "=== 验证安装 ===" >&2
python -c "
import torch
from gptqmodel import GPTQModel, QuantizeConfig

print(f'PyTorch: {torch.__version__}')
print(f'CUDA available: {torch.cuda.is_available()}')

# 测试小模型加载
try:
    model = GPTQModel.load('gpt2', QuantizeConfig(bits=4, group_size=128))
    print('模型加载测试: 成功')
except Exception as e:
    print(f'模型加载测试失败: {str(e)}')
    raise
"

echo "=== 安装完成 ===" >&2