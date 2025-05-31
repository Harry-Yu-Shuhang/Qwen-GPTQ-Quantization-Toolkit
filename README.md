# Qwen GPTQ Quantization Toolkit

专业工具包，用于在 SLURM 集群上量化 Qwen 大语言模型（使用 GPTQ 算法）

## 功能特性
- 🚀 一键式 SLURM 作业提交
- 🐳 Apptainer 容器支持，确保环境一致性
- ⚙️ 可配置的量化参数（YAML 配置）
- 🤖 支持 Qwen 全系列模型
- 💾 自动完成验证和模型保存

## 快速开始

```bash
# 克隆仓库（包含子模块）
git clone --recurse-submodules https://github.com/yourname/Qwen-GPTQ-Quantization-Toolkit.git

# 进入目录
cd Qwen-GPTQ-Quantization-Toolkit

# 提交 SLURM 作业
sbatch slurm/quantize.sub
```

## 配置说明
编辑 `configs/qwen_32b_gptq.yaml` 文件：
```yaml
quantization:
  bits: 4        # 量化位数
  group_size: 128 # 分组大小
  v2: true        # 是否使用 GPTQ v2
  # ... 其他参数 ...
```

## 自定义模型
1. 修改配置文件中 `model_path` 指向你的模型
2. 调整 `slurm/quantize.sub` 中的资源要求
3. 如需上传到 Hugging Face Hub：
   ```bash
   apptainer exec --nv build_GPTQModel/qwen-gptq.sif \
       python scripts/push_to_hub.py \
           --model_path ./quantized_models/qwen-32b-gptq-v2 \
           --repo_name your-username/Qwen-32B-GPTQ
   ```

## 支持模型
- Qwen2.5-32B-Instruct
- Qwen2.5-72B-Instruct
- Qwen2.5-VL-32B
- Qwen2.5-Omni

## 贡献指南
欢迎提交 PR！请确保：
1. 更新子模块到最新版本
2. 添加新的配置文件到 `configs/`
3. 测试脚本在 SLURM 环境运行正常
