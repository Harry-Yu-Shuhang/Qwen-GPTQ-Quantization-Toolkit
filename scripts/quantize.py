import os
import yaml
import argparse
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
import torch

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_id", type=str, required=True, 
                       help="Hugging Face 模型 ID (如 'Qwen/Qwen2.5-32B-Instruct')")
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/qwen_32b_gptq.yaml")
    parser.add_argument("--cache_dir", type=str, default="/tmp/hf_cache",
                       help="模型缓存目录")
    args = parser.parse_args()
    
    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置
    config = load_config(args.config)
    quant_config = QuantizeConfig(**config['quantization'])
    
    # 加载校准数据
    dataset_cfg = config['calibration_dataset']
    calibration_data = load_dataset(
        dataset_cfg['name'],
        dataset_cfg.get('config'),
        split=dataset_cfg['split'],
        cache_dir=os.getenv("HF_HOME", "/tmp/hf_cache")
    )[dataset_cfg['text_column']][:dataset_cfg['samples']]
    
    # 加载模型 - 直接使用远程 ID
    print(f"🚀 从 Hugging Face 加载模型: {args.model_id}")
    model = GPTQModel.load(
        args.model_id,
        quant_config,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir  # 指定缓存位置
    )
    
    # 量化
    model.quantize(
        calibration_data,
        batch_size=config['batch_size'],
        auto_gc=config['auto_gc'],
        buffered_fwd=config['buffered_fwd']
    )
    
    # 保存量化模型
    model.save(args.output_dir)
    
    print(f"✅ 量化模型已保存到: {args.output_dir}")

if __name__ == "__main__":
    main()
