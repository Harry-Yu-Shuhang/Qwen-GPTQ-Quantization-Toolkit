import argparse
import os
import yaml
import time
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
import torch

def load_config(config_path):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    # 解析命令行参数
    parser = argparse.ArgumentParser(description='Qwen GPTQ 量化工具')
    parser.add_argument('--model_id', type=str, required=True,
                        help='Hugging Face 模型 ID (如 Qwen/Qwen2.5-32B-Instruct)')
    parser.add_argument('--output_dir', type=str, required=True,
                        help='量化模型输出目录')
    parser.add_argument('--config', type=str, default='configs/qwen_32b_gptq.yaml',
                        help='量化配置文件路径')
    parser.add_argument('--cache_dir', type=str, default='/tmp/hf_cache',
                        help='Hugging Face 缓存目录')
    args = parser.parse_args()

    # 创建输出目录
    os.makedirs(args.output_dir, exist_ok=True)
    
    # 加载配置文件
    print(f"⚙️ 加载配置文件: {args.config}")
    config = load_config(args.config)
    quant_config = QuantizeConfig(**config['quantization'])
    
    # 加载校准数据集
    print(f"📊 加载校准数据集: {config['calibration_dataset']['name']}")
    dataset_cfg = config['calibration_dataset']
    calibration_data = load_dataset(
        dataset_cfg['name'],
        dataset_cfg.get('config'),
        split=dataset_cfg['split'],
        cache_dir=args.cache_dir
    )[dataset_cfg['text_column']][:dataset_cfg['samples']]
    
    # 加载模型
    print(f"🚀 从 Hugging Face 加载模型: {args.model_id}")
    print(f"💾 使用缓存目录: {args.cache_dir}")
    start_time = time.time()
    
    model = GPTQModel.load(
        args.model_id,
        quant_config,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir
    )
    
    load_time = time.time() - start_time
    print(f"✅ 模型加载完成! 耗时: {load_time:.2f}秒")
    
    # 量化配置
    quant_params = {
        'calibration_data': calibration_data,
        'batch_size': config['batch_size'],
        'auto_gc': config['auto_gc'],
        'buffered_fwd': config['buffered_fwd']
    }
    
    # 执行量化
    print(f"🔧 开始量化 (参数: {quant_params})")
    start_quant_time = time.time()
    model.quantize(**quant_params)
    quant_time = time.time() - start_quant_time
    print(f"🎉 量化完成! 总耗时: {quant_time:.2f}秒")
    
    # 保存量化模型
    print(f"💾 保存量化模型到: {args.output_dir}")
    model.save(args.output_dir)
    
    # 创建完成标记
    with open(os.path.join(args.output_dir, "COMPLETED"), "w") as f:
        f.write(f"success|model:{args.model_id}|time:{time.time()}")
    
    print("✅ 所有操作完成!")

if __name__ == "__main__":
    main()