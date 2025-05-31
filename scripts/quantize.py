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
    parser.add_argument("--model_path", type=str, required=True)
    parser.add_argument("--output_dir", type=str, required=True)
    parser.add_argument("--config", type=str, default="configs/qwen_32b_gptq.yaml")
    args = parser.parse_args()
    
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
    
    # 加载模型
    model = GPTQModel.load(
        args.model_path,
        quant_config,
        torch_dtype=torch.bfloat16
    )
    
    # 量化
    model.quantize(
        calibration_data,
        batch_size=config['batch_size'],
        auto_gc=config['auto_gc'],
        buffered_fwd=config['buffered_fwd']
    )
    
    # 保存
    model.save(args.output_dir)
    
    # 验证文件
    with open(os.path.join(args.output_dir, "COMPLETED"), "w") as f:
        f.write("success")

if __name__ == "__main__":
    main()
