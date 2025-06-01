import argparse
import os
import yaml
import time
from gptqmodel import GPTQModel, QuantizeConfig
from datasets import load_dataset
import torch
from torch.utils.data import DataLoader

def load_config(config_path):
    with open(config_path, 'r') as f:
        return yaml.safe_load(f)

def main():
    parser = argparse.ArgumentParser(description='Qwen GPTQ 量化工具')
    parser.add_argument('--model_id', type=str, required=True)
    parser.add_argument('--output_dir', type=str, required=True)
    parser.add_argument('--config', type=str, default='configs/qwen_32b_gptq.yaml')
    parser.add_argument('--cache_dir', type=str, default='/tmp/hf_cache')
    args = parser.parse_args()

    os.makedirs(args.output_dir, exist_ok=True)

    print(f"⚙️ 加载配置文件: {args.config}")
    config = load_config(args.config)
    quant_config = QuantizeConfig(**config['quantization'])

    # ✅ 加载模型
    print(f"🚀 加载模型: {args.model_id}")
    model = GPTQModel.load(
        args.model_id,
        quant_config,
        torch_dtype=torch.bfloat16,
        cache_dir=args.cache_dir
    )

    # ✅ 加载校准数据集
    dataset_cfg = config.get("calibration_dataset", {})
    if dataset_cfg:
        print(f"📊 加载校准数据集: {dataset_cfg['name']}")
        dataset = load_dataset(
            dataset_cfg['name'],
            dataset_cfg.get('config'),
            split=dataset_cfg['split'],
            cache_dir=args.cache_dir
        )
        texts = dataset[dataset_cfg['text_column']][:dataset_cfg['samples']]
        dataloader = DataLoader(texts, batch_size=config['batch_size'])
        model.collect_calibration_data(dataloader)
    else:
        print("⚠️ 未提供 calibration_dataset 字段，将跳过 collect_calibration_data() 步骤。")

    # ✅ 量化
    print("🔧 开始量化...")
    model.quantize(
        auto_gc=config['auto_gc'],
        buffered_fwd=config['buffered_fwd']
    )

    # ✅ 保存
    print(f"💾 保存量化模型到: {args.output_dir}")
    model.save(args.output_dir)

    with open(os.path.join(args.output_dir, "COMPLETED"), "w") as f:
        f.write(f"success|model:{args.model_id}|time:{time.time()}")

    print("✅ 所有操作完成!")

if __name__ == "__main__":
    main()
