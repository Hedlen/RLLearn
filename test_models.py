# test_models.py
import json
import argparse
from pathlib import Path
from src.evaluators import ModelEvaluator
from transformers import AutoTokenizer

def run_evaluation(model_path, test_data_path, output_dir):
    """运行模型评估"""
    
    # 加载测试数据
    with open(test_data_path, 'r', encoding='utf-8') as f:
        test_data = json.load(f)
    
    # 创建评估器
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
    evaluator = ModelEvaluator(tokenizer=tokenizer)
    
    # 配置评估
    config = {
        'policy_model_path': model_path,
        'eval_dataset_path': test_data_path,
        'batch_size': 8,
        'max_samples': len(test_data)
    }
    
    # 运行评估
    results = evaluator.evaluate(config)
    
    # 保存结果
    output_path = Path(output_dir) / f"eval_results_{Path(model_path).name}.json"
    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    
    print(f"评估完成，结果保存到: {output_path}")
    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--model_path", required=True, help="模型路径")
    parser.add_argument("--test_data", required=True, help="测试数据路径")
    parser.add_argument("--output_dir", default="./eval_results", help="输出目录")
    
    args = parser.parse_args()
    
    results = run_evaluation(args.model_path, args.test_data, args.output_dir)
    
    # 打印关键指标
    print("\n=== 评估结果 ===")
    print(f"BLEU-4: {results.get('bleu_4', 0):.4f}")
    print(f"ROUGE-L: {results.get('rouge_l', 0):.4f}")
    print(f"平均奖励: {results.get('mean_reward', 0):.4f}")
    print(f"生成多样性: {results.get('distinct_2', 0):.4f}")