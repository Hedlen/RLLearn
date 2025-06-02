#!/usr/bin/env python3
"""Quick test script for RL Learning Framework"""

import os
import json
import torch
from pathlib import Path
from transformers import AutoTokenizer

from src.data import DataProcessor
from src.models import create_policy_model, create_reward_model
from src.evaluators import ModelEvaluator, AutomaticEvaluator
from src.utils import setup_logger


def test_data_processing():
    """Test data processing functionality"""
    print("\n=== Testing Data Processing ===")
    
    try:
        # Create sample data
        sample_data = [
            {
                "prompt": "什么是机器学习？",
                "response": "机器学习是人工智能的一个分支，使计算机能够从数据中学习。"
            }
        ]
        
        # Save test data
        test_file = "./data/test_sft.json"
        Path("./data").mkdir(exist_ok=True)
        
        with open(test_file, "w", encoding="utf-8") as f:
            json.dump(sample_data, f, ensure_ascii=False, indent=2)
        
        # Test data processor
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        processor = DataProcessor(tokenizer=tokenizer, max_length=128)
        dataset = processor.load_dataset(test_file, dataset_type="sft")
        
        print(f"✓ Data processing successful! Dataset size: {len(dataset)}")
        print(f"✓ Sample item keys: {list(dataset[0].keys())}")
        
        # Clean up
        os.remove(test_file)
        
        return True
        
    except Exception as e:
        print(f"✗ Data processing failed: {e}")
        return False


def test_model_loading():
    """Test model loading functionality"""
    print("\n=== Testing Model Loading ===")
    
    try:
        # Test policy model
        policy_model, policy_config = create_policy_model("Qwen/Qwen2.5-3B-Instruct")
        print(f"✓ Policy model loaded: {type(policy_model).__name__}")
        
        # Test reward model
        reward_model, reward_config = create_reward_model("Qwen/Qwen2.5-3B-Instruct")
        print(f"✓ Reward model loaded: {type(reward_model).__name__}")
        
        # Test tokenizer
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        print(f"✓ Tokenizer loaded: vocab size {tokenizer.vocab_size}")
        
        return True
        
    except Exception as e:
        print(f"✗ Model loading failed: {e}")
        return False


def test_evaluation():
    """Test evaluation functionality"""
    print("\n=== Testing Evaluation ===")
    
    try:
        # Setup
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        Path("./eval_results").mkdir(exist_ok=True)
        
        # Test automatic evaluator
        evaluator = AutomaticEvaluator(
            tokenizer=tokenizer,
            output_dir="./eval_results"
        )
        
        # Test generation quality evaluation
        test_predictions = [
            "机器学习是人工智能的重要分支。",
            "深度学习使用神经网络进行学习。"
        ]
        
        results = evaluator.evaluate_generation_quality(
            predictions=test_predictions
        )
        
        print(f"✓ Generation quality evaluation successful!")
        print(f"  - Average length: {results.get('avg_length', 0):.2f}")
        print(f"  - Distinct-2: {results.get('distinct_2', 0):.4f}")
        
        # Test model evaluator
        model_evaluator = ModelEvaluator(
            model_name_or_path="Qwen/Qwen2.5-3B-Instruct",
            tokenizer=tokenizer
        )
        
        print(f"✓ Model evaluator initialized successfully!")
        
        return True
        
    except Exception as e:
        print(f"✗ Evaluation failed: {e}")
        return False


def test_generation():
    """Test text generation"""
    print("\n=== Testing Text Generation ===")
    
    try:
        # Load model and tokenizer
        model, _ = create_policy_model("Qwen/Qwen2.5-3B-Instruct")
        tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-3B-Instruct")
        if tokenizer.pad_token is None:
            tokenizer.pad_token = tokenizer.eos_token
        
        # Test generation
        test_prompt = "什么是人工智能？"
        inputs = tokenizer(test_prompt, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=50,
                temperature=0.7,
                do_sample=True,
                pad_token_id=tokenizer.eos_token_id
            )
        
        generated_text = tokenizer.decode(outputs[0], skip_special_tokens=True)
        response = generated_text[len(test_prompt):].strip()
        
        print(f"✓ Text generation successful!")
        print(f"  Prompt: {test_prompt}")
        print(f"  Response: {response[:100]}...")
        
        return True
        
    except Exception as e:
        print(f"✗ Text generation failed: {e}")
        return False


def test_config_loading():
    """Test configuration loading"""
    print("\n=== Testing Configuration Loading ===")
    
    try:
        from src.utils import load_config
        
        # Test loading config.yaml
        if os.path.exists("config.yaml"):
            config = load_config("config.yaml")
            print(f"✓ Config loaded successfully!")
            print(f"  Model: {config.get('model', {}).get('model_name', 'Unknown')}")
            print(f"  Training steps: {config.get('training', {}).get('max_steps', 'Unknown')}")
        else:
            print("⚠ config.yaml not found, but config loading function works")
        
        return True
        
    except Exception as e:
        print(f"✗ Config loading failed: {e}")
        return False


def test_directory_structure():
    """Test directory structure"""
    print("\n=== Testing Directory Structure ===")
    
    required_dirs = [
        "src", "src/algorithms", "src/data", "src/evaluators", 
        "src/models", "src/trainers", "src/utils"
    ]
    
    required_files = [
        "src/__init__.py", "src/data/__init__.py", "src/evaluators/__init__.py",
        "src/models/__init__.py", "src/trainers/__init__.py", "src/utils/__init__.py",
        "main.py", "config.yaml", "requirements.txt"
    ]
    
    missing_dirs = []
    missing_files = []
    
    for dir_path in required_dirs:
        if not os.path.isdir(dir_path):
            missing_dirs.append(dir_path)
    
    for file_path in required_files:
        if not os.path.isfile(file_path):
            missing_files.append(file_path)
    
    if not missing_dirs and not missing_files:
        print("✓ All required directories and files present!")
        return True
    else:
        if missing_dirs:
            print(f"✗ Missing directories: {missing_dirs}")
        if missing_files:
            print(f"✗ Missing files: {missing_files}")
        return False


def run_all_tests():
    """Run all tests"""
    print("=== RL Learning Framework Quick Test ===")
    
    tests = [
        ("Directory Structure", test_directory_structure),
        ("Configuration Loading", test_config_loading),
        ("Data Processing", test_data_processing),
        ("Model Loading", test_model_loading),
        ("Text Generation", test_generation),
        ("Evaluation", test_evaluation)
    ]
    
    results = []
    
    for test_name, test_func in tests:
        try:
            result = test_func()
            results.append((test_name, result))
        except Exception as e:
            print(f"\n✗ {test_name} crashed: {e}")
            results.append((test_name, False))
    
    # Summary
    print("\n" + "="*50)
    print("TEST SUMMARY")
    print("="*50)
    
    passed = 0
    total = len(results)
    
    for test_name, result in results:
        status = "✓ PASS" if result else "✗ FAIL"
        print(f"{test_name:<25} {status}")
        if result:
            passed += 1
    
    print(f"\nOverall: {passed}/{total} tests passed")
    
    if passed == total:
        print("🎉 All tests passed! The framework is ready to use.")
    else:
        print("⚠ Some tests failed. Please check the errors above.")
    
    return passed == total


if __name__ == "__main__":
    # Setup basic logging
    import logging
    logging.basicConfig(level=logging.WARNING)
    
    # Run tests
    success = run_all_tests()
    
    if success:
        print("\n🚀 You can now run the full training example with:")
        print("   python example_training.py --model_name Qwen/Qwen2.5-3B-Instruct --max_steps 50")
    else:
        print("\n🔧 Please fix the issues above before running the full training.")