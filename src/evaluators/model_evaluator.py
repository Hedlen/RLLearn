"""Model evaluation implementation"""

import os
import json
import torch
import numpy as np
from typing import Dict, List, Any, Optional, Tuple
from pathlib import Path
from tqdm import tqdm
from transformers import AutoTokenizer, AutoModelForCausalLM

try:
    from datasets import load_dataset
    HAS_DATASETS = True
except ImportError:
    HAS_DATASETS = False
    load_dataset = None

from ..models import PolicyModel, RewardModel
from ..utils.logger import get_logger
from .metrics import (
    compute_bleu_score,
    compute_rouge_score,
    compute_perplexity,
    compute_reward_metrics
)


class ModelEvaluator:
    """Comprehensive model evaluator for RLHF models"""
    
    def __init__(self, 
                 config: Optional[Dict[str, Any]] = None,
                 tokenizer: Optional[AutoTokenizer] = None,
                 device: Optional[torch.device] = None,
                 output_dir: str = "./eval_results"):
        
        if config is not None:
            # Initialize from config (for main.py usage)
            self.config = config
            self.eval_config = config.get('evaluation', {})
            self.model_config = config.get('model', {})
            self.logger = get_logger()
            
            # Initialize tokenizer
            tokenizer_name = (
                self.model_config.get('tokenizer_name_or_path') or 
                self.model_config.get('model_name_or_path')
            )
            self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_name)
            
            # Evaluation parameters from config
            self.max_new_tokens = self.eval_config.get('max_new_tokens', 1024)
            self.temperature = self.eval_config.get('temperature', 0.7)
            self.top_p = self.eval_config.get('top_p', 0.9)
            self.do_sample = self.eval_config.get('do_sample', True)
            self.batch_size = self.eval_config.get('eval_batch_size', 8)
        else:
            # Initialize from direct parameters (for example_training.py usage)
            if tokenizer is None:
                raise ValueError("Either config or tokenizer must be provided")
            
            self.config = {}
            self.eval_config = {}
            self.model_config = {}
            self.logger = get_logger()
            self.tokenizer = tokenizer
            
            # Default evaluation parameters
            self.max_new_tokens = 1024
            self.temperature = 0.7
            self.top_p = 0.9
            self.do_sample = True
            self.batch_size = 8
        
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
    def evaluate(self) -> Dict[str, Any]:
        """Run comprehensive model evaluation
        
        Returns:
            Evaluation results dictionary
        """
        self.logger.info("Starting model evaluation")
        
        results = {}
        
        # Load model
        model_path = self.config['training']['output_dir']
        if not os.path.exists(model_path):
            raise ValueError(f"Model path does not exist: {model_path}")
        
        model = self._load_model(model_path)
        
        # Load evaluation dataset
        eval_dataset = self._load_eval_dataset()
        
        if eval_dataset is None:
            self.logger.warning("No evaluation dataset found, using default prompts")
            eval_dataset = self._get_default_prompts()
        
        # Run different evaluation metrics
        results['generation_quality'] = self._evaluate_generation_quality(model, eval_dataset)
        results['perplexity'] = self._evaluate_perplexity(model, eval_dataset)
        
        # If reward model is available, evaluate reward scores
        reward_model_path = self.config.get('rlhf', {}).get('reward_model_path')
        if reward_model_path and os.path.exists(reward_model_path):
            reward_model = self._load_reward_model(reward_model_path)
            results['reward_scores'] = self._evaluate_reward_scores(model, reward_model, eval_dataset)
        
        # Save results
        self._save_results(results)
        
        self.logger.info("Model evaluation completed")
        return results
    
    def _load_model(self, model_path: str) -> PolicyModel:
        """Load trained model"""
        self.logger.info(f"Loading model from {model_path}")
        
        try:
            # Try to load as PolicyModel first
            model = PolicyModel.from_pretrained(model_path)
        except:
            # Fallback to standard transformers model
            model = AutoModelForCausalLM.from_pretrained(
                model_path,
                torch_dtype=torch.float16 if self.config.get('hardware', {}).get('torch_dtype') == 'float16' else torch.float32,
                device_map='auto'
            )
        
        model.eval()
        return model
    
    def _load_reward_model(self, model_path: str) -> RewardModel:
        """Load reward model"""
        self.logger.info(f"Loading reward model from {model_path}")
        reward_model = RewardModel.from_pretrained(model_path)
        reward_model.eval()
        return reward_model
    
    def _load_eval_dataset(self) -> Optional[List[Dict[str, Any]]]:
        """Load evaluation dataset"""
        eval_dataset_name = self.eval_config.get('eval_dataset')
        
        if not eval_dataset_name:
            return None
        
        try:
            if os.path.exists(eval_dataset_name):
                # Load from local file
                with open(eval_dataset_name, 'r', encoding='utf-8') as f:
                    if eval_dataset_name.endswith('.json'):
                        data = json.load(f)
                    elif eval_dataset_name.endswith('.jsonl'):
                        data = [json.loads(line) for line in f]
                    else:
                        raise ValueError(f"Unsupported file format: {eval_dataset_name}")
            else:
                # Load from HuggingFace datasets
                if not HAS_DATASETS:
                    raise ImportError("datasets library is required for loading HuggingFace datasets. Please install it with: pip install datasets")
                dataset = load_dataset(eval_dataset_name, split='test')
                data = list(dataset)
            
            self.logger.info(f"Loaded {len(data)} evaluation samples")
            return data
            
        except Exception as e:
            self.logger.error(f"Failed to load evaluation dataset: {e}")
            return None
    
    def _get_default_prompts(self) -> List[Dict[str, Any]]:
        """Get default evaluation prompts"""
        return [
            {"prompt": "What is the capital of France?", "reference": "The capital of France is Paris."},
            {"prompt": "Explain the concept of machine learning.", "reference": "Machine learning is a subset of artificial intelligence that enables computers to learn and improve from experience without being explicitly programmed."},
            {"prompt": "Write a short story about a robot.", "reference": "Once upon a time, there was a friendly robot named R2 who helped humans with their daily tasks."},
            {"prompt": "How do you make a paper airplane?", "reference": "To make a paper airplane, fold a piece of paper in half lengthwise, then fold the top corners down to form wings."},
            {"prompt": "What are the benefits of renewable energy?", "reference": "Renewable energy sources like solar and wind power are sustainable, reduce greenhouse gas emissions, and help combat climate change."}
        ]
    
    def _evaluate_generation_quality(self, model, eval_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate text generation quality using BLEU and ROUGE scores"""
        self.logger.info("Evaluating generation quality")
        
        generated_texts = []
        reference_texts = []
        
        for item in tqdm(eval_dataset, desc="Generating responses"):
            prompt = item.get('prompt', item.get('input', ''))
            reference = item.get('reference', item.get('output', item.get('response', '')))
            
            if not prompt:
                continue
            
            # Generate response
            generated = self._generate_response(model, prompt)
            generated_texts.append(generated)
            reference_texts.append(reference)
        
        # Compute metrics
        results = {}
        
        if reference_texts and any(ref for ref in reference_texts):
            # Only compute reference-based metrics if references are available
            results['bleu_score'] = compute_bleu_score(generated_texts, reference_texts)
            results['rouge_scores'] = compute_rouge_score(generated_texts, reference_texts)
        
        # Compute reference-free metrics
        results['avg_length'] = np.mean([len(text.split()) for text in generated_texts])
        results['unique_tokens'] = self._compute_diversity(generated_texts)
        
        return results
    
    def _evaluate_perplexity(self, model, eval_dataset: List[Dict[str, Any]]) -> float:
        """Evaluate model perplexity"""
        self.logger.info("Evaluating perplexity")
        
        texts = []
        for item in eval_dataset:
            text = item.get('text', item.get('prompt', ''))
            if item.get('response'):
                text += " " + item['response']
            texts.append(text)
        
        if not texts:
            return float('inf')
        
        return compute_perplexity(model, texts, self.tokenizer, self.device)
    
    def _evaluate_reward_scores(self, model, reward_model, eval_dataset: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate reward scores"""
        self.logger.info("Evaluating reward scores")
        
        reward_scores = []
        
        for item in tqdm(eval_dataset, desc="Computing reward scores"):
            prompt = item.get('prompt', item.get('input', ''))
            
            if not prompt:
                continue
            
            # Generate response
            generated = self._generate_response(model, prompt)
            
            # Compute reward
            full_text = f"{prompt} {generated}"
            inputs = self.tokenizer(
                full_text,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(self.device)
            
            with torch.no_grad():
                outputs = reward_model(**inputs, return_dict=True)
                reward = outputs['rewards'].squeeze().item()
                reward_scores.append(reward)
        
        return compute_reward_metrics(reward_scores)
    
    def _generate_response(self, model, prompt: str) -> str:
        """Generate response for a given prompt"""
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            padding=True,
            truncation=True
        ).to(self.device)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_new_tokens=self.max_new_tokens,
                temperature=self.temperature,
                top_p=self.top_p,
                do_sample=self.do_sample,
                pad_token_id=self.tokenizer.pad_token_id,
                eos_token_id=self.tokenizer.eos_token_id
            )
        
        # Extract generated text (excluding prompt)
        prompt_length = inputs['input_ids'].shape[1]
        generated_tokens = outputs[0][prompt_length:]
        generated_text = self.tokenizer.decode(generated_tokens, skip_special_tokens=True)
        
        return generated_text.strip()
    
    def _compute_diversity(self, texts: List[str]) -> Dict[str, float]:
        """Compute diversity metrics for generated texts"""
        all_tokens = []
        for text in texts:
            tokens = text.lower().split()
            all_tokens.extend(tokens)
        
        if not all_tokens:
            return {'unique_ratio': 0.0, 'vocab_size': 0}
        
        unique_tokens = set(all_tokens)
        
        return {
            'unique_ratio': len(unique_tokens) / len(all_tokens),
            'vocab_size': len(unique_tokens)
        }
    
    def _save_results(self, results: Dict[str, Any]):
        """Save evaluation results"""
        output_dir = self.config['training']['output_dir']
        results_path = os.path.join(output_dir, 'evaluation_results.json')
        
        os.makedirs(output_dir, exist_ok=True)
        
        with open(results_path, 'w', encoding='utf-8') as f:
            json.dump(results, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Evaluation results saved to {results_path}")
        
        # Print summary
        self.logger.info("=== Evaluation Summary ===")
        for category, metrics in results.items():
            self.logger.info(f"{category.upper()}:")
            if isinstance(metrics, dict):
                for metric, value in metrics.items():
                    if isinstance(value, float):
                        self.logger.info(f"  {metric}: {value:.4f}")
                    else:
                        self.logger.info(f"  {metric}: {value}")
            else:
                self.logger.info(f"  {metrics}")