"""Automatic evaluation for RLHF models"""

import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
import logging
from pathlib import Path
import json
from datetime import datetime

from .metrics import (
    compute_all_metrics,
    compute_perplexity,
    compute_preference_accuracy
)


class AutomaticEvaluator:
    """Automatic evaluator for RLHF models"""
    
    def __init__(self, 
                 tokenizer,
                 device: torch.device = None,
                 output_dir: str = "./eval_results"):
        """
        Initialize automatic evaluator
        
        Args:
            tokenizer: Tokenizer for text processing
            device: Device to run evaluation on
            output_dir: Directory to save evaluation results
        """
        self.tokenizer = tokenizer
        self.device = device or torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
    
    def evaluate_generation_quality(self,
                                  predictions: List[str],
                                  references: List[str] = None,
                                  prompts: List[str] = None) -> Dict[str, Any]:
        """
        Evaluate generation quality using automatic metrics
        
        Args:
            predictions: Generated texts
            references: Reference texts (optional)
            prompts: Input prompts (optional)
            
        Returns:
            Dictionary with evaluation metrics
        """
        self.logger.info(f"Evaluating generation quality for {len(predictions)} samples")
        
        # Compute all available metrics
        metrics = compute_all_metrics(
            predictions=predictions,
            references=references
        )
        
        # Add prompt-specific metrics if prompts are provided
        if prompts:
            prompt_metrics = self._compute_prompt_metrics(prompts, predictions)
            metrics.update(prompt_metrics)
        
        return metrics
    
    def evaluate_model_perplexity(self,
                                model,
                                texts: List[str],
                                batch_size: int = 8) -> Dict[str, float]:
        """
        Evaluate model perplexity on given texts
        
        Args:
            model: Language model to evaluate
            texts: Texts to compute perplexity on
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with perplexity metrics
        """
        self.logger.info(f"Computing perplexity for {len(texts)} texts")
        
        perplexity = compute_perplexity(
            model=model,
            texts=texts,
            tokenizer=self.tokenizer,
            device=self.device,
            batch_size=batch_size
        )
        
        return {'perplexity': perplexity}
    
    def evaluate_reward_model(self,
                            reward_model,
                            chosen_texts: List[str],
                            rejected_texts: List[str],
                            batch_size: int = 8) -> Dict[str, Any]:
        """
        Evaluate reward model performance
        
        Args:
            reward_model: Reward model to evaluate
            chosen_texts: Preferred texts
            rejected_texts: Non-preferred texts
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with reward model metrics
        """
        self.logger.info(f"Evaluating reward model on {len(chosen_texts)} pairs")
        
        # Get reward scores
        chosen_rewards = self._get_reward_scores(reward_model, chosen_texts, batch_size)
        rejected_rewards = self._get_reward_scores(reward_model, rejected_texts, batch_size)
        
        # Compute preference accuracy
        preference_metrics = compute_preference_accuracy(chosen_rewards, rejected_rewards)
        
        # Compute reward statistics
        all_rewards = chosen_rewards + rejected_rewards
        from .metrics import compute_reward_metrics
        reward_stats = compute_reward_metrics(all_rewards)
        
        return {
            **preference_metrics,
            **reward_stats,
            'chosen_reward_mean': np.mean(chosen_rewards),
            'rejected_reward_mean': np.mean(rejected_rewards)
        }
    
    def evaluate_policy_model(self,
                            policy_model,
                            reward_model,
                            prompts: List[str],
                            references: List[str] = None,
                            generation_kwargs: Dict[str, Any] = None,
                            batch_size: int = 8) -> Dict[str, Any]:
        """
        Comprehensive evaluation of policy model
        
        Args:
            policy_model: Policy model to evaluate
            reward_model: Reward model for scoring
            prompts: Input prompts
            references: Reference texts (optional)
            generation_kwargs: Generation parameters
            batch_size: Batch size for evaluation
            
        Returns:
            Dictionary with comprehensive metrics
        """
        self.logger.info(f"Evaluating policy model on {len(prompts)} prompts")
        
        # Default generation parameters
        if generation_kwargs is None:
            generation_kwargs = {
                'max_new_tokens': 128,
                'do_sample': True,
                'temperature': 0.7,
                'top_p': 0.9,
                'pad_token_id': self.tokenizer.eos_token_id
            }
        
        # Generate responses
        predictions = self._generate_responses(
            policy_model, prompts, generation_kwargs, batch_size
        )
        
        # Evaluate generation quality
        quality_metrics = self.evaluate_generation_quality(
            predictions=predictions,
            references=references,
            prompts=prompts
        )
        
        # Evaluate with reward model
        reward_scores = self._get_reward_scores(reward_model, predictions, batch_size)
        from .metrics import compute_reward_metrics
        reward_metrics = compute_reward_metrics(reward_scores)
        
        # Evaluate perplexity
        perplexity_metrics = self.evaluate_model_perplexity(
            policy_model, predictions, batch_size
        )
        
        return {
            **quality_metrics,
            **reward_metrics,
            **perplexity_metrics,
            'num_samples': len(predictions)
        }
    
    def _generate_responses(self,
                          model,
                          prompts: List[str],
                          generation_kwargs: Dict[str, Any],
                          batch_size: int) -> List[str]:
        """
        Generate responses using the model
        
        Args:
            model: Model to generate with
            prompts: Input prompts
            generation_kwargs: Generation parameters
            batch_size: Batch size
            
        Returns:
            List of generated responses
        """
        model.eval()
        responses = []
        
        with torch.no_grad():
            for i in range(0, len(prompts), batch_size):
                batch_prompts = prompts[i:i + batch_size]
                
                # Tokenize prompts
                inputs = self.tokenizer(
                    batch_prompts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                # Generate responses
                outputs = model.generate(
                    **inputs,
                    **generation_kwargs
                )
                
                # Decode responses
                for j, output in enumerate(outputs):
                    # Remove prompt tokens
                    prompt_length = inputs['input_ids'][j].shape[0]
                    response_tokens = output[prompt_length:]
                    
                    response = self.tokenizer.decode(
                        response_tokens,
                        skip_special_tokens=True
                    )
                    responses.append(response.strip())
        
        return responses
    
    def _get_reward_scores(self,
                          reward_model,
                          texts: List[str],
                          batch_size: int) -> List[float]:
        """
        Get reward scores for texts
        
        Args:
            reward_model: Reward model
            texts: Texts to score
            batch_size: Batch size
            
        Returns:
            List of reward scores
        """
        reward_model.eval()
        scores = []
        
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch_texts = texts[i:i + batch_size]
                
                # Tokenize texts
                inputs = self.tokenizer(
                    batch_texts,
                    return_tensors="pt",
                    padding=True,
                    truncation=True,
                    max_length=2048
                ).to(self.device)
                
                # Get reward scores
                outputs = reward_model(**inputs)
                
                # Extract scores (assuming reward model outputs logits)
                if hasattr(outputs, 'logits'):
                    batch_scores = outputs.logits.squeeze(-1).cpu().numpy()
                else:
                    batch_scores = outputs.cpu().numpy()
                
                if batch_scores.ndim == 0:
                    batch_scores = [float(batch_scores)]
                elif batch_scores.ndim == 1:
                    batch_scores = batch_scores.tolist()
                else:
                    batch_scores = batch_scores.mean(axis=-1).tolist()
                
                scores.extend(batch_scores)
        
        return scores
    
    def _compute_prompt_metrics(self,
                              prompts: List[str],
                              predictions: List[str]) -> Dict[str, float]:
        """
        Compute prompt-specific metrics
        
        Args:
            prompts: Input prompts
            predictions: Generated predictions
            
        Returns:
            Dictionary with prompt metrics
        """
        # Compute response relevance (simple token overlap)
        relevance_scores = []
        
        for prompt, pred in zip(prompts, predictions):
            prompt_tokens = set(prompt.lower().split())
            pred_tokens = set(pred.lower().split())
            
            if not prompt_tokens:
                relevance = 0.0
            else:
                overlap = len(prompt_tokens & pred_tokens)
                relevance = overlap / len(prompt_tokens)
            
            relevance_scores.append(relevance)
        
        return {
            'prompt_relevance': np.mean(relevance_scores) if relevance_scores else 0.0
        }
    
    def save_results(self,
                    results: Dict[str, Any],
                    filename: str = None) -> str:
        """
        Save evaluation results to file
        
        Args:
            results: Evaluation results
            filename: Output filename (optional)
            
        Returns:
            Path to saved file
        """
        if filename is None:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            filename = f"eval_results_{timestamp}.json"
        
        filepath = self.output_dir / filename
        
        # Add metadata
        results_with_meta = {
            'timestamp': datetime.now().isoformat(),
            'device': str(self.device),
            'results': results
        }
        
        with open(filepath, 'w', encoding='utf-8') as f:
            json.dump(results_with_meta, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Results saved to {filepath}")
        return str(filepath)
    
    def load_results(self, filepath: str) -> Dict[str, Any]:
        """
        Load evaluation results from file
        
        Args:
            filepath: Path to results file
            
        Returns:
            Loaded results
        """
        with open(filepath, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        return data.get('results', data)