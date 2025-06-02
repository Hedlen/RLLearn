"""Evaluation metrics for model assessment"""

import torch
import numpy as np
from typing import List, Dict, Any, Union
from collections import Counter
import math

try:
    from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction
    from rouge_score import rouge_scorer
    NLTK_AVAILABLE = True
except ImportError:
    NLTK_AVAILABLE = False
    print("Warning: NLTK or rouge-score not available. Some metrics will be unavailable.")


def compute_bleu_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute BLEU scores
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with BLEU scores
    """
    if not NLTK_AVAILABLE:
        return {'bleu_1': 0.0, 'bleu_2': 0.0, 'bleu_3': 0.0, 'bleu_4': 0.0}
    
    smoothing = SmoothingFunction().method1
    
    bleu_1_scores = []
    bleu_2_scores = []
    bleu_3_scores = []
    bleu_4_scores = []
    
    for pred, ref in zip(predictions, references):
        if not ref.strip():  # Skip empty references
            continue
            
        pred_tokens = pred.lower().split()
        ref_tokens = [ref.lower().split()]  # BLEU expects list of reference token lists
        
        try:
            bleu_1 = sentence_bleu(ref_tokens, pred_tokens, weights=(1, 0, 0, 0), smoothing_function=smoothing)
            bleu_2 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.5, 0.5, 0, 0), smoothing_function=smoothing)
            bleu_3 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.33, 0.33, 0.33, 0), smoothing_function=smoothing)
            bleu_4 = sentence_bleu(ref_tokens, pred_tokens, weights=(0.25, 0.25, 0.25, 0.25), smoothing_function=smoothing)
            
            bleu_1_scores.append(bleu_1)
            bleu_2_scores.append(bleu_2)
            bleu_3_scores.append(bleu_3)
            bleu_4_scores.append(bleu_4)
        except:
            # Handle edge cases
            continue
    
    return {
        'bleu_1': np.mean(bleu_1_scores) if bleu_1_scores else 0.0,
        'bleu_2': np.mean(bleu_2_scores) if bleu_2_scores else 0.0,
        'bleu_3': np.mean(bleu_3_scores) if bleu_3_scores else 0.0,
        'bleu_4': np.mean(bleu_4_scores) if bleu_4_scores else 0.0
    }


def compute_rouge_score(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute ROUGE scores
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with ROUGE scores
    """
    if not NLTK_AVAILABLE:
        return {'rouge_1': 0.0, 'rouge_2': 0.0, 'rouge_l': 0.0}
    
    scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
    
    rouge_1_scores = []
    rouge_2_scores = []
    rouge_l_scores = []
    
    for pred, ref in zip(predictions, references):
        if not ref.strip():  # Skip empty references
            continue
            
        scores = scorer.score(ref, pred)
        rouge_1_scores.append(scores['rouge1'].fmeasure)
        rouge_2_scores.append(scores['rouge2'].fmeasure)
        rouge_l_scores.append(scores['rougeL'].fmeasure)
    
    return {
        'rouge_1': np.mean(rouge_1_scores) if rouge_1_scores else 0.0,
        'rouge_2': np.mean(rouge_2_scores) if rouge_2_scores else 0.0,
        'rouge_l': np.mean(rouge_l_scores) if rouge_l_scores else 0.0
    }


def compute_perplexity(model, texts: List[str], tokenizer, device: torch.device, batch_size: int = 8) -> float:
    """Compute perplexity of model on given texts
    
    Args:
        model: Language model
        texts: List of texts to evaluate
        tokenizer: Tokenizer
        device: Device to run on
        batch_size: Batch size for evaluation
        
    Returns:
        Perplexity score
    """
    model.eval()
    total_loss = 0.0
    total_tokens = 0
    
    with torch.no_grad():
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=2048
            ).to(device)
            
            # Forward pass
            outputs = model(**inputs, labels=inputs['input_ids'])
            loss = outputs.loss
            
            # Accumulate loss and token count
            batch_tokens = (inputs['attention_mask'] == 1).sum().item()
            total_loss += loss.item() * batch_tokens
            total_tokens += batch_tokens
    
    if total_tokens == 0:
        return float('inf')
    
    avg_loss = total_loss / total_tokens
    perplexity = math.exp(avg_loss)
    
    return perplexity


def compute_reward_metrics(reward_scores: List[float]) -> Dict[str, float]:
    """Compute statistics for reward scores
    
    Args:
        reward_scores: List of reward scores
        
    Returns:
        Dictionary with reward statistics
    """
    if not reward_scores:
        return {
            'mean_reward': 0.0,
            'std_reward': 0.0,
            'min_reward': 0.0,
            'max_reward': 0.0,
            'median_reward': 0.0
        }
    
    rewards = np.array(reward_scores)
    
    return {
        'mean_reward': float(np.mean(rewards)),
        'std_reward': float(np.std(rewards)),
        'min_reward': float(np.min(rewards)),
        'max_reward': float(np.max(rewards)),
        'median_reward': float(np.median(rewards))
    }


def compute_preference_accuracy(chosen_rewards: List[float], rejected_rewards: List[float]) -> Dict[str, float]:
    """Compute preference accuracy metrics
    
    Args:
        chosen_rewards: Rewards for chosen responses
        rejected_rewards: Rewards for rejected responses
        
    Returns:
        Dictionary with preference metrics
    """
    if len(chosen_rewards) != len(rejected_rewards):
        raise ValueError("Chosen and rejected rewards must have same length")
    
    if not chosen_rewards:
        return {'accuracy': 0.0, 'margin': 0.0}
    
    chosen = np.array(chosen_rewards)
    rejected = np.array(rejected_rewards)
    
    # Compute accuracy (how often chosen > rejected)
    accuracy = np.mean(chosen > rejected)
    
    # Compute average margin
    margin = np.mean(chosen - rejected)
    
    return {
        'accuracy': float(accuracy),
        'margin': float(margin)
    }


def compute_diversity_metrics(texts: List[str]) -> Dict[str, float]:
    """Compute diversity metrics for generated texts
    
    Args:
        texts: List of generated texts
        
    Returns:
        Dictionary with diversity metrics
    """
    if not texts:
        return {
            'distinct_1': 0.0,
            'distinct_2': 0.0,
            'entropy': 0.0,
            'unique_ratio': 0.0
        }
    
    # Tokenize all texts
    all_tokens = []
    all_bigrams = []
    
    for text in texts:
        tokens = text.lower().split()
        all_tokens.extend(tokens)
        
        # Create bigrams
        bigrams = [(tokens[i], tokens[i+1]) for i in range(len(tokens)-1)]
        all_bigrams.extend(bigrams)
    
    # Compute distinct-1 (unique unigrams / total unigrams)
    distinct_1 = len(set(all_tokens)) / len(all_tokens) if all_tokens else 0.0
    
    # Compute distinct-2 (unique bigrams / total bigrams)
    distinct_2 = len(set(all_bigrams)) / len(all_bigrams) if all_bigrams else 0.0
    
    # Compute entropy
    token_counts = Counter(all_tokens)
    total_tokens = len(all_tokens)
    entropy = 0.0
    
    if total_tokens > 0:
        for count in token_counts.values():
            prob = count / total_tokens
            entropy -= prob * math.log2(prob)
    
    # Compute unique ratio (unique texts / total texts)
    unique_ratio = len(set(texts)) / len(texts)
    
    return {
        'distinct_1': distinct_1,
        'distinct_2': distinct_2,
        'entropy': entropy,
        'unique_ratio': unique_ratio
    }


def compute_semantic_similarity(predictions: List[str], references: List[str]) -> Dict[str, float]:
    """Compute semantic similarity metrics
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts
        
    Returns:
        Dictionary with similarity metrics
    """
    # Simple token-based similarity as fallback
    similarities = []
    
    for pred, ref in zip(predictions, references):
        if not ref.strip():
            continue
            
        pred_tokens = set(pred.lower().split())
        ref_tokens = set(ref.lower().split())
        
        if not pred_tokens and not ref_tokens:
            similarity = 1.0
        elif not pred_tokens or not ref_tokens:
            similarity = 0.0
        else:
            # Jaccard similarity
            intersection = len(pred_tokens & ref_tokens)
            union = len(pred_tokens | ref_tokens)
            similarity = intersection / union if union > 0 else 0.0
        
        similarities.append(similarity)
    
    return {
        'jaccard_similarity': np.mean(similarities) if similarities else 0.0
    }


def compute_length_metrics(texts: List[str]) -> Dict[str, float]:
    """Compute length-related metrics
    
    Args:
        texts: List of texts
        
    Returns:
        Dictionary with length metrics
    """
    if not texts:
        return {
            'avg_length': 0.0,
            'std_length': 0.0,
            'min_length': 0.0,
            'max_length': 0.0
        }
    
    lengths = [len(text.split()) for text in texts]
    
    return {
        'avg_length': float(np.mean(lengths)),
        'std_length': float(np.std(lengths)),
        'min_length': float(np.min(lengths)),
        'max_length': float(np.max(lengths))
    }


def compute_all_metrics(predictions: List[str], 
                       references: List[str] = None,
                       reward_scores: List[float] = None) -> Dict[str, Any]:
    """Compute all available metrics
    
    Args:
        predictions: List of predicted texts
        references: List of reference texts (optional)
        reward_scores: List of reward scores (optional)
        
    Returns:
        Dictionary with all computed metrics
    """
    metrics = {}
    
    # Length metrics
    metrics.update(compute_length_metrics(predictions))
    
    # Diversity metrics
    metrics.update(compute_diversity_metrics(predictions))
    
    # Reference-based metrics
    if references:
        metrics.update(compute_bleu_score(predictions, references))
        metrics.update(compute_rouge_score(predictions, references))
        metrics.update(compute_semantic_similarity(predictions, references))
    
    # Reward-based metrics
    if reward_scores:
        metrics.update(compute_reward_metrics(reward_scores))
    
    return metrics