"""Custom reward functions for GRPO training

This module provides example implementations of custom reward functions
that can be used instead of a trained reward model.
"""

import re
import torch
from typing import Union


def length_based_reward(prompt: str, response: str) -> float:
    """Simple length-based reward function
    
    Rewards longer responses up to a certain limit, then penalizes very long responses.
    
    Args:
        prompt: The input prompt
        response: The generated response
        
    Returns:
        Reward score between -1.0 and 1.0
    """
    response_length = len(response.split())
    
    # Optimal length range: 50-200 words
    if 50 <= response_length <= 200:
        return 1.0
    elif response_length < 50:
        # Penalize too short responses
        return max(-1.0, (response_length - 50) / 50)
    else:
        # Penalize too long responses
        return max(-1.0, (200 - response_length) / 100)


def keyword_based_reward(prompt: str, response: str) -> float:
    """Keyword-based reward function
    
    Rewards responses that contain certain keywords or phrases.
    
    Args:
        prompt: The input prompt
        response: The generated response
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    # Define positive and negative keywords
    positive_keywords = [
        "helpful", "accurate", "detailed", "clear", "informative",
        "thank you", "please", "sorry", "understand", "explain"
    ]
    
    negative_keywords = [
        "harmful", "offensive", "inappropriate", "illegal", "dangerous",
        "hate", "violence", "discrimination"
    ]
    
    response_lower = response.lower()
    
    # Count positive and negative keywords
    positive_count = sum(1 for keyword in positive_keywords if keyword in response_lower)
    negative_count = sum(1 for keyword in negative_keywords if keyword in response_lower)
    
    # Calculate reward
    reward = (positive_count * 0.2) - (negative_count * 0.5)
    return max(0.0, min(1.0, reward))


def quality_based_reward(prompt: str, response: str) -> float:
    """Quality-based reward function
    
    Evaluates response quality based on multiple criteria:
    - Grammar and punctuation
    - Coherence and structure
    - Relevance to prompt
    
    Args:
        prompt: The input prompt
        response: The generated response
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    if not response.strip():
        return 0.0
    
    score = 0.0
    
    # Check for proper sentence structure (basic)
    sentences = re.split(r'[.!?]+', response)
    valid_sentences = [s.strip() for s in sentences if s.strip()]
    
    if valid_sentences:
        # Reward proper capitalization
        capitalized_sentences = sum(1 for s in valid_sentences if s[0].isupper())
        score += (capitalized_sentences / len(valid_sentences)) * 0.3
        
        # Reward appropriate length sentences
        avg_sentence_length = sum(len(s.split()) for s in valid_sentences) / len(valid_sentences)
        if 5 <= avg_sentence_length <= 25:
            score += 0.3
        
        # Check for question words if prompt contains a question
        if '?' in prompt:
            question_indicators = ['because', 'since', 'due to', 'therefore', 'thus', 'so']
            if any(indicator in response.lower() for indicator in question_indicators):
                score += 0.2
    
    # Penalize repetition
    words = response.lower().split()
    if len(words) > 0:
        unique_words = len(set(words))
        repetition_ratio = unique_words / len(words)
        score += repetition_ratio * 0.2
    
    return min(1.0, score)


def task_specific_reward(prompt: str, response: str) -> float:
    """Task-specific reward function
    
    Customize this function based on your specific task requirements.
    This example is for a coding assistance task.
    
    Args:
        prompt: The input prompt
        response: The generated response
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    score = 0.0
    response_lower = response.lower()
    
    # Check if it's a coding-related prompt
    coding_keywords = ['code', 'function', 'class', 'variable', 'algorithm', 'programming']
    is_coding_task = any(keyword in prompt.lower() for keyword in coding_keywords)
    
    if is_coding_task:
        # Reward code blocks
        if '```' in response or 'def ' in response or 'class ' in response:
            score += 0.4
        
        # Reward explanations
        explanation_words = ['because', 'this', 'here', 'example', 'step']
        if any(word in response_lower for word in explanation_words):
            score += 0.3
        
        # Reward proper formatting
        if '\n' in response:  # Multi-line response
            score += 0.2
        
        # Penalize very short responses for coding tasks
        if len(response.split()) < 20:
            score -= 0.3
    else:
        # For non-coding tasks, use general quality metrics
        score = quality_based_reward(prompt, response)
    
    return max(0.0, min(1.0, score))


def combined_reward(prompt: str, response: str) -> float:
    """Combined reward function
    
    Combines multiple reward criteria with different weights.
    
    Args:
        prompt: The input prompt
        response: The generated response
        
    Returns:
        Reward score between 0.0 and 1.0
    """
    # Get individual scores
    length_score = length_based_reward(prompt, response)
    keyword_score = keyword_based_reward(prompt, response)
    quality_score = quality_based_reward(prompt, response)
    
    # Combine with weights
    combined_score = (
        0.3 * max(0, length_score) +  # Convert length score to 0-1 range
        0.3 * keyword_score +
        0.4 * quality_score
    )
    
    return min(1.0, combined_score)


# Example of a configurable reward function
class ConfigurableReward:
    """Configurable reward function class
    
    Allows for easy customization of reward parameters.
    """
    
    def __init__(self, 
                 length_weight: float = 0.3,
                 keyword_weight: float = 0.3,
                 quality_weight: float = 0.4,
                 target_length_range: tuple = (50, 200),
                 positive_keywords: list = None,
                 negative_keywords: list = None):
        self.length_weight = length_weight
        self.keyword_weight = keyword_weight
        self.quality_weight = quality_weight
        self.target_length_range = target_length_range
        
        self.positive_keywords = positive_keywords or [
            "helpful", "accurate", "detailed", "clear", "informative"
        ]
        self.negative_keywords = negative_keywords or [
            "harmful", "offensive", "inappropriate", "illegal", "dangerous"
        ]
    
    def __call__(self, prompt: str, response: str) -> float:
        """Calculate reward score
        
        Args:
            prompt: The input prompt
            response: The generated response
            
        Returns:
            Reward score between 0.0 and 1.0
        """
        # Length score
        response_length = len(response.split())
        min_len, max_len = self.target_length_range
        
        if min_len <= response_length <= max_len:
            length_score = 1.0
        elif response_length < min_len:
            length_score = max(0.0, response_length / min_len)
        else:
            length_score = max(0.0, max_len / response_length)
        
        # Keyword score
        response_lower = response.lower()
        positive_count = sum(1 for kw in self.positive_keywords if kw in response_lower)
        negative_count = sum(1 for kw in self.negative_keywords if kw in response_lower)
        keyword_score = max(0.0, min(1.0, (positive_count * 0.2) - (negative_count * 0.5)))
        
        # Quality score (simplified)
        quality_score = min(1.0, len(response.split()) / 100)  # Simple length-based quality
        
        # Combine scores
        total_score = (
            self.length_weight * length_score +
            self.keyword_weight * keyword_score +
            self.quality_weight * quality_score
        )
        
        return min(1.0, total_score)


# Create a default configurable reward instance
default_configurable_reward = ConfigurableReward()