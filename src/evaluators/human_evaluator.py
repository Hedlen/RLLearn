"""Human evaluation interface for RLHF models"""

import json
import random
from typing import List, Dict, Any, Optional, Tuple
from pathlib import Path
from datetime import datetime
import logging


class HumanEvaluator:
    """Human evaluation interface for model outputs"""
    
    def __init__(self, output_dir: str = "./human_eval"):
        """
        Initialize human evaluator
        
        Args:
            output_dir: Directory to save evaluation data
        """
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        
        self.logger = logging.getLogger(__name__)
        
        # Evaluation criteria
        self.quality_criteria = {
            'helpfulness': 'How helpful is the response?',
            'harmlessness': 'How safe and harmless is the response?',
            'honesty': 'How honest and truthful is the response?',
            'coherence': 'How coherent and well-structured is the response?',
            'relevance': 'How relevant is the response to the prompt?'
        }
        
        self.rating_scale = {
            1: 'Very Poor',
            2: 'Poor', 
            3: 'Fair',
            4: 'Good',
            5: 'Excellent'
        }
    
    def create_evaluation_task(self,
                             prompts: List[str],
                             responses_a: List[str],
                             responses_b: List[str],
                             model_a_name: str = "Model A",
                             model_b_name: str = "Model B",
                             task_name: str = None,
                             sample_size: int = None) -> str:
        """
        Create a human evaluation task for comparing two models
        
        Args:
            prompts: List of input prompts
            responses_a: Responses from model A
            responses_b: Responses from model B
            model_a_name: Name of model A
            model_b_name: Name of model B
            task_name: Name of the evaluation task
            sample_size: Number of samples to include (random subset)
            
        Returns:
            Path to the evaluation task file
        """
        if len(prompts) != len(responses_a) or len(prompts) != len(responses_b):
            raise ValueError("All input lists must have the same length")
        
        # Sample subset if requested
        if sample_size and sample_size < len(prompts):
            indices = random.sample(range(len(prompts)), sample_size)
            prompts = [prompts[i] for i in indices]
            responses_a = [responses_a[i] for i in indices]
            responses_b = [responses_b[i] for i in indices]
        
        # Create evaluation data
        evaluation_data = {
            'task_info': {
                'task_name': task_name or f"eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_a_name': model_a_name,
                'model_b_name': model_b_name,
                'created_at': datetime.now().isoformat(),
                'num_samples': len(prompts),
                'criteria': self.quality_criteria,
                'rating_scale': self.rating_scale
            },
            'samples': []
        }
        
        # Create evaluation samples
        for i, (prompt, resp_a, resp_b) in enumerate(zip(prompts, responses_a, responses_b)):
            # Randomize order to avoid bias
            if random.random() < 0.5:
                first_response = resp_a
                second_response = resp_b
                first_model = model_a_name
                second_model = model_b_name
                true_order = "A_first"
            else:
                first_response = resp_b
                second_response = resp_a
                first_model = model_b_name
                second_model = model_a_name
                true_order = "B_first"
            
            sample = {
                'id': i,
                'prompt': prompt,
                'response_1': {
                    'text': first_response,
                    'model': first_model
                },
                'response_2': {
                    'text': second_response,
                    'model': second_model
                },
                'true_order': true_order,  # For analysis purposes
                'evaluation': {
                    'overall_preference': None,  # 1 or 2
                    'criteria_ratings': {
                        criterion: {'response_1': None, 'response_2': None}
                        for criterion in self.quality_criteria.keys()
                    },
                    'comments': "",
                    'evaluator_id': "",
                    'evaluation_time': None
                }
            }
            
            evaluation_data['samples'].append(sample)
        
        # Save evaluation task
        task_file = self.output_dir / f"{evaluation_data['task_info']['task_name']}.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Created evaluation task with {len(prompts)} samples: {task_file}")
        return str(task_file)
    
    def create_single_model_evaluation(self,
                                     prompts: List[str],
                                     responses: List[str],
                                     model_name: str = "Model",
                                     task_name: str = None,
                                     sample_size: int = None) -> str:
        """
        Create a human evaluation task for a single model
        
        Args:
            prompts: List of input prompts
            responses: Model responses
            model_name: Name of the model
            task_name: Name of the evaluation task
            sample_size: Number of samples to include
            
        Returns:
            Path to the evaluation task file
        """
        if len(prompts) != len(responses):
            raise ValueError("Prompts and responses must have the same length")
        
        # Sample subset if requested
        if sample_size and sample_size < len(prompts):
            indices = random.sample(range(len(prompts)), sample_size)
            prompts = [prompts[i] for i in indices]
            responses = [responses[i] for i in indices]
        
        # Create evaluation data
        evaluation_data = {
            'task_info': {
                'task_name': task_name or f"single_eval_{datetime.now().strftime('%Y%m%d_%H%M%S')}",
                'model_name': model_name,
                'created_at': datetime.now().isoformat(),
                'num_samples': len(prompts),
                'criteria': self.quality_criteria,
                'rating_scale': self.rating_scale,
                'evaluation_type': 'single_model'
            },
            'samples': []
        }
        
        # Create evaluation samples
        for i, (prompt, response) in enumerate(zip(prompts, responses)):
            sample = {
                'id': i,
                'prompt': prompt,
                'response': {
                    'text': response,
                    'model': model_name
                },
                'evaluation': {
                    'criteria_ratings': {
                        criterion: None for criterion in self.quality_criteria.keys()
                    },
                    'overall_rating': None,  # 1-5 scale
                    'comments': "",
                    'evaluator_id': "",
                    'evaluation_time': None
                }
            }
            
            evaluation_data['samples'].append(sample)
        
        # Save evaluation task
        task_file = self.output_dir / f"{evaluation_data['task_info']['task_name']}.json"
        with open(task_file, 'w', encoding='utf-8') as f:
            json.dump(evaluation_data, f, indent=2, ensure_ascii=False)
        
        self.logger.info(f"Created single model evaluation task with {len(prompts)} samples: {task_file}")
        return str(task_file)
    
    def analyze_evaluation_results(self, task_file: str) -> Dict[str, Any]:
        """
        Analyze completed human evaluation results
        
        Args:
            task_file: Path to the evaluation task file
            
        Returns:
            Analysis results
        """
        with open(task_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        task_info = data['task_info']
        samples = data['samples']
        
        if task_info.get('evaluation_type') == 'single_model':
            return self._analyze_single_model_results(task_info, samples)
        else:
            return self._analyze_comparison_results(task_info, samples)
    
    def _analyze_comparison_results(self, task_info: Dict, samples: List[Dict]) -> Dict[str, Any]:
        """
        Analyze comparison evaluation results
        
        Args:
            task_info: Task information
            samples: Evaluation samples
            
        Returns:
            Analysis results
        """
        model_a_name = task_info['model_a_name']
        model_b_name = task_info['model_b_name']
        
        # Count preferences
        model_a_wins = 0
        model_b_wins = 0
        ties = 0
        completed_evaluations = 0
        
        # Criteria ratings
        criteria_scores = {
            model_a_name: {criterion: [] for criterion in self.quality_criteria.keys()},
            model_b_name: {criterion: [] for criterion in self.quality_criteria.keys()}
        }
        
        for sample in samples:
            evaluation = sample['evaluation']
            
            if evaluation['overall_preference'] is not None:
                completed_evaluations += 1
                
                # Determine actual winner based on true order
                true_order = sample['true_order']
                preference = evaluation['overall_preference']
                
                if preference == 1:
                    if true_order == "A_first":
                        model_a_wins += 1
                    else:
                        model_b_wins += 1
                elif preference == 2:
                    if true_order == "A_first":
                        model_b_wins += 1
                    else:
                        model_a_wins += 1
                else:
                    ties += 1
                
                # Collect criteria ratings
                for criterion, ratings in evaluation['criteria_ratings'].items():
                    if ratings['response_1'] is not None and ratings['response_2'] is not None:
                        if true_order == "A_first":
                            criteria_scores[model_a_name][criterion].append(ratings['response_1'])
                            criteria_scores[model_b_name][criterion].append(ratings['response_2'])
                        else:
                            criteria_scores[model_a_name][criterion].append(ratings['response_2'])
                            criteria_scores[model_b_name][criterion].append(ratings['response_1'])
        
        # Calculate statistics
        results = {
            'task_info': task_info,
            'completion_rate': completed_evaluations / len(samples) if samples else 0,
            'overall_preferences': {
                model_a_name: model_a_wins,
                model_b_name: model_b_wins,
                'ties': ties
            },
            'win_rates': {
                model_a_name: model_a_wins / completed_evaluations if completed_evaluations > 0 else 0,
                model_b_name: model_b_wins / completed_evaluations if completed_evaluations > 0 else 0
            },
            'criteria_analysis': {}
        }
        
        # Analyze criteria scores
        for model_name in [model_a_name, model_b_name]:
            results['criteria_analysis'][model_name] = {}
            for criterion, scores in criteria_scores[model_name].items():
                if scores:
                    results['criteria_analysis'][model_name][criterion] = {
                        'mean': sum(scores) / len(scores),
                        'count': len(scores),
                        'scores': scores
                    }
        
        return results
    
    def _analyze_single_model_results(self, task_info: Dict, samples: List[Dict]) -> Dict[str, Any]:
        """
        Analyze single model evaluation results
        
        Args:
            task_info: Task information
            samples: Evaluation samples
            
        Returns:
            Analysis results
        """
        model_name = task_info['model_name']
        
        completed_evaluations = 0
        overall_ratings = []
        criteria_scores = {criterion: [] for criterion in self.quality_criteria.keys()}
        
        for sample in samples:
            evaluation = sample['evaluation']
            
            if evaluation['overall_rating'] is not None:
                completed_evaluations += 1
                overall_ratings.append(evaluation['overall_rating'])
                
                # Collect criteria ratings
                for criterion, rating in evaluation['criteria_ratings'].items():
                    if rating is not None:
                        criteria_scores[criterion].append(rating)
        
        # Calculate statistics
        results = {
            'task_info': task_info,
            'completion_rate': completed_evaluations / len(samples) if samples else 0,
            'overall_rating': {
                'mean': sum(overall_ratings) / len(overall_ratings) if overall_ratings else 0,
                'count': len(overall_ratings),
                'ratings': overall_ratings
            },
            'criteria_analysis': {}
        }
        
        # Analyze criteria scores
        for criterion, scores in criteria_scores.items():
            if scores:
                results['criteria_analysis'][criterion] = {
                    'mean': sum(scores) / len(scores),
                    'count': len(scores),
                    'scores': scores
                }
        
        return results
    
    def export_evaluation_interface(self, task_file: str, output_format: str = "html") -> str:
        """
        Export evaluation task as a user-friendly interface
        
        Args:
            task_file: Path to the evaluation task file
            output_format: Output format ('html' or 'csv')
            
        Returns:
            Path to the exported file
        """
        with open(task_file, 'r', encoding='utf-8') as f:
            data = json.load(f)
        
        if output_format == "html":
            return self._export_html_interface(data)
        elif output_format == "csv":
            return self._export_csv_interface(data)
        else:
            raise ValueError(f"Unsupported output format: {output_format}")
    
    def _export_html_interface(self, data: Dict) -> str:
        """
        Export evaluation task as HTML interface
        
        Args:
            data: Evaluation task data
            
        Returns:
            Path to HTML file
        """
        task_name = data['task_info']['task_name']
        html_file = self.output_dir / f"{task_name}_interface.html"
        
        # Simple HTML template
        html_content = f"""
<!DOCTYPE html>
<html>
<head>
    <title>Human Evaluation: {task_name}</title>
    <style>
        body {{ font-family: Arial, sans-serif; margin: 20px; }}
        .sample {{ border: 1px solid #ccc; margin: 20px 0; padding: 15px; }}
        .prompt {{ background-color: #f0f0f0; padding: 10px; margin-bottom: 10px; }}
        .response {{ background-color: #f9f9f9; padding: 10px; margin: 10px 0; }}
        .evaluation {{ background-color: #fff; padding: 10px; border-top: 1px solid #eee; }}
        .criteria {{ margin: 10px 0; }}
        select, textarea {{ margin: 5px; }}
    </style>
</head>
<body>
    <h1>Human Evaluation: {data['task_info']['task_name']}</h1>
    <p><strong>Instructions:</strong> Please evaluate the responses according to the given criteria.</p>
    
    <div class="criteria-info">
        <h3>Evaluation Criteria:</h3>
        <ul>
"""
        
        for criterion, description in data['task_info']['criteria'].items():
            html_content += f"<li><strong>{criterion}:</strong> {description}</li>"
        
        html_content += """
        </ul>
        <h3>Rating Scale:</h3>
        <ul>
"""
        
        for rating, description in data['task_info']['rating_scale'].items():
            html_content += f"<li><strong>{rating}:</strong> {description}</li>"
        
        html_content += "</ul></div>"
        
        # Add samples
        for sample in data['samples']:
            html_content += f"""
    <div class="sample">
        <h3>Sample {sample['id'] + 1}</h3>
        <div class="prompt">
            <strong>Prompt:</strong> {sample['prompt']}
        </div>
"""
            
            if 'response_1' in sample:  # Comparison evaluation
                html_content += f"""
        <div class="response">
            <strong>Response 1:</strong><br>
            {sample['response_1']['text']}
        </div>
        <div class="response">
            <strong>Response 2:</strong><br>
            {sample['response_2']['text']}
        </div>
        <div class="evaluation">
            <p><strong>Which response is better overall?</strong></p>
            <select name="preference_{sample['id']}">
                <option value="">Select...</option>
                <option value="1">Response 1</option>
                <option value="2">Response 2</option>
                <option value="tie">Tie</option>
            </select>
        </div>
"""
            else:  # Single model evaluation
                html_content += f"""
        <div class="response">
            <strong>Response:</strong><br>
            {sample['response']['text']}
        </div>
        <div class="evaluation">
            <p><strong>Overall Rating:</strong></p>
            <select name="overall_{sample['id']}">
                <option value="">Select...</option>
                <option value="1">1 - Very Poor</option>
                <option value="2">2 - Poor</option>
                <option value="3">3 - Fair</option>
                <option value="4">4 - Good</option>
                <option value="5">5 - Excellent</option>
            </select>
        </div>
"""
            
            html_content += f"""
        <div class="evaluation">
            <p><strong>Comments:</strong></p>
            <textarea name="comments_{sample['id']}" rows="3" cols="50" placeholder="Optional comments..."></textarea>
        </div>
    </div>
"""
        
        html_content += """
</body>
</html>
"""
        
        with open(html_file, 'w', encoding='utf-8') as f:
            f.write(html_content)
        
        self.logger.info(f"Exported HTML interface: {html_file}")
        return str(html_file)
    
    def _export_csv_interface(self, data: Dict) -> str:
        """
        Export evaluation task as CSV file
        
        Args:
            data: Evaluation task data
            
        Returns:
            Path to CSV file
        """
        import csv
        
        task_name = data['task_info']['task_name']
        csv_file = self.output_dir / f"{task_name}_data.csv"
        
        with open(csv_file, 'w', newline='', encoding='utf-8') as f:
            if 'response_1' in data['samples'][0]:  # Comparison evaluation
                fieldnames = ['id', 'prompt', 'response_1', 'response_2', 'model_1', 'model_2']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for sample in data['samples']:
                    writer.writerow({
                        'id': sample['id'],
                        'prompt': sample['prompt'],
                        'response_1': sample['response_1']['text'],
                        'response_2': sample['response_2']['text'],
                        'model_1': sample['response_1']['model'],
                        'model_2': sample['response_2']['model']
                    })
            else:  # Single model evaluation
                fieldnames = ['id', 'prompt', 'response', 'model']
                writer = csv.DictWriter(f, fieldnames=fieldnames)
                writer.writeheader()
                
                for sample in data['samples']:
                    writer.writerow({
                        'id': sample['id'],
                        'prompt': sample['prompt'],
                        'response': sample['response']['text'],
                        'model': sample['response']['model']
                    })
        
        self.logger.info(f"Exported CSV data: {csv_file}")
        return str(csv_file)