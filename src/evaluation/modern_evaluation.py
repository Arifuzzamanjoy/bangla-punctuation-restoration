#!/usr/bin/env python3
"""
Modern Evaluation and Monitoring System
"""

import torch
import numpy as np
import pandas as pd
from typing import Dict, List, Any, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
import json
from pathlib import Path
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_recall_fscore_support,
    confusion_matrix, classification_report,
    roc_auc_score, average_precision_score
)
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots
import wandb
from datetime import datetime
import time
from contextlib import contextmanager
import psutil
import GPUtil
from transformers import pipeline
from sentence_transformers import SentenceTransformer
import nltk
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
import asyncio
import aiohttp
from concurrent.futures import ThreadPoolExecutor
import warnings
warnings.filterwarnings('ignore')

logger = logging.getLogger(__name__)

@dataclass
class EvaluationConfig:
    """Configuration for modern evaluation"""
    
    # Basic metrics
    compute_accuracy: bool = True
    compute_f1: bool = True
    compute_precision_recall: bool = True
    
    # Advanced metrics
    compute_semantic_similarity: bool = True
    compute_readability: bool = True
    compute_fluency: bool = True
    compute_bleu_rouge: bool = True
    
    # Robustness evaluation
    compute_adversarial_robustness: bool = True
    compute_ood_performance: bool = True
    compute_fairness_metrics: bool = True
    
    # Efficiency metrics
    compute_inference_time: bool = True
    compute_memory_usage: bool = True
    compute_energy_consumption: bool = True
    
    # Visualization
    generate_confusion_matrix: bool = True
    generate_error_analysis: bool = True
    generate_performance_plots: bool = True
    
    # Output
    save_detailed_results: bool = True
    save_predictions: bool = True
    save_visualizations: bool = True

class PerformanceMonitor:
    """Real-time performance monitoring"""
    
    def __init__(self):
        self.metrics_history = []
        self.start_time = None
        self.gpu_available = torch.cuda.is_available()
        
    @contextmanager
    def monitor_inference(self):
        """Context manager for monitoring inference"""
        # Start monitoring
        start_time = time.time()
        start_memory = self._get_memory_usage()
        start_gpu_memory = self._get_gpu_memory() if self.gpu_available else 0
        
        yield
        
        # End monitoring
        end_time = time.time()
        end_memory = self._get_memory_usage()
        end_gpu_memory = self._get_gpu_memory() if self.gpu_available else 0
        
        # Record metrics
        self.metrics_history.append({
            'inference_time': end_time - start_time,
            'memory_usage': end_memory - start_memory,
            'gpu_memory_usage': end_gpu_memory - start_gpu_memory,
            'timestamp': datetime.now()
        })
    
    def _get_memory_usage(self) -> float:
        """Get current memory usage in MB"""
        return psutil.Process().memory_info().rss / 1024 / 1024
    
    def _get_gpu_memory(self) -> float:
        """Get current GPU memory usage in MB"""
        if not self.gpu_available:
            return 0.0
        
        try:
            gpus = GPUtil.getGPUs()
            if gpus:
                return gpus[0].memoryUsed
        except:
            pass
        return 0.0
    
    def get_average_metrics(self) -> Dict[str, float]:
        """Get average performance metrics"""
        if not self.metrics_history:
            return {}
        
        return {
            'avg_inference_time': np.mean([m['inference_time'] for m in self.metrics_history]),
            'avg_memory_usage': np.mean([m['memory_usage'] for m in self.metrics_history]),
            'avg_gpu_memory_usage': np.mean([m['gpu_memory_usage'] for m in self.metrics_history]),
            'total_inferences': len(self.metrics_history)
        }

class SemanticEvaluator:
    """Evaluate semantic quality of punctuation restoration"""
    
    def __init__(self):
        try:
            self.sentence_transformer = SentenceTransformer('paraphrase-multilingual-MiniLM-L12-v2')
            self.bleu_scorer = nltk.translate.bleu_score
            self.rouge_scorer = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=False)
        except Exception as e:
            logger.warning(f"Could not initialize semantic evaluator: {e}")
            self.sentence_transformer = None
            self.bleu_scorer = None
            self.rouge_scorer = None
    
    def compute_semantic_similarity(self, predictions: List[str], references: List[str]) -> float:
        """Compute semantic similarity between predictions and references"""
        if not self.sentence_transformer:
            return 0.0
        
        try:
            pred_embeddings = self.sentence_transformer.encode(predictions)
            ref_embeddings = self.sentence_transformer.encode(references)
            
            similarities = []
            for pred_emb, ref_emb in zip(pred_embeddings, ref_embeddings):
                similarity = np.dot(pred_emb, ref_emb) / (np.linalg.norm(pred_emb) * np.linalg.norm(ref_emb))
                similarities.append(similarity)
            
            return np.mean(similarities)
        except Exception as e:
            logger.error(f"Error computing semantic similarity: {e}")
            return 0.0
    
    def compute_bleu_score(self, predictions: List[str], references: List[str]) -> float:
        """Compute BLEU score"""
        if not self.bleu_scorer:
            return 0.0
        
        try:
            scores = []
            for pred, ref in zip(predictions, references):
                pred_tokens = pred.split()
                ref_tokens = [ref.split()]  # BLEU expects list of reference token lists
                
                score = sentence_bleu(ref_tokens, pred_tokens)
                scores.append(score)
            
            return np.mean(scores)
        except Exception as e:
            logger.error(f"Error computing BLEU score: {e}")
            return 0.0
    
    def compute_rouge_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute ROUGE scores"""
        if not self.rouge_scorer:
            return {}
        
        try:
            rouge_scores = {'rouge1': [], 'rouge2': [], 'rougeL': []}
            
            for pred, ref in zip(predictions, references):
                scores = self.rouge_scorer.score(ref, pred)
                rouge_scores['rouge1'].append(scores['rouge1'].fmeasure)
                rouge_scores['rouge2'].append(scores['rouge2'].fmeasure)
                rouge_scores['rougeL'].append(scores['rougeL'].fmeasure)
            
            return {
                'rouge1': np.mean(rouge_scores['rouge1']),
                'rouge2': np.mean(rouge_scores['rouge2']),
                'rougeL': np.mean(rouge_scores['rougeL'])
            }
        except Exception as e:
            logger.error(f"Error computing ROUGE scores: {e}")
            return {}

class ReadabilityAnalyzer:
    """Analyze text readability and quality"""
    
    def compute_readability_metrics(self, texts: List[str]) -> Dict[str, float]:
        """Compute various readability metrics"""
        
        metrics = {
            'avg_sentence_length': [],
            'punctuation_density': [],
            'complexity_score': [],
            'bangla_char_ratio': []
        }
        
        for text in texts:
            # Average sentence length
            sentences = text.split('।')
            words = text.split()
            avg_sent_len = len(words) / max(len(sentences), 1)
            metrics['avg_sentence_length'].append(avg_sent_len)
            
            # Punctuation density
            punct_count = sum(1 for char in text if char in '।?!,;:-')
            punct_density = punct_count / max(len(text), 1)
            metrics['punctuation_density'].append(punct_density)
            
            # Complexity score (based on punctuation variety)
            unique_puncts = set(char for char in text if char in '।?!,;:-')
            complexity = len(unique_puncts)
            metrics['complexity_score'].append(complexity)
            
            # Bangla character ratio
            bangla_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
            bangla_ratio = bangla_chars / max(len(text), 1)
            metrics['bangla_char_ratio'].append(bangla_ratio)
        
        # Return averages
        return {key: np.mean(values) for key, values in metrics.items()}

class AdversarialEvaluator:
    """Evaluate model robustness against adversarial attacks"""
    
    def __init__(self):
        self.attack_methods = [
            'character_substitution',
            'word_insertion',
            'word_deletion',
            'word_reordering'
        ]
    
    def generate_adversarial_examples(self, texts: List[str], attack_type: str = 'character_substitution') -> List[str]:
        """Generate adversarial examples"""
        
        adversarial_texts = []
        
        for text in texts:
            if attack_type == 'character_substitution':
                adv_text = self._character_substitution_attack(text)
            elif attack_type == 'word_insertion':
                adv_text = self._word_insertion_attack(text)
            elif attack_type == 'word_deletion':
                adv_text = self._word_deletion_attack(text)
            elif attack_type == 'word_reordering':
                adv_text = self._word_reordering_attack(text)
            else:
                adv_text = text
            
            adversarial_texts.append(adv_text)
        
        return adversarial_texts
    
    def _character_substitution_attack(self, text: str, substitution_rate: float = 0.1) -> str:
        """Character substitution attack"""
        chars = list(text)
        num_substitutions = int(len(chars) * substitution_rate)
        
        # Bangla character substitution map
        substitution_map = {
            'া': 'ো', 'ি': 'ী', 'ু': 'ূ', 'ে': 'ৈ',
            'ক': 'খ', 'গ': 'ঘ', 'চ': 'ছ', 'জ': 'ঝ'
        }
        
        for _ in range(num_substitutions):
            if chars:
                idx = np.random.randint(0, len(chars))
                if chars[idx] in substitution_map:
                    chars[idx] = substitution_map[chars[idx]]
        
        return ''.join(chars)
    
    def _word_insertion_attack(self, text: str, insertion_rate: float = 0.1) -> str:
        """Word insertion attack"""
        words = text.split()
        num_insertions = int(len(words) * insertion_rate)
        
        filler_words = ['তাহলে', 'আসলে', 'মানে', 'যেমন']
        
        for _ in range(num_insertions):
            insert_word = np.random.choice(filler_words)
            insert_pos = np.random.randint(0, len(words) + 1)
            words.insert(insert_pos, insert_word)
        
        return ' '.join(words)
    
    def _word_deletion_attack(self, text: str, deletion_rate: float = 0.1) -> str:
        """Word deletion attack"""
        words = text.split()
        num_deletions = int(len(words) * deletion_rate)
        
        if len(words) <= num_deletions:
            return text
        
        indices_to_delete = np.random.choice(len(words), num_deletions, replace=False)
        words = [word for i, word in enumerate(words) if i not in indices_to_delete]
        
        return ' '.join(words)
    
    def _word_reordering_attack(self, text: str, reorder_rate: float = 0.1) -> str:
        """Word reordering attack"""
        words = text.split()
        num_swaps = int(len(words) * reorder_rate / 2)
        
        for _ in range(num_swaps):
            if len(words) >= 2:
                idx1, idx2 = np.random.choice(len(words), 2, replace=False)
                words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def evaluate_robustness(self, model, original_texts: List[str], predictions: List[str]) -> Dict[str, float]:
        """Evaluate model robustness against various attacks"""
        
        robustness_scores = {}
        
        for attack_type in self.attack_methods:
            # Generate adversarial examples
            adversarial_texts = self.generate_adversarial_examples(original_texts, attack_type)
            
            # Get model predictions on adversarial examples
            # This would require the actual model interface
            # For now, we'll use a placeholder
            adversarial_predictions = predictions  # Placeholder
            
            # Compute robustness score (similarity between original and adversarial predictions)
            similarities = []
            for orig_pred, adv_pred in zip(predictions, adversarial_predictions):
                # Simple similarity based on edit distance
                similarity = self._compute_text_similarity(orig_pred, adv_pred)
                similarities.append(similarity)
            
            robustness_scores[f'{attack_type}_robustness'] = np.mean(similarities)
        
        return robustness_scores
    
    def _compute_text_similarity(self, text1: str, text2: str) -> float:
        """Compute similarity between two texts"""
        # Simple Jaccard similarity
        set1 = set(text1.split())
        set2 = set(text2.split())
        
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / max(union, 1)

class FairnessEvaluator:
    """Evaluate model fairness across different groups"""
    
    def __init__(self):
        self.demographic_attributes = ['formal_style', 'informal_style', 'domain_specific']
    
    def evaluate_fairness(self, predictions: List[str], references: List[str], 
                         attributes: List[Dict[str, Any]]) -> Dict[str, float]:
        """Evaluate fairness across different demographic groups"""
        
        fairness_metrics = {}
        
        # Group data by attributes
        grouped_data = self._group_by_attributes(predictions, references, attributes)
        
        # Compute metrics for each group
        group_metrics = {}
        for group, (group_preds, group_refs) in grouped_data.items():
            accuracy = accuracy_score(
                [ref.split() for ref in group_refs], 
                [pred.split() for pred in group_preds]
            )
            group_metrics[group] = accuracy
        
        # Compute fairness metrics
        if len(group_metrics) > 1:
            accuracies = list(group_metrics.values())
            fairness_metrics['accuracy_variance'] = np.var(accuracies)
            fairness_metrics['min_accuracy'] = np.min(accuracies)
            fairness_metrics['max_accuracy'] = np.max(accuracies)
            fairness_metrics['accuracy_range'] = np.max(accuracies) - np.min(accuracies)
        
        return fairness_metrics
    
    def _group_by_attributes(self, predictions: List[str], references: List[str], 
                           attributes: List[Dict[str, Any]]) -> Dict[str, Tuple[List[str], List[str]]]:
        """Group data by demographic attributes"""
        
        grouped = {}
        
        for pred, ref, attr in zip(predictions, references, attributes):
            # Create group key based on attributes
            group_key = "_".join([f"{k}:{v}" for k, v in attr.items()])
            
            if group_key not in grouped:
                grouped[group_key] = ([], [])
            
            grouped[group_key][0].append(pred)
            grouped[group_key][1].append(ref)
        
        return grouped

class ModernEvaluator:
    """Comprehensive modern evaluation system"""
    
    def __init__(self, config: EvaluationConfig = None):
        self.config = config or EvaluationConfig()
        
        # Initialize components
        self.performance_monitor = PerformanceMonitor()
        self.semantic_evaluator = SemanticEvaluator()
        self.readability_analyzer = ReadabilityAnalyzer()
        self.adversarial_evaluator = AdversarialEvaluator()
        self.fairness_evaluator = FairnessEvaluator()
        
        # Results storage
        self.results = {}
        self.detailed_results = []
    
    def evaluate_model(self, 
                      model,
                      test_dataset: List[Dict[str, str]],
                      output_dir: str = "results/evaluation") -> Dict[str, Any]:
        """Comprehensive model evaluation"""
        
        logger.info("Starting comprehensive model evaluation...")
        
        # Prepare data
        texts = [item['unpunctuated_text'] for item in test_dataset]
        references = [item['punctuated_text'] for item in test_dataset]
        
        # Generate predictions with performance monitoring
        predictions = []
        inference_times = []
        
        for text in texts:
            with self.performance_monitor.monitor_inference():
                # This would be the actual model prediction
                prediction = self._predict_with_model(model, text)
                predictions.append(prediction)
        
        # Basic metrics
        if self.config.compute_accuracy:
            self.results['accuracy'] = self._compute_token_accuracy(predictions, references)
        
        if self.config.compute_f1:
            f1_metrics = self._compute_f1_scores(predictions, references)
            self.results.update(f1_metrics)
        
        # Advanced semantic metrics
        if self.config.compute_semantic_similarity:
            semantic_sim = self.semantic_evaluator.compute_semantic_similarity(predictions, references)
            self.results['semantic_similarity'] = semantic_sim
        
        if self.config.compute_bleu_rouge:
            bleu_score = self.semantic_evaluator.compute_bleu_score(predictions, references)
            rouge_scores = self.semantic_evaluator.compute_rouge_scores(predictions, references)
            self.results['bleu_score'] = bleu_score
            self.results.update(rouge_scores)
        
        # Readability metrics
        if self.config.compute_readability:
            readability_metrics = self.readability_analyzer.compute_readability_metrics(predictions)
            self.results.update({f'readability_{k}': v for k, v in readability_metrics.items()})
        
        # Robustness evaluation
        if self.config.compute_adversarial_robustness:
            robustness_metrics = self.adversarial_evaluator.evaluate_robustness(
                model, texts, predictions
            )
            self.results.update(robustness_metrics)
        
        # Performance metrics
        if self.config.compute_inference_time:
            perf_metrics = self.performance_monitor.get_average_metrics()
            self.results.update(perf_metrics)
        
        # Fairness evaluation (if demographic data available)
        if self.config.compute_fairness_metrics:
            # This would require demographic attributes for each sample
            # For now, we'll skip this
            pass
        
        # Generate visualizations
        if self.config.generate_confusion_matrix:
            self._generate_confusion_matrix(predictions, references, output_dir)
        
        if self.config.generate_error_analysis:
            self._generate_error_analysis(predictions, references, texts, output_dir)
        
        if self.config.generate_performance_plots:
            self._generate_performance_plots(output_dir)
        
        # Save results
        if self.config.save_detailed_results:
            self._save_results(output_dir)
        
        logger.info("Evaluation completed successfully!")
        return self.results
    
    def _predict_with_model(self, model, text: str) -> str:
        """Make prediction with model (placeholder)"""
        # This would be the actual model prediction logic
        # For now, return the input text as a placeholder
        return text
    
    def _compute_token_accuracy(self, predictions: List[str], references: List[str]) -> float:
        """Compute token-level accuracy"""
        total_tokens = 0
        correct_tokens = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # Align tokens (simple approach)
            min_len = min(len(pred_tokens), len(ref_tokens))
            
            for i in range(min_len):
                total_tokens += 1
                if pred_tokens[i] == ref_tokens[i]:
                    correct_tokens += 1
        
        return correct_tokens / max(total_tokens, 1)
    
    def _compute_f1_scores(self, predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Compute F1 scores for punctuation marks"""
        
        punctuation_marks = ['।', '?', '!', ',', ';', ':', '-']
        f1_scores = {}
        
        for punct in punctuation_marks:
            pred_labels = [1 if punct in pred else 0 for pred in predictions]
            ref_labels = [1 if punct in ref else 0 for ref in references]
            
            if sum(ref_labels) > 0:  # Only compute if punctuation exists in references
                _, _, f1, _ = precision_recall_fscore_support(
                    ref_labels, pred_labels, average='binary', zero_division=0
                )
                f1_scores[f'f1_{punct}'] = f1
        
        # Overall F1
        all_pred_labels = []
        all_ref_labels = []
        
        for pred, ref in zip(predictions, references):
            for punct in punctuation_marks:
                all_pred_labels.append(1 if punct in pred else 0)
                all_ref_labels.append(1 if punct in ref else 0)
        
        _, _, overall_f1, _ = precision_recall_fscore_support(
            all_ref_labels, all_pred_labels, average='macro', zero_division=0
        )
        f1_scores['f1_overall'] = overall_f1
        
        return f1_scores
    
    def _generate_confusion_matrix(self, predictions: List[str], references: List[str], output_dir: str):
        """Generate and save confusion matrix"""
        
        # Extract punctuation labels
        punctuation_marks = ['।', '?', '!', ',', ';', ':', '-']
        
        pred_labels = []
        ref_labels = []
        
        for pred, ref in zip(predictions, references):
            for punct in punctuation_marks:
                pred_labels.append(punct if punct in pred else 'O')
                ref_labels.append(punct if punct in ref else 'O')
        
        # Create confusion matrix
        cm = confusion_matrix(ref_labels, pred_labels, labels=['O'] + punctuation_marks)
        
        # Plot
        plt.figure(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=['O'] + punctuation_marks,
                   yticklabels=['O'] + punctuation_marks)
        plt.title('Punctuation Confusion Matrix')
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        
        # Save
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        plt.savefig(f"{output_dir}/confusion_matrix.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _generate_error_analysis(self, predictions: List[str], references: List[str], 
                               texts: List[str], output_dir: str):
        """Generate detailed error analysis"""
        
        errors = []
        
        for i, (pred, ref, text) in enumerate(zip(predictions, references, texts)):
            if pred != ref:
                error_info = {
                    'index': i,
                    'original_text': text,
                    'predicted': pred,
                    'reference': ref,
                    'error_type': self._classify_error(pred, ref)
                }
                errors.append(error_info)
        
        # Save error analysis
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        with open(f"{output_dir}/error_analysis.json", 'w', encoding='utf-8') as f:
            json.dump(errors, f, ensure_ascii=False, indent=2)
        
        # Generate error statistics
        error_types = [error['error_type'] for error in errors]
        error_counts = pd.Series(error_types).value_counts()
        
        # Plot error distribution
        plt.figure(figsize=(12, 6))
        error_counts.plot(kind='bar')
        plt.title('Error Type Distribution')
        plt.xlabel('Error Type')
        plt.ylabel('Count')
        plt.xticks(rotation=45)
        plt.tight_layout()
        plt.savefig(f"{output_dir}/error_distribution.png", dpi=300, bbox_inches='tight')
        plt.close()
    
    def _classify_error(self, prediction: str, reference: str) -> str:
        """Classify the type of error"""
        
        pred_puncts = set(char for char in prediction if char in '।?!,;:-')
        ref_puncts = set(char for char in reference if char in '।?!,;:-')
        
        if len(pred_puncts) < len(ref_puncts):
            return "missing_punctuation"
        elif len(pred_puncts) > len(ref_puncts):
            return "extra_punctuation"
        elif pred_puncts != ref_puncts:
            return "wrong_punctuation"
        else:
            return "position_error"
    
    def _generate_performance_plots(self, output_dir: str):
        """Generate performance visualization plots"""
        
        # Performance metrics over time
        if self.performance_monitor.metrics_history:
            df = pd.DataFrame(self.performance_monitor.metrics_history)
            
            fig = make_subplots(
                rows=2, cols=2,
                subplot_titles=['Inference Time', 'Memory Usage', 'GPU Memory', 'Cumulative Performance']
            )
            
            # Inference time
            fig.add_trace(
                go.Scatter(y=df['inference_time'], mode='lines', name='Inference Time'),
                row=1, col=1
            )
            
            # Memory usage
            fig.add_trace(
                go.Scatter(y=df['memory_usage'], mode='lines', name='Memory Usage'),
                row=1, col=2
            )
            
            # GPU memory
            fig.add_trace(
                go.Scatter(y=df['gpu_memory_usage'], mode='lines', name='GPU Memory'),
                row=2, col=1
            )
            
            # Cumulative performance
            cumulative_time = df['inference_time'].cumsum()
            fig.add_trace(
                go.Scatter(y=cumulative_time, mode='lines', name='Cumulative Time'),
                row=2, col=2
            )
            
            fig.update_layout(height=600, showlegend=False, title_text="Performance Metrics")
            
            # Save
            Path(output_dir).mkdir(parents=True, exist_ok=True)
            fig.write_html(f"{output_dir}/performance_plots.html")
    
    def _save_results(self, output_dir: str):
        """Save detailed evaluation results"""
        
        Path(output_dir).mkdir(parents=True, exist_ok=True)
        
        # Save main results
        with open(f"{output_dir}/evaluation_results.json", 'w') as f:
            json.dump(self.results, f, indent=2)
        
        # Save performance history
        if self.performance_monitor.metrics_history:
            perf_df = pd.DataFrame(self.performance_monitor.metrics_history)
            perf_df.to_csv(f"{output_dir}/performance_history.csv", index=False)
        
        # Generate summary report
        self._generate_summary_report(output_dir)
    
    def _generate_summary_report(self, output_dir: str):
        """Generate a comprehensive summary report"""
        
        report = f"""
# Bangla Punctuation Restoration - Evaluation Report

## Executive Summary
- **Overall Accuracy**: {self.results.get('accuracy', 'N/A'):.4f}
- **Overall F1 Score**: {self.results.get('f1_overall', 'N/A'):.4f}
- **Semantic Similarity**: {self.results.get('semantic_similarity', 'N/A'):.4f}
- **BLEU Score**: {self.results.get('bleu_score', 'N/A'):.4f}

## Performance Metrics
- **Average Inference Time**: {self.results.get('avg_inference_time', 'N/A'):.4f} seconds
- **Average Memory Usage**: {self.results.get('avg_memory_usage', 'N/A'):.2f} MB
- **Total Inferences**: {self.results.get('total_inferences', 'N/A')}

## Detailed Metrics
"""
        
        # Add all metrics
        for key, value in self.results.items():
            if isinstance(value, float):
                report += f"- **{key.replace('_', ' ').title()}**: {value:.4f}\n"
            else:
                report += f"- **{key.replace('_', ' ').title()}**: {value}\n"
        
        report += f"""

## Evaluation Date
{datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
"""
        
        # Save report
        with open(f"{output_dir}/evaluation_report.md", 'w', encoding='utf-8') as f:
            f.write(report)

# Usage example
if __name__ == "__main__":
    # Create evaluation config
    eval_config = EvaluationConfig(
        compute_semantic_similarity=True,
        compute_adversarial_robustness=True,
        generate_visualizations=True
    )
    
    # Initialize evaluator
    evaluator = ModernEvaluator(eval_config)
    
    # Example test dataset
    test_dataset = [
        {
            'unpunctuated_text': 'আমি ভালো আছি তুমি কেমন আছো',
            'punctuated_text': 'আমি ভালো আছি। তুমি কেমন আছো?'
        }
    ]
    
    # Evaluate model (placeholder model)
    results = evaluator.evaluate_model(
        model=None,  # Your model here
        test_dataset=test_dataset,
        output_dir="results/modern_evaluation"
    )
    
    print("Evaluation completed!")
    print(json.dumps(results, indent=2))
