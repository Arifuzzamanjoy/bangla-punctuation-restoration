#!/usr/bin/env python3
"""
Script to evaluate all trained models
"""

import sys
import os
import argparse
import json
import logging
from datetime import datetime
from typing import Dict, Any, List

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset_loader import BanglaDatasetLoader
from models.baseline_model import BaselineModel, PunctuationRestorer
from config import MODEL_CONFIG, EVALUATION_CONFIG
import evaluate

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'model_evaluation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class ModelEvaluator:
    """Comprehensive model evaluation class"""
    
    def __init__(self):
        self.loader = BanglaDatasetLoader()
        self.metrics = {
            "accuracy": evaluate.load("accuracy"),
            "precision": evaluate.load("precision"), 
            "recall": evaluate.load("recall"),
            "f1": evaluate.load("f1"),
            "bleu": evaluate.load("bleu"),
            "rouge": evaluate.load("rouge")
        }
    
    def evaluate_punctuation_restoration(self, model: BaselineModel, test_data: List[Dict[str, str]]) -> Dict[str, Any]:
        """Evaluate punctuation restoration performance"""
        predictions = []
        references = []
        
        logger.info(f"Evaluating on {len(test_data)} examples...")
        
        for example in test_data:
            unpunctuated = example["unpunctuated_text"]
            expected = example["punctuated_text"]
            
            try:
                predicted = model.predict(unpunctuated)
                predictions.append(predicted)
                references.append(expected)
            except Exception as e:
                logger.warning(f"Error predicting for text: {unpunctuated[:50]}... Error: {e}")
                predictions.append("")
                references.append(expected)
        
        # Calculate metrics
        results = {}
        
        # BLEU Score
        try:
            bleu_score = self.metrics["bleu"].compute(
                predictions=predictions, 
                references=[[ref] for ref in references]
            )
            results["bleu"] = bleu_score["bleu"]
        except Exception as e:
            logger.warning(f"Error calculating BLEU: {e}")
            results["bleu"] = 0.0
        
        # ROUGE Scores
        try:
            rouge_scores = self.metrics["rouge"].compute(
                predictions=predictions,
                references=references
            )
            results.update({
                "rouge1": rouge_scores["rouge1"],
                "rouge2": rouge_scores["rouge2"], 
                "rougeL": rouge_scores["rougeL"]
            })
        except Exception as e:
            logger.warning(f"Error calculating ROUGE: {e}")
            results.update({"rouge1": 0.0, "rouge2": 0.0, "rougeL": 0.0})
        
        # Sentence-level accuracy (exact match)
        exact_matches = sum(1 for pred, ref in zip(predictions, references) if pred.strip() == ref.strip())
        results["sentence_accuracy"] = exact_matches / len(references)
        
        # Token-level accuracy
        total_tokens = 0
        correct_tokens = 0
        
        for pred, ref in zip(predictions, references):
            pred_tokens = pred.split()
            ref_tokens = ref.split()
            
            # Align tokens (simple approach)
            min_len = min(len(pred_tokens), len(ref_tokens))
            total_tokens += len(ref_tokens)
            correct_tokens += sum(1 for i in range(min_len) if pred_tokens[i] == ref_tokens[i])
        
        results["token_accuracy"] = correct_tokens / total_tokens if total_tokens > 0 else 0.0
        
        # Punctuation-specific metrics
        punctuation_marks = [",", "।", "?", "!", ";", ":", "-"]
        punct_results = {}
        
        for punct in punctuation_marks:
            pred_count = sum(pred.count(punct) for pred in predictions)
            ref_count = sum(ref.count(punct) for ref in references)
            
            # Simple precision/recall for each punctuation
            if ref_count > 0:
                recall = min(pred_count, ref_count) / ref_count
            else:
                recall = 1.0 if pred_count == 0 else 0.0
            
            if pred_count > 0:
                precision = min(pred_count, ref_count) / pred_count
            else:
                precision = 1.0 if ref_count == 0 else 0.0
            
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
            
            punct_results[f"punct_{punct}"] = {
                "precision": precision,
                "recall": recall, 
                "f1": f1,
                "predicted_count": pred_count,
                "reference_count": ref_count
            }
        
        results["punctuation_metrics"] = punct_results
        
        return results
    
    def evaluate_model(self, model_path: str, model_type: str, test_dataset) -> Dict[str, Any]:
        """Evaluate a single model"""
        logger.info(f"Evaluating model: {model_path}")
        
        # Load model
        model = BaselineModel(model_type=model_type)
        if not model.load_model(model_path):
            logger.error(f"Failed to load model from {model_path}")
            return {}
        
        # Convert dataset to list of examples
        test_data = []
        for i in range(len(test_dataset)):
            test_data.append({
                "unpunctuated_text": test_dataset[i]["unpunctuated_text"],
                "punctuated_text": test_dataset[i]["punctuated_text"]
            })
        
        # Evaluate
        results = self.evaluate_punctuation_restoration(model, test_data)
        
        # Add model info
        results["model_info"] = {
            "model_path": model_path,
            "model_type": model_type,
            "evaluation_time": datetime.now().isoformat(),
            "test_examples": len(test_data)
        }
        
        return results
    
    def compare_models(self, model_results: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Compare multiple model results"""
        comparison = {
            "models": [],
            "best_model": {},
            "metric_comparison": {}
        }
        
        key_metrics = ["sentence_accuracy", "token_accuracy", "bleu", "rouge1", "rougeL"]
        
        # Collect model info and metrics
        for result in model_results:
            if not result:
                continue
                
            model_info = result.get("model_info", {})
            model_summary = {
                "model_path": model_info.get("model_path", "unknown"),
                "model_type": model_info.get("model_type", "unknown"),
                "metrics": {metric: result.get(metric, 0.0) for metric in key_metrics}
            }
            comparison["models"].append(model_summary)
        
        # Find best model for each metric
        best_models = {}
        for metric in key_metrics:
            best_score = -1
            best_model = None
            
            for model in comparison["models"]:
                score = model["metrics"].get(metric, 0.0)
                if score > best_score:
                    best_score = score
                    best_model = model["model_path"]
            
            best_models[metric] = {
                "model": best_model,
                "score": best_score
            }
        
        comparison["best_model"] = best_models
        
        # Create metric comparison table
        metric_table = {}
        for metric in key_metrics:
            metric_table[metric] = {}
            for model in comparison["models"]:
                model_name = os.path.basename(model["model_path"])
                metric_table[metric][model_name] = model["metrics"].get(metric, 0.0)
        
        comparison["metric_comparison"] = metric_table
        
        return comparison

def main():
    parser = argparse.ArgumentParser(description='Evaluate trained models')
    parser.add_argument('--model_paths', type=str, nargs='+', 
                       default=['models/baseline'],
                       help='Paths to trained models')
    parser.add_argument('--model_types', type=str, nargs='+',
                       default=['token_classification'],
                       help='Types of models (corresponding to model_paths)')
    parser.add_argument('--test_dataset', type=str, default=None,
                       help='Path to test dataset (uses original if not provided)')
    parser.add_argument('--output_dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    parser.add_argument('--include_adversarial', action='store_true',
                       help='Include adversarial dataset in evaluation')
    
    args = parser.parse_args()
    
    logger.info("Starting model evaluation...")
    
    # Ensure model_types matches model_paths
    if len(args.model_types) == 1 and len(args.model_paths) > 1:
        args.model_types = args.model_types * len(args.model_paths)
    elif len(args.model_types) != len(args.model_paths):
        logger.error("Number of model_types must match number of model_paths")
        return 1
    
    # Initialize evaluator
    evaluator = ModelEvaluator()
    
    # Load test dataset
    if args.test_dataset:
        logger.info(f"Loading test dataset from: {args.test_dataset}")
        test_dataset = evaluator.loader.load_dataset_from_path(args.test_dataset)
    else:
        logger.info("Loading original test dataset...")
        dataset = evaluator.loader.load_original_dataset()
        test_dataset = dataset
    
    if test_dataset is None:
        logger.error("Failed to load test dataset")
        return 1
    
    # Use test split
    if "test" in test_dataset:
        test_data = test_dataset["test"]
    else:
        # Use a portion of the data if no test split
        all_data = list(test_dataset.values())[0]
        test_size = min(1000, len(all_data) // 5)  # Use 20% or max 1000 examples
        test_data = all_data.select(range(test_size))
    
    logger.info(f"Test dataset size: {len(test_data)}")
    
    # Evaluate each model
    all_results = []
    
    for model_path, model_type in zip(args.model_paths, args.model_types):
        if not os.path.exists(model_path):
            logger.warning(f"Model path does not exist: {model_path}")
            continue
            
        logger.info(f"Evaluating {model_type} model from {model_path}")
        
        try:
            results = evaluator.evaluate_model(model_path, model_type, test_data)
            if results:
                all_results.append(results)
                
                # Log key metrics
                logger.info(f"Results for {model_path}:")
                logger.info(f"  Sentence Accuracy: {results.get('sentence_accuracy', 0):.4f}")
                logger.info(f"  Token Accuracy: {results.get('token_accuracy', 0):.4f}")
                logger.info(f"  BLEU Score: {results.get('bleu', 0):.4f}")
                logger.info(f"  ROUGE-L: {results.get('rougeL', 0):.4f}")
            
        except Exception as e:
            logger.error(f"Error evaluating model {model_path}: {e}")
            import traceback
            traceback.print_exc()
    
    if not all_results:
        logger.error("No models were successfully evaluated")
        return 1
    
    # Compare models
    logger.info("Comparing models...")
    comparison = evaluator.compare_models(all_results)
    
    # Save results
    os.makedirs(args.output_dir, exist_ok=True)
    
    # Save individual results
    for i, result in enumerate(all_results):
        model_name = os.path.basename(args.model_paths[i])
        result_file = os.path.join(args.output_dir, f"{model_name}_results.json")
        
        with open(result_file, 'w', encoding='utf-8') as f:
            json.dump(result, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved results to: {result_file}")
    
    # Save comparison
    comparison_file = os.path.join(args.output_dir, "model_comparison.json")
    with open(comparison_file, 'w', encoding='utf-8') as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    
    logger.info(f"Saved comparison to: {comparison_file}")
    
    # Print best models
    logger.info("Best models by metric:")
    for metric, best_info in comparison["best_model"].items():
        logger.info(f"  {metric}: {best_info['model']} (score: {best_info['score']:.4f})")
    
    logger.info("Model evaluation completed successfully!")
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
