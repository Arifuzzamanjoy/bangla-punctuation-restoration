#!/usr/bin/env python3
"""
Model utilities for Bangla punctuation restoration
"""

import os
import torch
import numpy as np
from typing import Dict, Any, List, Optional, Tuple
import logging
from transformers import AutoTokenizer, AutoModelForTokenClassification
from config import PUNCTUATION_LABELS, ID_TO_SYMBOL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class ModelUtils:
    """Utility functions for model operations"""
    
    @staticmethod
    def count_parameters(model) -> int:
        """Count trainable parameters in a model"""
        return sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    @staticmethod
    def get_model_size(model_path: str) -> float:
        """Get model size in MB"""
        if not os.path.exists(model_path):
            return 0.0
        
        total_size = 0
        for dirpath, dirnames, filenames in os.walk(model_path):
            for filename in filenames:
                filepath = os.path.join(dirpath, filename)
                total_size += os.path.getsize(filepath)
        
        return total_size / (1024 * 1024)  # Convert to MB
    
    @staticmethod
    def get_device() -> torch.device:
        """Get the best available device"""
        if torch.cuda.is_available():
            return torch.device("cuda")
        elif torch.backends.mps.is_available():
            return torch.device("mps")
        else:
            return torch.device("cpu")
    
    @staticmethod
    def optimize_model_for_inference(model):
        """Optimize model for inference"""
        model.eval()
        
        # Try to compile the model if PyTorch 2.0+
        try:
            if hasattr(torch, 'compile'):
                model = torch.compile(model)
                logger.info("Model compiled for inference optimization")
        except Exception as e:
            logger.warning(f"Failed to compile model: {e}")
        
        return model
    
    @staticmethod
    def create_model_wrapper_for_textattack(model_path: str, model_type: str = "token_classification"):
        """Create a model wrapper compatible with TextAttack"""
        try:
            from textattack.models.wrappers import HuggingFaceModelWrapper
            
            if model_type == "token_classification":
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForTokenClassification.from_pretrained(model_path)
                
                class PunctuationWrapper(HuggingFaceModelWrapper):
                    def __call__(self, text_inputs):
                        # Convert punctuation restoration to classification format
                        inputs = tokenizer(text_inputs, return_tensors="pt", padding=True, truncation=True)
                        
                        with torch.no_grad():
                            outputs = model(**inputs)
                        
                        # Return dummy classification scores (TextAttack expects classification)
                        batch_size = len(text_inputs)
                        return np.ones((batch_size, 2))  # Binary classification dummy
                
                return PunctuationWrapper(model)
            else:
                logger.warning("TextAttack wrapper not implemented for seq2seq models")
                return None
                
        except ImportError:
            logger.warning("TextAttack not available. Install with: pip install textattack")
            return None
        except Exception as e:
            logger.error(f"Failed to create TextAttack wrapper: {e}")
            return None
    
    @staticmethod
    def evaluate_punctuation_accuracy(predictions: List[str], references: List[str]) -> Dict[str, float]:
        """Evaluate punctuation-specific accuracy"""
        punctuation_marks = [",", "।", "?", "!", ";", ":", "-"]
        results = {}
        
        for punct in punctuation_marks:
            true_positive = 0
            false_positive = 0
            false_negative = 0
            
            for pred, ref in zip(predictions, references):
                pred_count = pred.count(punct)
                ref_count = ref.count(punct)
                
                # Simple counting approach
                true_positive += min(pred_count, ref_count)
                false_positive += max(0, pred_count - ref_count)
                false_negative += max(0, ref_count - pred_count)
            
            # Calculate metrics
            precision = true_positive / (true_positive + false_positive) if (true_positive + false_positive) > 0 else 0
            recall = true_positive / (true_positive + false_negative) if (true_positive + false_negative) > 0 else 0
            f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0
            
            results[f"punct_{punct}_precision"] = precision
            results[f"punct_{punct}_recall"] = recall
            results[f"punct_{punct}_f1"] = f1
        
        # Overall metrics
        all_precisions = [results[k] for k in results.keys() if k.endswith("_precision")]
        all_recalls = [results[k] for k in results.keys() if k.endswith("_recall")]
        all_f1s = [results[k] for k in results.keys() if k.endswith("_f1")]
        
        results["macro_precision"] = np.mean(all_precisions)
        results["macro_recall"] = np.mean(all_recalls)
        results["macro_f1"] = np.mean(all_f1s)
        
        return results
    
    @staticmethod
    def analyze_error_patterns(predictions: List[str], references: List[str]) -> Dict[str, Any]:
        """Analyze common error patterns"""
        error_analysis = {
            "total_examples": len(predictions),
            "perfect_matches": 0,
            "punctuation_errors": {},
            "length_differences": [],
            "common_mistakes": []
        }
        
        punctuation_marks = [",", "।", "?", "!", ";", ":", "-"]
        
        for punct in punctuation_marks:
            error_analysis["punctuation_errors"][punct] = {
                "over_predicted": 0,
                "under_predicted": 0,
                "correct": 0
            }
        
        for i, (pred, ref) in enumerate(zip(predictions, references)):
            # Perfect match check
            if pred.strip() == ref.strip():
                error_analysis["perfect_matches"] += 1
            
            # Length difference
            error_analysis["length_differences"].append(len(pred.split()) - len(ref.split()))
            
            # Punctuation analysis
            for punct in punctuation_marks:
                pred_count = pred.count(punct)
                ref_count = ref.count(punct)
                
                if pred_count > ref_count:
                    error_analysis["punctuation_errors"][punct]["over_predicted"] += 1
                elif pred_count < ref_count:
                    error_analysis["punctuation_errors"][punct]["under_predicted"] += 1
                else:
                    error_analysis["punctuation_errors"][punct]["correct"] += 1
        
        # Calculate statistics
        error_analysis["accuracy"] = error_analysis["perfect_matches"] / error_analysis["total_examples"]
        error_analysis["avg_length_diff"] = np.mean(error_analysis["length_differences"])
        
        return error_analysis
    
    @staticmethod
    def convert_token_predictions_to_text(token_predictions: List[int], 
                                        tokens: List[str], 
                                        tokenizer) -> str:
        """Convert token-level predictions back to punctuated text"""
        punctuated_text = ""
        
        for i, (token, pred_id) in enumerate(zip(tokens, token_predictions)):
            # Skip special tokens
            if token in [tokenizer.cls_token, tokenizer.sep_token, tokenizer.pad_token]:
                continue
            
            # Handle subword tokens
            if token.startswith("##"):
                punctuated_text += token[2:]
            else:
                if punctuated_text:  # Add space except for first token
                    punctuated_text += " "
                punctuated_text += token
            
            # Add punctuation based on prediction
            if pred_id in ID_TO_SYMBOL and ID_TO_SYMBOL[pred_id]:
                punctuated_text += ID_TO_SYMBOL[pred_id]
        
        return punctuated_text.strip()
    
    @staticmethod
    def batch_predict(model, tokenizer, texts: List[str], batch_size: int = 32) -> List[str]:
        """Perform batch prediction efficiently"""
        results = []
        device = ModelUtils.get_device()
        model = model.to(device)
        model.eval()
        
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            
            # Tokenize batch
            inputs = tokenizer(
                batch_texts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=128
            ).to(device)
            
            # Predict
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
            
            # Convert predictions to text
            for j, (text, pred) in enumerate(zip(batch_texts, predictions)):
                tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[j])
                punctuated = ModelUtils.convert_token_predictions_to_text(
                    pred.cpu().tolist(), tokens, tokenizer
                )
                results.append(punctuated)
        
        return results

class ModelEnsemble:
    """Ensemble of multiple models for improved performance"""
    
    def __init__(self, model_paths: List[str], model_types: List[str]):
        """
        Initialize ensemble with multiple models
        
        Args:
            model_paths: List of paths to trained models
            model_types: List of model types corresponding to paths
        """
        self.models = []
        self.tokenizers = []
        
        for path, model_type in zip(model_paths, model_types):
            if os.path.exists(path):
                try:
                    if model_type == "token_classification":
                        tokenizer = AutoTokenizer.from_pretrained(path)
                        model = AutoModelForTokenClassification.from_pretrained(path)
                    else:
                        logger.warning(f"Model type {model_type} not supported in ensemble")
                        continue
                    
                    self.models.append(model)
                    self.tokenizers.append(tokenizer)
                    logger.info(f"Added model to ensemble: {path}")
                    
                except Exception as e:
                    logger.error(f"Failed to load model {path}: {e}")
            else:
                logger.warning(f"Model path does not exist: {path}")
    
    def predict(self, text: str, voting: str = "majority") -> str:
        """
        Predict using ensemble of models
        
        Args:
            text: Input text
            voting: Voting strategy ("majority" or "average")
            
        Returns:
            Punctuated text
        """
        if not self.models:
            raise ValueError("No models loaded in ensemble")
        
        predictions = []
        
        # Get predictions from all models
        for model, tokenizer in zip(self.models, self.tokenizers):
            try:
                inputs = tokenizer(text, return_tensors="pt", truncation=True)
                
                with torch.no_grad():
                    outputs = model(**inputs)
                    pred = torch.argmax(outputs.logits, dim=2)[0]
                
                tokens = tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
                punctuated = ModelUtils.convert_token_predictions_to_text(
                    pred.tolist(), tokens, tokenizer
                )
                predictions.append(punctuated)
                
            except Exception as e:
                logger.warning(f"Model prediction failed: {e}")
        
        if not predictions:
            return text  # Fallback to original text
        
        # Simple ensemble: return the most common prediction
        if voting == "majority":
            from collections import Counter
            return Counter(predictions).most_common(1)[0][0]
        else:
            # For simplicity, return first prediction
            return predictions[0]

# Example usage
if __name__ == "__main__":
    # Test model utilities
    utils = ModelUtils()
    
    # Test device detection
    device = utils.get_device()
    print(f"Using device: {device}")
    
    # Test punctuation accuracy evaluation
    predictions = ["আমি ভালো আছি।", "তুমি কেমন আছো?"]
    references = ["আমি ভালো আছি।", "তুমি কেমন আছো?"]
    
    accuracy = utils.evaluate_punctuation_accuracy(predictions, references)
    print(f"Accuracy metrics: {accuracy}")
    
    # Test error analysis
    error_analysis = utils.analyze_error_patterns(predictions, references)
    print(f"Error analysis: {error_analysis}")
