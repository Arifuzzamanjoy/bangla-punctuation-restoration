#!/usr/bin/env python3
"""
Baseline model for Bangla Punctuation Restoration
"""

import os
import torch
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoModelForSeq2SeqLM,
    DataCollatorForTokenClassification,
    DataCollatorForSeq2Seq,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    TrainingArguments,
    Trainer
)
import evaluate
from datasets import Dataset, DatasetDict
import re
from tqdm import tqdm
import logging
from config import MODEL_CONFIG, PUNCTUATION_LABELS, ID_TO_SYMBOL

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BaselineModel:
    """
    Baseline model for Bangla punctuation restoration using token classification
    """
    
    def __init__(self, model_type: str = "token_classification", config: Optional[Dict[str, Any]] = None):
        """
        Initialize the baseline model
        
        Args:
            model_type: Type of model ("token_classification" or "seq2seq")
            config: Model configuration
        """
        self.model_type = model_type
        
        # Map model type to config key
        if config is None:
            if model_type == "token_classification":
                self.config = MODEL_CONFIG["baseline_model"]
            elif model_type == "seq2seq":
                self.config = MODEL_CONFIG["seq2seq_model"]
            else:
                raise ValueError(f"Unknown model type: {model_type}")
        else:
            self.config = config
            
        self.tokenizer = None
        self.model = None
        self.trained = False
    
    def load_model(self, model_path: str) -> bool:
        """
        Load a trained model from path
        
        Args:
            model_path: Path to the saved model
            
        Returns:
            True if successful, False otherwise
        """
        try:
            if self.model_type == "token_classification":
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForTokenClassification.from_pretrained(model_path)
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_path)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_path)
            
            self.trained = True
            logger.info(f"Successfully loaded model from {model_path}")
            return True
            
        except Exception as e:
            logger.error(f"Error loading model from {model_path}: {e}")
            return False
    
    def initialize_model(self) -> bool:
        """
        Initialize model and tokenizer
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_name = self.config["name"]
            
            if self.model_type == "token_classification":
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForTokenClassification.from_pretrained(
                    model_name,
                    num_labels=len(PUNCTUATION_LABELS)
                )
            else:
                self.tokenizer = AutoTokenizer.from_pretrained(model_name)
                self.model = AutoModelForSeq2SeqLM.from_pretrained(model_name)
            
            logger.info(f"Successfully initialized model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing model: {e}")
            return False
    
    def preprocess_for_token_classification(self, examples: Dict[str, List[str]]) -> Dict[str, Any]:
        """Preprocess data for token classification"""
        max_length = self.config["max_length"]
        
        # Tokenize the unpunctuated text
        tokenized_inputs = self.tokenizer(
            examples["unpunctuated_text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        labels = []
        
        for i, (unpunctuated, punctuated) in enumerate(zip(examples["unpunctuated_text"], examples["punctuated_text"])):
            # Create labels for this example
            example_labels = [PUNCTUATION_LABELS["O"]] * max_length
            
            # Simple approach: find punctuation in the punctuated text
            words_unpunc = unpunctuated.split()
            words_punc = punctuated.split()
            
            # Tokenize both versions
            tokens_unpunc = self.tokenizer.tokenize(unpunctuated)
            
            # Find punctuation positions (simplified approach)
            for j, char in enumerate(punctuated):
                if char in [",", "।", "?", "!", ";", ":", "-"]:
                    # Map to token position (approximate)
                    char_pos = j
                    word_boundaries = []
                    current_pos = 0
                    
                    for word in words_punc:
                        word_start = punctuated.find(word, current_pos)
                        word_end = word_start + len(word)
                        word_boundaries.append((word_start, word_end))
                        current_pos = word_end
                    
                    # Find which word this punctuation belongs to
                    for word_idx, (start, end) in enumerate(word_boundaries):
                        if start <= char_pos <= end + 1:  # Allow punctuation right after word
                            if word_idx < len(example_labels):
                                if char == ",":
                                    example_labels[word_idx] = PUNCTUATION_LABELS["COMMA"]
                                elif char == "।":
                                    example_labels[word_idx] = PUNCTUATION_LABELS["PERIOD"]
                                elif char == "?":
                                    example_labels[word_idx] = PUNCTUATION_LABELS["QUESTION"]
                                elif char == "!":
                                    example_labels[word_idx] = PUNCTUATION_LABELS["EXCLAMATION"]
                                elif char == ";":
                                    example_labels[word_idx] = PUNCTUATION_LABELS["SEMICOLON"]
                                elif char == ":":
                                    example_labels[word_idx] = PUNCTUATION_LABELS["COLON"]
                                elif char == "-":
                                    example_labels[word_idx] = PUNCTUATION_LABELS["HYPHEN"]
                            break
            
            labels.append(example_labels)
        
        tokenized_inputs["labels"] = labels
        return tokenized_inputs
    
    def preprocess_for_seq2seq(self, examples: Dict[str, List[str]]) -> Dict[str, Any]:
        """Preprocess data for sequence-to-sequence"""
        max_length = self.config["max_length"]
        
        # Tokenize source (unpunctuated text)
        model_inputs = self.tokenizer(
            examples["unpunctuated_text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        # Tokenize target (punctuated text)
        labels = self.tokenizer(
            examples["punctuated_text"],
            truncation=True,
            max_length=max_length,
            padding="max_length",
            return_tensors="pt"
        )
        
        model_inputs["labels"] = labels["input_ids"]
        return model_inputs
    
    def compute_metrics_token_classification(self, eval_pred):
        """Compute metrics for token classification"""
        predictions, labels = eval_pred
        predictions = np.argmax(predictions, axis=2)
        
        # Load metrics
        accuracy_metric = evaluate.load("accuracy")
        precision_metric = evaluate.load("precision")
        recall_metric = evaluate.load("recall")
        f1_metric = evaluate.load("f1")
        
        # Flatten arrays and filter out padding
        true_labels = []
        true_predictions = []
        
        for prediction, label in zip(predictions, labels):
            for pred, lab in zip(prediction, label):
                if lab != -100:  # Filter out padding tokens
                    true_labels.append(lab)
                    true_predictions.append(pred)
        
        # Calculate metrics
        result = {
            "accuracy": accuracy_metric.compute(predictions=true_predictions, references=true_labels)["accuracy"],
            "precision": precision_metric.compute(predictions=true_predictions, references=true_labels, average="macro")["precision"],
            "recall": recall_metric.compute(predictions=true_predictions, references=true_labels, average="macro")["recall"],
            "f1": f1_metric.compute(predictions=true_predictions, references=true_labels, average="macro")["f1"]
        }
        
        return result
    
    def compute_metrics_seq2seq(self, eval_pred):
        """Compute metrics for seq2seq"""
        predictions, labels = eval_pred
        
        # Decode predictions and labels
        decoded_preds = self.tokenizer.batch_decode(predictions, skip_special_tokens=True)
        
        # Replace -100 with pad token id
        labels = np.where(labels != -100, labels, self.tokenizer.pad_token_id)
        decoded_labels = self.tokenizer.batch_decode(labels, skip_special_tokens=True)
        
        # BLEU score
        bleu_metric = evaluate.load("bleu")
        bleu_score = bleu_metric.compute(predictions=decoded_preds, references=[[ref] for ref in decoded_labels])
        
        # ROUGE score
        rouge_metric = evaluate.load("rouge")
        rouge_scores = rouge_metric.compute(predictions=decoded_preds, references=decoded_labels)
        
        # Sentence-level accuracy
        exact_matches = sum(pred.strip() == label.strip() for pred, label in zip(decoded_preds, decoded_labels))
        sentence_accuracy = exact_matches / len(decoded_preds)
        
        return {
            "bleu": bleu_score["bleu"],
            "rouge1": rouge_scores["rouge1"],
            "rouge2": rouge_scores["rouge2"],
            "rougeL": rouge_scores["rougeL"],
            "sentence_accuracy": sentence_accuracy
        }
    
    def train(self, dataset: DatasetDict, output_dir: str = "models/baseline") -> str:
        """
        Train the baseline model
        
        Args:
            dataset: Training dataset
            output_dir: Output directory for saving the model
            
        Returns:
            Path to the saved model
        """
        if not self.initialize_model():
            raise ValueError("Failed to initialize model")
        
        logger.info(f"Training {self.model_type} model...")
        
        # Create validation split if it doesn't exist
        if "validation" not in dataset:
            train_test_split = dataset["train"].train_test_split(test_size=0.1, seed=42)
            dataset = DatasetDict({
                "train": train_test_split["train"],
                "validation": train_test_split["test"]
            })
            logger.info(f"Created validation split: train={len(dataset['train'])}, validation={len(dataset['validation'])}")
        
        # Preprocess dataset
        if self.model_type == "token_classification":
            tokenized_dataset = dataset.map(
                self.preprocess_for_token_classification,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
            compute_metrics = self.compute_metrics_token_classification
            
            training_args = TrainingArguments(
                output_dir=f"results/{output_dir}",
                eval_strategy="epoch",
                learning_rate=self.config["learning_rate"],
                per_device_train_batch_size=self.config["batch_size"],
                per_device_eval_batch_size=self.config["batch_size"],
                num_train_epochs=self.config["num_epochs"],
                weight_decay=self.config["weight_decay"],
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="f1",
                report_to="none"
            )
            
            trainer = Trainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
        
        else:  # seq2seq
            tokenized_dataset = dataset.map(
                self.preprocess_for_seq2seq,
                batched=True,
                remove_columns=dataset["train"].column_names
            )
            data_collator = DataCollatorForSeq2Seq(tokenizer=self.tokenizer, model=self.model)
            compute_metrics = self.compute_metrics_seq2seq
            
            training_args = Seq2SeqTrainingArguments(
                output_dir=f"results/{output_dir}",
                eval_strategy="epoch",
                learning_rate=self.config["learning_rate"],
                per_device_train_batch_size=self.config["batch_size"],
                per_device_eval_batch_size=self.config["batch_size"],
                num_train_epochs=self.config["num_epochs"],
                weight_decay=self.config["weight_decay"],
                predict_with_generate=True,
                generation_max_length=self.config.get("generation_max_length", 128),
                save_strategy="epoch",
                load_best_model_at_end=True,
                metric_for_best_model="bleu",
                report_to="none"
            )
            
            trainer = Seq2SeqTrainer(
                model=self.model,
                args=training_args,
                train_dataset=tokenized_dataset["train"],
                eval_dataset=tokenized_dataset["validation"],
                tokenizer=self.tokenizer,
                data_collator=data_collator,
                compute_metrics=compute_metrics,
            )
        
        # Train the model
        trainer.train()
        
        # Save the model
        model_path = output_dir
        os.makedirs(model_path, exist_ok=True)
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Evaluate on test set
        if "test" in tokenized_dataset:
            test_results = trainer.evaluate(tokenized_dataset["test"], metric_key_prefix="test")
            logger.info(f"Test results: {test_results}")
        
        self.trained = True
        logger.info(f"Model saved to {model_path}")
        return model_path
    
    def predict(self, text: str) -> str:
        """
        Predict punctuation for a given text
        
        Args:
            text: Unpunctuated text
            
        Returns:
            Punctuated text
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained or loaded")
        
        if self.model_type == "token_classification":
            return self._predict_token_classification(text)
        else:
            return self._predict_seq2seq(text)
    
    def _predict_token_classification(self, text: str) -> str:
        """Predict using token classification model"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["max_length"]
        )
        
        # Forward pass
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Get predictions
        predictions = torch.argmax(outputs.logits, dim=2)[0].tolist()
        
        # More robust approach: work with the original text and token positions
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        
        # Build result by aligning predictions with original text
        result_text = text  # Start with original text
        punctuation_positions = []
        
        # Find positions where punctuation should be added
        current_pos = 0
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            # Skip special tokens
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue
            
            # For SentencePiece tokenizer, handle subword tokens
            if token.startswith('▁'):  # SentencePiece word boundary
                token_text = token[1:]  # Remove the ▁ prefix
                # Find this token in the original text
                if token_text:
                    token_pos = result_text.find(token_text, current_pos)
                    if token_pos >= 0:
                        current_pos = token_pos + len(token_text)
                        # Add punctuation after this token if predicted
                        if pred in ID_TO_SYMBOL and ID_TO_SYMBOL[pred]:
                            punctuation_positions.append((current_pos, ID_TO_SYMBOL[pred]))
            else:
                # Continuation of previous token
                token_text = token
                if token_text:
                    token_pos = result_text.find(token_text, current_pos)
                    if token_pos >= 0:
                        current_pos = token_pos + len(token_text)
                        # Add punctuation after this token if predicted
                        if pred in ID_TO_SYMBOL and ID_TO_SYMBOL[pred]:
                            punctuation_positions.append((current_pos, ID_TO_SYMBOL[pred]))
        
        # Insert punctuation marks from right to left to preserve positions
        for pos, punct in reversed(punctuation_positions):
            if pos <= len(result_text):
                result_text = result_text[:pos] + punct + result_text[pos:]
        
        return result_text.strip()
    
    def _predict_seq2seq(self, text: str) -> str:
        """Predict using seq2seq model"""
        # Tokenize input
        inputs = self.tokenizer(
            text,
            return_tensors="pt",
            truncation=True,
            max_length=self.config["max_length"]
        )
        
        # Generate output
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                max_length=self.config.get("generation_max_length", 128),
                num_beams=4,
                early_stopping=True
            )
        
        # Decode output
        punctuated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        return punctuated_text


class PunctuationRestorer:
    """
    High-level interface for punctuation restoration
    """
    
    def __init__(self, model_path: Optional[str] = None, model_type: str = "token_classification"):
        """
        Initialize the punctuation restorer
        
        Args:
            model_path: Path to a trained model (optional)
            model_type: Type of model to use
        """
        self.model = BaselineModel(model_type=model_type)
        
        if model_path and os.path.exists(model_path):
            self.model.load_model(model_path)
    
    def restore_punctuation(self, text: str) -> str:
        """
        Restore punctuation in text
        
        Args:
            text: Unpunctuated text
            
        Returns:
            Punctuated text
        """
        if not self.model.trained:
            raise ValueError("Model not trained. Please train or load a model first.")
        
        return self.model.predict(text)
    
    def train_model(self, dataset: DatasetDict, output_dir: str = "models/baseline") -> str:
        """
        Train the model
        
        Args:
            dataset: Training dataset
            output_dir: Output directory for saving the model
            
        Returns:
            Path to saved model
        """
        return self.model.train(dataset, output_dir)


# Example usage
if __name__ == "__main__":
    # Example dataset (you would load real data)
    sample_data = {
        "train": Dataset.from_dict({
            "unpunctuated_text": [
                "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে",
                "তুমি কি আজ বাজারে যাবে",
                "এই বইটি খুবই ভালো"
            ],
            "punctuated_text": [
                "আমি তোমাকে বলেছিলাম, তুমি কেন আসোনি আজ স্কুলে?",
                "তুমি কি আজ বাজারে যাবে?",
                "এই বইটি খুবই ভালো।"
            ]
        }),
        "validation": Dataset.from_dict({
            "unpunctuated_text": ["আমি ভালো আছি"],
            "punctuated_text": ["আমি ভালো আছি।"]
        }),
        "test": Dataset.from_dict({
            "unpunctuated_text": ["তুমি কেমন আছো"],
            "punctuated_text": ["তুমি কেমন আছো?"]
        })
    }
    
    dataset = DatasetDict(sample_data)
    
    # Initialize and train model
    restorer = PunctuationRestorer()
    model_path = restorer.train_model(dataset)
    
    # Test prediction
    test_text = "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে"
    result = restorer.restore_punctuation(test_text)
    print(f"Input: {test_text}")
    print(f"Output: {result}")
