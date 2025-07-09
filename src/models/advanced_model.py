#!/usr/bin/env python3
"""
Advanced model for Bangla Punctuation Restoration with improved techniques
"""

import os
import torch
import torch.nn as nn
import numpy as np
from typing import List, Dict, Any, Optional, Union
from transformers import (
    AutoTokenizer, 
    AutoModelForTokenClassification,
    AutoConfig,
    TrainingArguments,
    Trainer,
    EarlyStoppingCallback
)
from datasets import Dataset, DatasetDict
import logging
from config import MODEL_CONFIG, PUNCTUATION_LABELS, ID_TO_SYMBOL
from .baseline_model import BaselineModel

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class AdvancedPunctuationModel(nn.Module):
    """
    Advanced model with additional layers and techniques
    """
    
    def __init__(self, base_model_name: str, num_labels: int, dropout_rate: float = 0.3):
        super().__init__()
        
        # Load base model
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.backbone = AutoModelForTokenClassification.from_pretrained(
            base_model_name, 
            num_labels=num_labels,
            config=self.config
        )
        
        # Additional layers
        hidden_size = self.config.hidden_size
        
        # Bi-LSTM layer for sequence modeling
        self.lstm = nn.LSTM(
            hidden_size, 
            hidden_size // 2, 
            batch_first=True, 
            bidirectional=True,
            dropout=dropout_rate
        )
        
        # Attention mechanism
        self.attention = nn.MultiheadAttention(
            embed_dim=hidden_size,
            num_heads=8,
            dropout=dropout_rate,
            batch_first=True
        )
        
        # Layer normalization
        self.layer_norm = nn.LayerNorm(hidden_size)
        
        # Dropout for regularization
        self.dropout = nn.Dropout(dropout_rate)
        
        # Enhanced classifier
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.ReLU(),
            nn.Dropout(dropout_rate),
            nn.Linear(hidden_size // 2, num_labels)
        )
    
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Get embeddings from backbone (without final classification layer)
        outputs = self.backbone.base_model(
            input_ids=input_ids,
            attention_mask=attention_mask
        )
        
        sequence_output = outputs.last_hidden_state
        
        # Apply LSTM
        lstm_output, _ = self.lstm(sequence_output)
        
        # Apply self-attention
        attended_output, _ = self.attention(lstm_output, lstm_output, lstm_output)
        
        # Residual connection and layer norm
        enhanced_output = self.layer_norm(attended_output + sequence_output)
        
        # Apply dropout
        enhanced_output = self.dropout(enhanced_output)
        
        # Final classification
        logits = self.classifier(enhanced_output)
        
        loss = None
        if labels is not None:
            loss_fct = nn.CrossEntropyLoss(ignore_index=-100)
            loss = loss_fct(logits.view(-1, self.config.num_labels), labels.view(-1))
        
        return {
            'loss': loss,
            'logits': logits,
            'hidden_states': enhanced_output
        }

class AdvancedModel(BaselineModel):
    """
    Advanced model class with enhanced training techniques
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the advanced model
        
        Args:
            config: Model configuration
        """
        self.config = config or MODEL_CONFIG["advanced_model"]
        self.tokenizer = None
        self.model = None
        self.trained = False
    
    def initialize_model(self) -> bool:
        """
        Initialize the advanced model and tokenizer
        
        Returns:
            True if successful, False otherwise
        """
        try:
            model_name = self.config["name"]
            
            # Initialize tokenizer
            self.tokenizer = AutoTokenizer.from_pretrained(model_name)
            
            # Initialize advanced model
            self.model = AdvancedPunctuationModel(
                base_model_name=model_name,
                num_labels=len(PUNCTUATION_LABELS),
                dropout_rate=0.3
            )
            
            logger.info(f"Successfully initialized advanced model: {model_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error initializing advanced model: {e}")
            return False
    
    def create_enhanced_training_arguments(self, output_dir: str) -> TrainingArguments:
        """Create enhanced training arguments with advanced techniques"""
        return TrainingArguments(
            output_dir=f"results/{output_dir}",
            eval_strategy="steps",
            eval_steps=500,
            save_strategy="steps",
            save_steps=500,
            logging_steps=100,
            
            # Learning schedule
            learning_rate=self.config["learning_rate"],
            warmup_ratio=self.config.get("warmup_ratio", 0.1),
            weight_decay=self.config["weight_decay"],
            
            # Batch settings
            per_device_train_batch_size=self.config["batch_size"],
            per_device_eval_batch_size=self.config["batch_size"],
            gradient_accumulation_steps=self.config.get("gradient_accumulation_steps", 1),
            
            # Training settings
            num_train_epochs=self.config["num_epochs"],
            max_grad_norm=1.0,
            
            # Optimization
            fp16=self.config.get("fp16", True),
            dataloader_num_workers=4,
            
            # Model selection
            load_best_model_at_end=True,
            metric_for_best_model="eval_f1",
            greater_is_better=True,
            
            # Regularization
            label_smoothing_factor=0.1,
            
            # Reporting
            report_to="none",
            
            # Early stopping will be handled by callback
        )
    
    def apply_advanced_data_preprocessing(self, dataset: DatasetDict) -> DatasetDict:
        """Apply advanced data preprocessing techniques"""
        logger.info("Applying advanced data preprocessing...")
        
        def advanced_preprocess(examples):
            # Enhanced tokenization with special handling for Bengali
            tokenized = self.tokenizer(
                examples["unpunctuated_text"],
                truncation=True,
                max_length=self.config["max_length"],
                padding="max_length",
                return_tensors="pt"
            )
            
            # Create more sophisticated labels
            labels = []
            for unpunctuated, punctuated in zip(examples["unpunctuated_text"], examples["punctuated_text"]):
                example_labels = self._create_enhanced_labels(unpunctuated, punctuated)
                labels.append(example_labels)
            
            tokenized["labels"] = labels
            return tokenized
        
        # Apply preprocessing to all splits
        processed_dataset = dataset.map(
            advanced_preprocess,
            batched=True,
            remove_columns=dataset["train"].column_names
        )
        
        return processed_dataset
    
    def _create_enhanced_labels(self, unpunctuated: str, punctuated: str) -> List[int]:
        """Create enhanced labels with better alignment"""
        max_length = self.config["max_length"]
        labels = [PUNCTUATION_LABELS["O"]] * max_length
        
        # Tokenize both versions for better alignment
        unpunc_tokens = self.tokenizer.tokenize(unpunctuated)
        punc_tokens = self.tokenizer.tokenize(punctuated)
        
        # Simple approach: find punctuation positions
        # In a production system, you'd use more sophisticated alignment
        for i, char in enumerate(punctuated):
            if char in [",", "।", "?", "!", ";", ":", "-"]:
                # Estimate token position
                char_context = punctuated[:i]
                estimated_pos = len(self.tokenizer.tokenize(char_context))
                
                if estimated_pos < max_length:
                    if char == ",":
                        labels[estimated_pos] = PUNCTUATION_LABELS["COMMA"]
                    elif char == "।":
                        labels[estimated_pos] = PUNCTUATION_LABELS["PERIOD"]
                    elif char == "?":
                        labels[estimated_pos] = PUNCTUATION_LABELS["QUESTION"]
                    elif char == "!":
                        labels[estimated_pos] = PUNCTUATION_LABELS["EXCLAMATION"]
                    elif char == ";":
                        labels[estimated_pos] = PUNCTUATION_LABELS["SEMICOLON"]
                    elif char == ":":
                        labels[estimated_pos] = PUNCTUATION_LABELS["COLON"]
                    elif char == "-":
                        labels[estimated_pos] = PUNCTUATION_LABELS["HYPHEN"]
        
        return labels
    
    def train(self, dataset: DatasetDict, output_dir: str = "models/advanced") -> str:
        """
        Train the advanced model with enhanced techniques
        
        Args:
            dataset: Training dataset
            output_dir: Output directory for saving the model
            
        Returns:
            Path to the saved model
        """
        if not self.initialize_model():
            raise ValueError("Failed to initialize advanced model")
        
        logger.info("Training advanced model with enhanced techniques...")
        
        # Apply advanced preprocessing
        processed_dataset = self.apply_advanced_data_preprocessing(dataset)
        
        # Create training arguments
        training_args = self.create_enhanced_training_arguments(output_dir)
        
        # Enhanced data collator
        from transformers import DataCollatorForTokenClassification
        data_collator = DataCollatorForTokenClassification(tokenizer=self.tokenizer)
        
        # Custom trainer with additional features
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=processed_dataset["train"],
            eval_dataset=processed_dataset["validation"],
            tokenizer=self.tokenizer,
            data_collator=data_collator,
            compute_metrics=self.compute_metrics_token_classification,
            callbacks=[
                EarlyStoppingCallback(
                    early_stopping_patience=self.config.get("early_stopping_patience", 3)
                )
            ]
        )
        
        # Train the model
        trainer.train()
        
        # Save the model
        model_path = output_dir
        os.makedirs(model_path, exist_ok=True)
        trainer.save_model(model_path)
        self.tokenizer.save_pretrained(model_path)
        
        # Evaluate on test set
        if "test" in processed_dataset:
            test_results = trainer.evaluate(processed_dataset["test"], metric_key_prefix="test")
            logger.info(f"Test results: {test_results}")
        
        self.trained = True
        logger.info(f"Advanced model saved to {model_path}")
        return model_path
    
    def predict_with_confidence(self, text: str) -> tuple[str, float]:
        """
        Predict with confidence score
        
        Args:
            text: Input text
            
        Returns:
            Tuple of (punctuated_text, confidence_score)
        """
        if not self.trained or self.model is None:
            raise ValueError("Model not trained or loaded")
        
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
        
        # Get predictions and confidence
        logits = outputs['logits']
        probabilities = torch.softmax(logits, dim=2)
        predictions = torch.argmax(logits, dim=2)[0].tolist()
        confidence = torch.max(probabilities, dim=2)[0].mean().item()
        
        # Convert to text
        tokens = self.tokenizer.convert_ids_to_tokens(inputs.input_ids[0])
        punctuated_text = self._convert_predictions_to_text(tokens, predictions)
        
        return punctuated_text, confidence
    
    def _convert_predictions_to_text(self, tokens: List[str], predictions: List[int]) -> str:
        """Convert token predictions to punctuated text"""
        punctuated_text = ""
        
        for i, (token, pred) in enumerate(zip(tokens, predictions)):
            # Skip special tokens
            if token in [self.tokenizer.cls_token, self.tokenizer.sep_token, self.tokenizer.pad_token]:
                continue
            
            # Add the token
            if token.startswith("##"):
                punctuated_text += token[2:]
            else:
                if punctuated_text:  # Add space except for the first token
                    punctuated_text += " "
                punctuated_text += token
            
            # Add punctuation based on prediction
            if pred in ID_TO_SYMBOL:
                punctuated_text += ID_TO_SYMBOL[pred]
        
        return punctuated_text.strip()

class MultiTaskModel(AdvancedModel):
    """Multi-task model that learns punctuation restoration with auxiliary tasks"""
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        super().__init__(config)
        self.auxiliary_tasks = ["pos_tagging", "ner"]  # Placeholder for future implementation
    
    def initialize_model(self) -> bool:
        """Initialize multi-task model"""
        # This would be implemented with multiple heads for different tasks
        logger.info("Multi-task model initialization not fully implemented in this demo")
        return super().initialize_model()

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
    
    # Initialize and train advanced model
    advanced_model = AdvancedModel()
    model_path = advanced_model.train(dataset)
    
    # Test prediction with confidence
    test_text = "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে"
    result, confidence = advanced_model.predict_with_confidence(test_text)
    print(f"Input: {test_text}")
    print(f"Output: {result}")
    print(f"Confidence: {confidence:.4f}")
