#!/usr/bin/env python3
"""
Data processor utilities for Bangla punctuation restoration
"""

import re
import random
from typing import List, Dict, Any, Tuple, Optional
from datasets import Dataset, DatasetDict
import logging

# Handle config import for different execution contexts
try:
    from config import PUNCTUATION_LABELS, ID_TO_SYMBOL, AUGMENTATION_CONFIG
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from config import PUNCTUATION_LABELS, ID_TO_SYMBOL, AUGMENTATION_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class DataProcessor:
    """
    Utility class for processing and augmenting Bangla text data
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the data processor
        
        Args:
            config: Configuration dictionary for data processing
        """
        self.config = config or AUGMENTATION_CONFIG
        self.punctuation_marks = list(ID_TO_SYMBOL.values())[1:]  # Exclude empty string
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation marks from text"""
        pattern = r'[!?,;:\-।]'
        return re.sub(pattern, '', text).strip()
    
    def clean_text(self, text: str) -> str:
        """Clean and normalize text"""
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text)
        
        # Remove leading/trailing whitespace
        text = text.strip()
        
        # Normalize quotation marks
        text = re.sub(r'["""]', '"', text)
        text = re.sub(r"[''']", "'", text)
        
        return text
    
    def tokenize_bangla_text(self, text: str) -> List[str]:
        """Simple tokenization for Bangla text"""
        # Split on whitespace and punctuation
        tokens = re.findall(r'\S+', text)
        return tokens
    
    def create_token_labels(self, unpunctuated_text: str, punctuated_text: str) -> List[int]:
        """
        Create token-level labels for punctuation restoration
        
        Args:
            unpunctuated_text: Text without punctuation
            punctuated_text: Text with punctuation
            
        Returns:
            List of label IDs for each token
        """
        unpunc_tokens = self.tokenize_bangla_text(unpunctuated_text)
        labels = [PUNCTUATION_LABELS["O"]] * len(unpunc_tokens)
        
        # This is a simplified approach - in practice, you'd need more sophisticated alignment
        # Find punctuation positions in the punctuated text
        punctuation_positions = []
        for i, char in enumerate(punctuated_text):
            if char in self.punctuation_marks:
                punctuation_positions.append((i, char))
        
        # Map punctuation to token positions (simplified)
        for pos, punct in punctuation_positions:
            # Estimate token position based on character position
            preceding_text = punctuated_text[:pos]
            estimated_token_pos = len(self.tokenize_bangla_text(preceding_text)) - 1
            
            if 0 <= estimated_token_pos < len(labels):
                if punct == ",":
                    labels[estimated_token_pos] = PUNCTUATION_LABELS["COMMA"]
                elif punct == "।":
                    labels[estimated_token_pos] = PUNCTUATION_LABELS["PERIOD"]
                elif punct == "?":
                    labels[estimated_token_pos] = PUNCTUATION_LABELS["QUESTION"]
                elif punct == "!":
                    labels[estimated_token_pos] = PUNCTUATION_LABELS["EXCLAMATION"]
                elif punct == ";":
                    labels[estimated_token_pos] = PUNCTUATION_LABELS["SEMICOLON"]
                elif punct == ":":
                    labels[estimated_token_pos] = PUNCTUATION_LABELS["COLON"]
                elif punct == "-":
                    labels[estimated_token_pos] = PUNCTUATION_LABELS["HYPHEN"]
        
        return labels
    
    def apply_character_noise(self, text: str) -> str:
        """Apply character-level noise to text"""
        if not self.config["enable_augmentation"]:
            return text
        
        chars = list(text)
        num_operations = max(1, len(chars) // 20)
        
        for _ in range(num_operations):
            if len(chars) < 3:
                break
            
            operation = random.choice(["swap", "delete", "insert"])
            
            if operation == "swap" and len(chars) >= 2:
                # Swap two adjacent characters
                idx = random.randint(0, len(chars) - 2)
                chars[idx], chars[idx + 1] = chars[idx + 1], chars[idx]
            
            elif operation == "delete" and len(chars) >= 3:
                # Delete a character
                idx = random.randint(0, len(chars) - 1)
                chars.pop(idx)
            
            elif operation == "insert":
                # Insert a random Bengali character
                bengali_chars = self.config["bengali_chars"]
                char_to_insert = random.choice(bengali_chars)
                idx = random.randint(0, len(chars))
                chars.insert(idx, char_to_insert)
        
        return "".join(chars)
    
    def apply_word_deletion(self, text: str) -> str:
        """Apply word-level deletion to text"""
        if not self.config["enable_augmentation"]:
            return text
        
        words = text.split()
        
        if len(words) <= 3:
            return text
        
        # Delete 1-2 random words
        num_to_delete = min(random.randint(1, 2), len(words) - 3)
        indices_to_delete = random.sample(range(len(words)), num_to_delete)
        
        new_words = [word for i, word in enumerate(words) if i not in indices_to_delete]
        return " ".join(new_words)
    
    def apply_word_substitution(self, text: str) -> str:
        """Apply word-level substitution (simplified)"""
        if not self.config["enable_augmentation"]:
            return text
        
        # Simple substitution dictionary for common Bangla words
        substitutions = {
            'আমি': ['আমরা'],
            'তুমি': ['তোমরা', 'আপনি'],
            'সে': ['তারা', 'তিনি'],
            'এটি': ['এগুলি', 'এই'],
            'ভালো': ['চমৎকার', 'সুন্দর', 'উত্তম'],
            'খারাপ': ['মন্দ', 'বাজে'],
            'বড়': ['বিশাল', 'বৃহৎ'],
            'ছোট': ['ক্ষুদ্র', 'সামান্য']
        }
        
        words = text.split()
        
        for i, word in enumerate(words):
            if word in substitutions and random.random() < 0.1:
                words[i] = random.choice(substitutions[word])
        
        return " ".join(words)
    
    def augment_dataset(self, dataset: DatasetDict) -> DatasetDict:
        """
        Apply data augmentation to the training split
        
        Args:
            dataset: Original dataset
            
        Returns:
            Augmented dataset
        """
        if not self.config["enable_augmentation"]:
            return dataset
        
        augmented_dataset = {}
        
        for split_name, split_data in dataset.items():
            if split_name != "train":
                # Don't augment validation and test sets
                augmented_dataset[split_name] = split_data
                continue
            
            logger.info(f"Augmenting {split_name} split...")
            
            # Original data
            unpunctuated_texts = list(split_data["unpunctuated_text"])
            punctuated_texts = list(split_data["punctuated_text"])
            
            # Apply augmentations
            for i, (unpunc_text, punc_text) in enumerate(zip(split_data["unpunctuated_text"], 
                                                             split_data["punctuated_text"])):
                
                # Character noise augmentation
                if random.random() < self.config["character_noise_ratio"]:
                    noisy_text = self.apply_character_noise(unpunc_text)
                    unpunctuated_texts.append(noisy_text)
                    punctuated_texts.append(punc_text)
                
                # Word deletion augmentation
                if random.random() < self.config["word_deletion_ratio"]:
                    deleted_text = self.apply_word_deletion(unpunc_text)
                    unpunctuated_texts.append(deleted_text)
                    punctuated_texts.append(punc_text)
                
                # Word substitution augmentation
                if random.random() < 0.2:
                    substituted_text = self.apply_word_substitution(unpunc_text)
                    unpunctuated_texts.append(substituted_text)
                    punctuated_texts.append(punc_text)
            
            # Create augmented dataset
            augmented_dataset[split_name] = Dataset.from_dict({
                "unpunctuated_text": unpunctuated_texts,
                "punctuated_text": punctuated_texts
            })
            
            logger.info(f"Original {split_name} size: {len(split_data)}")
            logger.info(f"Augmented {split_name} size: {len(augmented_dataset[split_name])}")
        
        return DatasetDict(augmented_dataset)
    
    def validate_data_quality(self, dataset: DatasetDict) -> Dict[str, Any]:
        """
        Validate the quality of the dataset
        
        Args:
            dataset: Dataset to validate
            
        Returns:
            Dictionary containing quality metrics
        """
        quality_report = {}
        
        for split_name, split_data in dataset.items():
            split_report = {
                "total_examples": len(split_data),
                "avg_sentence_length": 0,
                "punctuation_distribution": {},
                "quality_issues": []
            }
            
            sentence_lengths = []
            punctuation_counts = {punct: 0 for punct in self.punctuation_marks}
            
            for i, (unpunc, punc) in enumerate(zip(split_data["unpunctuated_text"], 
                                                  split_data["punctuated_text"])):
                # Check sentence length
                length = len(unpunc.split())
                sentence_lengths.append(length)
                
                # Count punctuation marks
                for punct in self.punctuation_marks:
                    punctuation_counts[punct] += punc.count(punct)
                
                # Check for quality issues
                if length < 3:
                    split_report["quality_issues"].append(f"Example {i}: Too short ({length} words)")
                
                if length > 50:
                    split_report["quality_issues"].append(f"Example {i}: Too long ({length} words)")
                
                # Check if unpunctuated text actually has no punctuation
                if any(punct in unpunc for punct in self.punctuation_marks):
                    split_report["quality_issues"].append(f"Example {i}: Unpunctuated text contains punctuation")
            
            split_report["avg_sentence_length"] = sum(sentence_lengths) / len(sentence_lengths)
            split_report["punctuation_distribution"] = punctuation_counts
            split_report["num_quality_issues"] = len(split_report["quality_issues"])
            
            quality_report[split_name] = split_report
        
        return quality_report
    
    def filter_dataset_by_quality(self, dataset: DatasetDict, 
                                 min_length: int = 3, 
                                 max_length: int = 50) -> DatasetDict:
        """
        Filter dataset based on quality criteria
        
        Args:
            dataset: Dataset to filter
            min_length: Minimum sentence length in words
            max_length: Maximum sentence length in words
            
        Returns:
            Filtered dataset
        """
        filtered_dataset = {}
        
        for split_name, split_data in dataset.items():
            filtered_unpunc = []
            filtered_punc = []
            
            for unpunc, punc in zip(split_data["unpunctuated_text"], 
                                   split_data["punctuated_text"]):
                
                # Check length
                length = len(unpunc.split())
                if not (min_length <= length <= max_length):
                    continue
                
                # Check if unpunctuated text has no punctuation
                if any(punct in unpunc for punct in self.punctuation_marks):
                    continue
                
                # Check if punctuated text has some punctuation
                if not any(punct in punc for punct in self.punctuation_marks):
                    continue
                
                filtered_unpunc.append(unpunc)
                filtered_punc.append(punc)
            
            filtered_dataset[split_name] = Dataset.from_dict({
                "unpunctuated_text": filtered_unpunc,
                "punctuated_text": filtered_punc
            })
            
            logger.info(f"Filtered {split_name}: {len(split_data)} -> {len(filtered_dataset[split_name])}")
        
        return DatasetDict(filtered_dataset)

# Example usage
if __name__ == "__main__":
    # Initialize processor
    processor = DataProcessor()
    
    # Example data
    sample_data = {
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
    }
    
    # Create sample dataset
    sample_dataset = DatasetDict({
        "train": Dataset.from_dict(sample_data),
        "validation": Dataset.from_dict(sample_data),
        "test": Dataset.from_dict(sample_data)
    })
    
    # Validate quality
    quality_report = processor.validate_data_quality(sample_dataset)
    print("Quality report:", quality_report)
    
    # Apply augmentation
    augmented_dataset = processor.augment_dataset(sample_dataset)
    print(f"Augmented train size: {len(augmented_dataset['train'])}")
