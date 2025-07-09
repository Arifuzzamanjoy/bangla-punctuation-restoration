#!/usr/bin/env python3
"""
Test cases for data processing utilities
"""

import unittest
import tempfile
import os
from unittest.mock import patch, MagicMock
from datasets import Dataset, DatasetDict

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.data_processor import DataProcessor
from src.data.dataset_loader import BanglaDatasetLoader
from src.data.dataset_generator import BanglaDatasetGenerator
from src.data.adversarial_attacks import BanglaAdversarialGenerator
from config import PUNCTUATION_LABELS, ID_TO_SYMBOL


class TestDataProcessor(unittest.TestCase):
    """Test cases for DataProcessor class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.processor = DataProcessor()
    
    def test_remove_punctuation(self):
        """Test punctuation removal"""
        text = "আমি বাংলাদেশে থাকি। আপনি কেমন আছেন?"
        expected = "আমি বাংলাদেশে থাকি আপনি কেমন আছেন"
        result = self.processor.remove_punctuation(text)
        self.assertEqual(result, expected)
    
    def test_clean_text(self):
        """Test text cleaning"""
        text = "  আমি   বাংলায়   কথা  বলি  "
        expected = "আমি বাংলায় কথা বলি"
        result = self.processor.clean_text(text)
        self.assertEqual(result, expected)
    
    def test_tokenize_bangla_text(self):
        """Test Bangla text tokenization"""
        text = "আমি বাংলাদেশে থাকি।"
        tokens = self.processor.tokenize_bangla_text(text)
        self.assertIsInstance(tokens, list)
        self.assertGreater(len(tokens), 0)
    
    def test_create_token_labels(self):
        """Test token label creation"""
        unpunctuated = "আমি বাংলাদেশে থাকি"
        punctuated = "আমি বাংলাদেশে থাকি।"
        labels = self.processor.create_token_labels(unpunctuated, punctuated)
        self.assertIsInstance(labels, list)
        self.assertEqual(len(labels), len(self.processor.tokenize_bangla_text(unpunctuated)))
    
    def test_apply_character_noise(self):
        """Test character noise application"""
        text = "আমি বাংলায় কথা বলি"
        # Test with augmentation disabled
        config = {"enable_augmentation": False}
        processor = DataProcessor(config)
        result = processor.apply_character_noise(text)
        self.assertEqual(result, text)
        
        # Test with augmentation enabled
        config = {"enable_augmentation": True}
        processor = DataProcessor(config)
        result = processor.apply_character_noise(text)
        self.assertIsInstance(result, str)


class TestDatasetLoader(unittest.TestCase):
    """Test cases for DatasetLoader class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.loader = BanglaDatasetLoader()
    
    @patch('src.data.dataset_loader.load_dataset')
    def test_load_huggingface_dataset(self, mock_load_dataset):
        """Test loading dataset from Hugging Face"""
        # Mock the dataset
        mock_dataset = MagicMock()
        mock_load_dataset.return_value = mock_dataset
        
        result = self.loader.load_huggingface_dataset("test/dataset")
        self.assertEqual(result, mock_dataset)
        mock_load_dataset.assert_called_once_with("test/dataset")
    
    def test_load_local_dataset(self):
        """Test loading local dataset"""
        # Create a temporary file
        with tempfile.NamedTemporaryFile(mode='w', suffix='.txt', delete=False) as f:
            f.write("আমি বাংলাদেশে থাকি।\n")
            f.write("আপনি কেমন আছেন?\n")
            temp_file = f.name
        
        try:
            result = self.loader.load_local_dataset(temp_file)
            self.assertIsInstance(result, Dataset)
            self.assertGreater(len(result), 0)
        finally:
            os.unlink(temp_file)
    
    def test_preprocess_dataset(self):
        """Test dataset preprocessing"""
        # Create a mock dataset
        data = {
            'text': ["আমি বাংলাদেশে থাকি।", "আপনি কেমন আছেন?"]
        }
        dataset = Dataset.from_dict(data)
        
        result = self.loader.preprocess_dataset(dataset)
        self.assertIsInstance(result, Dataset)
        self.assertIn('tokens', result.column_names)
        self.assertIn('labels', result.column_names)


class TestDatasetGenerator(unittest.TestCase):
    """Test cases for DatasetGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.generator = BanglaDatasetGenerator()

    def test_generate_synthetic_sentences(self):
        """Test synthetic sentence generation"""
        base_sentences = ["আমি স্কুলে যাই।", "সে বই পড়ে।"]
        patterns = self.generator.generate_synthetic_sentences(base_sentences, 5)
        self.assertIsInstance(patterns, list)
        self.assertEqual(len(patterns), 5)

    def test_generate_dataset(self):
        """Test dataset generation"""
        dataset_dict = self.generator.generate_dataset()
        self.assertIsInstance(dataset_dict, DatasetDict)
        # Check for expected splits
        self.assertIn('train', dataset_dict)
        self.assertIn('validation', dataset_dict)
        self.assertIn('test', dataset_dict)
        self.assertIn('labels', dataset.column_names)


class TestAdversarialAttackGenerator(unittest.TestCase):
    """Test cases for AdversarialAttackGenerator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.attack_generator = BanglaAdversarialGenerator()
    
    def test_character_substitution(self):
        """Test character substitution attack"""
        text = "আমি বাংলায় কথা বলি"
        result = self.attack_generator.character_substitution(text)
        self.assertIsInstance(result, str)
        # Text should be different (with some probability)
        # but we can't guarantee it due to randomness
    
    def test_word_order_shuffle(self):
        """Test word order shuffle attack"""
        text = "আমি বাংলাদেশে থাকি"
        result = self.attack_generator.word_order_shuffle(text)
        self.assertIsInstance(result, str)
        # Check that all words are still present
        original_words = set(text.split())
        result_words = set(result.split())
        self.assertEqual(original_words, result_words)
    
    def test_generate_adversarial_examples(self):
        """Test adversarial example generation"""
        data = {
            'text': ["আমি বাংলাদেশে থাকি।", "আপনি কেমন আছেন?"],
            'tokens': [["আমি", "বাংলাদেশে", "থাকি।"], ["আপনি", "কেমন", "আছেন?"]],
            'labels': [[0, 0, 1], [0, 0, 2]]
        }
        dataset = Dataset.from_dict(data)
        
        result = self.attack_generator.generate_adversarial_examples(dataset, num_samples=1)
        self.assertIsInstance(result, Dataset)
        self.assertGreater(len(result), 0)


if __name__ == '__main__':
    unittest.main()
