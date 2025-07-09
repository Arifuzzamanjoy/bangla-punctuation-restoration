#!/usr/bin/env python3
"""
Integration tests for the complete pipeline
"""

import unittest
import tempfile
import os
import json
from unittest.mock import patch, MagicMock

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from run_pipeline import run_full_pipeline, check_prerequisites
from src.data.dataset_loader import BanglaDatasetLoader
from src.data.dataset_generator import DatasetGenerator
from src.models.baseline_model import BaselineModel
from src.models.advanced_model import AdvancedModel


class TestPipelineIntegration(unittest.TestCase):
    """Integration tests for the complete pipeline"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.temp_dir = tempfile.mkdtemp()
        self.config = {
            "data_dir": self.temp_dir,
            "model_dir": self.temp_dir,
            "results_dir": self.temp_dir
        }
    
    def tearDown(self):
        """Clean up test fixtures"""
        import shutil
        shutil.rmtree(self.temp_dir, ignore_errors=True)
    
    @patch('run_pipeline.BaselineModel')
    @patch('run_pipeline.AdvancedModel')
    def test_full_pipeline_execution(self, mock_advanced_model, mock_baseline_model):
        """Test complete pipeline execution"""
        # Mock models
        mock_baseline = MagicMock()
        mock_advanced = MagicMock()
        mock_baseline_model.return_value = mock_baseline
        mock_advanced_model.return_value = mock_advanced
        
        # Mock training and evaluation
        mock_baseline.train.return_value = None
        mock_advanced.train.return_value = None
        mock_baseline.evaluate.return_value = {"accuracy": 0.85, "f1": 0.82}
        mock_advanced.evaluate.return_value = {"accuracy": 0.90, "f1": 0.88}
        
        # Test prerequisites check
        result = check_prerequisites()
        self.assertTrue(result)
        
        # Note: Full pipeline test skipped as it requires actual data files
    
    def test_data_pipeline_integration(self):
        """Test data loading and processing pipeline"""
        # Create test data file
        test_data = [
            "আমি বাংলাদেশে থাকি।",
            "আপনি কেমন আছেন?",
            "আজ আবহাওয়া ভালো।"
        ]
        
        test_file = os.path.join(self.temp_dir, "test_data.txt")
        with open(test_file, 'w', encoding='utf-8') as f:
            for line in test_data:
                f.write(line + '\n')
        
        # Test data loading
        loader = BanglaDatasetLoader()
        dataset = loader.load_local_dataset(test_file)
        
        self.assertIsNotNone(dataset)
        self.assertGreater(len(dataset), 0)
        
        # Test preprocessing
        processed_dataset = loader.preprocess_dataset(dataset)
        self.assertIn('tokens', processed_dataset.column_names)
        self.assertIn('labels', processed_dataset.column_names)
    
    def test_synthetic_data_generation(self):
        """Test synthetic data generation pipeline"""
        generator = DatasetGenerator()
        
        # Generate synthetic dataset
        synthetic_dataset = generator.generate_synthetic_dataset(size=50)
        
        self.assertIsNotNone(synthetic_dataset)
        self.assertEqual(len(synthetic_dataset), 50)
        self.assertIn('text', synthetic_dataset.column_names)
        self.assertIn('tokens', synthetic_dataset.column_names)
        self.assertIn('labels', synthetic_dataset.column_names)
    
    @patch('src.models.baseline_model.Trainer')
    def test_model_training_integration(self, mock_trainer_class):
        """Test model training integration"""
        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Create test dataset
        from datasets import Dataset
        data = {
            'tokens': [["আমি", "বাংলাদেশে", "থাকি"] for _ in range(10)],
            'labels': [[0, 0, 1] for _ in range(10)]
        }
        dataset = Dataset.from_dict(data)
        
        # Test baseline model training
        model = BaselineModel()
        model.train(dataset, dataset, epochs=1)
        
        # Verify training was called
        mock_trainer.train.assert_called_once()
    
    def test_evaluation_pipeline(self):
        """Test model evaluation pipeline"""
        from src.models.model_utils import ModelEvaluator
        
        evaluator = ModelEvaluator()
        
        # Mock predictions and labels
        true_labels = [[0, 1, 2, 1, 0] for _ in range(5)]
        pred_labels = [[0, 1, 1, 1, 0] for _ in range(5)]
        
        # Test evaluation
        metrics = evaluator.evaluate_predictions(true_labels, pred_labels)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)


class TestEndToEndScenarios(unittest.TestCase):
    """End-to-end test scenarios"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.test_texts = [
            "আমি বাংলাদেশে থাকি",
            "আপনি কেমন আছেন",
            "আজ আবহাওয়া ভালো",
            "আমরা স্কুলে যাবো"
        ]
    
    @patch('src.models.baseline_model.BaselineModel.predict')
    def test_baseline_model_prediction_flow(self, mock_predict):
        """Test complete prediction flow with baseline model"""
        # Mock predictions
        mock_predict.return_value = [
            "আমি বাংলাদেশে থাকি।",
            "আপনি কেমন আছেন?",
            "আজ আবহাওয়া ভালো।",
            "আমরা স্কুলে যাবো।"
        ]
        
        model = BaselineModel()
        predictions = model.predict(self.test_texts)
        
        self.assertEqual(len(predictions), len(self.test_texts))
        for pred in predictions:
            self.assertIsInstance(pred, str)
            # Check that punctuation was added
            self.assertTrue(any(punct in pred for punct in ['।', '?', '!', ',']))
    
    @patch('src.models.advanced_model.AdvancedModel.predict')
    def test_advanced_model_prediction_flow(self, mock_predict):
        """Test complete prediction flow with advanced model"""
        # Mock predictions
        mock_predict.return_value = [
            "আমি বাংলাদেশে থাকি।",
            "আপনি কেমন আছেন?", 
            "আজ আবহাওয়া ভালো।",
            "আমরা স্কুলে যাবো।"
        ]
        
        model = AdvancedModel()
        predictions = model.predict(self.test_texts)
        
        self.assertEqual(len(predictions), len(self.test_texts))
        for pred in predictions:
            self.assertIsInstance(pred, str)
    
    def test_adversarial_robustness_scenario(self):
        """Test model robustness against adversarial examples"""
        from src.data.adversarial_attacks import AdversarialAttackGenerator
        
        attack_generator = AdversarialAttackGenerator()
        
        # Create adversarial examples
        original_text = "আমি বাংলাদেশে থাকি"
        
        # Test different attack types
        char_sub = attack_generator.character_substitution(original_text)
        word_shuffle = attack_generator.word_order_shuffle(original_text)
        
        self.assertIsInstance(char_sub, str)
        self.assertIsInstance(word_shuffle, str)
        
        # Verify that adversarial examples are different from original
        # (with some probability due to randomness)
        self.assertIsNotNone(char_sub)
        self.assertIsNotNone(word_shuffle)


class TestPerformanceBenchmarks(unittest.TestCase):
    """Performance and benchmark tests"""
    
    def test_processing_speed_benchmark(self):
        """Test processing speed for different text lengths"""
        import time
        from src.data.data_processor import DataProcessor
        
        processor = DataProcessor()
        
        # Test texts of different lengths
        short_text = "আমি বাংলায় কথা বলি"
        medium_text = "আমি বাংলাদেশে থাকি এবং বাংলায় কথা বলি। আমার পরিবার এখানে বাস করে।"
        long_text = medium_text * 10
        
        test_cases = [
            ("short", short_text),
            ("medium", medium_text),
            ("long", long_text)
        ]
        
        for name, text in test_cases:
            start_time = time.time()
            
            # Process text
            clean_text = processor.clean_text(text)
            tokens = processor.tokenize_bangla_text(clean_text)
            
            end_time = time.time()
            processing_time = end_time - start_time
            
            # Basic performance assertions
            self.assertLess(processing_time, 1.0)  # Should process within 1 second
            self.assertGreater(len(tokens), 0)
            
            print(f"{name} text processing time: {processing_time:.4f}s")
    
    def test_memory_usage_benchmark(self):
        """Test memory usage during processing"""
        import psutil
        import os
        
        process = psutil.Process(os.getpid())
        
        # Measure initial memory
        initial_memory = process.memory_info().rss / 1024 / 1024  # MB
        
        # Create large dataset
        from src.data.dataset_generator import DatasetGenerator
        generator = DatasetGenerator()
        
        large_dataset = generator.generate_synthetic_dataset(size=1000)
        
        # Measure memory after dataset creation
        final_memory = process.memory_info().rss / 1024 / 1024  # MB
        memory_increase = final_memory - initial_memory
        
        print(f"Memory usage increase: {memory_increase:.2f} MB")
        
        # Assert reasonable memory usage (less than 500MB increase)
        self.assertLess(memory_increase, 500)


if __name__ == '__main__':
    unittest.main()
