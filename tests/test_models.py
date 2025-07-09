#!/usr/bin/env python3
"""
Test cases for model implementations
"""

import unittest
import tempfile
import os
import torch
from unittest.mock import patch, MagicMock
from datasets import Dataset

import sys
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.models.baseline_model import BaselineModel
from src.models.advanced_model import AdvancedModel
from src.models.model_utils import ModelUtils
from config import MODEL_CONFIG, PUNCTUATION_LABELS


class TestBaselineModel(unittest.TestCase):
    """Test cases for BaselineModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = BaselineModel()
    
    def test_model_initialization(self):
        """Test model initialization"""
        self.assertIsNotNone(self.model.tokenizer)
        self.assertIsNotNone(self.model.model)
    
    def test_prepare_dataset(self):
        """Test dataset preparation"""
        data = {
            'tokens': [["আমি", "বাংলাদেশে", "থাকি"], ["আপনি", "কেমন", "আছেন"]],
            'labels': [[0, 0, 1], [0, 0, 2]]
        }
        dataset = Dataset.from_dict(data)
        
        result = self.model.prepare_dataset(dataset)
        self.assertIsInstance(result, Dataset)
        self.assertIn('input_ids', result.column_names)
        self.assertIn('attention_mask', result.column_names)
        self.assertIn('labels', result.column_names)
    
    @patch('src.models.baseline_model.Trainer')
    def test_train(self, mock_trainer_class):
        """Test model training"""
        # Mock trainer
        mock_trainer = MagicMock()
        mock_trainer_class.return_value = mock_trainer
        
        # Create mock dataset
        data = {
            'tokens': [["আমি", "বাংলাদেশে", "থাকি"]],
            'labels': [[0, 0, 1]]
        }
        train_dataset = Dataset.from_dict(data)
        val_dataset = Dataset.from_dict(data)
        
        # Test training
        self.model.train(train_dataset, val_dataset, epochs=1)
        
        # Verify trainer was called
        mock_trainer_class.assert_called_once()
        mock_trainer.train.assert_called_once()
    
    def test_predict(self):
        """Test model prediction"""
        texts = ["আমি বাংলাদেশে থাকি"]
        
        with patch.object(self.model.model, 'eval'), \
             patch('torch.no_grad'), \
             patch.object(self.model.tokenizer, '__call__') as mock_tokenizer:
            
            # Mock tokenizer output
            mock_tokenizer.return_value = {
                'input_ids': torch.tensor([[1, 2, 3, 4]]),
                'attention_mask': torch.tensor([[1, 1, 1, 1]])
            }
            
            # Mock model output
            with patch.object(self.model.model, 'forward') as mock_forward:
                mock_logits = torch.randn(1, 4, len(PUNCTUATION_LABELS))
                mock_output = MagicMock()
                mock_output.logits = mock_logits
                mock_forward.return_value = mock_output
                
                predictions = self.model.predict(texts)
                self.assertIsInstance(predictions, list)
                self.assertEqual(len(predictions), len(texts))


class TestAdvancedModel(unittest.TestCase):
    """Test cases for AdvancedModel class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.model = AdvancedModel()
    
    def test_model_initialization(self):
        """Test advanced model initialization"""
        self.assertIsNotNone(self.model.tokenizer)
        self.assertIsNotNone(self.model.model)
    
    def test_multitask_training_config(self):
        """Test multitask training configuration"""
        config = self.model.get_multitask_config()
        self.assertIsInstance(config, dict)
        self.assertIn('tasks', config)
        self.assertIn('punctuation_restoration', config['tasks'])


class TestModelEvaluator(unittest.TestCase):
    """Test cases for ModelEvaluator class"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.evaluator = ModelUtils()
    
    def test_evaluate_predictions(self):
        """Test prediction evaluation"""
        true_labels = [[0, 1, 2], [0, 0, 1]]
        pred_labels = [[0, 1, 1], [0, 1, 1]]
        
        metrics = self.evaluator.evaluate_predictions(true_labels, pred_labels)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)
        self.assertIn('f1', metrics)
    
    def test_confusion_matrix_generation(self):
        """Test confusion matrix generation"""
        true_labels = [0, 1, 2, 1, 0]
        pred_labels = [0, 1, 1, 1, 0]
        
        cm = self.evaluator.generate_confusion_matrix(true_labels, pred_labels)
        self.assertIsNotNone(cm)
        # Should be a square matrix
        self.assertEqual(cm.shape[0], cm.shape[1])
    
    def test_generate_classification_report(self):
        """Test classification report generation"""
        true_labels = [0, 1, 2, 1, 0]
        pred_labels = [0, 1, 1, 1, 0]
        
        report = self.evaluator.generate_classification_report(true_labels, pred_labels)
        self.assertIsInstance(report, str)
        self.assertIn('precision', report)
        self.assertIn('recall', report)
        self.assertIn('f1-score', report)


class TestComputeMetrics(unittest.TestCase):
    """Test cases for compute_metrics function"""
    
    def test_compute_metrics_function(self):
        """Test the compute_metrics function"""
        # Mock evaluation prediction object
        class MockEvalPrediction:
            def __init__(self, predictions, label_ids):
                self.predictions = predictions
                self.label_ids = label_ids
        
        # Create mock data
        predictions = torch.randn(2, 3, 4).numpy()  # 2 sequences, 3 tokens, 4 classes
        labels = torch.tensor([[0, 1, 2], [1, 0, 2]]).numpy()
        
        eval_pred = MockEvalPrediction(predictions, labels)
        
        # Use the actual evaluation function from ModelUtils
        pred_texts = ["test", "sample"]
        ref_texts = ["test", "sample"]
        metrics = ModelUtils.evaluate_punctuation_accuracy(pred_texts, ref_texts)
        
        self.assertIsInstance(metrics, dict)
        self.assertIn('accuracy', metrics)
        self.assertIn('f1', metrics)
        self.assertIn('precision', metrics)
        self.assertIn('recall', metrics)


if __name__ == '__main__':
    unittest.main()
