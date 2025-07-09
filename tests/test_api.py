#!/usr/bin/env python3
"""
Test cases for API implementations
"""

import unittest
import json
from unittest.mock import patch, MagicMock
from fastapi.testclient import TestClient

import sys
import os
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.api.fastapi_server import app
from src.api.gradio_interface import GradioInterface


class TestFastAPIServer(unittest.TestCase):
    """Test cases for FastAPI server"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.client = TestClient(app)
    
    def test_health_endpoint(self):
        """Test health check endpoint"""
        response = self.client.get("/health")
        self.assertEqual(response.status_code, 200)
        data = response.json()
        self.assertEqual(data["status"], "healthy")
    
    @patch('src.api.fastapi_server.model')
    def test_punctuate_endpoint(self, mock_model):
        """Test punctuation restoration endpoint"""
        # Mock model prediction
        mock_model.predict.return_value = ["আমি বাংলাদেশে থাকি।"]
        
        # Test data
        test_data = {
            "text": "আমি বাংলাদেশে থাকি",
            "model_type": "baseline"
        }
        
        response = self.client.post("/punctuate", json=test_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("punctuated_text", data)
        self.assertIn("confidence", data)
        self.assertIn("processing_time", data)
    
    def test_punctuate_endpoint_validation(self):
        """Test input validation for punctuate endpoint"""
        # Test missing text
        response = self.client.post("/punctuate", json={})
        self.assertEqual(response.status_code, 422)
        
        # Test empty text
        response = self.client.post("/punctuate", json={"text": ""})
        self.assertEqual(response.status_code, 400)
    
    @patch('src.api.fastapi_server.model')
    def test_batch_punctuate_endpoint(self, mock_model):
        """Test batch punctuation restoration endpoint"""
        # Mock model prediction
        mock_model.predict.return_value = ["আমি বাংলাদেশে থাকি।", "আপনি কেমন আছেন?"]
        
        # Test data
        test_data = {
            "texts": ["আমি বাংলাদেশে থাকি", "আপনি কেমন আছেন"],
            "model_type": "baseline"
        }
        
        response = self.client.post("/punctuate/batch", json=test_data)
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("results", data)
        self.assertEqual(len(data["results"]), 2)
    
    def test_models_endpoint(self):
        """Test available models endpoint"""
        response = self.client.get("/models")
        self.assertEqual(response.status_code, 200)
        
        data = response.json()
        self.assertIn("available_models", data)
        self.assertIsInstance(data["available_models"], list)


class TestGradioInterface(unittest.TestCase):
    """Test cases for Gradio interface"""
    
    def setUp(self):
        """Set up test fixtures"""
        self.interface = GradioInterface()
    
    @patch('src.api.gradio_interface.AdvancedModel')
    def test_interface_initialization(self, mock_model_class):
        """Test Gradio interface initialization"""
        mock_model = MagicMock()
        mock_model_class.return_value = mock_model
        
        interface = GradioInterface()
        self.assertIsNotNone(interface.model)
    
    @patch.object(GradioInterface, 'model')
    def test_punctuate_text_function(self, mock_model):
        """Test text punctuation function"""
        # Mock model prediction
        mock_model.predict.return_value = ["আমি বাংলাদেশে থাকি।"]
        
        result = self.interface.punctuate_text("আমি বাংলাদেশে থাকি", "baseline")
        
        self.assertIsInstance(result, tuple)
        self.assertEqual(len(result), 2)  # (punctuated_text, confidence)
    
    @patch.object(GradioInterface, 'model')
    def test_analyze_text_function(self, mock_model):
        """Test text analysis function"""
        # Mock model prediction
        mock_model.predict.return_value = ["আমি বাংলাদেশে থাকি।"]
        
        result = self.interface.analyze_text("আমি বাংলাদেশে থাকি")
        
        self.assertIsInstance(result, dict)
        self.assertIn("original_text", result)
        self.assertIn("punctuated_text", result)
        self.assertIn("statistics", result)
    
    def test_create_interface(self):
        """Test Gradio interface creation"""
        with patch('gradio.Interface') as mock_interface:
            self.interface.create_interface()
            mock_interface.assert_called_once()


class TestAPIUtilities(unittest.TestCase):
    """Test cases for API utility functions"""
    
    def test_input_validation(self):
        """Test input validation functions"""
        from src.api.fastapi_server import validate_text_input
        
        # Test valid input
        self.assertTrue(validate_text_input("আমি বাংলাদেশে থাকি"))
        
        # Test invalid inputs
        self.assertFalse(validate_text_input(""))
        self.assertFalse(validate_text_input(None))
        self.assertFalse(validate_text_input("   "))
    
    def test_error_handling(self):
        """Test error handling in API endpoints"""
        # This would test specific error handling scenarios
        # Implementation depends on the specific error handling logic
        pass


if __name__ == '__main__':
    unittest.main()
