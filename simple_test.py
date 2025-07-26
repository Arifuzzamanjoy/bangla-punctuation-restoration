#!/usr/bin/env python3
"""
Simple test for Bangla Punctuation Restoration System
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def test_imports():
    """Test if all imports work"""
    print("Testing imports...")
    
    try:
        from src.data.dataset_generator import BanglaDatasetGenerator
        print("✅ BanglaDatasetGenerator imported")
    except Exception as e:
        print(f"❌ BanglaDatasetGenerator failed: {e}")
        return False
    
    try:
        from src.models.baseline_model import BaselineModel
        print("✅ BaselineModel imported")
    except Exception as e:
        print(f"❌ BaselineModel failed: {e}")
        return False
    
    try:
        from config import MODEL_CONFIG, PUNCTUATION_LABELS
        print("✅ Config imported")
    except Exception as e:
        print(f"❌ Config failed: {e}")
        return False
    
    return True

def test_simple_dataset():
    """Test creating a simple dataset"""
    print("\nTesting simple dataset creation...")
    
    try:
        from src.data.dataset_generator import BanglaDatasetGenerator
        from datasets import Dataset, DatasetDict
        
        # Create simple test data
        test_sentences = [
            "আমি ভাত খাই।",
            "তুমি কেমন আছো?",
            "সে খুব ভালো!",
            "আমরা স্কুলে যাই।",
            "বাংলাদেশ সুন্দর দেশ।"
        ]
        
        # Create simple dataset manually
        unpunctuated = [s.replace('।', '').replace('?', '').replace('!', '') for s in test_sentences]
        
        dataset = DatasetDict({
            'train': Dataset.from_dict({
                'unpunctuated_text': unpunctuated[:3],
                'punctuated_text': test_sentences[:3]
            }),
            'validation': Dataset.from_dict({
                'unpunctuated_text': unpunctuated[3:4],
                'punctuated_text': test_sentences[3:4]
            }),
            'test': Dataset.from_dict({
                'unpunctuated_text': unpunctuated[4:5],
                'punctuated_text': test_sentences[4:5]
            })
        })
        
        print(f"✅ Created simple dataset:")
        print(f"   - Train: {len(dataset['train'])} examples")
        print(f"   - Validation: {len(dataset['validation'])} examples")
        print(f"   - Test: {len(dataset['test'])} examples")
        
        return dataset
        
    except Exception as e:
        print(f"❌ Simple dataset creation failed: {e}")
        return None

def test_model_initialization():
    """Test model initialization"""
    print("\nTesting model initialization...")
    
    try:
        from src.models.baseline_model import BaselineModel
        
        # Simple config
        config = {
            "name": "ai4bharat/indic-bert",
            "max_length": 64,
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
        }
        
        model = BaselineModel(model_type="token_classification", config=config)
        success = model.initialize_model()
        
        if success:
            print("✅ Model initialized successfully")
            return model
        else:
            print("❌ Model initialization failed")
            return None
            
    except Exception as e:
        print(f"❌ Model initialization error: {e}")
        return None

def test_simple_prediction():
    """Test simple prediction"""
    print("\nTesting simple prediction...")
    
    try:
        dataset = test_simple_dataset()
        if dataset is None:
            return False
        
        model = test_model_initialization()
        if model is None:
            return False
        
        # Try simple training for 1 step
        print("Training model for 1 epoch...")
        model_path = model.train(dataset, "simple_test_model")
        
        if model_path:
            print("✅ Model trained successfully")
            
            # Test prediction
            test_text = "আমি ভাত খেয়েছি"
            prediction = model.predict(test_text)
            print(f"✅ Prediction: '{test_text}' -> '{prediction}'")
            return True
        else:
            print("❌ Model training failed")
            return False
            
    except Exception as e:
        print(f"❌ Prediction test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 Running Simple Bangla Punctuation Restoration Test")
    print("=" * 60)
    
    success = True
    
    # Test 1: Imports
    if not test_imports():
        success = False
    
    # Test 2: Simple prediction pipeline
    if not test_simple_prediction():
        success = False
    
    print("\n" + "=" * 60)
    if success:
        print("🎉 All tests passed!")
        return 0
    else:
        print("❌ Some tests failed!")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
