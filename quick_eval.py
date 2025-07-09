#!/usr/bin/env python3
"""
Quick test of improved model
"""

import sys
import os
import logging

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

try:
    from config import MODEL_CONFIG, DATASET_CONFIG
    from src.data.dataset_loader import BanglaDatasetLoader
    from src.models.baseline_model import BaselineModel
except ImportError as e:
    print(f"Import error: {e}")
    sys.exit(1)

def quick_evaluation():
    """Quick evaluation of improved model"""
    
    logging.basicConfig(level=logging.INFO)
    logger = logging.getLogger(__name__)
    
    try:
        # Load dataset
        loader = BanglaDatasetLoader()
        dataset = loader.load_generated_dataset()
        if dataset is None:
            dataset = loader.load_original_dataset()
        
        test_data = dataset['test'].select(range(10))  # Just 10 examples
        
        # Load model
        model = BaselineModel(model_type="token_classification")
        model.load_model("models/baseline")
        
        logger.info("Testing improved model on 10 examples:")
        
        correct = 0
        total = 0
        
        for i, example in enumerate(test_data):
            unpunctuated = example['unpunctuated_text']
            punctuated = example['punctuated_text']
            prediction = model.predict(unpunctuated)
            
            total += 1
            if prediction.strip() == punctuated.strip():
                correct += 1
                match_status = "✓"
            else:
                match_status = "✗"
            
            logger.info(f"\nExample {i+1} {match_status}:")
            logger.info(f"Input:     {unpunctuated}")
            logger.info(f"Expected:  {punctuated}")
            logger.info(f"Predicted: {prediction}")
        
        accuracy = correct / total
        logger.info(f"\nAccuracy: {correct}/{total} = {accuracy:.2%}")
        
        return True
        
    except Exception as e:
        logger.error(f"Error: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = quick_evaluation()
    if success:
        print("✓ Quick evaluation completed!")
    else:
        print("✗ Quick evaluation failed!")
        sys.exit(1)
