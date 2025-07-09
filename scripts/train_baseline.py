#!/usr/bin/env python3
"""
Script to train the baseline model for Bangla punctuation restoration
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset_loader import BanglaDatasetLoader
from src.data.data_processor import DataProcessor
from src.models.baseline_model import BaselineModel, PunctuationRestorer
from config import MODEL_CONFIG, DATASET_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'baseline_training_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Train baseline model for Bangla punctuation restoration')
    parser.add_argument('--model_type', type=str, default='token_classification', 
                       choices=['token_classification', 'seq2seq'],
                       help='Type of model to train')
    parser.add_argument('--output_dir', type=str, default='models/baseline',
                       help='Output directory for saving the model')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to custom dataset (optional)')
    parser.add_argument('--augment_data', action='store_true',
                       help='Apply data augmentation')
    parser.add_argument('--epochs', type=int, default=None,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=None,
                       help='Batch size for training')
    parser.add_argument('--learning_rate', type=float, default=None,
                       help='Learning rate for training')
    
    args = parser.parse_args()
    
    logger.info("Starting baseline model training...")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Initialize components
    loader = BanglaDatasetLoader()
    processor = DataProcessor()
    
    # Load dataset
    if args.dataset_path:
        logger.info(f"Loading dataset from: {args.dataset_path}")
        dataset = loader.load_dataset_from_path(args.dataset_path)
    else:
        logger.info("Loading original hishab dataset...")
        dataset = loader.load_original_dataset()
    
    if dataset is None:
        logger.error("Failed to load dataset. Exiting.")
        return 1
    
    logger.info("Dataset loaded successfully!")
    for split_name, split_data in dataset.items():
        logger.info(f"{split_name}: {len(split_data)} examples")
    
    # Validate and filter dataset
    logger.info("Validating dataset quality...")
    quality_report = processor.validate_data_quality(dataset)
    
    for split_name, split_report in quality_report.items():
        logger.info(f"{split_name} quality:")
        logger.info(f"  Total examples: {split_report['total_examples']}")
        logger.info(f"  Avg sentence length: {split_report['avg_sentence_length']:.2f}")
        logger.info(f"  Quality issues: {split_report['num_quality_issues']}")
    
    # Filter dataset for quality
    dataset = processor.filter_dataset_by_quality(dataset)
    
    # Apply data augmentation if requested
    if args.augment_data:
        logger.info("Applying data augmentation...")
        dataset = processor.augment_dataset(dataset)
    
    # Update model config with command line arguments
    if args.model_type == "token_classification":
        model_config = MODEL_CONFIG["baseline_model"].copy()
    elif args.model_type == "seq2seq":
        model_config = MODEL_CONFIG["seq2seq_model"].copy()
    else:
        logger.error(f"Unknown model type: {args.model_type}")
        return 1
    
    if args.epochs:
        model_config["num_epochs"] = args.epochs
    if args.batch_size:
        model_config["batch_size"] = args.batch_size
    if args.learning_rate:
        model_config["learning_rate"] = args.learning_rate
    
    logger.info(f"Model configuration: {model_config}")
    
    # Initialize and train model
    try:
        model = BaselineModel(model_type=args.model_type, config=model_config)
        
        logger.info("Starting model training...")
        model_path = model.train(dataset, args.output_dir)
        
        logger.info(f"Training completed successfully!")
        logger.info(f"Model saved to: {model_path}")
        
        # Test the model with a sample
        logger.info("Testing the trained model...")
        test_text = "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে"
        
        try:
            result = model.predict(test_text)
            logger.info(f"Test input: {test_text}")
            logger.info(f"Test output: {result}")
        except Exception as e:
            logger.warning(f"Error during test prediction: {e}")
        
        # Save training info
        training_info = {
            "model_type": args.model_type,
            "model_config": model_config,
            "dataset_info": {
                split_name: len(split_data) 
                for split_name, split_data in dataset.items()
            },
            "training_time": datetime.now().isoformat(),
            "model_path": model_path
        }
        
        import json
        info_path = os.path.join(args.output_dir, "training_info.json")
        with open(info_path, 'w', encoding='utf-8') as f:
            json.dump(training_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Training info saved to: {info_path}")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during training: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
