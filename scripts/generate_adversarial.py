#!/usr/bin/env python3
"""
Script to generate adversarial examples for robustness testing
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from data.dataset_loader import BanglaDatasetLoader
from data.adversarial_attacks import BanglaAdversarialGenerator, AdversarialDatasetBuilder, AttackConfig
from config import ADVERSARIAL_CONFIG, DATASET_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'adversarial_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate adversarial examples for Bangla punctuation restoration')
    parser.add_argument('--dataset_path', type=str, default=None,
                       help='Path to dataset (uses original if not provided)')
    parser.add_argument('--output_dir', type=str, default='data/adversarial_dataset',
                       help='Output directory for adversarial dataset')
    parser.add_argument('--variants_per_sample', type=int, default=2,
                       help='Number of adversarial variants per original sample')
    parser.add_argument('--attack_ratio', type=float, default=None,
                       help='Ratio of samples to attack')
    parser.add_argument('--upload_to_hf', action='store_true',
                       help='Upload adversarial dataset to Hugging Face')
    parser.add_argument('--evaluate_model', type=str, default=None,
                       help='Path to model for evaluating attack success')
    
    args = parser.parse_args()
    
    logger.info("Starting adversarial example generation...")
    logger.info(f"Output directory: {args.output_dir}")
    logger.info(f"Variants per sample: {args.variants_per_sample}")
    
    # Initialize components
    loader = BanglaDatasetLoader()
    
    # Create attack configuration
    attack_config = AttackConfig()
    if args.attack_ratio:
        attack_config.char_swap_prob = args.attack_ratio * 0.1
        attack_config.char_delete_prob = args.attack_ratio * 0.05
        attack_config.typo_prob = args.attack_ratio * 0.1
    
    generator = BanglaAdversarialGenerator(config=attack_config)
    builder = AdversarialDatasetBuilder(generator)
    
    # Load dataset
    if args.dataset_path:
        logger.info(f"Loading dataset from: {args.dataset_path}")
        dataset = loader.load_dataset_from_path(args.dataset_path)
    else:
        logger.info("Loading original dataset...")
        dataset = loader.load_original_dataset()
    
    if dataset is None:
        logger.error("Failed to load dataset. Exiting.")
        return 1
    
    # Convert dataset to format expected by adversarial generator
    all_texts = []
    all_labels = []
    
    for split_name, split_data in dataset.items():
        for example in split_data:
            all_texts.append(example["unpunctuated_text"])
            all_labels.append(example["punctuated_text"])
    
    logger.info(f"Processing {len(all_texts)} examples")
    
    # Build adversarial dataset
    try:
        original_dataset = {
            'text': all_texts,
            'labels': all_labels
        }
        
        adversarial_dataset = builder.build_adversarial_dataset(
            original_dataset,
            output_path=os.path.join(args.output_dir, "adversarial_data.json"),
            variants_per_sample=args.variants_per_sample
        )
        
        logger.info("Adversarial dataset generation completed!")
        logger.info(f"Generated {len(adversarial_dataset['adversarial_text'])} adversarial examples")
        
        # Save in splits format
        total_samples = len(adversarial_dataset['adversarial_text'])
        train_size = int(total_samples * 0.8)
        val_size = int(total_samples * 0.1)
        
        # Create splits
        splits = {
            'train': {
                'unpunctuated_text': adversarial_dataset['adversarial_text'][:train_size],
                'punctuated_text': adversarial_dataset['adversarial_labels'][:train_size]
            },
            'validation': {
                'unpunctuated_text': adversarial_dataset['adversarial_text'][train_size:train_size+val_size],
                'punctuated_text': adversarial_dataset['adversarial_labels'][train_size:train_size+val_size]
            },
            'test': {
                'unpunctuated_text': adversarial_dataset['adversarial_text'][train_size+val_size:],
                'punctuated_text': adversarial_dataset['adversarial_labels'][train_size+val_size:]
            }
        }
        
        # Save splits
        import json
        os.makedirs(args.output_dir, exist_ok=True)
        
        for split_name, split_data in splits.items():
            output_file = os.path.join(args.output_dir, f"{split_name}.json")
            with open(output_file, 'w', encoding='utf-8') as f:
                json.dump(split_data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved {len(split_data['unpunctuated_text'])} examples to {output_file}")
        
        # Evaluate attack success if model provided
        if args.evaluate_model and os.path.exists(args.evaluate_model):
            logger.info("Evaluating attack success...")
            try:
                from models.baseline_model import PunctuationRestorer
                
                model = PunctuationRestorer(model_path=args.evaluate_model)
                
                # Sample evaluation on a subset
                sample_size = min(100, len(all_texts))
                sample_original = all_texts[:sample_size]
                sample_adversarial = adversarial_dataset['adversarial_text'][:sample_size]
                
                success_metrics = generator.evaluate_attack_success(
                    sample_original, 
                    sample_adversarial,
                    model.model.model,
                    model.model.tokenizer
                )
                
                logger.info(f"Attack success metrics: {success_metrics}")
                
                # Save evaluation results
                eval_file = os.path.join(args.output_dir, "attack_evaluation.json")
                with open(eval_file, 'w', encoding='utf-8') as f:
                    json.dump(success_metrics, f, indent=2)
                
            except Exception as e:
                logger.warning(f"Attack evaluation failed: {e}")
        
        # Upload to Hugging Face if requested
        if args.upload_to_hf:
            logger.info("Uploading to Hugging Face...")
            try:
                from datasets import Dataset, DatasetDict
                from huggingface_hub import login
                
                # Convert to DatasetDict format
                hf_dataset = DatasetDict({
                    split_name: Dataset.from_dict(split_data)
                    for split_name, split_data in splits.items()
                })
                
                # Upload
                dataset_name = DATASET_CONFIG["adversarial_dataset_name"]
                hf_dataset.push_to_hub(dataset_name)
                logger.info(f"Uploaded to Hugging Face: {dataset_name}")
                
            except Exception as e:
                logger.error(f"Failed to upload to Hugging Face: {e}")
        
        # Generate summary report
        summary = {
            "generation_time": datetime.now().isoformat(),
            "attack_config": attack_config.__dict__,
            "dataset_statistics": {
                "original_samples": len(all_texts),
                "adversarial_samples": len(adversarial_dataset['adversarial_text']),
                "variants_per_sample": args.variants_per_sample
            },
            "split_sizes": {
                split: len(data['unpunctuated_text']) 
                for split, data in splits.items()
            }
        }
        
        summary_file = os.path.join(args.output_dir, "generation_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump(summary, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generation summary saved to: {summary_file}")
        logger.info("Adversarial example generation completed successfully!")
        
        return 0
        
    except Exception as e:
        logger.error(f"Error during adversarial generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
