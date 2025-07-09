#!/usr/bin/env python3
"""
Script to generate a new Bangla punctuation dataset from diverse sources
"""

import sys
import os
import argparse
import logging
from datetime import datetime

# Add project root to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from src.data.dataset_generator import BanglaDatasetGenerator
from src.data.data_processor import DataProcessor
from config import GENERATION_CONFIG, DATASET_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'dataset_generation_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def main():
    parser = argparse.ArgumentParser(description='Generate new Bangla punctuation dataset')
    parser.add_argument('--output_dir', type=str, default='data/generated_dataset',
                       help='Output directory for saving the dataset')
    parser.add_argument('--min_sentences', type=int, default=None,
                       help='Minimum number of sentences to generate')
    parser.add_argument('--wikipedia_articles', type=int, default=None,
                       help='Number of Wikipedia articles to scrape')
    parser.add_argument('--literary_works_dir', type=str, default=None,
                       help='Directory containing literary works text files')
    parser.add_argument('--upload_to_hf', action='store_true',
                       help='Upload dataset to Hugging Face Hub')
    parser.add_argument('--dataset_name', type=str, default=None,
                       help='Custom dataset name for Hugging Face')
    parser.add_argument('--validate_quality', action='store_true',
                       help='Validate dataset quality after generation')
    
    args = parser.parse_args()
    
    logger.info("Starting dataset generation...")
    logger.info(f"Output directory: {args.output_dir}")
    
    # Update generation config with command line arguments
    generation_config = GENERATION_CONFIG.copy()
    
    if args.min_sentences:
        generation_config["min_sentences"] = args.min_sentences
    if args.wikipedia_articles:
        generation_config["wikipedia_articles"] = args.wikipedia_articles
    if args.literary_works_dir:
        generation_config["literary_works_dir"] = args.literary_works_dir
    
    logger.info(f"Generation configuration: {generation_config}")
    
    # Initialize generator
    generator = BanglaDatasetGenerator(config=generation_config)
    
    try:
        # Generate dataset
        logger.info("Generating dataset from diverse sources...")
        dataset = generator.generate_dataset()
        
        if dataset is None:
            logger.error("Failed to generate dataset. Exiting.")
            return 1
        
        logger.info("Dataset generation completed!")
        
        # Display dataset statistics
        total_examples = 0
        for split_name, split_data in dataset.items():
            split_size = len(split_data)
            total_examples += split_size
            logger.info(f"{split_name}: {split_size} examples")
        
        logger.info(f"Total examples: {total_examples}")
        
        # Validate quality if requested
        if args.validate_quality:
            logger.info("Validating dataset quality...")
            processor = DataProcessor()
            quality_report = processor.validate_data_quality(dataset)
            
            for split_name, split_report in quality_report.items():
                logger.info(f"{split_name} quality:")
                logger.info(f"  Total examples: {split_report['total_examples']}")
                logger.info(f"  Avg sentence length: {split_report['avg_sentence_length']:.2f}")
                logger.info(f"  Punctuation distribution: {split_report['punctuation_distribution']}")
                logger.info(f"  Quality issues: {split_report['num_quality_issues']}")
                
                if split_report['num_quality_issues'] > 0:
                    logger.warning(f"Quality issues found in {split_name} split:")
                    for issue in split_report['quality_issues'][:5]:  # Show first 5 issues
                        logger.warning(f"  {issue}")
        
        # Save dataset locally
        logger.info(f"Saving dataset to {args.output_dir}...")
        success = generator.save_dataset_locally(dataset, args.output_dir)
        
        if not success:
            logger.error("Failed to save dataset locally.")
            return 1
        
        logger.info("Dataset saved successfully!")
        
        # Upload to Hugging Face if requested
        if args.upload_to_hf:
            logger.info("Uploading dataset to Hugging Face Hub...")
            
            dataset_name = args.dataset_name
            success = generator.upload_to_huggingface(dataset, dataset_name)
            
            if success:
                final_name = dataset_name or DATASET_CONFIG["generated_dataset_name"]
                logger.info(f"Dataset uploaded successfully to: {final_name}")
            else:
                logger.warning("Failed to upload dataset to Hugging Face")
        
        # Save generation metadata
        generation_info = {
            "generation_config": generation_config,
            "dataset_statistics": {
                split_name: {
                    "num_examples": len(split_data),
                    "features": list(split_data.features.keys())
                }
                for split_name, split_data in dataset.items()
            },
            "generation_time": datetime.now().isoformat(),
            "total_examples": total_examples,
            "output_directory": args.output_dir
        }
        
        import json
        metadata_path = os.path.join(args.output_dir, "generation_metadata.json")
        os.makedirs(args.output_dir, exist_ok=True)
        
        with open(metadata_path, 'w', encoding='utf-8') as f:
            json.dump(generation_info, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Generation metadata saved to: {metadata_path}")
        
        # Show some examples
        logger.info("Sample examples from the generated dataset:")
        train_data = dataset["train"]
        for i in range(min(3, len(train_data))):
            example = train_data[i]
            logger.info(f"Example {i+1}:")
            logger.info(f"  Unpunctuated: {example['unpunctuated_text']}")
            logger.info(f"  Punctuated: {example['punctuated_text']}")
        
        logger.info("Dataset generation process completed successfully!")
        return 0
        
    except Exception as e:
        logger.error(f"Error during dataset generation: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
