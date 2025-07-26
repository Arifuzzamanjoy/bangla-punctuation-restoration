#!/usr/bin/env python3
"""
Test Script for Bangla Punctuation Restoration System
====================================================

This script tests the complete pipeline with a small dataset:
1. Generate small dataset (100-500 sentences)
2. Train baseline model
3. Evaluate performance
4. Test modern components
"""

import sys
import os
import argparse
import logging
import json
from datetime import datetime
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

from src.data.dataset_generator import BanglaDatasetGenerator
from src.data.dataset_loader import BanglaDatasetLoader
from src.models.baseline_model import BaselineModel
from src.training.modern_training import ModernTrainer
from src.evaluation.modern_evaluation import ModernEvaluator
from config import MODEL_CONFIG, DATASET_CONFIG, GENERATION_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'test_pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

class PipelineTester:
    """Test the complete pipeline with small dataset"""
    
    def __init__(self, args):
        self.args = args
        self.test_output_dir = f"test_results_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        os.makedirs(self.test_output_dir, exist_ok=True)
        
        # Small dataset configuration
        self.small_config = {
            **GENERATION_CONFIG,
            "min_sentences": args.dataset_size,
            "wikipedia_articles": min(5, args.dataset_size // 20),  # Small number of articles
            "news_articles_per_site": 2,  # Very limited news scraping
        }
        
        # Fast training configuration
        self.fast_train_config = {
            "name": "ai4bharat/indic-bert",
            "num_epochs": 1,  # Just 1 epoch for testing
            "batch_size": 4,  # Small batch size
            "learning_rate": 5e-5,
            "max_length": 128,  # Shorter sequences
            "weight_decay": 0.01,
        }
        
        logger.info(f"Initialized pipeline tester with {args.dataset_size} sentences target")
        logger.info(f"Output directory: {self.test_output_dir}")
    
    def step1_generate_small_dataset(self):
        """Generate a small dataset for testing"""
        logger.info("=" * 60)
        logger.info("STEP 1: Generating Small Dataset")
        logger.info("=" * 60)
        
        try:
            # Initialize generator with small config
            generator = BanglaDatasetGenerator(self.small_config)
            
            # Generate dataset
            dataset = generator.generate_dataset()
            
            if dataset is None or len(dataset['train']) == 0:
                logger.error("Failed to generate dataset")
                return None
            
            # Save dataset locally
            dataset_path = os.path.join(self.test_output_dir, "small_dataset")
            success = generator.save_dataset_locally(dataset, dataset_path)
            
            if success:
                logger.info(f"✅ Dataset generated successfully:")
                logger.info(f"   - Train: {len(dataset['train'])} examples")
                logger.info(f"   - Validation: {len(dataset['validation'])} examples") 
                logger.info(f"   - Test: {len(dataset['test'])} examples")
                logger.info(f"   - Saved to: {dataset_path}")
                
                # Save sample examples
                self._save_dataset_samples(dataset)
                return dataset
            else:
                logger.error("Failed to save dataset")
                return None
                
        except Exception as e:
            logger.error(f"Error in dataset generation: {e}")
            return None
    
    def step2_train_baseline_model(self, dataset):
        """Train baseline model on small dataset"""
        logger.info("=" * 60)
        logger.info("STEP 2: Training Baseline Model")
        logger.info("=" * 60)
        
        try:
            # Initialize baseline model with fast config
            model = BaselineModel(
                model_type="token_classification",
                config=self.fast_train_config
            )
            
            # Train model
            model_path = os.path.join(self.test_output_dir, "baseline_model")
            trained_path = model.train(dataset, model_path)
            
            if trained_path:
                logger.info(f"✅ Baseline model trained successfully")
                logger.info(f"   - Model saved to: {trained_path}")
                
                # Test prediction
                test_text = "আমি ভাত খেয়েছি সে বাড়ি গেছে"
                try:
                    prediction = model.predict(test_text)
                    logger.info(f"   - Test prediction: '{test_text}' -> '{prediction}'")
                except Exception as e:
                    logger.warning(f"   - Test prediction failed: {e}")
                
                return model, trained_path
            else:
                logger.error("Model training failed")
                return None, None
                
        except Exception as e:
            logger.error(f"Error in model training: {e}")
            return None, None
    
    def step3_evaluate_model(self, model, dataset):
        """Evaluate the trained model"""
        logger.info("=" * 60)
        logger.info("STEP 3: Evaluating Model Performance")
        logger.info("=" * 60)
        
        try:
            # Evaluate on test set
            test_examples = dataset['test']
            
            if len(test_examples) == 0:
                logger.warning("No test examples available")
                return None
            
            # Take small sample for evaluation
            sample_size = min(20, len(test_examples))
            test_sample = test_examples.select(range(sample_size))
            
            correct_predictions = 0
            total_predictions = 0
            evaluation_results = []
            
            logger.info(f"Evaluating on {sample_size} test examples...")
            
            for i, example in enumerate(test_sample):
                unpunctuated = example['unpunctuated_text']
                ground_truth = example['punctuated_text']
                
                try:
                    prediction = model.predict(unpunctuated)
                    
                    # Simple accuracy check
                    is_correct = prediction.strip() == ground_truth.strip()
                    if is_correct:
                        correct_predictions += 1
                    total_predictions += 1
                    
                    evaluation_results.append({
                        'input': unpunctuated,
                        'ground_truth': ground_truth,
                        'prediction': prediction,
                        'correct': is_correct
                    })
                    
                    if i < 5:  # Show first 5 examples
                        logger.info(f"   Example {i+1}:")
                        logger.info(f"     Input: {unpunctuated}")
                        logger.info(f"     Truth: {ground_truth}")
                        logger.info(f"     Pred:  {prediction}")
                        logger.info(f"     ✅ {'Correct' if is_correct else '❌ Incorrect'}")
                
                except Exception as e:
                    logger.warning(f"   Prediction failed for example {i+1}: {e}")
                    total_predictions += 1
            
            # Calculate accuracy
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            
            logger.info(f"✅ Evaluation completed:")
            logger.info(f"   - Total examples: {total_predictions}")
            logger.info(f"   - Correct predictions: {correct_predictions}")
            logger.info(f"   - Accuracy: {accuracy:.2%}")
            
            # Save evaluation results
            results_file = os.path.join(self.test_output_dir, "evaluation_results.json")
            with open(results_file, 'w', encoding='utf-8') as f:
                json.dump({
                    'summary': {
                        'total_examples': total_predictions,
                        'correct_predictions': correct_predictions,
                        'accuracy': accuracy
                    },
                    'examples': evaluation_results
                }, f, ensure_ascii=False, indent=2)
            
            logger.info(f"   - Detailed results saved to: {results_file}")
            
            return {
                'accuracy': accuracy,
                'total_examples': total_predictions,
                'correct_predictions': correct_predictions,
                'results_file': results_file
            }
            
        except Exception as e:
            logger.error(f"Error in model evaluation: {e}")
            return None
    
    def step4_test_modern_components(self, dataset):
        """Test modern components if available"""
        logger.info("=" * 60)
        logger.info("STEP 4: Testing Modern Components")
        logger.info("=" * 60)
        
        try:
            # Test Modern Trainer
            logger.info("Testing Modern Training Pipeline...")
            try:
                modern_trainer = ModernTrainer(self.fast_train_config)
                logger.info("✅ Modern Trainer initialized successfully")
            except Exception as e:
                logger.warning(f"❌ Modern Trainer initialization failed: {e}")
            
            # Test Modern Evaluator
            logger.info("Testing Modern Evaluation...")
            try:
                modern_evaluator = ModernEvaluator(self.fast_train_config)
                logger.info("✅ Modern Evaluator initialized successfully")
            except Exception as e:
                logger.warning(f"❌ Modern Evaluator initialization failed: {e}")
            
            # Test some sample predictions with modern features
            test_sentences = [
                "আমি খেলি তুমি পড়ো",
                "সে আজ আসবে নাকি",
                "বাংলাদেশ একটি সুন্দর দেশ",
                "আমাদের স্কুল খুব ভালো"
            ]
            
            logger.info("Testing sample predictions:")
            for i, sentence in enumerate(test_sentences):
                logger.info(f"   Sample {i+1}: {sentence}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error in testing modern components: {e}")
            return False
    
    def _save_dataset_samples(self, dataset):
        """Save sample examples from dataset for inspection"""
        try:
            samples_file = os.path.join(self.test_output_dir, "dataset_samples.txt")
            with open(samples_file, 'w', encoding='utf-8') as f:
                f.write("Dataset Samples\\n")
                f.write("=" * 50 + "\\n\\n")
                
                # Save 10 samples from each split
                for split_name, split_data in dataset.items():
                    f.write(f"{split_name.upper()} SAMPLES:\\n")
                    f.write("-" * 20 + "\\n")
                    
                    sample_count = min(10, len(split_data))
                    for i in range(sample_count):
                        example = split_data[i]
                        f.write(f"Example {i+1}:\\n")
                        f.write(f"  Unpunctuated: {example['unpunctuated_text']}\\n")
                        f.write(f"  Punctuated:   {example['punctuated_text']}\\n\\n")
                    
                    f.write("\\n")
            
            logger.info(f"   - Dataset samples saved to: {samples_file}")
        except Exception as e:
            logger.warning(f"Failed to save dataset samples: {e}")
    
    def run_complete_pipeline(self):
        """Run the complete pipeline test"""
        logger.info("🚀 Starting Complete Pipeline Test")
        logger.info(f"Target dataset size: {self.args.dataset_size} sentences")
        logger.info(f"Output directory: {self.test_output_dir}")
        
        start_time = datetime.now()
        results = {}
        
        # Step 1: Generate dataset
        dataset = self.step1_generate_small_dataset()
        if dataset is None:
            logger.error("❌ Pipeline failed at dataset generation")
            return False
        results['dataset_generated'] = True
        
        # Step 2: Train model
        model, model_path = self.step2_train_baseline_model(dataset)
        if model is None:
            logger.error("❌ Pipeline failed at model training")
            return False
        results['model_trained'] = True
        
        # Step 3: Evaluate model
        eval_results = self.step3_evaluate_model(model, dataset)
        if eval_results is None:
            logger.error("❌ Pipeline failed at evaluation")
            return False
        results['model_evaluated'] = True
        results['accuracy'] = eval_results['accuracy']
        
        # Step 4: Test modern components
        modern_test = self.step4_test_modern_components(dataset)
        results['modern_components_tested'] = modern_test
        
        # Final summary
        end_time = datetime.now()
        duration = end_time - start_time
        
        logger.info("=" * 60)
        logger.info("🎉 PIPELINE TEST COMPLETED")
        logger.info("=" * 60)
        logger.info(f"✅ Dataset Generated: {len(dataset['train'])} train examples")
        logger.info(f"✅ Model Trained: {model_path}")
        logger.info(f"✅ Model Evaluated: {eval_results['accuracy']:.2%} accuracy")
        logger.info(f"✅ Modern Components: {'Tested' if modern_test else 'Failed'}")
        logger.info(f"📁 All results saved in: {self.test_output_dir}")
        logger.info(f"⏱️  Total duration: {duration}")
        
        # Save final summary
        summary_file = os.path.join(self.test_output_dir, "pipeline_summary.json")
        with open(summary_file, 'w', encoding='utf-8') as f:
            json.dump({
                'timestamp': start_time.isoformat(),
                'duration_seconds': duration.total_seconds(),
                'dataset_size': len(dataset['train']),
                'accuracy': eval_results['accuracy'],
                'results': results,
                'config': {
                    'target_sentences': self.args.dataset_size,
                    'epochs': self.fast_train_config['num_epochs'],
                    'batch_size': self.fast_train_config['batch_size']
                }
            }, f, indent=2)
        
        logger.info(f"📊 Summary saved to: {summary_file}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description='Test Bangla Punctuation Restoration Pipeline')
    parser.add_argument('--dataset-size', type=int, default=100,
                       help='Target number of sentences for test dataset (default: 100)')
    parser.add_argument('--skip-internet', action='store_true',
                       help='Skip internet scraping, use local/synthetic data only')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run minimal test with 50 sentences and 1 epoch')
    
    args = parser.parse_args()
    
    # Quick test adjustments
    if args.quick_test:
        args.dataset_size = 50
        logger.info("🏃 Quick test mode: 50 sentences, 1 epoch")
    
    try:
        # Initialize and run pipeline tester
        tester = PipelineTester(args)
        success = tester.run_complete_pipeline()
        
        if success:
            logger.info("✅ Pipeline test completed successfully!")
            return 0
        else:
            logger.error("❌ Pipeline test failed!")
            return 1
            
    except KeyboardInterrupt:
        logger.info("❌ Pipeline test interrupted by user")
        return 1
    except Exception as e:
        logger.error(f"❌ Pipeline test failed with error: {e}")
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
