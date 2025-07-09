#!/usr/bin/env python3
"""
Complete pipeline script for Bangla Punctuation Restoration project
"""

import os
import sys
import subprocess
import argparse
import logging
from datetime import datetime
import json

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'pipeline_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def run_script(script_path, args=None, check=True):
    """Run a Python script with optional arguments"""
    cmd = [sys.executable, script_path]
    if args:
        cmd.extend(args)
    
    logger.info(f"Running: {' '.join(cmd)}")
    
    try:
        result = subprocess.run(cmd, check=check, capture_output=True, text=True)
        logger.info(f"Script completed successfully: {script_path}")
        return True, result.stdout
    except subprocess.CalledProcessError as e:
        logger.error(f"Script failed: {script_path}")
        logger.error(f"Error output: {e.stderr}")
        return False, e.stderr

def check_prerequisites():
    """Check if all prerequisites are met"""
    logger.info("Checking prerequisites...")
    
    # Check if required directories exist
    required_dirs = ['src', 'scripts', 'models', 'data']
    for dir_name in required_dirs:
        if not os.path.exists(dir_name):
            logger.error(f"Required directory not found: {dir_name}")
            return False
    
    # Check if key scripts exist
    required_scripts = [
        'scripts/generate_dataset.py',
        'scripts/train_baseline.py',
        'scripts/evaluate_models.py',
        'scripts/deploy_api.py'
    ]
    
    for script in required_scripts:
        if not os.path.exists(script):
            logger.error(f"Required script not found: {script}")
            return False
    
    logger.info("All prerequisites met!")
    return True

def run_full_pipeline(args):
    """Run the complete pipeline"""
    logger.info("Starting full pipeline...")
    
    pipeline_results = {
        "start_time": datetime.now().isoformat(),
        "steps": [],
        "success": True,
        "final_model_path": None
    }
    
    # Step 1: Generate dataset
    if args.generate_dataset:
        logger.info("Step 1: Generating dataset...")
        step_args = ['--output_dir', args.dataset_output_dir]
        
        if args.min_sentences:
            step_args.extend(['--min_sentences', str(args.min_sentences)])
        if args.upload_datasets:
            step_args.append('--upload_to_hf')
        if args.validate_quality:
            step_args.append('--validate_quality')
        
        success, output = run_script('scripts/generate_dataset.py', step_args)
        pipeline_results["steps"].append({
            "step": "generate_dataset",
            "success": success,
            "output_snippet": output[:500] if output else ""
        })
        
        if not success:
            logger.error("Dataset generation failed. Stopping pipeline.")
            pipeline_results["success"] = False
            return pipeline_results
    
    # Step 2: Train baseline model
    if args.train_baseline:
        logger.info("Step 2: Training baseline model...")
        step_args = [
            '--model_type', args.baseline_model_type,
            '--output_dir', args.baseline_output_dir
        ]
        
        if args.epochs:
            step_args.extend(['--epochs', str(args.epochs)])
        if args.batch_size:
            step_args.extend(['--batch_size', str(args.batch_size)])
        if args.augment_data:
            step_args.append('--augment_data')
        
        success, output = run_script('scripts/train_baseline.py', step_args)
        pipeline_results["steps"].append({
            "step": "train_baseline",
            "success": success,
            "output_snippet": output[:500] if output else ""
        })
        
        if success:
            pipeline_results["final_model_path"] = args.baseline_output_dir
        else:
            logger.error("Baseline training failed. Continuing with evaluation if possible.")
    
    # Step 3: Generate adversarial examples (if baseline was trained)
    if args.generate_adversarial and pipeline_results["final_model_path"]:
        logger.info("Step 3: Generating adversarial examples...")
        # Note: This would require implementing adversarial_attacks.py
        logger.info("Adversarial generation script not implemented in this demo")
        pipeline_results["steps"].append({
            "step": "generate_adversarial",
            "success": False,
            "note": "Script not implemented"
        })
    
    # Step 4: Train advanced model (placeholder)
    if args.train_advanced:
        logger.info("Step 4: Training advanced model...")
        # Note: This would require implementing advanced_model.py
        logger.info("Advanced model training script not implemented in this demo")
        pipeline_results["steps"].append({
            "step": "train_advanced", 
            "success": False,
            "note": "Script not implemented"
        })
    
    # Step 5: Evaluate models
    if args.evaluate_models and pipeline_results["final_model_path"]:
        logger.info("Step 5: Evaluating models...")
        step_args = [
            '--model_paths', pipeline_results["final_model_path"],
            '--model_types', args.baseline_model_type,
            '--output_dir', args.evaluation_output_dir
        ]
        
        success, output = run_script('scripts/evaluate_models.py', step_args)
        pipeline_results["steps"].append({
            "step": "evaluate_models",
            "success": success,
            "output_snippet": output[:500] if output else ""
        })
    
    # Step 6: Deploy API
    if args.deploy_api and pipeline_results["final_model_path"]:
        logger.info("Step 6: Deploying API...")
        if args.create_docker:
            step_args = [
                '--model_path', pipeline_results["final_model_path"],
                '--model_type', args.baseline_model_type,
                '--create_docker'
            ]
            
            success, output = run_script('scripts/deploy_api.py', step_args, check=False)
            pipeline_results["steps"].append({
                "step": "deploy_api",
                "success": success,
                "note": "Docker files created" if success else "Deployment failed"
            })
        else:
            logger.info("API deployment requires manual intervention. Use:")
            logger.info(f"python scripts/deploy_api.py --model_path {pipeline_results['final_model_path']} --model_type {args.baseline_model_type}")
            pipeline_results["steps"].append({
                "step": "deploy_api",
                "success": True,
                "note": "Manual deployment required"
            })
    
    pipeline_results["end_time"] = datetime.now().isoformat()
    return pipeline_results

def save_pipeline_results(results, output_file):
    """Save pipeline results to a JSON file"""
    try:
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(results, f, ensure_ascii=False, indent=2)
        logger.info(f"Pipeline results saved to: {output_file}")
    except Exception as e:
        logger.error(f"Failed to save pipeline results: {e}")

def print_pipeline_summary(results):
    """Print a summary of the pipeline execution"""
    logger.info("\n" + "="*60)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("="*60)
    
    total_steps = len(results["steps"])
    successful_steps = sum(1 for step in results["steps"] if step["success"])
    
    logger.info(f"📊 Total Steps: {total_steps}")
    logger.info(f"✅ Successful: {successful_steps}")
    logger.info(f"❌ Failed: {total_steps - successful_steps}")
    logger.info(f"🎯 Overall Success: {'Yes' if results['success'] else 'No'}")
    
    if results["final_model_path"]:
        logger.info(f"🤖 Final Model: {results['final_model_path']}")
    
    logger.info("\n📋 Step Details:")
    for i, step in enumerate(results["steps"], 1):
        status = "✅" if step["success"] else "❌"
        logger.info(f"{i}. {status} {step['step']}")
        if "note" in step:
            logger.info(f"   Note: {step['note']}")
    
    # Calculate duration
    try:
        start = datetime.fromisoformat(results["start_time"])
        end = datetime.fromisoformat(results["end_time"])
        duration = end - start
        logger.info(f"\n⏱️ Total Duration: {duration}")
    except:
        pass
    
    logger.info("\n💡 Next Steps:")
    if results["final_model_path"]:
        logger.info(f"1. Test your model: python -c \"from src.models.baseline_model import PunctuationRestorer; r=PunctuationRestorer('{results['final_model_path']}'); print(r.restore_punctuation('আমি ভালো আছি'))\"")
        logger.info(f"2. Deploy API: python scripts/deploy_api.py --model_path {results['final_model_path']}")
        logger.info("3. Check evaluation results in results/evaluation/")
    else:
        logger.info("1. Check the logs for errors in model training")
        logger.info("2. Try running individual steps manually")
    
    logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description='Run complete Bangla punctuation restoration pipeline')
    
    # Pipeline control
    parser.add_argument('--generate_dataset', action='store_true',
                       help='Generate new dataset')
    parser.add_argument('--train_baseline', action='store_true',
                       help='Train baseline model')
    parser.add_argument('--generate_adversarial', action='store_true',
                       help='Generate adversarial examples')
    parser.add_argument('--train_advanced', action='store_true',
                       help='Train advanced model')
    parser.add_argument('--evaluate_models', action='store_true',
                       help='Evaluate trained models')
    parser.add_argument('--deploy_api', action='store_true',
                       help='Deploy API service')
    parser.add_argument('--full_pipeline', action='store_true',
                       help='Run complete pipeline (enables all steps)')
    
    # Dataset parameters
    parser.add_argument('--dataset_output_dir', type=str, default='data/generated_dataset',
                       help='Output directory for generated dataset')
    parser.add_argument('--min_sentences', type=int, default=10000,
                       help='Minimum number of sentences to generate')
    parser.add_argument('--upload_datasets', action='store_true',
                       help='Upload datasets to Hugging Face')
    parser.add_argument('--validate_quality', action='store_true',
                       help='Validate dataset quality')
    
    # Model training parameters
    parser.add_argument('--baseline_model_type', type=str, default='token_classification',
                       choices=['token_classification', 'seq2seq'],
                       help='Type of baseline model')
    parser.add_argument('--baseline_output_dir', type=str, default='models/baseline',
                       help='Output directory for baseline model')
    parser.add_argument('--epochs', type=int, default=5,
                       help='Number of training epochs')
    parser.add_argument('--batch_size', type=int, default=16,
                       help='Batch size for training')
    parser.add_argument('--augment_data', action='store_true',
                       help='Apply data augmentation')
    
    # Evaluation parameters
    parser.add_argument('--evaluation_output_dir', type=str, default='results/evaluation',
                       help='Output directory for evaluation results')
    
    # Deployment parameters
    parser.add_argument('--create_docker', action='store_true',
                       help='Create Docker deployment files')
    
    # Pipeline output
    parser.add_argument('--save_results', type=str, default='pipeline_results.json',
                       help='File to save pipeline results')
    
    args = parser.parse_args()
    
    # Enable all steps if full_pipeline is requested
    if args.full_pipeline:
        args.generate_dataset = True
        args.train_baseline = True
        args.evaluate_models = True
        args.deploy_api = True
        args.validate_quality = True
        args.create_docker = True
    
    # Check if at least one step is enabled
    steps_enabled = any([
        args.generate_dataset,
        args.train_baseline,
        args.generate_adversarial,
        args.train_advanced,
        args.evaluate_models,
        args.deploy_api
    ])
    
    if not steps_enabled:
        logger.error("No pipeline steps enabled. Use --full_pipeline or enable specific steps.")
        parser.print_help()
        return 1
    
    # Check prerequisites
    if not check_prerequisites():
        logger.error("Prerequisites not met. Run setup.py first.")
        return 1
    
    # Run pipeline
    try:
        results = run_full_pipeline(args)
        
        # Save results
        if args.save_results:
            save_pipeline_results(results, args.save_results)
        
        # Print summary
        print_pipeline_summary(results)
        
        # Return appropriate exit code
        return 0 if results["success"] else 1
        
    except Exception as e:
        logger.error(f"Pipeline execution failed: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
