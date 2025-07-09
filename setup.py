#!/usr/bin/env python3
"""
Setup script for Bangla Punctuation Restoration project
"""

import os
import sys
import subprocess
import logging
import argparse
from pathlib import Path

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

def run_command(command, check=True, shell=False):
    """Run a shell command and return the result"""
    try:
        if shell:
            result = subprocess.run(command, shell=True, check=check, capture_output=True, text=True)
        else:
            result = subprocess.run(command.split(), check=check, capture_output=True, text=True)
        return result
    except subprocess.CalledProcessError as e:
        logger.error(f"Command failed: {command}")
        logger.error(f"Error: {e.stderr}")
        return None

def check_python_version():
    """Check if Python version is compatible"""
    version = sys.version_info
    if version.major < 3 or (version.major == 3 and version.minor < 8):
        logger.error("Python 3.8 or higher is required")
        return False
    logger.info(f"Python version: {version.major}.{version.minor}.{version.micro}")
    return True

def install_dependencies():
    """Install required dependencies"""
    logger.info("Installing dependencies...")
    
    # Upgrade pip first
    logger.info("Upgrading pip...")
    result = run_command("python -m pip install --upgrade pip")
    if result is None:
        logger.warning("Failed to upgrade pip")
    
    # Install requirements
    if os.path.exists("requirements.txt"):
        logger.info("Installing requirements from requirements.txt...")
        result = run_command("pip install -r requirements.txt")
        if result is None:
            logger.error("Failed to install requirements")
            return False
    else:
        logger.warning("requirements.txt not found. Installing basic dependencies...")
        basic_deps = [
            "datasets>=2.14.0",
            "huggingface_hub>=0.16.4",
            "transformers>=4.30.0",
            "torch>=2.0.0",
            "evaluate>=0.4.0",
            "pandas>=1.5.0",
            "numpy>=1.24.0",
            "gradio>=3.40.0",
            "fastapi>=0.100.0",
            "uvicorn>=0.23.0",
            "nltk>=3.8.0",
            "scikit-learn>=1.3.0",
            "tqdm>=4.65.0"
        ]
        
        for dep in basic_deps:
            logger.info(f"Installing {dep}...")
            result = run_command(f"pip install {dep}")
            if result is None:
                logger.warning(f"Failed to install {dep}")
    
    logger.info("Dependencies installation completed")
    return True

def setup_directories():
    """Create necessary directories"""
    logger.info("Setting up directories...")
    
    directories = [
        "data",
        "models", 
        "results",
        "reports",
        "notebooks",
        "tests"
    ]
    
    for directory in directories:
        Path(directory).mkdir(exist_ok=True)
        logger.info(f"Created directory: {directory}")
    
    # Create subdirectories
    subdirs = [
        "data/original_dataset",
        "data/generated_dataset", 
        "data/adversarial_dataset",
        "models/baseline",
        "models/advanced",
        "results/baseline",
        "results/advanced",
        "results/evaluation"
    ]
    
    for subdir in subdirs:
        Path(subdir).mkdir(parents=True, exist_ok=True)
        logger.info(f"Created subdirectory: {subdir}")

def download_nltk_data():
    """Download required NLTK data"""
    logger.info("Downloading NLTK data...")
    
    try:
        import nltk
        nltk.download('punkt', quiet=True)
        nltk.download('stopwords', quiet=True)
        logger.info("NLTK data downloaded successfully")
    except Exception as e:
        logger.warning(f"Failed to download NLTK data: {e}")

def check_gpu_availability():
    """Check if GPU is available for training"""
    try:
        import torch
        if torch.cuda.is_available():
            gpu_count = torch.cuda.device_count()
            gpu_name = torch.cuda.get_device_name(0)
            logger.info(f"GPU available: {gpu_name} (Count: {gpu_count})")
            return True
        else:
            logger.info("No GPU available. Will use CPU for training.")
            return False
    except ImportError:
        logger.warning("PyTorch not installed. Cannot check GPU availability.")
        return False

def create_example_config():
    """Create an example configuration file"""
    logger.info("Creating example configuration...")
    
    config_content = '''# Example environment variables for Bangla Punctuation Restoration

# Hugging Face token for accessing private datasets/models
export HUGGINGFACE_TOKEN="your_huggingface_token_here"

# Optional: Set custom cache directory for Hugging Face
export TRANSFORMERS_CACHE="./cache/transformers"
export HF_DATASETS_CACHE="./cache/datasets"

# Optional: CUDA settings
export CUDA_VISIBLE_DEVICES="0"

# Optional: Python path
export PYTHONPATH="${PYTHONPATH}:$(pwd)/src"
'''
    
    with open(".env.example", "w") as f:
        f.write(config_content)
    
    logger.info("Created .env.example file. Copy to .env and customize as needed.")

def validate_installation():
    """Validate that the installation is working correctly"""
    logger.info("Validating installation...")
    
    try:
        # Test imports
        import datasets
        import transformers
        import torch
        import gradio
        import fastapi
        logger.info("✅ All major dependencies imported successfully")
        
        # Test model loading (basic)
        from transformers import AutoTokenizer
        tokenizer = AutoTokenizer.from_pretrained("bert-base-uncased")
        logger.info("✅ Tokenizer loading test passed")
        
        # Test dataset loading
        from datasets import Dataset
        sample_data = {"text": ["Hello world"], "label": [1]}
        dataset = Dataset.from_dict(sample_data)
        logger.info("✅ Dataset creation test passed")
        
        logger.info("🎉 Installation validation completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Installation validation failed: {e}")
        return False

def run_quick_test():
    """Run a quick test of the system"""
    logger.info("Running quick system test...")
    
    try:
        # Add src to path for testing
        sys.path.insert(0, os.path.join(os.getcwd(), 'src'))
        
        # Test configuration loading
        import config
        logger.info("✅ Configuration loaded successfully")
        
        # Test data processor
        from data.data_processor import DataProcessor
        processor = DataProcessor()
        
        # Test text processing
        test_text = "আমি ভালো আছি"
        cleaned_text = processor.clean_text(test_text)
        logger.info(f"✅ Text processing test passed: '{test_text}' -> '{cleaned_text}'")
        
        logger.info("🎉 Quick system test completed successfully!")
        return True
        
    except Exception as e:
        logger.error(f"❌ Quick system test failed: {e}")
        logger.error("This might be normal if you haven't set up the project structure yet.")
        return False

def print_next_steps():
    """Print next steps for the user"""
    logger.info("\n" + "="*60)
    logger.info("🎉 SETUP COMPLETED SUCCESSFULLY!")
    logger.info("="*60)
    logger.info("\n📋 Next Steps:")
    logger.info("1. Set up your Hugging Face token:")
    logger.info("   - Copy .env.example to .env")
    logger.info("   - Add your Hugging Face token")
    logger.info("   - Or run: huggingface-cli login")
    logger.info("\n2. Generate/load dataset:")
    logger.info("   python scripts/generate_dataset.py")
    logger.info("\n3. Train baseline model:")
    logger.info("   python scripts/train_baseline.py")
    logger.info("\n4. Evaluate models:")
    logger.info("   python scripts/evaluate_models.py")
    logger.info("\n5. Deploy API:")
    logger.info("   python scripts/deploy_api.py")
    logger.info("\n📚 For more information, see README.md")
    logger.info("🐛 For issues, check the logs or create an issue on GitHub")
    logger.info("="*60)

def main():
    parser = argparse.ArgumentParser(description='Setup Bangla Punctuation Restoration project')
    parser.add_argument('--skip-deps', action='store_true',
                       help='Skip dependency installation')
    parser.add_argument('--skip-validation', action='store_true',
                       help='Skip installation validation')
    parser.add_argument('--quick-test', action='store_true',
                       help='Run quick system test')
    
    args = parser.parse_args()
    
    logger.info("Starting Bangla Punctuation Restoration setup...")
    
    # Check Python version
    if not check_python_version():
        return 1
    
    # Setup directories
    setup_directories()
    
    # Install dependencies
    if not args.skip_deps:
        if not install_dependencies():
            logger.error("Failed to install dependencies")
            return 1
    else:
        logger.info("Skipping dependency installation")
    
    # Download NLTK data
    download_nltk_data()
    
    # Check GPU
    check_gpu_availability()
    
    # Create example config
    create_example_config()
    
    # Validate installation
    if not args.skip_validation:
        if not validate_installation():
            logger.warning("Installation validation failed, but continuing...")
    
    # Run quick test if requested
    if args.quick_test:
        run_quick_test()
    
    # Print next steps
    print_next_steps()
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
