#!/bin/bash

# Bangla Punctuation Restoration - Installation Script
# This script installs dependencies in the correct order

set -e  # Exit on any error

echo "🚀 Installing Bangla Punctuation Restoration Dependencies"
echo "========================================================="

# Colors for output
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# Function to print colored output
print_status() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# Check if virtual environment is activated
if [[ "$VIRTUAL_ENV" == "" ]]; then
    print_warning "No virtual environment detected. Creating one..."
    python3 -m venv venv
    source venv/bin/activate
    print_status "Virtual environment created and activated"
else
    print_status "Virtual environment detected: $VIRTUAL_ENV"
fi

# Upgrade pip
print_status "Upgrading pip..."
pip install --upgrade pip

# Install core requirements
print_status "Installing core requirements..."
pip install -r requirements.txt

# Ask about GPU support
echo ""
read -p "Do you have an NVIDIA GPU and want GPU acceleration? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installing GPU-specific packages..."
    pip install -r requirements-gpu.txt
else
    print_warning "Skipping GPU packages. Using CPU-only versions."
fi

# Ask about development tools
echo ""
read -p "Do you want to install development tools? (y/n): " -n 1 -r
echo
if [[ $REPLY =~ ^[Yy]$ ]]; then
    print_status "Installing development tools..."
    pip install -r requirements-dev.txt
else
    print_warning "Skipping development tools."
fi

# Download spaCy model
print_status "Downloading spaCy models..."
python -m spacy download en_core_web_sm
python -m spacy download xx_ent_wiki_sm  # Multilingual model

# Download NLTK data
print_status "Downloading NLTK data..."
python -c "
import nltk
nltk.download('punkt')
nltk.download('stopwords')
nltk.download('wordnet')
print('NLTK data downloaded successfully')
"

# Verify installation
print_status "Verifying installation..."
python -c "
import torch
import transformers
import fastapi
import gradio
print(f'✅ PyTorch: {torch.__version__}')
print(f'✅ Transformers: {transformers.__version__}')
print(f'✅ FastAPI: {fastapi.__version__}')
print(f'✅ Gradio: {gradio.__version__}')
print('🎉 All core packages installed successfully!')
"

echo ""
print_status "Installation completed successfully!"
print_status "You can now run the application using:"
echo "  python run_pipeline.py"
echo ""
print_warning "Note: For Triton Inference Server, use Docker:"
echo "  docker pull nvcr.io/nvidia/tritonserver:23.10-py3"
echo ""
print_warning "Note: For TensorRT, follow NVIDIA's installation guide:"
echo "  https://docs.nvidia.com/deeplearning/tensorrt/install-guide/"
