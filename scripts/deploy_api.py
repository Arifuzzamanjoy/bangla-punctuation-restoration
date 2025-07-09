#!/usr/bin/env python3
"""
Script to deploy the API service for Bangla punctuation restoration
"""

import sys
import os
import argparse
import logging
import subprocess
import threading
import time
from datetime import datetime

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

from api.fastapi_server import run_server
from api.gradio_interface import launch_interface
from config import API_CONFIG, GRADIO_CONFIG

# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(f'api_deployment_{datetime.now().strftime("%Y%m%d_%H%M%S")}.log'),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def check_model_exists(model_path: str) -> bool:
    """Check if model exists at the given path"""
    if not os.path.exists(model_path):
        logger.warning(f"Model path does not exist: {model_path}")
        return False
    
    # Check for required model files
    required_files = ['config.json']
    for file in required_files:
        if not os.path.exists(os.path.join(model_path, file)):
            logger.warning(f"Required model file not found: {file}")
            return False
    
    return True

def run_fastapi_server(model_path: str, model_type: str, host: str, port: int):
    """Run FastAPI server in a separate thread"""
    try:
        logger.info(f"Starting FastAPI server on {host}:{port}")
        run_server(
            model_path=model_path,
            model_type=model_type,
            host=host,
            port=port
        )
    except Exception as e:
        logger.error(f"Error running FastAPI server: {e}")

def run_gradio_interface(model_path: str, model_type: str, host: str, port: int, share: bool):
    """Run Gradio interface in a separate thread"""
    try:
        logger.info(f"Starting Gradio interface on {host}:{port}")
        launch_interface(
            model_path=model_path,
            model_type=model_type,
            share=share,
            host=host,
            port=port
        )
    except Exception as e:
        logger.error(f"Error running Gradio interface: {e}")

def deploy_to_huggingface_spaces(model_path: str, space_name: str):
    """Deploy to Hugging Face Spaces (placeholder implementation)"""
    logger.info(f"Deploying to Hugging Face Spaces: {space_name}")
    
    # This is a placeholder - in a real implementation, you would:
    # 1. Create a Space repository
    # 2. Upload the model and code
    # 3. Configure the Space
    
    logger.info("Hugging Face Spaces deployment is not implemented in this demo")
    logger.info("To deploy to HF Spaces:")
    logger.info("1. Create a new Space at https://huggingface.co/spaces")
    logger.info("2. Upload your model and code")
    logger.info("3. Add requirements.txt and app.py")
    logger.info("4. Configure the Space settings")

def create_docker_files(model_path: str, model_type: str):
    """Create Docker files for containerized deployment"""
    logger.info("Creating Docker files...")
    
    # Dockerfile
    dockerfile_content = f"""
FROM python:3.9-slim

WORKDIR /app

# Install system dependencies
RUN apt-get update && apt-get install -y \\
    build-essential \\
    && rm -rf /var/lib/apt/lists/*

# Copy requirements
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy source code
COPY src/ src/
COPY config.py .
COPY models/ models/

# Expose ports
EXPOSE 8000 7860

# Set environment variables
ENV MODEL_PATH={model_path}
ENV MODEL_TYPE={model_type}

# Run the application
CMD ["python", "src/api/fastapi_server.py", "--model_path", "$MODEL_PATH", "--model_type", "$MODEL_TYPE", "--host", "0.0.0.0", "--port", "8000"]
"""
    
    with open("Dockerfile", 'w') as f:
        f.write(dockerfile_content)
    
    # docker-compose.yml
    compose_content = f"""
version: '3.8'

services:
  bangla-punctuation-api:
    build: .
    ports:
      - "8000:8000"
      - "7860:7860"
    volumes:
      - ./models:/app/models
    environment:
      - MODEL_PATH={model_path}
      - MODEL_TYPE={model_type}
    restart: unless-stopped

  nginx:
    image: nginx:alpine
    ports:
      - "80:80"
    volumes:
      - ./nginx.conf:/etc/nginx/nginx.conf
    depends_on:
      - bangla-punctuation-api
    restart: unless-stopped
"""
    
    with open("docker-compose.yml", 'w') as f:
        f.write(compose_content)
    
    # nginx.conf
    nginx_content = """
events {
    worker_connections 1024;
}

http {
    upstream api {
        server bangla-punctuation-api:8000;
    }
    
    upstream gradio {
        server bangla-punctuation-api:7860;
    }
    
    server {
        listen 80;
        
        location /api/ {
            proxy_pass http://api/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
        }
        
        location / {
            proxy_pass http://gradio/;
            proxy_set_header Host $host;
            proxy_set_header X-Real-IP $remote_addr;
            proxy_set_header Upgrade $http_upgrade;
            proxy_set_header Connection "upgrade";
        }
    }
}
"""
    
    with open("nginx.conf", 'w') as f:
        f.write(nginx_content)
    
    logger.info("Docker files created successfully!")
    logger.info("To deploy with Docker:")
    logger.info("1. Build: docker-compose build")
    logger.info("2. Run: docker-compose up -d")

def main():
    parser = argparse.ArgumentParser(description='Deploy API service for Bangla punctuation restoration')
    parser.add_argument('--model_path', type=str, default='models/baseline',
                       help='Path to trained model')
    parser.add_argument('--model_type', type=str, default='token_classification',
                       choices=['token_classification', 'seq2seq'],
                       help='Type of model to deploy')
    parser.add_argument('--service', type=str, choices=['fastapi', 'gradio', 'both'], default='both',
                       help='Which service to deploy')
    parser.add_argument('--fastapi_host', type=str, default=None,
                       help='FastAPI host address')
    parser.add_argument('--fastapi_port', type=int, default=None,
                       help='FastAPI port number')
    parser.add_argument('--gradio_host', type=str, default=None,
                       help='Gradio host address')
    parser.add_argument('--gradio_port', type=int, default=None,
                       help='Gradio port number')
    parser.add_argument('--share', action='store_true',
                       help='Create public Gradio link')
    parser.add_argument('--create_docker', action='store_true',
                       help='Create Docker deployment files')
    parser.add_argument('--deploy_hf_spaces', type=str, default=None,
                       help='Deploy to Hugging Face Spaces (provide space name)')
    
    args = parser.parse_args()
    
    logger.info("Starting API deployment...")
    logger.info(f"Model path: {args.model_path}")
    logger.info(f"Model type: {args.model_type}")
    logger.info(f"Service: {args.service}")
    
    # Check if model exists
    if not check_model_exists(args.model_path):
        logger.error("Model validation failed. Please ensure the model is properly trained and saved.")
        logger.info("To train a model, run: python scripts/train_baseline.py")
        return 1
    
    # Create Docker files if requested
    if args.create_docker:
        create_docker_files(args.model_path, args.model_type)
    
    # Deploy to Hugging Face Spaces if requested
    if args.deploy_hf_spaces:
        deploy_to_huggingface_spaces(args.model_path, args.deploy_hf_spaces)
    
    # Set up service parameters
    fastapi_host = args.fastapi_host or API_CONFIG["host"]
    fastapi_port = args.fastapi_port or API_CONFIG["port"]
    gradio_host = args.gradio_host or GRADIO_CONFIG["host"]
    gradio_port = args.gradio_port or GRADIO_CONFIG["port"]
    
    try:
        threads = []
        
        # Start FastAPI server
        if args.service in ['fastapi', 'both']:
            fastapi_thread = threading.Thread(
                target=run_fastapi_server,
                args=(args.model_path, args.model_type, fastapi_host, fastapi_port),
                daemon=True
            )
            fastapi_thread.start()
            threads.append(fastapi_thread)
            
            # Wait a bit for FastAPI to start
            time.sleep(2)
            logger.info(f"FastAPI server should be available at: http://{fastapi_host}:{fastapi_port}")
            logger.info(f"API documentation at: http://{fastapi_host}:{fastapi_port}/docs")
        
        # Start Gradio interface
        if args.service in ['gradio', 'both']:
            gradio_thread = threading.Thread(
                target=run_gradio_interface,
                args=(args.model_path, args.model_type, gradio_host, gradio_port, args.share),
                daemon=True
            )
            gradio_thread.start()
            threads.append(gradio_thread)
            
            # Wait a bit for Gradio to start
            time.sleep(2)
            logger.info(f"Gradio interface should be available at: http://{gradio_host}:{gradio_port}")
        
        # Display service information
        logger.info("\n" + "="*50)
        logger.info("DEPLOYMENT SUCCESSFUL!")
        logger.info("="*50)
        
        if args.service in ['fastapi', 'both']:
            logger.info(f"🚀 FastAPI Server: http://{fastapi_host}:{fastapi_port}")
            logger.info(f"📚 API Docs: http://{fastapi_host}:{fastapi_port}/docs")
            logger.info(f"🔍 Health Check: http://{fastapi_host}:{fastapi_port}/health")
        
        if args.service in ['gradio', 'both']:
            logger.info(f"🎨 Gradio Interface: http://{gradio_host}:{gradio_port}")
            if args.share:
                logger.info("🌐 Public link will be displayed in the Gradio logs")
        
        logger.info("\n📝 Usage Examples:")
        if args.service in ['fastapi', 'both']:
            logger.info("FastAPI:")
            logger.info(f"curl -X POST http://{fastapi_host}:{fastapi_port}/restore-punctuation \\")
            logger.info('     -H "Content-Type: application/json" \\')
            logger.info('     -d \'{"text": "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে"}\'')
        
        if args.service in ['gradio', 'both']:
            logger.info(f"Gradio: Open http://{gradio_host}:{gradio_port} in your browser")
        
        logger.info("\n💡 Press Ctrl+C to stop all services")
        logger.info("="*50)
        
        # Keep the main thread alive
        try:
            while True:
                time.sleep(1)
        except KeyboardInterrupt:
            logger.info("Shutting down services...")
            return 0
        
    except Exception as e:
        logger.error(f"Error during deployment: {e}")
        import traceback
        traceback.print_exc()
        return 1

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
