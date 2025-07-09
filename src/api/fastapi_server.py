#!/usr/bin/env python3
"""
FastAPI server for Bangla punctuation restoration
"""

import os
import sys
import logging
from typing import Optional, Dict, Any
from datetime import datetime

from fastapi import FastAPI, HTTPException, Request
from fastapi.middleware.cors import CORSMiddleware
from fastapi.responses import JSONResponse
from pydantic import BaseModel, Field
import uvicorn

# Add src to path
sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))

from src.models.baseline_model import PunctuationRestorer
from config import API_CONFIG, GRADIO_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Global model instance for API and testing
model = None

# Simple input validation function for testing and API
def validate_text_input(text: str) -> bool:
    """Validate that input is a non-empty Bangla string (very basic)."""
    return isinstance(text, str) and len(text.strip()) > 0

class PunctuationRequest(BaseModel):
    """Request model for punctuation restoration"""
    text: str = Field(..., description="Unpunctuated text to restore punctuation", max_length=API_CONFIG["max_text_length"])
    model_type: Optional[str] = Field("token_classification", description="Type of model to use")

class PunctuationResponse(BaseModel):
    """Response model for punctuation restoration"""
    original_text: str
    punctuated_text: str
    processing_time: float
    model_info: Dict[str, Any]

class HealthResponse(BaseModel):
    """Health check response"""
    status: str
    timestamp: str
    model_loaded: bool

def create_app(model_path: Optional[str] = None, model_type: str = "token_classification") -> FastAPI:
    """
    Create FastAPI application
    
    Args:
        model_path: Path to trained model
        model_type: Type of model to load
        
    Returns:
        FastAPI application instance
    """
    app = FastAPI(
        title="Bangla Punctuation Restoration API",
        description="API for restoring punctuation in Bangla text using machine learning models",
        version="1.0.0",
        docs_url="/docs",
        redoc_url="/redoc"
    )
    
    # Add CORS middleware
    app.add_middleware(
        CORSMiddleware,
        allow_origins=API_CONFIG["cors_origins"],
        allow_credentials=True,
        allow_methods=["*"],
        allow_headers=["*"],
    )
    
    @app.on_event("startup")
    async def startup_event():
        """Load model on startup"""
        global model
        
        try:
            if model_path and os.path.exists(model_path):
                logger.info(f"Loading model from: {model_path}")
                model = PunctuationRestorer(model_path=model_path, model_type=model_type)
                logger.info("Model loaded successfully")
            else:
                logger.warning("No valid model path provided. Model will need to be loaded manually.")
                model = None
                
        except Exception as e:
            logger.error(f"Error loading model: {e}")
            model = None
    
    @app.get("/health", response_model=HealthResponse)
    async def health_check():
        """Health check endpoint"""
        return HealthResponse(
            status="healthy",
            timestamp=datetime.now().isoformat(),
            model_loaded=model is not None and hasattr(model, 'model') and model.model.trained
        )
    
    @app.post("/restore-punctuation", response_model=PunctuationResponse)
    async def restore_punctuation(request: PunctuationRequest):
        """
        Restore punctuation in Bangla text
        
        Args:
            request: Request containing text to punctuate
            
        Returns:
            Response with punctuated text
        """
        if model is None or not model.model.trained:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded or not trained. Please check server configuration."
            )
        
        if not request.text.strip():
            raise HTTPException(
                status_code=400,
                detail="Text cannot be empty"
            )
        
        try:
            start_time = datetime.now()
            
            # Restore punctuation
            punctuated_text = model.restore_punctuation(request.text)
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return PunctuationResponse(
                original_text=request.text,
                punctuated_text=punctuated_text,
                processing_time=processing_time,
                model_info={
                    "model_type": model.model.model_type,
                    "config": model.model.config
                }
            )
            
        except Exception as e:
            logger.error(f"Error during punctuation restoration: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing text: {str(e)}"
            )
    
    @app.post("/batch-restore-punctuation")
    async def batch_restore_punctuation(texts: list[str]):
        """
        Restore punctuation for multiple texts
        
        Args:
            texts: List of texts to punctuate
            
        Returns:
            List of punctuated texts
        """
        if model is None or not model.model.trained:
            raise HTTPException(
                status_code=503,
                detail="Model not loaded or not trained. Please check server configuration."
            )
        
        if not texts:
            raise HTTPException(
                status_code=400,
                detail="Text list cannot be empty"
            )
        
        if len(texts) > 100:  # Limit batch size
            raise HTTPException(
                status_code=400,
                detail="Batch size too large. Maximum 100 texts allowed."
            )
        
        try:
            start_time = datetime.now()
            results = []
            
            for text in texts:
                if text.strip():
                    punctuated = model.restore_punctuation(text)
                    results.append({
                        "original": text,
                        "punctuated": punctuated
                    })
                else:
                    results.append({
                        "original": text,
                        "punctuated": text,
                        "error": "Empty text"
                    })
            
            end_time = datetime.now()
            processing_time = (end_time - start_time).total_seconds()
            
            return {
                "results": results,
                "processing_time": processing_time,
                "total_texts": len(texts)
            }
            
        except Exception as e:
            logger.error(f"Error during batch punctuation restoration: {e}")
            raise HTTPException(
                status_code=500,
                detail=f"Error processing texts: {str(e)}"
            )
    
    @app.get("/model-info")
    async def get_model_info():
        """Get information about the loaded model"""
        if model is None:
            return {"status": "No model loaded"}
        
        return {
            "model_type": model.model.model_type,
            "model_config": model.model.config,
            "trained": model.model.trained,
            "model_path": getattr(model.model, 'model_path', 'unknown')
        }
    
    @app.exception_handler(Exception)
    async def general_exception_handler(request: Request, exc: Exception):
        """Handle general exceptions"""
        logger.error(f"Unhandled exception: {exc}")
        return JSONResponse(
            status_code=500,
            content={"detail": "Internal server error"}
        )
    
    return app

def run_server(model_path: Optional[str] = None, 
               model_type: str = "token_classification",
               host: str = None,
               port: int = None):
    """
    Run the FastAPI server
    
    Args:
        model_path: Path to trained model
        model_type: Type of model to load
        host: Host address
        port: Port number
    """
    app = create_app(model_path, model_type)
    
    # Use config defaults if not provided
    host = host or API_CONFIG["host"]
    port = port or API_CONFIG["port"]
    
    logger.info(f"Starting server on {host}:{port}")
    logger.info(f"Model path: {model_path}")
    logger.info(f"Model type: {model_type}")
    
    uvicorn.run(
        app,
        host=host,
        port=port,
        workers=API_CONFIG["workers"],
        log_level="info"
    )

if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Run FastAPI server for punctuation restoration')
    parser.add_argument('--model_path', type=str, default="models/baseline",
                       help='Path to trained model')
    parser.add_argument('--model_type', type=str, default="token_classification",
                       choices=['token_classification', 'seq2seq'],
                       help='Type of model to load')
    parser.add_argument('--host', type=str, default=None,
                       help='Host address')
    parser.add_argument('--port', type=int, default=None,
                       help='Port number')
    
    args = parser.parse_args()
    
    run_server(
        model_path=args.model_path,
        model_type=args.model_type,
        host=args.host,
        port=args.port
    )

# Create a default app instance for testing
app = create_app()
