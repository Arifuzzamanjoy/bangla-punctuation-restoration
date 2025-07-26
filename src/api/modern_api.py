#!/usr/bin/env python3
"""
Modern API and Deployment System with Latest Technologies
"""

import asyncio
import aioredis
from fastapi import FastAPI, HTTPException, BackgroundTasks, Depends, Security
from fastapi.security import HTTPBearer, HTTPAuthorizationCredentials
from fastapi.middleware.cors import CORSMiddleware
from fastapi.middleware.gzip import GZipMiddleware
from fastapi.responses import StreamingResponse
import uvicorn
from pydantic import BaseModel, Field, validator
from typing import List, Dict, Any, Optional, Generator, AsyncGenerator
import torch
import numpy as np
from transformers import AutoTokenizer, AutoModelForTokenClassification
from contextlib import asynccontextmanager
import logging
import time
import json
from datetime import datetime, timedelta
import hashlib
import jwt
from prometheus_client import Counter, Histogram, Gauge, generate_latest
from prometheus_client.core import CollectorRegistry
import asyncpg
from sqlalchemy.ext.asyncio import create_async_engine, AsyncSession
from sqlalchemy.orm import sessionmaker
import redis
from celery import Celery
import websockets
from sse_starlette.sse import EventSourceResponse
import aiofiles
from pathlib import Path
import httpx
import gradio as gr
from gradio import mount_gradio_app
import streamlit as st
from starlette.background import BackgroundTask
from starlette.responses import JSONResponse
import structlog
from opentelemetry import trace
from opentelemetry.exporter.jaeger.thrift import JaegerExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor

# Setup structured logging
structlog.configure(
    processors=[
        structlog.stdlib.filter_by_level,
        structlog.stdlib.add_logger_name,
        structlog.stdlib.add_log_level,
        structlog.stdlib.PositionalArgumentsFormatter(),
        structlog.processors.TimeStamper(fmt="iso"),
        structlog.processors.StackInfoRenderer(),
        structlog.processors.format_exc_info,
        structlog.processors.UnicodeDecoder(),
        structlog.processors.JSONRenderer()
    ],
    context_class=dict,
    logger_factory=structlog.stdlib.LoggerFactory(),
    wrapper_class=structlog.stdlib.BoundLogger,
    cache_logger_on_first_use=True,
)

logger = structlog.get_logger()

# Prometheus metrics
REGISTRY = CollectorRegistry()
REQUEST_COUNT = Counter('punctuation_requests_total', 'Total requests', registry=REGISTRY)
REQUEST_DURATION = Histogram('punctuation_request_duration_seconds', 'Request duration', registry=REGISTRY)
ACTIVE_CONNECTIONS = Gauge('punctuation_active_connections', 'Active connections', registry=REGISTRY)
MODEL_INFERENCE_TIME = Histogram('model_inference_duration_seconds', 'Model inference time', registry=REGISTRY)

# Tracing setup
trace.set_tracer_provider(TracerProvider())
tracer = trace.get_tracer(__name__)

# JWT Configuration
JWT_SECRET = "your-secret-key-here"  # Should be in environment variables
JWT_ALGORITHM = "HS256"
JWT_EXPIRATION_HOURS = 24

class PunctuationRequest(BaseModel):
    """Request model for punctuation restoration"""
    text: str = Field(..., min_length=1, max_length=10000, description="Text to punctuate")
    model_name: Optional[str] = Field("baseline", description="Model to use")
    include_confidence: Optional[bool] = Field(False, description="Include confidence scores")
    stream_response: Optional[bool] = Field(False, description="Stream response")
    
    @validator('text')
    def validate_text(cls, v):
        if not v.strip():
            raise ValueError('Text cannot be empty')
        # Check for Bangla characters
        bangla_chars = sum(1 for char in v if '\u0980' <= char <= '\u09FF')
        if bangla_chars / len(v) < 0.3:
            raise ValueError('Text must contain at least 30% Bangla characters')
        return v

class PunctuationResponse(BaseModel):
    """Response model for punctuation restoration"""
    punctuated_text: str
    confidence_score: Optional[float] = None
    processing_time: float
    model_used: str
    request_id: str
    metadata: Optional[Dict[str, Any]] = None

class BatchPunctuationRequest(BaseModel):
    """Batch request model"""
    texts: List[str] = Field(..., min_items=1, max_items=100)
    model_name: Optional[str] = "baseline"
    include_confidence: Optional[bool] = False

class BatchPunctuationResponse(BaseModel):
    """Batch response model"""
    results: List[PunctuationResponse]
    total_processing_time: float
    batch_id: str

class ModelManager:
    """Advanced model management with caching and optimization"""
    
    def __init__(self):
        self.models = {}
        self.tokenizers = {}
        self.model_configs = {
            "baseline": "ai4bharat/indic-bert",
            "advanced": "xlm-roberta-base",
            "llm": "microsoft/DialoGPT-medium"
        }
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.redis_client = None
        
    async def initialize(self):
        """Initialize Redis and load models"""
        try:
            self.redis_client = await aioredis.from_url("redis://localhost")
            await self.load_models()
            logger.info("Model manager initialized successfully")
        except Exception as e:
            logger.error(f"Failed to initialize model manager: {e}")
    
    async def load_models(self):
        """Load all models asynchronously"""
        for model_name, model_path in self.model_configs.items():
            try:
                # Check cache first
                cached_model = await self.get_cached_model(model_name)
                if cached_model:
                    self.models[model_name] = cached_model
                    continue
                
                # Load model
                tokenizer = AutoTokenizer.from_pretrained(model_path)
                model = AutoModelForTokenClassification.from_pretrained(model_path)
                model.to(self.device)
                model.eval()
                
                # Optimize for inference
                if hasattr(torch, 'jit'):
                    try:
                        model = torch.jit.script(model)
                    except:
                        pass  # JIT compilation failed, use original model
                
                self.tokenizers[model_name] = tokenizer
                self.models[model_name] = model
                
                # Cache model
                await self.cache_model(model_name, model)
                
                logger.info(f"Loaded model: {model_name}")
                
            except Exception as e:
                logger.error(f"Failed to load model {model_name}: {e}")
    
    async def get_cached_model(self, model_name: str):
        """Get model from cache"""
        if not self.redis_client:
            return None
        
        try:
            cached_data = await self.redis_client.get(f"model:{model_name}")
            if cached_data:
                # In practice, you'd deserialize the model here
                # For now, return None to force loading
                pass
        except Exception as e:
            logger.warning(f"Cache retrieval failed: {e}")
        
        return None
    
    async def cache_model(self, model_name: str, model):
        """Cache model in Redis"""
        if not self.redis_client:
            return
        
        try:
            # In practice, you'd serialize the model here
            # For now, just set a flag
            await self.redis_client.set(f"model:{model_name}:loaded", "true", ex=3600)
        except Exception as e:
            logger.warning(f"Cache storage failed: {e}")
    
    @tracer.start_as_current_span("predict")
    async def predict(self, text: str, model_name: str = "baseline", 
                     include_confidence: bool = False) -> Dict[str, Any]:
        """Make prediction with specified model"""
        
        start_time = time.time()
        
        with MODEL_INFERENCE_TIME.time():
            # Get model and tokenizer
            if model_name not in self.models:
                raise HTTPException(status_code=400, detail=f"Model {model_name} not available")
            
            model = self.models[model_name]
            tokenizer = self.tokenizers[model_name]
            
            # Tokenize input
            inputs = tokenizer(
                text,
                return_tensors="pt",
                truncation=True,
                max_length=512,
                padding=True
            ).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = model(**inputs)
                predictions = torch.argmax(outputs.logits, dim=2)
                
                # Calculate confidence if requested
                confidence = None
                if include_confidence:
                    probabilities = torch.softmax(outputs.logits, dim=2)
                    confidence = torch.max(probabilities, dim=2)[0].mean().item()
            
            # Convert to text (simplified)
            punctuated_text = self._convert_predictions_to_text(text, predictions, tokenizer)
            
            processing_time = time.time() - start_time
            
            return {
                "punctuated_text": punctuated_text,
                "confidence_score": confidence,
                "processing_time": processing_time,
                "model_used": model_name
            }
    
    def _convert_predictions_to_text(self, original_text: str, predictions: torch.Tensor, 
                                   tokenizer) -> str:
        """Convert model predictions to punctuated text"""
        # Simplified implementation
        # In practice, this would be more sophisticated
        return original_text + "।"  # Placeholder

class AuthManager:
    """JWT-based authentication manager"""
    
    @staticmethod
    def create_access_token(user_id: str) -> str:
        """Create JWT access token"""
        payload = {
            "user_id": user_id,
            "exp": datetime.utcnow() + timedelta(hours=JWT_EXPIRATION_HOURS),
            "iat": datetime.utcnow()
        }
        return jwt.encode(payload, JWT_SECRET, algorithm=JWT_ALGORITHM)
    
    @staticmethod
    def verify_token(token: str) -> Dict[str, Any]:
        """Verify JWT token"""
        try:
            payload = jwt.decode(token, JWT_SECRET, algorithms=[JWT_ALGORITHM])
            return payload
        except jwt.ExpiredSignatureError:
            raise HTTPException(status_code=401, detail="Token expired")
        except jwt.InvalidTokenError:
            raise HTTPException(status_code=401, detail="Invalid token")

class RateLimiter:
    """Redis-based rate limiter"""
    
    def __init__(self, redis_client):
        self.redis = redis_client
        self.default_limit = 100  # requests per hour
    
    async def is_allowed(self, key: str, limit: Optional[int] = None) -> bool:
        """Check if request is allowed"""
        limit = limit or self.default_limit
        
        try:
            current = await self.redis.get(key)
            if current is None:
                await self.redis.setex(key, 3600, 1)  # 1 hour expiry
                return True
            
            if int(current) >= limit:
                return False
            
            await self.redis.incr(key)
            return True
            
        except Exception as e:
            logger.warning(f"Rate limiting error: {e}")
            return True  # Allow on error

# Global instances
model_manager = ModelManager()
auth_manager = AuthManager()
rate_limiter = None

@asynccontextmanager
async def lifespan(app: FastAPI):
    """Application lifespan manager"""
    # Startup
    logger.info("Starting up application...")
    await model_manager.initialize()
    
    global rate_limiter
    try:
        redis_client = await aioredis.from_url("redis://localhost")
        rate_limiter = RateLimiter(redis_client)
    except Exception as e:
        logger.warning(f"Redis not available: {e}")
    
    yield
    
    # Shutdown
    logger.info("Shutting down application...")

# FastAPI app
app = FastAPI(
    title="Modern Bangla Punctuation Restoration API",
    description="Advanced AI-powered punctuation restoration for Bangla text",
    version="2.0.0",
    lifespan=lifespan
)

# Middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

app.add_middleware(GZipMiddleware, minimum_size=1000)

# Instrument with OpenTelemetry
FastAPIInstrumentor.instrument_app(app)

# Security
security = HTTPBearer()

async def get_current_user(credentials: HTTPAuthorizationCredentials = Security(security)):
    """Get current user from JWT token"""
    token = credentials.credentials
    payload = auth_manager.verify_token(token)
    return payload["user_id"]

async def check_rate_limit(request_key: str):
    """Check rate limit"""
    if rate_limiter and not await rate_limiter.is_allowed(request_key):
        raise HTTPException(status_code=429, detail="Rate limit exceeded")

@app.get("/health")
async def health_check():
    """Health check endpoint"""
    return {
        "status": "healthy",
        "timestamp": datetime.utcnow(),
        "models_loaded": list(model_manager.models.keys())
    }

@app.get("/metrics")
async def metrics():
    """Prometheus metrics endpoint"""
    return generate_latest(REGISTRY)

@app.post("/v1/punctuate", response_model=PunctuationResponse)
async def punctuate_text(
    request: PunctuationRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user)
):
    """Punctuate single text"""
    
    REQUEST_COUNT.inc()
    
    # Rate limiting
    await check_rate_limit(f"user:{user_id}")
    
    # Generate request ID
    request_id = hashlib.md5(f"{user_id}{request.text}{time.time()}".encode()).hexdigest()
    
    with REQUEST_DURATION.time():
        # Make prediction
        result = await model_manager.predict(
            request.text,
            request.model_name,
            request.include_confidence
        )
        
        # Create response
        response = PunctuationResponse(
            punctuated_text=result["punctuated_text"],
            confidence_score=result.get("confidence_score"),
            processing_time=result["processing_time"],
            model_used=result["model_used"],
            request_id=request_id
        )
        
        # Log request (background task)
        background_tasks.add_task(log_request, user_id, request, response)
        
        return response

@app.post("/v1/punctuate/batch", response_model=BatchPunctuationResponse)
async def punctuate_batch(
    request: BatchPunctuationRequest,
    background_tasks: BackgroundTasks,
    user_id: str = Depends(get_current_user)
):
    """Punctuate multiple texts in batch"""
    
    REQUEST_COUNT.inc()
    
    # Rate limiting
    await check_rate_limit(f"user:{user_id}:batch")
    
    batch_id = hashlib.md5(f"{user_id}{time.time()}".encode()).hexdigest()
    start_time = time.time()
    
    # Process all texts
    results = []
    for text in request.texts:
        result = await model_manager.predict(
            text,
            request.model_name,
            request.include_confidence
        )
        
        response = PunctuationResponse(
            punctuated_text=result["punctuated_text"],
            confidence_score=result.get("confidence_score"),
            processing_time=result["processing_time"],
            model_used=result["model_used"],
            request_id=f"{batch_id}_{len(results)}"
        )
        results.append(response)
    
    total_time = time.time() - start_time
    
    batch_response = BatchPunctuationResponse(
        results=results,
        total_processing_time=total_time,
        batch_id=batch_id
    )
    
    # Log batch request
    background_tasks.add_task(log_batch_request, user_id, request, batch_response)
    
    return batch_response

@app.get("/v1/punctuate/stream")
async def punctuate_stream(
    text: str,
    model_name: str = "baseline",
    user_id: str = Depends(get_current_user)
):
    """Stream punctuation results"""
    
    async def generate_stream():
        # Split text into chunks
        words = text.split()
        chunk_size = 10
        
        for i in range(0, len(words), chunk_size):
            chunk = " ".join(words[i:i+chunk_size])
            
            result = await model_manager.predict(chunk, model_name)
            
            yield f"data: {json.dumps(result)}\n\n"
            await asyncio.sleep(0.1)  # Small delay for streaming effect
    
    return EventSourceResponse(generate_stream())

@app.websocket("/ws/punctuate")
async def websocket_punctuate(websocket):
    """WebSocket endpoint for real-time punctuation"""
    
    await websocket.accept()
    ACTIVE_CONNECTIONS.inc()
    
    try:
        while True:
            # Receive text
            data = await websocket.receive_json()
            text = data.get("text", "")
            model_name = data.get("model_name", "baseline")
            
            if text:
                # Process text
                result = await model_manager.predict(text, model_name)
                
                # Send result
                await websocket.send_json(result)
            
    except Exception as e:
        logger.error(f"WebSocket error: {e}")
    finally:
        ACTIVE_CONNECTIONS.dec()
        await websocket.close()

@app.post("/v1/auth/token")
async def create_token(user_id: str):
    """Create authentication token"""
    token = auth_manager.create_access_token(user_id)
    return {"access_token": token, "token_type": "bearer"}

async def log_request(user_id: str, request: PunctuationRequest, response: PunctuationResponse):
    """Log request for analytics"""
    log_data = {
        "user_id": user_id,
        "text_length": len(request.text),
        "model_used": response.model_used,
        "processing_time": response.processing_time,
        "timestamp": datetime.utcnow(),
        "confidence_score": response.confidence_score
    }
    
    logger.info("Request processed", **log_data)

async def log_batch_request(user_id: str, request: BatchPunctuationRequest, 
                          response: BatchPunctuationResponse):
    """Log batch request"""
    log_data = {
        "user_id": user_id,
        "batch_size": len(request.texts),
        "total_processing_time": response.total_processing_time,
        "timestamp": datetime.utcnow()
    }
    
    logger.info("Batch request processed", **log_data)

# Gradio Interface
def create_gradio_interface():
    """Create Gradio interface"""
    
    def punctuate_gradio(text: str, model_name: str = "baseline"):
        """Gradio wrapper for punctuation"""
        import asyncio
        
        async def async_predict():
            return await model_manager.predict(text, model_name)
        
        # Run async function
        loop = asyncio.new_event_loop()
        asyncio.set_event_loop(loop)
        result = loop.run_until_complete(async_predict())
        loop.close()
        
        return result["punctuated_text"]
    
    # Create Gradio interface
    iface = gr.Interface(
        fn=punctuate_gradio,
        inputs=[
            gr.Textbox(label="Unpunctuated Bangla Text", lines=5),
            gr.Dropdown(choices=["baseline", "advanced", "llm"], label="Model", value="baseline")
        ],
        outputs=gr.Textbox(label="Punctuated Text", lines=5),
        title="Modern Bangla Punctuation Restoration",
        description="Advanced AI-powered punctuation restoration for Bangla text",
        examples=[
            ["আমি ভালো আছি তুমি কেমন আছো", "baseline"],
            ["আজ আবহাওয়া খুব সুন্দর তাই না", "advanced"]
        ]
    )
    
    return iface

# Mount Gradio app
gradio_app = create_gradio_interface()
app = mount_gradio_app(app, gradio_app, path="/gradio")

# Celery for background tasks
celery_app = Celery(
    "punctuation_tasks",
    broker="redis://localhost:6379/0",
    backend="redis://localhost:6379/0"
)

@celery_app.task
def process_large_batch(texts: List[str], model_name: str) -> List[Dict]:
    """Process large batch in background"""
    # This would process the batch asynchronously
    results = []
    for text in texts:
        # Placeholder processing
        result = {
            "text": text,
            "punctuated": text + "।",
            "model": model_name
        }
        results.append(result)
    
    return results

# Container deployment configuration
class DeploymentConfig:
    """Configuration for modern deployment"""
    
    @staticmethod
    def get_docker_config():
        """Get Docker configuration"""
        return {
            "base_image": "python:3.11-slim",
            "requirements": [
                "fastapi[all]",
                "uvicorn[standard]",
                "torch",
                "transformers",
                "redis",
                "celery",
                "prometheus-client",
                "opentelemetry-api",
                "opentelemetry-sdk",
                "opentelemetry-instrumentation-fastapi",
                "structlog",
                "gradio"
            ],
            "environment": {
                "PYTHONPATH": "/app",
                "REDIS_URL": "redis://redis:6379",
                "MODEL_CACHE_DIR": "/app/models"
            },
            "ports": ["8000:8000"],
            "volumes": ["/app/models:/app/models"]
        }
    
    @staticmethod
    def get_kubernetes_config():
        """Get Kubernetes configuration"""
        return {
            "apiVersion": "apps/v1",
            "kind": "Deployment",
            "metadata": {"name": "bangla-punctuation-api"},
            "spec": {
                "replicas": 3,
                "selector": {"matchLabels": {"app": "bangla-punctuation"}},
                "template": {
                    "metadata": {"labels": {"app": "bangla-punctuation"}},
                    "spec": {
                        "containers": [{
                            "name": "api",
                            "image": "bangla-punctuation:latest",
                            "ports": [{"containerPort": 8000}],
                            "resources": {
                                "requests": {"memory": "2Gi", "cpu": "1"},
                                "limits": {"memory": "4Gi", "cpu": "2"}
                            },
                            "env": [
                                {"name": "REDIS_URL", "value": "redis://redis-service:6379"}
                            ]
                        }]
                    }
                }
            }
        }

# CLI for development
if __name__ == "__main__":
    # Development server
    uvicorn.run(
        "modern_api:app",
        host="0.0.0.0",
        port=8000,
        reload=True,
        workers=1,
        log_config={
            "version": 1,
            "disable_existing_loggers": False,
            "formatters": {
                "default": {
                    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
                },
            },
            "handlers": {
                "default": {
                    "formatter": "default",
                    "class": "logging.StreamHandler",
                    "stream": "ext://sys.stdout",
                },
            },
            "root": {
                "level": "INFO",
                "handlers": ["default"],
            },
        }
    )
