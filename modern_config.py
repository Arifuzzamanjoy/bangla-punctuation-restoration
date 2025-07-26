#!/usr/bin/env python3
"""
Modern Configuration Management with Environment Support
"""

import os
from pathlib import Path
from typing import Dict, Any, Optional, List
import yaml
import json
from dataclasses import dataclass, field
from enum import Enum
import logging

logger = logging.getLogger(__name__)

class Environment(Enum):
    """Environment types"""
    DEVELOPMENT = "development"
    STAGING = "staging"
    PRODUCTION = "production"
    TESTING = "testing"

@dataclass
class ModelConfig:
    """Modern model configuration"""
    # Base model settings
    name: str = "ai4bharat/indic-bert"
    max_length: int = 512
    num_labels: int = 8
    
    # Training parameters
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    num_epochs: int = 10
    batch_size: int = 16
    gradient_accumulation_steps: int = 4
    max_grad_norm: float = 1.0
    
    # Modern optimization
    use_amp: bool = True  # Automatic Mixed Precision
    use_gradient_checkpointing: bool = True
    use_deepspeed: bool = False
    deepspeed_config: Optional[str] = None
    
    # LoRA settings
    use_lora: bool = True
    lora_r: int = 16
    lora_alpha: int = 32
    lora_dropout: float = 0.1
    lora_target_modules: List[str] = field(default_factory=lambda: ["q_proj", "k_proj", "v_proj", "o_proj"])
    
    # Quantization
    use_8bit: bool = False
    use_4bit: bool = True
    bnb_4bit_compute_dtype: str = "bfloat16"
    bnb_4bit_quant_type: str = "nf4"
    
    # Advanced training techniques
    use_swa: bool = True  # Stochastic Weight Averaging
    use_ema: bool = True  # Exponential Moving Average
    use_sam: bool = True  # Sharpness Aware Minimization
    use_lookahead: bool = True
    
    # Regularization
    dropout_rate: float = 0.1
    attention_dropout: float = 0.1
    hidden_dropout: float = 0.1
    label_smoothing: float = 0.1
    
    # Data augmentation
    use_mixup: bool = True
    mixup_alpha: float = 0.2
    use_cutmix: bool = True
    cutmix_alpha: float = 1.0
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_strategy: str = "length_based"
    
    # Multi-task learning
    use_multitask: bool = False
    auxiliary_tasks: List[str] = field(default_factory=list)

@dataclass
class DataConfig:
    """Modern data configuration"""
    # Dataset paths
    base_dataset: str = "hishab/hishab-pr-bn-v1"
    custom_dataset_path: Optional[str] = None
    cache_dir: str = "data/cache"
    
    # Data splits
    train_ratio: float = 0.8
    validation_ratio: float = 0.1
    test_ratio: float = 0.1
    
    # Data processing
    min_text_length: int = 5
    max_text_length: int = 512
    min_bangla_ratio: float = 0.7
    
    # Augmentation settings
    use_augmentation: bool = True
    augmentation_factor: int = 2
    synonym_replacement_prob: float = 0.1
    random_insertion_prob: float = 0.1
    random_swap_prob: float = 0.1
    random_deletion_prob: float = 0.05
    back_translation_prob: float = 0.2
    paraphrasing_prob: float = 0.15
    
    # Quality filtering
    use_quality_filter: bool = True
    remove_duplicates: bool = True
    language_detection_threshold: float = 0.8
    
    # Streaming and batching
    use_streaming: bool = True
    streaming_batch_size: int = 1000
    dataloader_num_workers: int = 4
    pin_memory: bool = True

@dataclass
class WebScrapingConfig:
    """Enhanced web scraping configuration"""
    # Basic settings
    max_concurrent_requests: int = 10
    request_timeout: int = 30
    delay_between_requests: float = 1.0
    max_retries: int = 3
    
    # User agents rotation
    rotate_user_agents: bool = True
    custom_user_agents: List[str] = field(default_factory=list)
    
    # Proxy settings
    use_proxies: bool = False
    proxy_list: List[str] = field(default_factory=list)
    proxy_rotation: bool = True
    
    # Advanced features
    use_selenium: bool = False
    headless_browser: bool = True
    javascript_execution: bool = False
    
    # Content filtering
    min_content_length: int = 100
    max_content_length: int = 10000
    bangla_content_threshold: float = 0.5
    
    # Storage settings
    use_distributed_storage: bool = False
    storage_backend: str = "local"  # local, s3, gcs
    compression: bool = True
    
    # Monitoring
    track_scraping_metrics: bool = True
    alert_on_failures: bool = True

@dataclass
class APIConfig:
    """Modern API configuration"""
    # Server settings
    host: str = "0.0.0.0"
    port: int = 8000
    workers: int = 4
    
    # Security
    secret_key: str = "your-secret-key-change-in-production"
    jwt_algorithm: str = "HS256"
    jwt_expiration_hours: int = 24
    enable_https: bool = False
    
    # Rate limiting
    enable_rate_limiting: bool = True
    requests_per_minute: int = 100
    requests_per_hour: int = 1000
    
    # CORS settings
    allow_origins: List[str] = field(default_factory=lambda: ["*"])
    allow_methods: List[str] = field(default_factory=lambda: ["*"])
    allow_headers: List[str] = field(default_factory=lambda: ["*"])
    
    # API versioning
    api_version: str = "v1"
    enable_versioning: bool = True
    
    # Documentation
    enable_docs: bool = True
    docs_url: str = "/docs"
    redoc_url: str = "/redoc"
    
    # WebSocket settings
    enable_websocket: bool = True
    websocket_timeout: int = 300
    
    # Background tasks
    enable_celery: bool = True
    celery_broker: str = "redis://localhost:6379/0"
    celery_backend: str = "redis://localhost:6379/0"

@dataclass
class MonitoringConfig:
    """Monitoring and observability configuration"""
    # Logging
    log_level: str = "INFO"
    log_format: str = "json"  # json, structured, simple
    log_file: Optional[str] = None
    
    # Metrics
    enable_prometheus: bool = True
    metrics_port: int = 9090
    metrics_path: str = "/metrics"
    
    # Tracing
    enable_tracing: bool = True
    jaeger_endpoint: Optional[str] = None
    trace_sampling_rate: float = 0.1
    
    # Health checks
    enable_health_checks: bool = True
    health_check_interval: int = 30
    
    # Alerting
    enable_alerting: bool = True
    alert_channels: List[str] = field(default_factory=list)
    
    # Performance monitoring
    monitor_inference_time: bool = True
    monitor_memory_usage: bool = True
    monitor_gpu_usage: bool = True

@dataclass
class InfrastructureConfig:
    """Infrastructure and deployment configuration"""
    # Container settings
    enable_docker: bool = True
    docker_image: str = "bangla-punctuation:latest"
    
    # Kubernetes
    enable_kubernetes: bool = False
    namespace: str = "default"
    replicas: int = 3
    
    # Storage
    model_storage_path: str = "models/"
    data_storage_path: str = "data/"
    use_shared_storage: bool = False
    
    # Database
    database_url: Optional[str] = None
    enable_connection_pooling: bool = True
    
    # Cache
    redis_url: str = "redis://localhost:6379"
    cache_ttl: int = 3600
    
    # Auto-scaling
    enable_autoscaling: bool = True
    min_replicas: int = 1
    max_replicas: int = 10
    target_cpu_utilization: int = 70

@dataclass
class ModernConfig:
    """Complete modern configuration"""
    # Environment
    environment: Environment = Environment.DEVELOPMENT
    debug: bool = True
    
    # Component configurations
    models: Dict[str, ModelConfig] = field(default_factory=dict)
    data: DataConfig = field(default_factory=DataConfig)
    web_scraping: WebScrapingConfig = field(default_factory=WebScrapingConfig)
    api: APIConfig = field(default_factory=APIConfig)
    monitoring: MonitoringConfig = field(default_factory=MonitoringConfig)
    infrastructure: InfrastructureConfig = field(default_factory=InfrastructureConfig)
    
    # Feature flags
    features: Dict[str, bool] = field(default_factory=dict)
    
    # Experimental features
    experimental: Dict[str, Any] = field(default_factory=dict)

class ConfigManager:
    """Advanced configuration manager"""
    
    def __init__(self, config_path: Optional[str] = None, environment: Optional[str] = None):
        self.config_path = config_path or "config/"
        self.environment = Environment(environment or os.getenv("ENVIRONMENT", "development"))
        self.config = self._load_config()
    
    def _load_config(self) -> ModernConfig:
        """Load configuration from files and environment"""
        
        # Start with default configuration
        config_dict = self._get_default_config()
        
        # Load from files
        config_dict.update(self._load_from_files())
        
        # Override with environment variables
        config_dict.update(self._load_from_environment())
        
        # Create configuration object
        return self._create_config_object(config_dict)
    
    def _get_default_config(self) -> Dict[str, Any]:
        """Get default configuration"""
        return {
            "environment": self.environment.value,
            "debug": self.environment == Environment.DEVELOPMENT,
            "models": {
                "baseline": {
                    "name": "ai4bharat/indic-bert",
                    "learning_rate": 2e-5,
                    "batch_size": 16,
                    "num_epochs": 5
                },
                "advanced": {
                    "name": "xlm-roberta-base",
                    "learning_rate": 3e-5,
                    "batch_size": 16,
                    "num_epochs": 10,
                    "use_lora": True,
                    "use_4bit": True
                },
                "llm": {
                    "name": "microsoft/DialoGPT-medium",
                    "learning_rate": 1e-4,
                    "batch_size": 8,
                    "num_epochs": 3,
                    "use_lora": True,
                    "use_4bit": True
                }
            }
        }
    
    def _load_from_files(self) -> Dict[str, Any]:
        """Load configuration from YAML/JSON files"""
        config_dict = {}
        
        config_dir = Path(self.config_path)
        if not config_dir.exists():
            logger.warning(f"Config directory {config_dir} does not exist")
            return config_dict
        
        # Load base configuration
        base_config_file = config_dir / "config.yaml"
        if base_config_file.exists():
            with open(base_config_file, 'r') as f:
                base_config = yaml.safe_load(f)
                if base_config:
                    config_dict.update(base_config)
        
        # Load environment-specific configuration
        env_config_file = config_dir / f"{self.environment.value}.yaml"
        if env_config_file.exists():
            with open(env_config_file, 'r') as f:
                env_config = yaml.safe_load(f)
                if env_config:
                    config_dict.update(env_config)
        
        return config_dict
    
    def _load_from_environment(self) -> Dict[str, Any]:
        """Load configuration from environment variables"""
        config_dict = {}
        
        # Map environment variables to config keys
        env_mappings = {
            "ENVIRONMENT": "environment",
            "DEBUG": "debug",
            "API_HOST": "api.host",
            "API_PORT": "api.port",
            "SECRET_KEY": "api.secret_key",
            "REDIS_URL": "infrastructure.redis_url",
            "DATABASE_URL": "infrastructure.database_url",
            "LOG_LEVEL": "monitoring.log_level",
            "ENABLE_PROMETHEUS": "monitoring.enable_prometheus",
            "MODEL_STORAGE_PATH": "infrastructure.model_storage_path"
        }
        
        for env_var, config_key in env_mappings.items():
            env_value = os.getenv(env_var)
            if env_value is not None:
                # Convert string values to appropriate types
                if env_value.lower() in ['true', 'false']:
                    env_value = env_value.lower() == 'true'
                elif env_value.isdigit():
                    env_value = int(env_value)
                elif self._is_float(env_value):
                    env_value = float(env_value)
                
                # Set nested configuration
                self._set_nested_config(config_dict, config_key, env_value)
        
        return config_dict
    
    def _is_float(self, value: str) -> bool:
        """Check if string can be converted to float"""
        try:
            float(value)
            return True
        except ValueError:
            return False
    
    def _set_nested_config(self, config_dict: Dict, key_path: str, value: Any):
        """Set nested configuration value"""
        keys = key_path.split('.')
        current_dict = config_dict
        
        for key in keys[:-1]:
            if key not in current_dict:
                current_dict[key] = {}
            current_dict = current_dict[key]
        
        current_dict[keys[-1]] = value
    
    def _create_config_object(self, config_dict: Dict[str, Any]) -> ModernConfig:
        """Create configuration object from dictionary"""
        
        # Create model configurations
        models = {}
        if "models" in config_dict:
            for model_name, model_config in config_dict["models"].items():
                models[model_name] = ModelConfig(**model_config)
        
        # Create other configurations
        data_config = DataConfig(**(config_dict.get("data", {})))
        web_scraping_config = WebScrapingConfig(**(config_dict.get("web_scraping", {})))
        api_config = APIConfig(**(config_dict.get("api", {})))
        monitoring_config = MonitoringConfig(**(config_dict.get("monitoring", {})))
        infrastructure_config = InfrastructureConfig(**(config_dict.get("infrastructure", {})))
        
        return ModernConfig(
            environment=Environment(config_dict.get("environment", "development")),
            debug=config_dict.get("debug", True),
            models=models,
            data=data_config,
            web_scraping=web_scraping_config,
            api=api_config,
            monitoring=monitoring_config,
            infrastructure=infrastructure_config,
            features=config_dict.get("features", {}),
            experimental=config_dict.get("experimental", {})
        )
    
    def get_model_config(self, model_name: str) -> ModelConfig:
        """Get configuration for specific model"""
        if model_name not in self.config.models:
            raise ValueError(f"Model {model_name} not found in configuration")
        return self.config.models[model_name]
    
    def save_config(self, output_path: str = None):
        """Save current configuration to file"""
        output_path = output_path or f"config/{self.environment.value}_generated.yaml"
        
        # Convert configuration to dictionary
        config_dict = self._config_to_dict(self.config)
        
        # Save to file
        Path(output_path).parent.mkdir(parents=True, exist_ok=True)
        with open(output_path, 'w') as f:
            yaml.dump(config_dict, f, default_flow_style=False, indent=2)
        
        logger.info(f"Configuration saved to {output_path}")
    
    def _config_to_dict(self, config: ModernConfig) -> Dict[str, Any]:
        """Convert configuration object to dictionary"""
        # This is a simplified implementation
        # In practice, you'd want more sophisticated serialization
        import dataclasses
        return dataclasses.asdict(config)

# Global configuration instance
config_manager = None

def get_config(config_path: Optional[str] = None, environment: Optional[str] = None) -> ModernConfig:
    """Get global configuration instance"""
    global config_manager
    
    if config_manager is None:
        config_manager = ConfigManager(config_path, environment)
    
    return config_manager.config

def reload_config():
    """Reload configuration"""
    global config_manager
    if config_manager:
        config_manager.config = config_manager._load_config()

# Legacy compatibility
PROJECT_NAME = "bangla-punctuation-restoration"
VERSION = "2.0.0"
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"
RESULTS_DIR = BASE_DIR / "results"

# Punctuation configuration (unchanged for compatibility)
PUNCTUATION_MARKS = {
    "comma": ",",
    "period": "।",
    "question": "?",
    "exclamation": "!",
    "semicolon": ";",
    "colon": ":",
    "hyphen": "-",
}

PUNCTUATION_LABELS = {
    "O": 0,
    "COMMA": 1,
    "PERIOD": 2,
    "QUESTION": 3,
    "EXCLAMATION": 4,
    "SEMICOLON": 5,
    "COLON": 6,
    "HYPHEN": 7
}

ID_TO_PUNCTUATION = {v: k for k, v in PUNCTUATION_LABELS.items()}
ID_TO_SYMBOL = {
    0: "",
    1: ",",
    2: "।",
    3: "?",
    4: "!",
    5: ";",
    6: ":",
    7: "-"
}

if __name__ == "__main__":
    # Example usage
    config = get_config()
    print(f"Environment: {config.environment}")
    print(f"Available models: {list(config.models.keys())}")
    
    # Get specific model config
    baseline_config = config_manager.get_model_config("baseline")
    print(f"Baseline model: {baseline_config.name}")
    print(f"Learning rate: {baseline_config.learning_rate}")
    
    # Save configuration
    config_manager.save_config("config/current_config.yaml")
