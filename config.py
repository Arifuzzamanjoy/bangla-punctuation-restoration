#!/usr/bin/env python3
"""
Configuration file for the Bangla Punctuation Restoration project
"""

import os

# Project Configuration
PROJECT_NAME = "bangla-punctuation-restoration"
VERSION = "1.0.0"
APPLICANT_NAME = "your_name"  # Replace with your actual name

# Paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, "data")
MODELS_DIR = os.path.join(BASE_DIR, "models")
RESULTS_DIR = os.path.join(BASE_DIR, "results")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

# Dataset Configuration
DATASET_CONFIG = {
    "original_dataset": "hishab/hishab-pr-bn-v1",
    "generated_dataset_name": f"ha-pr-bn-{APPLICANT_NAME}-generated",
    "adversarial_dataset_name": f"ha-pr-bn-{APPLICANT_NAME}-attack",
    "min_sentence_length": 5,
    "max_sentence_length": 128,
    "train_ratio": 0.8,
    "validation_ratio": 0.1,
    "test_ratio": 0.1,
}

# Punctuation Configuration
PUNCTUATION_MARKS = {
    "comma": ",",
    "period": "।",  # Dari in Bangla
    "question": "?",
    "exclamation": "!",
    "semicolon": ";",
    "colon": ":",
    "hyphen": "-",
}

# Label mapping for token classification
PUNCTUATION_LABELS = {
    "O": 0,           # No punctuation
    "COMMA": 1,       # ,
    "PERIOD": 2,      # ।
    "QUESTION": 3,    # ?
    "EXCLAMATION": 4, # !
    "SEMICOLON": 5,   # ;
    "COLON": 6,       # :
    "HYPHEN": 7       # -
}

# Inverse mapping for decoding
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

# Model Configuration
MODEL_CONFIG = {
    "model_type": "token_classification",  # Options: "token_classification" or "seq2seq"
    "baseline_model": {
        "name": "ai4bharat/indic-bert",
        "max_length": 128,
        "num_epochs": 5,
        "batch_size": 16,
        "learning_rate": 2e-5,
        "weight_decay": 0.01,
    },
    "advanced_model": {
        "name": "xlm-roberta-base",
        "max_length": 128,
        "num_epochs": 10,
        "batch_size": 16,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "warmup_ratio": 0.1,
        "gradient_accumulation_steps": 2,
        "fp16": True,
        "early_stopping_patience": 3,
    },
    "seq2seq_model": {
        "name": "google/mt5-small",
        "max_length": 128,
        "num_epochs": 8,
        "batch_size": 8,
        "learning_rate": 3e-5,
        "weight_decay": 0.01,
        "generation_max_length": 128,
    }
}

# Data Augmentation Configuration
AUGMENTATION_CONFIG = {
    "enable_augmentation": True,
    "character_noise_ratio": 0.3,
    "word_deletion_ratio": 0.3,
    "max_char_operations": 3,
    "bengali_chars": "অআইঈউঊঋএঐওঔকখগঘঙচছজঝঞটঠডঢণতথদধনপফবভমযরলশষসহড়ঢ়য়",
}

# Adversarial Attack Configuration
ADVERSARIAL_CONFIG = {
    "attack_ratio": 0.3,
    "target_success_rate": 0.7,
    "max_candidates": 10,
    "min_cos_sim": 0.7,
    "attack_methods": ["word_swap", "char_swap", "word_deletion"],
}

# Dataset Generation Configuration
GENERATION_CONFIG = {
    "min_sentences": 20000,
    "wikipedia_articles": 200,  # Increased for comprehensive collection
    "news_articles_per_site": 25,  # NEW: Articles per news site
    "include_blogs": True,  # NEW: Include blog content
    "include_educational": True,  # NEW: Include educational content
    "include_social_media": False,  # NEW: Social media (requires API)
    "include_academic": False,  # NEW: Academic content
    "news_portals": [
        "https://www.prothomalo.com/",
        "https://www.anandabazar.com/",
        "https://www.bbc.com/bengali",
        "https://www.jugantor.com/",
        "https://www.kalerkantho.com/",
        "https://www.ittefaq.com.bd/",
        "https://www.samakal.com/",
        "https://bangla.bdnews24.com/",
        "https://www.jagonews24.com/",
    ],
    "literary_works_dir": "literary_works",
    "request_timeout": 15,  # Increased timeout
    "max_retries": 3,
    "respect_robots_txt": True,  # NEW: Respect robots.txt
    "delay_between_requests": 2.0,  # NEW: Delay between requests (seconds)
    "max_concurrent_requests": 3,  # NEW: Maximum concurrent requests
    "quality_filter": True,  # NEW: Apply quality filtering
    "deduplicate": True,  # NEW: Remove duplicates
}

# Evaluation Configuration
EVALUATION_CONFIG = {
    "metrics": ["accuracy", "precision", "recall", "f1", "bleu", "rouge"],
    "per_class_metrics": True,
    "sentence_level_accuracy": True,
    "token_level_accuracy": True,
    "save_predictions": True,
    "generate_error_analysis": True,
}

# API Configuration
API_CONFIG = {
    "host": "0.0.0.0",
    "port": 8000,
    "workers": 1,
    "max_text_length": 1000,
    "rate_limit": "100/minute",
    "cors_origins": ["*"],
}

# Gradio Configuration
GRADIO_CONFIG = {
    "host": "0.0.0.0",
    "port": 7860,
    "share": False,
    "enable_queue": True,
    "max_text_length": 1000,
    "examples": [
        "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে",
        "তুমি কি আজ বাজারে যাবে না",
        "আমার নাম রহিম আমি ঢাকায় থাকি",
        "বইটি খুবই ভালো ছিল আমি অনেক কিছু শিখেছি",
        "আমি কাল তোমার সাথে দেখা করব ইনশাআল্লাহ",
    ]
}

# Logging Configuration
LOGGING_CONFIG = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": "bangla_punctuation.log",
    "max_file_size": "10MB",
    "backup_count": 5,
}

# Hardware Configuration
HARDWARE_CONFIG = {
    "use_cuda": True,
    "mixed_precision": True,
    "num_workers": 4,
    "pin_memory": True,
}

# Hugging Face Configuration
HF_CONFIG = {
    "token_env_var": "HUGGINGFACE_TOKEN",
    "cache_dir": os.path.join(BASE_DIR, ".hf_cache"),
    "upload_to_hub": True,
    "private_repo": False,
}

# Create directories if they don't exist
for directory in [DATA_DIR, MODELS_DIR, RESULTS_DIR, REPORTS_DIR]:
    os.makedirs(directory, exist_ok=True)
