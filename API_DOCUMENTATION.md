# API Documentation

## Overview

The Bangla Punctuation Restoration system provides both REST API and Python API interfaces for restoring punctuation in Bangla text.

## REST API

### Base URL
```
http://localhost:8000
```

### Authentication
Currently, no authentication is required for local deployment.

### Endpoints

#### 1. Restore Punctuation

**Endpoint:** `POST /restore-punctuation`

**Description:** Restores punctuation in the provided Bangla text.

**Request Body:**
```json
{
    "text": "string (required) - The unpunctuated Bangla text",
    "model_type": "string (optional) - 'baseline' or 'advanced', default: 'baseline'",
    "confidence_threshold": "float (optional) - Minimum confidence for predictions, default: 0.5",
    "include_details": "boolean (optional) - Include token-level details, default: false"
}
```

**Response:**
```json
{
    "punctuated_text": "string - Text with restored punctuation",
    "confidence": "float - Overall confidence score",
    "processing_time": "float - Processing time in seconds",
    "model_used": "string - Model type used",
    "details": {
        "token_predictions": [
            {
                "token": "string - Original token",
                "predicted_punct": "string - Predicted punctuation",
                "confidence": "float - Token confidence"
            }
        ]
    }
}
```

**Example Request:**
```bash
curl -X POST "http://localhost:8000/restore-punctuation" \
     -H "Content-Type: application/json" \
     -d '{
         "text": "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে",
         "model_type": "advanced",
         "include_details": true
     }'
```

**Example Response:**
```json
{
    "punctuated_text": "আমি তোমাকে বলেছিলাম, তুমি কেন আসোনি আজ স্কুলে?",
    "confidence": 0.87,
    "processing_time": 0.123,
    "model_used": "advanced",
    "details": {
        "token_predictions": [
            {"token": "বলেছিলাম", "predicted_punct": ",", "confidence": 0.92},
            {"token": "স্কুলে", "predicted_punct": "?", "confidence": 0.89}
        ]
    }
}
```

#### 2. Batch Processing

**Endpoint:** `POST /restore-punctuation/batch`

**Description:** Processes multiple texts in a single request.

**Request Body:**
```json
{
    "texts": ["string", "string", ...],
    "model_type": "string (optional)",
    "confidence_threshold": "float (optional)"
}
```

**Response:**
```json
{
    "results": [
        {
            "input_text": "string",
            "punctuated_text": "string",
            "confidence": "float"
        }
    ],
    "total_processing_time": "float",
    "model_used": "string"
}
```

#### 3. Model Information

**Endpoint:** `GET /models`

**Description:** Get information about available models.

**Response:**
```json
{
    "available_models": [
        {
            "name": "baseline",
            "description": "Token classification model",
            "accuracy": 0.852,
            "f1_score": 0.83
        },
        {
            "name": "advanced",
            "description": "Enhanced model with adversarial training",
            "accuracy": 0.897,
            "f1_score": 0.87
        }
    ]
}
```

#### 4. Health Check

**Endpoint:** `GET /health`

**Description:** Check API health status.

**Response:**
```json
{
    "status": "healthy",
    "timestamp": "2025-01-XX XX:XX:XX",
    "version": "1.0.0",
    "models_loaded": ["baseline", "advanced"]
}
```

#### 5. Model Evaluation

**Endpoint:** `POST /evaluate`

**Description:** Evaluate model performance on provided test data.

**Request Body:**
```json
{
    "test_data": [
        {
            "input": "string - unpunctuated text",
            "expected": "string - expected punctuated text"
        }
    ],
    "model_type": "string (optional)"
}
```

**Response:**
```json
{
    "evaluation_results": {
        "token_accuracy": "float",
        "sentence_accuracy": "float",
        "f1_score": "float",
        "bleu_score": "float",
        "rouge_l": "float"
    },
    "detailed_results": [
        {
            "input": "string",
            "predicted": "string",
            "expected": "string",
            "correct": "boolean"
        }
    ]
}
```

### Error Responses

All endpoints return error responses in the following format:

```json
{
    "error": "string - Error type",
    "message": "string - Detailed error message",
    "timestamp": "string - ISO timestamp"
}
```

**Common Error Codes:**
- `400 Bad Request` - Invalid input data
- `422 Unprocessable Entity` - Validation errors
- `500 Internal Server Error` - Server errors

## Python API

### Installation

```python
from src.models.baseline_model import BaselinePunctuationModel
from src.models.advanced_model import AdvancedPunctuationModel
```

### Basic Usage

#### BaselinePunctuationModel

```python
# Initialize model
model = BaselinePunctuationModel()

# Simple restoration
result = model.restore_punctuation("আমি তোমাকে বলেছিলাম")
print(result['punctuated_text'])  # "আমি তোমাকে বলেছিলাম।"

# Batch processing
texts = ["text1", "text2", "text3"]
results = model.restore_punctuation_batch(texts)

# Custom configuration
config = {
    "confidence_threshold": 0.8,
    "max_length": 256
}
model = BaselinePunctuationModel(config=config)
```

#### AdvancedPunctuationModel

```python
# Initialize advanced model
model = AdvancedPunctuationModel()

# Detailed restoration
result = model.restore_punctuation_detailed("আমি তোমাকে বলেছিলাম")
print(result['token_predictions'])  # Token-level details

# With ensemble
model = AdvancedPunctuationModel(use_ensemble=True)
result = model.restore_punctuation("text")
```

### API Methods

#### restore_punctuation(text, **kwargs)

**Parameters:**
- `text` (str): Input text without punctuation
- `confidence_threshold` (float, optional): Minimum confidence for predictions
- `return_confidence` (bool, optional): Include confidence scores

**Returns:**
```python
{
    'punctuated_text': str,
    'confidence': float,
    'processing_time': float
}
```

#### restore_punctuation_batch(texts, **kwargs)

**Parameters:**
- `texts` (List[str]): List of input texts
- `batch_size` (int, optional): Processing batch size

**Returns:**
```python
[
    {
        'punctuated_text': str,
        'confidence': float
    }
]
```

#### restore_punctuation_detailed(text, **kwargs)

**Parameters:**
- `text` (str): Input text
- `include_attention` (bool, optional): Include attention weights

**Returns:**
```python
{
    'punctuated_text': str,
    'overall_confidence': float,
    'token_predictions': [
        {
            'token': str,
            'predicted_punct': str,
            'confidence': float,
            'attention_weights': List[float]  # if included
        }
    ]
}
```

### Evaluation API

```python
from src.models.model_utils import ModelEvaluator

# Initialize evaluator
evaluator = ModelEvaluator()

# Evaluate on test data
test_data = [
    {'input': 'text1', 'expected': 'text1.'},
    {'input': 'text2', 'expected': 'text2?'}
]

metrics = evaluator.evaluate_on_samples(test_data, model_type="baseline")
print(metrics)
```

### Adversarial Testing API

```python
from src.data.adversarial_attacks import AdversarialAttackGenerator

# Initialize attack generator
attack_gen = AdversarialAttackGenerator()

# Generate adversarial examples
original_text = "আমি তোমাকে বলেছিলাম"
adversarial_examples = attack_gen.generate_attacks(
    original_text,
    target_model=model,
    attack_types=['synonym_replacement', 'word_insertion'],
    num_examples=5
)

# Evaluate robustness
robustness_score = attack_gen.evaluate_robustness(
    original_text,
    model,
    num_attacks=100
)
```

### Configuration API

```python
import config

# View current configuration
print(config.MODEL_NAME)
print(config.PUNCTUATION_LABELS)

# Update configuration
config.CONFIDENCE_THRESHOLD = 0.8
config.MAX_SEQUENCE_LENGTH = 256

# Load custom configuration
from src.models.baseline_model import BaselinePunctuationModel

custom_config = {
    'model_name': 'custom-model',
    'confidence_threshold': 0.9
}

model = BaselinePunctuationModel(config=custom_config)
```

## Gradio Web Interface

The Gradio interface provides a user-friendly web interface for testing the models.

### Access

Navigate to: `http://localhost:7860` (when running locally)

### Features

1. **Text Input**: Enter unpunctuated Bangla text
2. **Model Selection**: Choose between baseline and advanced models
3. **Confidence Threshold**: Adjust prediction confidence
4. **Live Results**: Real-time punctuation restoration
5. **Token-level Details**: View per-token predictions
6. **Example Texts**: Pre-loaded sample texts for testing

### Usage

1. Start the Gradio interface:
   ```bash
   python scripts/deploy_api.py --interface gradio
   ```

2. Open browser and navigate to the provided URL

3. Enter text and click "Restore Punctuation"

4. View results with confidence scores and token details

## Deployment

### Local Deployment

```bash
# Start FastAPI server
python scripts/deploy_api.py --interface fastapi --port 8000

# Start Gradio interface
python scripts/deploy_api.py --interface gradio --port 7860

# Start both interfaces
python scripts/deploy_api.py --interface both
```

### Docker Deployment

```bash
# Build Docker image
docker build -t bangla-punctuation-api .

# Run container
docker run -p 8000:8000 bangla-punctuation-api
```

### Production Deployment

For production deployment, consider:

1. **Load Balancing**: Use multiple API instances
2. **Caching**: Cache model predictions for common texts
3. **Rate Limiting**: Implement request rate limiting
4. **Monitoring**: Add performance monitoring and logging
5. **Security**: Implement authentication and input validation

## Rate Limits

For production deployment, recommended rate limits:

- **Standard Users**: 100 requests per minute
- **Premium Users**: 1000 requests per minute
- **Batch Endpoint**: 10 requests per minute (max 100 texts per request)

## SDK Examples

### JavaScript/Node.js

```javascript
const axios = require('axios');

async function restorePunctuation(text) {
    try {
        const response = await axios.post('http://localhost:8000/restore-punctuation', {
            text: text,
            model_type: 'advanced'
        });
        return response.data;
    } catch (error) {
        console.error('Error:', error.response.data);
    }
}

// Usage
restorePunctuation('আমি তোমাকে বলেছিলাম').then(result => {
    console.log(result.punctuated_text);
});
```

### Python Requests

```python
import requests

def restore_punctuation(text, model_type='baseline'):
    url = 'http://localhost:8000/restore-punctuation'
    data = {
        'text': text,
        'model_type': model_type
    }
    
    response = requests.post(url, json=data)
    if response.status_code == 200:
        return response.json()
    else:
        raise Exception(f"API Error: {response.status_code}")

# Usage
result = restore_punctuation('আমি তোমাকে বলেছিলাম', 'advanced')
print(result['punctuated_text'])
```

## Troubleshooting

### Common Issues

1. **Model Loading Error**
   - Ensure model files are in the correct directory
   - Check model path in configuration

2. **Out of Memory**
   - Reduce batch size
   - Use shorter input sequences

3. **Slow Processing**
   - Use GPU if available
   - Optimize model configuration

4. **API Connection Error**
   - Verify server is running
   - Check port configuration

### Debug Mode

Enable debug mode for detailed logging:

```python
import logging
logging.basicConfig(level=logging.DEBUG)

# Or via environment variable
export DEBUG=True
python scripts/deploy_api.py
```

## Support

For additional support:

1. Check the [examples.py](examples.py) file for usage examples
2. Review the test files in the `tests/` directory
3. Open an issue on the project repository
4. Contact the development team

## Changelog

### Version 1.0.0
- Initial release with baseline and advanced models
- REST API and Gradio interface
- Adversarial testing capabilities
- Comprehensive evaluation metrics
