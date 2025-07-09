"""
Bangla Punctuation Restoration - Usage Examples
===============================================

This file demonstrates various ways to use the Bangla Punctuation Restoration system.
"""

import os
import sys
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

# Example 1: Basic punctuation restoration using baseline model
def example_basic_usage():
    """Basic usage example with baseline model."""
    print("=== Example 1: Basic Usage ===")
    
    from src.models.baseline_model import BaselinePunctuationModel
    
    # Sample Bangla text without punctuation
    sample_texts = [
        "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে",
        "তুমি কি জানো আজ কি দিন",
        "দয়া করে এখানে বসুন আমি এখনই আসছি",
        "আমার খুব ভালো লাগছে তোমার সাথে কথা বলতে",
        "তুমি কোথায় যাচ্ছো এত তাড়াতাড়ি"
    ]
    
    # Initialize model
    model = BaselinePunctuationModel()
    
    # Process each text
    for text in sample_texts:
        result = model.restore_punctuation(text)
        print(f"Input:  {text}")
        print(f"Output: {result['punctuated_text']}")
        print(f"Confidence: {result['confidence']:.3f}")
        print()

# Example 2: Advanced model with confidence scores
def example_advanced_usage():
    """Advanced usage with confidence scores and detailed output."""
    print("=== Example 2: Advanced Usage ===")
    
    from src.models.advanced_model import AdvancedPunctuationModel
    
    # Initialize advanced model
    model = AdvancedPunctuationModel()
    
    text = "তুমি কি আজ আমার সাথে বাজারে যাবে আমার কিছু জিনিস কিনতে হবে"
    
    # Get detailed predictions
    result = model.restore_punctuation_detailed(text)
    
    print(f"Input: {text}")
    print(f"Output: {result['punctuated_text']}")
    print(f"Overall Confidence: {result['overall_confidence']:.3f}")
    
    print("\nToken-level predictions:")
    for token_info in result['token_predictions']:
        if token_info['predicted_punct'] != 'O':
            print(f"  '{token_info['token']}' -> '{token_info['predicted_punct']}' "
                  f"(confidence: {token_info['confidence']:.3f})")

# Example 3: Batch processing
def example_batch_processing():
    """Batch processing multiple texts."""
    print("=== Example 3: Batch Processing ===")
    
    from src.models.baseline_model import BaselinePunctuationModel
    
    # Load model once
    model = BaselinePunctuationModel()
    
    # Sample batch
    texts = [
        "আমি কাল তোমার বাড়িতে যাবো",
        "তুমি কি এখানে এসেছো",
        "আমার মনে হয় তুমি ভুল বুঝেছো",
        "এটা খুবই গুরুত্বপূর্ণ বিষয়",
        "তুমি কি আমার কথা শুনছো"
    ]
    
    # Process batch
    results = model.restore_punctuation_batch(texts)
    
    for i, (original, result) in enumerate(zip(texts, results)):
        print(f"{i+1}. {original}")
        print(f"   -> {result['punctuated_text']}")

# Example 4: API usage
def example_api_usage():
    """Example of using the REST API."""
    print("=== Example 4: API Usage ===")
    
    import requests
    import json
    
    # API endpoint (assuming server is running)
    url = "http://localhost:8000/restore-punctuation"
    
    # Sample data
    data = {
        "text": "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে",
        "model_type": "baseline"  # or "advanced"
    }
    
    try:
        # Make request
        response = requests.post(url, json=data)
        
        if response.status_code == 200:
            result = response.json()
            print(f"Input: {data['text']}")
            print(f"Output: {result['punctuated_text']}")
            print(f"Processing time: {result['processing_time']:.3f}s")
        else:
            print(f"API Error: {response.status_code}")
            
    except requests.exceptions.ConnectionError:
        print("API server not running. Start with: python scripts/deploy_api.py")

# Example 5: Adversarial testing
def example_adversarial_testing():
    """Example of testing model robustness."""
    print("=== Example 5: Adversarial Testing ===")
    
    from src.data.adversarial_attacks import AdversarialAttackGenerator
    from src.models.baseline_model import BaselinePunctuationModel
    
    # Initialize components
    model = BaselinePunctuationModel()
    attack_generator = AdversarialAttackGenerator()
    
    # Original text
    original_text = "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে"
    original_result = model.restore_punctuation(original_text)
    
    print(f"Original: {original_text}")
    print(f"Punctuated: {original_result['punctuated_text']}")
    print()
    
    # Generate adversarial examples
    adversarial_examples = attack_generator.generate_attacks(
        original_text, 
        target_model=model,
        num_examples=3
    )
    
    print("Adversarial examples:")
    for i, adv_text in enumerate(adversarial_examples):
        adv_result = model.restore_punctuation(adv_text)
        print(f"{i+1}. {adv_text}")
        print(f"   -> {adv_result['punctuated_text']}")
        print(f"   Confidence drop: {original_result['confidence'] - adv_result['confidence']:.3f}")
        print()

# Example 6: Model comparison
def example_model_comparison():
    """Compare different models on the same text."""
    print("=== Example 6: Model Comparison ===")
    
    from src.models.baseline_model import BaselinePunctuationModel
    from src.models.advanced_model import AdvancedPunctuationModel
    
    text = "তুমি কি জানো আজ কি দিন আমি ভুলে গেছি"
    
    # Test with both models
    baseline_model = BaselinePunctuationModel()
    advanced_model = AdvancedPunctuationModel()
    
    baseline_result = baseline_model.restore_punctuation(text)
    advanced_result = advanced_model.restore_punctuation(text)
    
    print(f"Input: {text}")
    print(f"Baseline:  {baseline_result['punctuated_text']} (conf: {baseline_result['confidence']:.3f})")
    print(f"Advanced:  {advanced_result['punctuated_text']} (conf: {advanced_result['confidence']:.3f})")

# Example 7: Custom evaluation
def example_custom_evaluation():
    """Custom evaluation on user data."""
    print("=== Example 7: Custom Evaluation ===")
    
    from src.models.model_utils import ModelEvaluator
    
    # Sample test data (ground truth pairs)
    test_data = [
        {
            "input": "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি আজ স্কুলে",
            "expected": "আমি তোমাকে বলেছিলাম, তুমি কেন আসোনি আজ স্কুলে?"
        },
        {
            "input": "তুমি কি জানো আজ কি দিন",
            "expected": "তুমি কি জানো আজ কি দিন?"
        },
        {
            "input": "দয়া করে এখানে বসুন আমি এখনই আসছি",
            "expected": "দয়া করে এখানে বসুন, আমি এখনই আসছি।"
        }
    ]
    
    # Evaluate model
    evaluator = ModelEvaluator()
    metrics = evaluator.evaluate_on_samples(test_data, model_type="baseline")
    
    print("Evaluation Results:")
    print(f"Token Accuracy: {metrics['token_accuracy']:.3f}")
    print(f"Sentence Accuracy: {metrics['sentence_accuracy']:.3f}")
    print(f"F1 Score: {metrics['f1_score']:.3f}")
    print(f"BLEU Score: {metrics['bleu_score']:.3f}")

# Example 8: Configuration customization
def example_config_customization():
    """Example of customizing model configuration."""
    print("=== Example 8: Configuration Customization ===")
    
    import config
    from src.models.baseline_model import BaselinePunctuationModel
    
    print("Current configuration:")
    print(f"Model name: {config.MODEL_NAME}")
    print(f"Max sequence length: {config.MAX_SEQUENCE_LENGTH}")
    print(f"Batch size: {config.BATCH_SIZE}")
    print(f"Supported punctuation: {list(config.PUNCTUATION_LABELS.keys())}")
    
    # Create model with custom settings
    custom_config = {
        "max_length": 256,
        "confidence_threshold": 0.8,
        "use_ensemble": True
    }
    
    model = BaselinePunctuationModel(config=custom_config)
    
    text = "আমি তোমাকে বলেছিলাম তুমি কেন আসোনি"
    result = model.restore_punctuation(text)
    
    print(f"\nWith custom config:")
    print(f"Input: {text}")
    print(f"Output: {result['punctuated_text']}")

# Example 9: Error analysis
def example_error_analysis():
    """Analyze common errors and patterns."""
    print("=== Example 9: Error Analysis ===")
    
    from src.models.model_utils import ErrorAnalyzer
    
    # Sample error cases
    error_cases = [
        {
            "input": "তুমি কি এসেছো",
            "predicted": "তুমি কি এসেছো।",
            "expected": "তুমি কি এসেছো?"
        },
        {
            "input": "আমি যাবো তুমি আসবে",
            "predicted": "আমি যাবো তুমি আসবে।",
            "expected": "আমি যাবো, তুমি আসবে।"
        }
    ]
    
    analyzer = ErrorAnalyzer()
    error_patterns = analyzer.analyze_errors(error_cases)
    
    print("Common error patterns:")
    for pattern, count in error_patterns.items():
        print(f"  {pattern}: {count} occurrences")

# Example 10: Data augmentation
def example_data_augmentation():
    """Example of data augmentation techniques."""
    print("=== Example 10: Data Augmentation ===")
    
    from src.data.data_processor import DataAugmentationPipeline
    
    # Original text
    text = "আমি কাল তোমার বাড়িতে যাবো"
    
    # Initialize augmentation pipeline
    augmentor = DataAugmentationPipeline()
    
    # Generate augmented versions
    augmented_texts = augmentor.augment_text(text, num_augmentations=5)
    
    print(f"Original: {text}")
    print("Augmented versions:")
    for i, aug_text in enumerate(augmented_texts, 1):
        print(f"{i}. {aug_text}")

def main():
    """Run all examples."""
    print("Bangla Punctuation Restoration - Usage Examples")
    print("=" * 50)
    
    examples = [
        example_basic_usage,
        example_advanced_usage,
        example_batch_processing,
        example_api_usage,
        example_adversarial_testing,
        example_model_comparison,
        example_custom_evaluation,
        example_config_customization,
        example_error_analysis,
        example_data_augmentation
    ]
    
    for example_func in examples:
        try:
            example_func()
            print("-" * 50)
        except Exception as e:
            print(f"Error in {example_func.__name__}: {e}")
            print("-" * 50)

if __name__ == "__main__":
    main()
