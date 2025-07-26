#!/usr/bin/env python3
"""
Quick Fix Test: Improved Punctuation Prediction
==============================================

This script tests a simple rule-based improvement to add missing punctuation.
"""

import sys
import os
from pathlib import Path

# Add src to path
sys.path.append(str(Path(__file__).parent / "src"))

def simple_punctuation_rule_fix(text):
    """Apply simple rules to add punctuation"""
    text = text.strip()
    
    # Rule 1: If text ends with question words, add ?
    question_words = ['কি', 'কেন', 'কোথায়', 'কখন', 'কেমন', 'কে', 'কার']
    for word in question_words:
        if word in text:
            if not text.endswith('?'):
                text += '?'
            return text
    
    # Rule 2: If text contains exclamation context, add !
    exclamation_words = ['খুব', 'অনেক', 'চমৎকার', 'দারুণ', 'বাহ']
    for word in exclamation_words:
        if word in text:
            if not any(punct in text for punct in ['।', '?', '!']):
                text += '!'
            return text
    
    # Rule 3: Default case - add period (।)
    if not any(punct in text for punct in ['।', '?', '!', '.', ',']):
        text += '।'
    
    return text

def test_improved_prediction():
    """Test improved punctuation prediction with rule-based post-processing"""
    
    print("🔧 Testing Improved Punctuation Prediction")
    print("=" * 50)
    
    # Test cases
    test_cases = [
        ("আমি ভাত খাই", "আমি ভাত খাই।"),
        ("তুমি কেমন আছো", "তুমি কেমন আছো?"),
        ("সে খুব ভালো", "সে খুব ভালো!"),
        ("আজ আবহাওয়া কেমন", "আজ আবহাওয়া কেমন?"),
        ("বাংলাদেশ অনেক সুন্দর", "বাংলাদেশ অনেক সুন্দর!"),
        ("আমরা স্কুলে যাই", "আমরা স্কুলে যাই।"),
        ("তুমি কি খেলতে পারো", "তুমি কি খেলতে পারো?"),
        ("এটা দারুণ লাগছে", "এটা দারুণ লাগছে!"),
    ]
    
    correct_predictions = 0
    total_predictions = len(test_cases)
    
    print("📝 Test Results:")
    print("-" * 30)
    
    for i, (input_text, expected) in enumerate(test_cases):
        # Apply rule-based fix
        prediction = simple_punctuation_rule_fix(input_text)
        
        # Check if correct
        is_correct = prediction == expected
        if is_correct:
            correct_predictions += 1
        
        # Display result
        status = "✅" if is_correct else "❌"
        print(f"{i+1}. {status}")
        print(f"   Input:    {input_text}")
        print(f"   Expected: {expected}")
        print(f"   Got:      {prediction}")
        print()
    
    # Calculate accuracy
    accuracy = correct_predictions / total_predictions
    
    print("📊 IMPROVED RESULTS:")
    print(f"   Total: {total_predictions}")
    print(f"   Correct: {correct_predictions}")
    print(f"   Accuracy: {accuracy:.1%}")
    
    return accuracy

def test_hybrid_approach():
    """Test combining model with rule-based post-processing"""
    
    print("\n🤖 Testing Hybrid Model + Rules Approach")
    print("=" * 50)
    
    try:
        from src.models.baseline_model import BaselineModel
        from datasets import Dataset, DatasetDict
        
        # Quick training on simple examples
        train_examples = [
            ("আমি ভাত খাই", "আমি ভাত খাই।"),
            ("তুমি কেমন আছো", "তুমি কেমন আছো?"),
            ("সে খুব ভালো", "সে খুব ভালো!"),
        ]
        
        dataset = DatasetDict({
            'train': Dataset.from_dict({
                'unpunctuated_text': [ex[0] for ex in train_examples],
                'punctuated_text': [ex[1] for ex in train_examples]
            }),
            'validation': Dataset.from_dict({
                'unpunctuated_text': [train_examples[0][0]],
                'punctuated_text': [train_examples[0][1]]
            })
        })
        
        # Initialize and train model quickly
        config = {
            "name": "ai4bharat/indic-bert",
            "max_length": 32,
            "num_epochs": 1,
            "batch_size": 2,
            "learning_rate": 5e-5,
            "weight_decay": 0.01,
        }
        
        print("Training quick model...")
        model = BaselineModel(model_type="token_classification", config=config)
        
        if model.initialize_model():
            model_path = model.train(dataset, "quick_fix_model")
            
            if model_path:
                print("✅ Quick model trained")
                
                # Test hybrid approach
                test_inputs = [
                    "আমি খেলি",
                    "তুমি কি আসবে", 
                    "এটা চমৎকার",
                    "আমরা বাড়ি যাই"
                ]
                
                print("\n📝 Hybrid Predictions:")
                print("-" * 30)
                
                for i, text in enumerate(test_inputs):
                    # Get model prediction
                    model_pred = model.predict(text)
                    
                    # Apply rule-based fix if model didn't add punctuation
                    if not any(punct in model_pred for punct in ['।', '?', '!']):
                        final_pred = simple_punctuation_rule_fix(model_pred)
                        approach = "Model + Rules"
                    else:
                        final_pred = model_pred
                        approach = "Model Only"
                    
                    print(f"{i+1}. Input: {text}")
                    print(f"   Model: {model_pred}")
                    print(f"   Final: {final_pred}")
                    print(f"   Used: {approach}")
                    print()
                
                return True
            
        print("❌ Quick model training failed")
        return False
        
    except Exception as e:
        print(f"❌ Hybrid test failed: {e}")
        return False

def main():
    """Run improvement tests"""
    
    print("🚀 Testing Improved Bangla Punctuation Restoration")
    print("=" * 60)
    
    # Test 1: Rule-based improvement
    rule_accuracy = test_improved_prediction()
    
    # Test 2: Hybrid approach
    hybrid_success = test_hybrid_approach()
    
    print("\n" + "=" * 60)
    print("📊 IMPROVEMENT TEST SUMMARY")
    print("=" * 60)
    print(f"✅ Rule-based accuracy: {rule_accuracy:.1%}")
    print(f"✅ Hybrid approach: {'Working' if hybrid_success else 'Failed'}")
    
    if rule_accuracy >= 0.75:  # 75% threshold
        print(f"\n🎉 SIGNIFICANT IMPROVEMENT! Rule-based accuracy: {rule_accuracy:.1%}")
        print("💡 Recommendation: Implement rule-based post-processing")
    else:
        print(f"\n⚠️  Rule-based accuracy: {rule_accuracy:.1%} (needs refinement)")
    
    print("\n🔧 Next Steps:")
    print("   1. Implement rule-based post-processing")
    print("   2. Train with larger, more diverse dataset")
    print("   3. Try sequence-to-sequence model")
    print("   4. Add more sophisticated punctuation rules")
    
    return 0

if __name__ == "__main__":
    exit_code = main()
    sys.exit(exit_code)
