"""
Adversarial attack generation for Bangla punctuation restoration.
Implements various attack strategies to create robust test datasets.
"""

import logging
import random
import re
from typing import List, Dict, Tuple, Optional, Set
import unicodedata
from dataclasses import dataclass

import numpy as np
from transformers import AutoTokenizer
import torch

logger = logging.getLogger(__name__)


@dataclass
class AttackConfig:
    """Configuration for adversarial attacks."""
    char_swap_prob: float = 0.1
    char_delete_prob: float = 0.05
    char_insert_prob: float = 0.05
    word_swap_prob: float = 0.1
    punct_noise_prob: float = 0.15
    typo_prob: float = 0.1
    space_noise_prob: float = 0.08
    max_attacks_per_text: int = 3


class BanglaAdversarialGenerator:
    """Generate adversarial examples for Bangla text."""
    
    def __init__(self, config: AttackConfig = None):
        self.config = config or AttackConfig()
        self.bangla_chars = self._get_bangla_chars()
        self.common_typos = self._get_common_typos()
        self.punctuation_marks = set('।,;:!?""''()[]{}')
        
    def _get_bangla_chars(self) -> List[str]:
        """Get common Bangla characters for substitution."""
        # Bangla Unicode range: 0980-09FF
        bangla_chars = []
        for i in range(0x0980, 0x09FF):
            char = chr(i)
            if unicodedata.category(char) in ['Lo', 'Mn', 'Mc']:  # Letter, Mark
                bangla_chars.append(char)
        return bangla_chars
    
    def _get_common_typos(self) -> Dict[str, List[str]]:
        """Common typos in Bangla typing."""
        return {
            'া': ['ো', 'ে'],
            'ি': ['ী', 'ে'],
            'ু': ['ূ', 'ো'],
            'ে': ['া', 'ি'],
            'ো': ['া', 'ু'],
            'ক': ['খ', 'গ'],
            'ত': ['থ', 'দ'],
            'প': ['ফ', 'ব'],
            'স': ['শ', 'ষ'],
            'ন': ['ণ', 'ম'],
            'র': ['ল', 'য'],
        }
    
    def generate_adversarial_dataset(
        self, 
        texts: List[str], 
        labels: List[str] = None,
        num_variants: int = 2
    ) -> Tuple[List[str], List[str], List[Dict]]:
        """
        Generate adversarial dataset from original texts.
        
        Args:
            texts: Original texts
            labels: Original labels (if available)
            num_variants: Number of adversarial variants per text
            
        Returns:
            Tuple of (adversarial_texts, adversarial_labels, attack_info)
        """
        adversarial_texts = []
        adversarial_labels = []
        attack_info = []
        
        for i, text in enumerate(texts):
            original_label = labels[i] if labels else None
            
            for variant in range(num_variants):
                attacked_text, attacks_applied = self._apply_random_attacks(text)
                
                # Generate corresponding label if original label exists
                if original_label:
                    attacked_label = self._adjust_label_for_attacks(
                        original_label, text, attacked_text, attacks_applied
                    )
                else:
                    attacked_label = None
                
                adversarial_texts.append(attacked_text)
                adversarial_labels.append(attacked_label)
                attack_info.append({
                    'original_index': i,
                    'variant': variant,
                    'attacks': attacks_applied,
                    'original_text': text,
                    'attacked_text': attacked_text
                })
        
        logger.info(f"Generated {len(adversarial_texts)} adversarial examples")
        return adversarial_texts, adversarial_labels, attack_info
    
    def _apply_random_attacks(self, text: str) -> Tuple[str, List[str]]:
        """Apply random combination of attacks to text."""
        attacks_applied = []
        modified_text = text
        
        # Randomly select number of attacks to apply
        num_attacks = random.randint(1, self.config.max_attacks_per_text)
        
        # Available attack methods
        attack_methods = [
            ('character_substitution', self._character_substitution_attack),
            ('character_deletion', self._character_deletion_attack),
            ('character_insertion', self._character_insertion_attack),
            ('word_swapping', self._word_swapping_attack),
            ('punctuation_noise', self._punctuation_noise_attack),
            ('typo_injection', self._typo_injection_attack),
            ('space_manipulation', self._space_manipulation_attack),
        ]
        
        # Randomly select and apply attacks
        selected_attacks = random.sample(attack_methods, min(num_attacks, len(attack_methods)))
        
        for attack_name, attack_func in selected_attacks:
            if random.random() < 0.7:  # 70% chance to apply each selected attack
                try:
                    modified_text = attack_func(modified_text)
                    attacks_applied.append(attack_name)
                except Exception as e:
                    logger.warning(f"Failed to apply {attack_name}: {e}")
        
        return modified_text, attacks_applied
    
    def _character_substitution_attack(self, text: str) -> str:
        """Substitute random characters with similar ones."""
        chars = list(text)
        for i in range(len(chars)):
            if random.random() < self.config.char_swap_prob:
                char = chars[i]
                if char in self.common_typos:
                    chars[i] = random.choice(self.common_typos[char])
                elif char in self.bangla_chars:
                    chars[i] = random.choice(self.bangla_chars)
        return ''.join(chars)
    
    def _character_deletion_attack(self, text: str) -> str:
        """Randomly delete characters."""
        chars = list(text)
        chars_to_keep = []
        
        for char in chars:
            if random.random() > self.config.char_delete_prob:
                chars_to_keep.append(char)
            # Don't delete too many characters to maintain readability
            elif len(chars_to_keep) < len(chars) * 0.8:
                chars_to_keep.append(char)
        
        return ''.join(chars_to_keep)
    
    def _character_insertion_attack(self, text: str) -> str:
        """Randomly insert characters."""
        chars = list(text)
        result = []
        
        for char in chars:
            result.append(char)
            if random.random() < self.config.char_insert_prob:
                # Insert a random Bangla character
                random_char = random.choice(self.bangla_chars)
                result.append(random_char)
        
        return ''.join(result)
    
    def _word_swapping_attack(self, text: str) -> str:
        """Swap adjacent words randomly."""
        words = text.split()
        if len(words) < 2:
            return text
        
        for i in range(len(words) - 1):
            if random.random() < self.config.word_swap_prob:
                words[i], words[i + 1] = words[i + 1], words[i]
        
        return ' '.join(words)
    
    def _punctuation_noise_attack(self, text: str) -> str:
        """Add, remove, or modify punctuation."""
        chars = list(text)
        result = []
        
        for i, char in enumerate(chars):
            if char in self.punctuation_marks:
                if random.random() < self.config.punct_noise_prob:
                    # Remove punctuation
                    continue
                elif random.random() < 0.3:
                    # Replace with different punctuation
                    result.append(random.choice(list(self.punctuation_marks)))
                else:
                    result.append(char)
            else:
                result.append(char)
                # Randomly add punctuation
                if random.random() < self.config.punct_noise_prob / 3:
                    result.append(random.choice(list(self.punctuation_marks)))
        
        return ''.join(result)
    
    def _typo_injection_attack(self, text: str) -> str:
        """Inject common typing errors."""
        chars = list(text)
        
        for i in range(len(chars)):
            if random.random() < self.config.typo_prob:
                char = chars[i]
                if char in self.common_typos:
                    chars[i] = random.choice(self.common_typos[char])
        
        return ''.join(chars)
    
    def _space_manipulation_attack(self, text: str) -> str:
        """Manipulate spaces (add, remove, modify)."""
        result = []
        chars = list(text)
        
        for i, char in enumerate(chars):
            if char == ' ':
                if random.random() < self.config.space_noise_prob:
                    # Remove space
                    continue
                elif random.random() < 0.2:
                    # Double space
                    result.append('  ')
                else:
                    result.append(char)
            else:
                result.append(char)
                # Randomly add space
                if random.random() < self.config.space_noise_prob / 4:
                    result.append(' ')
        
        return ''.join(result)
    
    def _adjust_label_for_attacks(
        self, 
        original_label: str, 
        original_text: str, 
        attacked_text: str, 
        attacks: List[str]
    ) -> str:
        """
        Adjust labels based on applied attacks.
        This is a simplified version - more sophisticated alignment needed.
        """
        # For now, return the original label
        # In practice, you'd need to carefully track character-level changes
        # and adjust the label positions accordingly
        
        if len(attacked_text) == len(original_text):
            return original_label
        
        # For length changes, we need more sophisticated alignment
        # This is a placeholder implementation
        if len(attacked_text) > len(original_text):
            # Text expanded - add O labels
            extra_chars = len(attacked_text) - len(original_text)
            return original_label + ' O' * extra_chars
        else:
            # Text contracted - truncate labels
            return original_label[:len(attacked_text)]
    
    def evaluate_attack_success(
        self, 
        original_texts: List[str],
        attacked_texts: List[str],
        model,
        tokenizer
    ) -> Dict[str, float]:
        """
        Evaluate the success rate of adversarial attacks.
        
        Args:
            original_texts: Original texts
            attacked_texts: Adversarially modified texts
            model: The model to test against
            tokenizer: Tokenizer for the model
            
        Returns:
            Dictionary with attack success metrics
        """
        success_count = 0
        total_count = len(original_texts)
        
        for orig_text, attacked_text in zip(original_texts, attacked_texts):
            try:
                # Get predictions for both texts
                orig_pred = self._get_model_prediction(orig_text, model, tokenizer)
                attacked_pred = self._get_model_prediction(attacked_text, model, tokenizer)
                
                # Check if predictions differ significantly
                if self._predictions_differ(orig_pred, attacked_pred):
                    success_count += 1
                    
            except Exception as e:
                logger.warning(f"Error evaluating attack: {e}")
        
        success_rate = success_count / total_count if total_count > 0 else 0
        
        return {
            'success_rate': success_rate,
            'successful_attacks': success_count,
            'total_attacks': total_count,
            'robustness_score': 1 - success_rate
        }
    
    def _get_model_prediction(self, text: str, model, tokenizer) -> str:
        """Get model prediction for text."""
        # This is a placeholder - implement based on your model interface
        try:
            inputs = tokenizer(text, return_tensors="pt", truncation=True, padding=True)
            with torch.no_grad():
                outputs = model(**inputs)
            # Convert outputs to prediction string
            # This depends on your model's output format
            return str(outputs.logits.argmax(-1).tolist())
        except Exception as e:
            logger.warning(f"Prediction failed: {e}")
            return ""
    
    def _predictions_differ(self, pred1: str, pred2: str, threshold: float = 0.1) -> bool:
        """Check if two predictions differ significantly."""
        if pred1 == pred2:
            return False
        
        # Simple character-level difference
        if len(pred1) != len(pred2):
            return True
        
        diff_count = sum(c1 != c2 for c1, c2 in zip(pred1, pred2))
        diff_ratio = diff_count / len(pred1) if len(pred1) > 0 else 0
        
        return diff_ratio > threshold


class AdversarialDatasetBuilder:
    """Build comprehensive adversarial datasets."""
    
    def __init__(self, generator: BanglaAdversarialGenerator = None):
        self.generator = generator or BanglaAdversarialGenerator()
    
    def build_adversarial_dataset(
        self,
        original_dataset: Dict[str, List[str]],
        output_path: str = None,
        variants_per_sample: int = 2
    ) -> Dict[str, List]:
        """
        Build a comprehensive adversarial dataset.
        
        Args:
            original_dataset: Original dataset with 'text' and optionally 'labels'
            output_path: Path to save the adversarial dataset
            variants_per_sample: Number of adversarial variants per original sample
            
        Returns:
            Adversarial dataset dictionary
        """
        texts = original_dataset.get('text', [])
        labels = original_dataset.get('labels', None)
        
        logger.info(f"Building adversarial dataset from {len(texts)} samples")
        
        adv_texts, adv_labels, attack_info = self.generator.generate_adversarial_dataset(
            texts, labels, variants_per_sample
        )
        
        adversarial_dataset = {
            'original_text': texts,
            'adversarial_text': adv_texts,
            'original_labels': labels,
            'adversarial_labels': adv_labels,
            'attack_info': attack_info,
            'metadata': {
                'total_samples': len(adv_texts),
                'variants_per_original': variants_per_sample,
                'attack_config': self.generator.config.__dict__
            }
        }
        
        if output_path:
            self._save_adversarial_dataset(adversarial_dataset, output_path)
        
        return adversarial_dataset
    
    def _save_adversarial_dataset(self, dataset: Dict, output_path: str):
        """Save adversarial dataset to file."""
        import json
        import os
        
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        
        # Convert to JSON-serializable format
        serializable_dataset = {
            k: v for k, v in dataset.items() 
            if k != 'attack_info'  # Skip complex attack info for JSON
        }
        
        with open(output_path, 'w', encoding='utf-8') as f:
            json.dump(serializable_dataset, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Adversarial dataset saved to {output_path}")


def main():
    """Demo of adversarial attack generation."""
    # Sample Bangla texts
    sample_texts = [
        "আমি বাংলাদেশে থাকি আমার নাম রহিম",
        "তুমি কেমন আছো আজকে আবহাওয়া খুব ভালো",
        "বই পড়া আমার খুব প্রিয় কাজ প্রতিদিন আমি বই পড়ি"
    ]
    
    # Initialize generator
    generator = BanglaAdversarialGenerator()
    
    # Generate adversarial examples
    adv_texts, _, attack_info = generator.generate_adversarial_dataset(
        sample_texts, num_variants=2
    )
    
    # Display results
    for i, info in enumerate(attack_info):
        print(f"\n--- Example {i+1} ---")
        print(f"Original: {info['original_text']}")
        print(f"Attacked: {info['attacked_text']}")
        print(f"Attacks: {', '.join(info['attacks'])}")


if __name__ == "__main__":
    main()
