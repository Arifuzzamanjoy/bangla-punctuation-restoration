#!/usr/bin/env python3
"""
Modern Data Pipeline with Advanced Augmentation Techniques
"""

import torch
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional
import random
import re
from dataclasses import dataclass
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import multiprocessing as mp
from functools import partial
import logging
from pathlib import Path
import json
import pickle
from sklearn.model_selection import StratifiedKFold
from transformers import pipeline
import asyncio
import aiohttp
from datasets import Dataset, DatasetDict
import nltk
from nltk.corpus import wordnet
import spacy

logger = logging.getLogger(__name__)

@dataclass
class AugmentationConfig:
    """Configuration for data augmentation"""
    synonym_replacement_prob: float = 0.1
    random_insertion_prob: float = 0.1
    random_swap_prob: float = 0.1
    random_deletion_prob: float = 0.05
    back_translation_prob: float = 0.2
    paraphrasing_prob: float = 0.15
    noise_injection_prob: float = 0.05
    style_transfer_prob: float = 0.1

class ModernDataAugmentor:
    """State-of-the-art data augmentation for Bangla text"""
    
    def __init__(self, config: AugmentationConfig = None):
        self.config = config or AugmentationConfig()
        self.setup_models()
    
    def setup_models(self):
        """Setup models for augmentation"""
        try:
            # Paraphrasing model
            self.paraphraser = pipeline(
                "text2text-generation",
                model="humarin/chatgpt_paraphraser_on_T5_base",
                device=0 if torch.cuda.is_available() else -1
            )
            
            # Translation models for back-translation
            self.bn_to_en = pipeline(
                "translation",
                model="Helsinki-NLP/opus-mt-bn-en",
                device=0 if torch.cuda.is_available() else -1
            )
            
            self.en_to_bn = pipeline(
                "translation", 
                model="Helsinki-NLP/opus-mt-en-bn",
                device=0 if torch.cuda.is_available() else -1
            )
            
        except Exception as e:
            logger.warning(f"Could not load all augmentation models: {e}")
            self.paraphraser = None
            self.bn_to_en = None
            self.en_to_bn = None
    
    def augment_dataset(self, 
                       dataset: List[Dict], 
                       augmentation_factor: int = 2) -> List[Dict]:
        """Augment dataset with multiple techniques"""
        
        augmented_data = []
        original_data = dataset.copy()
        
        # Apply different augmentation techniques
        for _ in range(augmentation_factor):
            for item in original_data:
                augmented_item = self.apply_random_augmentation(item)
                if augmented_item:
                    augmented_data.append(augmented_item)
        
        # Combine original and augmented data
        return original_data + augmented_data
    
    def apply_random_augmentation(self, item: Dict) -> Optional[Dict]:
        """Apply random augmentation to a single item"""
        text = item['unpunctuated_text']
        punctuated = item['punctuated_text']
        
        # Choose random augmentation technique
        techniques = [
            (self.synonym_replacement, self.config.synonym_replacement_prob),
            (self.random_insertion, self.config.random_insertion_prob),
            (self.random_swap, self.config.random_swap_prob),
            (self.random_deletion, self.config.random_deletion_prob),
            (self.back_translate, self.config.back_translation_prob),
            (self.paraphrase, self.config.paraphrasing_prob),
            (self.add_noise, self.config.noise_injection_prob),
            (self.style_transfer, self.config.style_transfer_prob)
        ]
        
        # Apply augmentation based on probability
        for technique, prob in techniques:
            if random.random() < prob:
                try:
                    aug_text = technique(text)
                    if aug_text and aug_text != text:
                        # Generate new punctuated version
                        aug_punctuated = self.transfer_punctuation(
                            text, punctuated, aug_text
                        )
                        return {
                            'unpunctuated_text': aug_text,
                            'punctuated_text': aug_punctuated,
                            'augmentation_type': technique.__name__
                        }
                except Exception as e:
                    logger.warning(f"Augmentation failed: {e}")
                    continue
        
        return None
    
    def synonym_replacement(self, text: str, n: int = 1) -> str:
        """Replace words with synonyms"""
        words = text.split()
        new_words = words.copy()
        
        # Simple synonym replacement using word variations
        synonym_map = {
            'ভালো': ['উত্তম', 'চমৎকার', 'সুন্দর'],
            'খারাপ': ['মন্দ', 'নিকৃষ্ট', 'দুর্বল'],
            'বড়': ['বৃহৎ', 'বিশাল', 'প্রকাণ্ড'],
            'ছোট': ['ক্ষুদ্র', 'সূক্ষ্ম', 'নিচু']
        }
        
        random_words = random.sample(range(len(words)), min(n, len(words)))
        
        for word_idx in random_words:
            word = words[word_idx]
            if word in synonym_map:
                new_words[word_idx] = random.choice(synonym_map[word])
        
        return ' '.join(new_words)
    
    def random_insertion(self, text: str, n: int = 1) -> str:
        """Randomly insert words"""
        words = text.split()
        
        # Common Bangla filler words
        filler_words = ['তাহলে', 'আসলে', 'মানে', 'যেমন', 'অর্থাৎ']
        
        for _ in range(n):
            new_word = random.choice(filler_words)
            position = random.randint(0, len(words))
            words.insert(position, new_word)
        
        return ' '.join(words)
    
    def random_swap(self, text: str, n: int = 1) -> str:
        """Randomly swap words"""
        words = text.split()
        
        for _ in range(n):
            if len(words) < 2:
                break
            idx1, idx2 = random.sample(range(len(words)), 2)
            words[idx1], words[idx2] = words[idx2], words[idx1]
        
        return ' '.join(words)
    
    def random_deletion(self, text: str, p: float = 0.1) -> str:
        """Randomly delete words"""
        words = text.split()
        
        if len(words) == 1:
            return text
        
        new_words = []
        for word in words:
            if random.random() > p:
                new_words.append(word)
        
        return ' '.join(new_words) if new_words else text
    
    def back_translate(self, text: str) -> str:
        """Back-translation augmentation"""
        if not (self.bn_to_en and self.en_to_bn):
            return text
        
        try:
            # Translate to English
            en_text = self.bn_to_en(text)[0]['translation_text']
            
            # Translate back to Bangla
            back_translated = self.en_to_bn(en_text)[0]['translation_text']
            
            return back_translated
        except:
            return text
    
    def paraphrase(self, text: str) -> str:
        """Paraphrase using T5 model"""
        if not self.paraphraser:
            return text
        
        try:
            paraphrased = self.paraphraser(f"paraphrase: {text}")
            return paraphrased[0]['generated_text']
        except:
            return text
    
    def add_noise(self, text: str, noise_level: float = 0.02) -> str:
        """Add character-level noise"""
        chars = list(text)
        num_changes = int(len(chars) * noise_level)
        
        for _ in range(num_changes):
            if chars:
                idx = random.randint(0, len(chars) - 1)
                # Random character substitution with similar Bangla characters
                similar_chars = {
                    'া': 'ো', 'ি': 'ী', 'ু': 'ূ', 'ে': 'ৈ',
                    'ক': 'খ', 'গ': 'ঘ', 'চ': 'ছ', 'জ': 'ঝ',
                    'ট': 'ঠ', 'ড': 'ঢ', 'ত': 'থ', 'দ': 'ধ',
                    'প': 'ফ', 'ব': 'ভ', 'ম': 'ন', 'র': 'ল'
                }
                
                if chars[idx] in similar_chars:
                    chars[idx] = similar_chars[chars[idx]]
        
        return ''.join(chars)
    
    def style_transfer(self, text: str) -> str:
        """Transfer between formal/informal styles"""
        # Simple style transfer rules
        formal_to_informal = {
            'আপনি': 'তুমি',
            'আপনার': 'তোমার',
            'করবেন': 'করবে',
            'পারবেন': 'পারবে'
        }
        
        informal_to_formal = {v: k for k, v in formal_to_informal.items()}
        
        # Randomly choose direction
        if random.random() < 0.5:
            replacements = formal_to_informal
        else:
            replacements = informal_to_formal
        
        for old, new in replacements.items():
            text = text.replace(old, new)
        
        return text
    
    def transfer_punctuation(self, 
                           original: str, 
                           original_punctuated: str, 
                           augmented: str) -> str:
        """Transfer punctuation pattern to augmented text"""
        # Simple approach: align words and transfer punctuation
        orig_words = original.split()
        aug_words = augmented.split()
        
        # Extract punctuation pattern
        punctuation_pattern = self.extract_punctuation_pattern(
            original, original_punctuated
        )
        
        # Apply pattern to augmented text
        return self.apply_punctuation_pattern(augmented, punctuation_pattern)
    
    def extract_punctuation_pattern(self, 
                                  unpunctuated: str, 
                                  punctuated: str) -> List[str]:
        """Extract punctuation pattern from text pair"""
        pattern = []
        words = unpunctuated.split()
        
        # Simple extraction logic
        for word in words:
            word_end = punctuated.find(word) + len(word)
            if word_end < len(punctuated):
                next_char = punctuated[word_end]
                if next_char in '।?!,;:-':
                    pattern.append(next_char)
                else:
                    pattern.append('')
            else:
                pattern.append('')
        
        return pattern
    
    def apply_punctuation_pattern(self, 
                                text: str, 
                                pattern: List[str]) -> str:
        """Apply punctuation pattern to text"""
        words = text.split()
        result = []
        
        for i, word in enumerate(words):
            result.append(word)
            if i < len(pattern) and pattern[i]:
                result.append(pattern[i])
        
        return ' '.join(result)

class AdvancedDataLoader:
    """Modern data loading with streaming and caching"""
    
    def __init__(self, cache_dir: str = "data/cache"):
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
    
    def load_streaming_dataset(self, 
                             source_urls: List[str],
                             batch_size: int = 1000) -> Dataset:
        """Load data in streaming fashion"""
        
        async def fetch_data(session, url):
            try:
                async with session.get(url) as response:
                    return await response.json()
            except Exception as e:
                logger.error(f"Failed to fetch {url}: {e}")
                return None
        
        async def fetch_all_data(urls):
            async with aiohttp.ClientSession() as session:
                tasks = [fetch_data(session, url) for url in urls]
                results = await asyncio.gather(*tasks)
                return [r for r in results if r is not None]
        
        # Fetch data asynchronously
        loop = asyncio.get_event_loop()
        data = loop.run_until_complete(fetch_all_data(source_urls))
        
        return Dataset.from_list(data)
    
    def create_cached_dataset(self, 
                            dataset: Dataset, 
                            cache_key: str) -> Dataset:
        """Create cached version of dataset"""
        cache_file = self.cache_dir / f"{cache_key}.pkl"
        
        if cache_file.exists():
            logger.info(f"Loading cached dataset: {cache_file}")
            with open(cache_file, 'rb') as f:
                return pickle.load(f)
        
        logger.info(f"Caching dataset: {cache_file}")
        with open(cache_file, 'wb') as f:
            pickle.dump(dataset, f)
        
        return dataset

class SmartDataSplitter:
    """Intelligent data splitting with stratification"""
    
    @staticmethod
    def stratified_split_by_length(dataset: List[Dict], 
                                 splits: Tuple[float, float, float] = (0.8, 0.1, 0.1),
                                 random_state: int = 42) -> Tuple[List[Dict], List[Dict], List[Dict]]:
        """Split dataset stratified by text length"""
        
        # Categorize by length
        def get_length_category(text: str) -> str:
            length = len(text.split())
            if length < 10:
                return "short"
            elif length < 30:
                return "medium"
            else:
                return "long"
        
        # Add length categories
        for item in dataset:
            item['length_category'] = get_length_category(item['unpunctuated_text'])
        
        # Create DataFrame for easier manipulation
        df = pd.DataFrame(dataset)
        
        # Stratified split
        train_data, temp_data = train_test_split(
            df, 
            test_size=1-splits[0], 
            stratify=df['length_category'],
            random_state=random_state
        )
        
        val_ratio = splits[1] / (splits[1] + splits[2])
        val_data, test_data = train_test_split(
            temp_data,
            test_size=1-val_ratio,
            stratify=temp_data['length_category'],
            random_state=random_state
        )
        
        return (train_data.to_dict('records'), 
                val_data.to_dict('records'), 
                test_data.to_dict('records'))

class DataQualityAnalyzer:
    """Analyze and improve data quality"""
    
    def __init__(self):
        self.quality_metrics = {}
    
    def analyze_dataset(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Comprehensive data quality analysis"""
        
        analysis = {
            'total_samples': len(dataset),
            'length_distribution': self._analyze_length_distribution(dataset),
            'punctuation_distribution': self._analyze_punctuation_distribution(dataset),
            'language_quality': self._analyze_language_quality(dataset),
            'duplicates': self._find_duplicates(dataset),
            'anomalies': self._detect_anomalies(dataset)
        }
        
        return analysis
    
    def _analyze_length_distribution(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Analyze text length distribution"""
        lengths = [len(item['unpunctuated_text'].split()) for item in dataset]
        
        return {
            'mean': np.mean(lengths),
            'std': np.std(lengths),
            'min': np.min(lengths),
            'max': np.max(lengths),
            'percentiles': {
                '25': np.percentile(lengths, 25),
                '50': np.percentile(lengths, 50),
                '75': np.percentile(lengths, 75),
                '95': np.percentile(lengths, 95)
            }
        }
    
    def _analyze_punctuation_distribution(self, dataset: List[Dict]) -> Dict[str, int]:
        """Analyze punctuation mark distribution"""
        punctuation_counts = {}
        punctuation_marks = ['।', '?', '!', ',', ';', ':', '-']
        
        for mark in punctuation_marks:
            punctuation_counts[mark] = 0
        
        for item in dataset:
            text = item['punctuated_text']
            for mark in punctuation_marks:
                punctuation_counts[mark] += text.count(mark)
        
        return punctuation_counts
    
    def _analyze_language_quality(self, dataset: List[Dict]) -> Dict[str, Any]:
        """Analyze language quality metrics"""
        
        bangla_char_ratio = []
        for item in dataset:
            text = item['unpunctuated_text']
            bangla_chars = sum(1 for char in text if '\u0980' <= char <= '\u09FF')
            total_chars = len(text)
            ratio = bangla_chars / max(total_chars, 1)
            bangla_char_ratio.append(ratio)
        
        return {
            'avg_bangla_char_ratio': np.mean(bangla_char_ratio),
            'min_bangla_char_ratio': np.min(bangla_char_ratio),
            'samples_below_threshold': sum(1 for r in bangla_char_ratio if r < 0.8)
        }
    
    def _find_duplicates(self, dataset: List[Dict]) -> Dict[str, int]:
        """Find duplicate entries"""
        texts = [item['unpunctuated_text'] for item in dataset]
        unique_texts = set(texts)
        
        return {
            'total_duplicates': len(texts) - len(unique_texts),
            'duplicate_ratio': 1 - len(unique_texts) / len(texts)
        }
    
    def _detect_anomalies(self, dataset: List[Dict]) -> List[Dict]:
        """Detect anomalous entries"""
        anomalies = []
        
        for i, item in enumerate(dataset):
            issues = []
            
            # Check for extremely short/long texts
            word_count = len(item['unpunctuated_text'].split())
            if word_count < 3:
                issues.append("too_short")
            elif word_count > 100:
                issues.append("too_long")
            
            # Check for missing punctuation
            if not any(p in item['punctuated_text'] for p in '।?!,;:-'):
                issues.append("no_punctuation")
            
            # Check for excessive punctuation
            punct_ratio = sum(1 for c in item['punctuated_text'] if c in '।?!,;:-') / len(item['punctuated_text'])
            if punct_ratio > 0.3:
                issues.append("excessive_punctuation")
            
            if issues:
                anomalies.append({
                    'index': i,
                    'text': item['unpunctuated_text'][:50] + "...",
                    'issues': issues
                })
        
        return anomalies
    
    def clean_dataset(self, dataset: List[Dict]) -> List[Dict]:
        """Clean dataset based on quality analysis"""
        cleaned_dataset = []
        
        for item in dataset:
            # Filter criteria
            word_count = len(item['unpunctuated_text'].split())
            
            # Skip if too short or too long
            if word_count < 3 or word_count > 100:
                continue
            
            # Skip if no Bangla characters
            bangla_chars = sum(1 for char in item['unpunctuated_text'] 
                             if '\u0980' <= char <= '\u09FF')
            if bangla_chars / len(item['unpunctuated_text']) < 0.5:
                continue
            
            # Skip if no punctuation in punctuated version
            if not any(p in item['punctuated_text'] for p in '।?!,;:-'):
                continue
            
            cleaned_dataset.append(item)
        
        return cleaned_dataset

class ModernDataPipeline:
    """Complete modern data pipeline"""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.augmentor = ModernDataAugmentor()
        self.loader = AdvancedDataLoader()
        self.splitter = SmartDataSplitter()
        self.quality_analyzer = DataQualityAnalyzer()
    
    def process_dataset(self, 
                       raw_data: List[Dict],
                       augment: bool = True,
                       clean: bool = True,
                       cache_key: str = None) -> DatasetDict:
        """Complete dataset processing pipeline"""
        
        logger.info(f"Starting data processing pipeline with {len(raw_data)} samples")
        
        # Step 1: Quality analysis and cleaning
        if clean:
            logger.info("Analyzing data quality...")
            quality_report = self.quality_analyzer.analyze_dataset(raw_data)
            logger.info(f"Quality report: {quality_report}")
            
            logger.info("Cleaning dataset...")
            clean_data = self.quality_analyzer.clean_dataset(raw_data)
            logger.info(f"Cleaned dataset: {len(clean_data)} samples")
        else:
            clean_data = raw_data
        
        # Step 2: Data augmentation
        if augment:
            logger.info("Augmenting dataset...")
            augmented_data = self.augmentor.augment_dataset(
                clean_data, 
                augmentation_factor=self.config.get('augmentation_factor', 2)
            )
            logger.info(f"Augmented dataset: {len(augmented_data)} samples")
        else:
            augmented_data = clean_data
        
        # Step 3: Smart splitting
        logger.info("Splitting dataset...")
        train_data, val_data, test_data = self.splitter.stratified_split_by_length(
            augmented_data,
            splits=self.config.get('splits', (0.8, 0.1, 0.1))
        )
        
        # Step 4: Create Dataset objects
        dataset_dict = DatasetDict({
            'train': Dataset.from_list(train_data),
            'validation': Dataset.from_list(val_data),
            'test': Dataset.from_list(test_data)
        })
        
        # Step 5: Cache if requested
        if cache_key:
            self.loader.create_cached_dataset(dataset_dict, cache_key)
        
        logger.info("Data processing pipeline completed!")
        logger.info(f"Final splits - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        return dataset_dict

# Usage example
if __name__ == "__main__":
    # Example usage
    pipeline = ModernDataPipeline({
        'augmentation_factor': 3,
        'splits': (0.8, 0.1, 0.1)
    })
    
    # Sample data
    sample_data = [
        {
            'unpunctuated_text': 'আমি ভালো আছি তুমি কেমন আছো',
            'punctuated_text': 'আমি ভালো আছি। তুমি কেমন আছো?'
        }
    ]
    
    # Process dataset
    processed_dataset = pipeline.process_dataset(
        sample_data,
        augment=True,
        clean=True,
        cache_key="bangla_punctuation_v1"
    )
