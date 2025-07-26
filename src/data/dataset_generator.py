#!/usr/bin/env python3
"""
Dataset generator for creating new Bangla punctuation datasets from diverse sources
"""

import os
import re
import json
import random
import requests
import wikipedia
from bs4 import BeautifulSoup
from typing import List, Dict, Any, Optional
from datasets import Dataset, DatasetDict
from huggingface_hub import login, HfApi
from tqdm import tqdm
import logging

# Import the enhanced web scraper
from .web_scraper import BanglaWebScraper

# Handle config import for different execution contexts
try:
    from config import GENERATION_CONFIG, DATASET_CONFIG, HF_CONFIG
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from config import GENERATION_CONFIG, DATASET_CONFIG, HF_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Set Wikipedia language to Bengali
wikipedia.set_lang('bn')

class BanglaDatasetGenerator:
    """
    Generate new Bangla punctuation datasets from diverse sources
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dataset generator
        
        Args:
            config: Configuration dictionary
        """
        self.config = config or GENERATION_CONFIG
        self.dataset_config = DATASET_CONFIG
        self.hf_token = os.getenv(HF_CONFIG["token_env_var"])
        
        # Initialize the enhanced web scraper
        self.web_scraper = BanglaWebScraper(config)
    
    def remove_punctuation(self, text: str) -> str:
        """Remove punctuation marks from text"""
        pattern = r'[!?,;:\-।]'
        return re.sub(pattern, '', text)
    
    def extract_sentences(self, text: str) -> List[str]:
        """Extract well-formed sentences from text"""
        # Split on sentence ending markers (। ! ?)
        sentence_pattern = r'[^।!?]+[।!?]'
        sentences = re.findall(sentence_pattern, text)
        
        # Clean up sentences
        cleaned_sentences = []
        for sentence in sentences:
            sentence = sentence.strip()
            # Check minimum length and ensure it contains Bengali characters
            if (len(sentence.split()) >= self.dataset_config["min_sentence_length"] and
                any('\u0980' <= c <= '\u09FF' for c in sentence)):
                cleaned_sentences.append(sentence)
        
        return cleaned_sentences
    
    def scrape_wikipedia_articles(self, num_articles: int = None) -> List[str]:
        """Scrape Bengali Wikipedia articles"""
        if num_articles is None:
            num_articles = self.config["wikipedia_articles"]
        
        logger.info(f"Scraping {num_articles} Wikipedia articles...")
        sentences = []
        
        # Get random article titles
        random_titles = []
        for _ in range(num_articles):
            try:
                random_titles.append(wikipedia.random())
            except Exception as e:
                logger.warning(f"Error getting random title: {e}")
        
        # Fetch content for each article
        for title in tqdm(random_titles, desc="Scraping Wikipedia"):
            try:
                page = wikipedia.page(title)
                content = page.content
                article_sentences = self.extract_sentences(content)
                sentences.extend(article_sentences)
                
                # Add some delay to be respectful
                import time
                time.sleep(0.1)
                
            except Exception as e:
                logger.warning(f"Error fetching article {title}: {e}")
        
        logger.info(f"Extracted {len(sentences)} sentences from Wikipedia")
        return sentences
    
    def scrape_comprehensive_internet_data(self) -> List[str]:
        """
        Scrape comprehensive Bangla data from the entire internet
        
        Returns:
            List of sentences scraped from diverse internet sources
        """
        logger.info("Starting comprehensive internet data collection...")
        
        # Use the enhanced web scraper for comprehensive data collection
        sentences = self.web_scraper.scrape_comprehensive_bangla_data(
            wikipedia_articles=self.config.get("wikipedia_articles", 200),
            news_articles_per_site=self.config.get("news_articles_per_site", 20),
            include_blogs=self.config.get("include_blogs", True),
            include_educational=self.config.get("include_educational", True)
        )
        
        logger.info(f"Collected {len(sentences)} sentences from comprehensive internet scraping")
        return sentences
    
    def scrape_social_media_content(self) -> List[str]:
        """
        Scrape Bangla content from social media and discussion platforms
        Note: This would require API access for platforms like Facebook, Twitter, etc.
        """
        sentences = []
        
        # Placeholder for social media scraping
        # In a real implementation, you would:
        # 1. Use official APIs (Twitter API, Facebook Graph API, etc.)
        # 2. Scrape public discussion forums
        # 3. Collect content from comment sections
        
        logger.info("Social media scraping not implemented (requires API access)")
        return sentences
    
    def scrape_academic_content(self) -> List[str]:
        """
        Scrape academic and research content in Bangla
        """
        sentences = []
        
        academic_sources = [
            "https://www.researchgate.net/search/publication?q=bengali",
            "https://scholar.google.com/scholar?q=bengali+language",
            # Add more academic sources
        ]
        
        # Implementation would scrape academic papers, theses, etc.
        logger.info("Academic content scraping not fully implemented")
        return sentences
    
    def load_literary_works(self, directory: Optional[str] = None) -> List[str]:
        """Load Bengali literary works from text files"""
        if directory is None:
            directory = self.config["literary_works_dir"]
        
        if not os.path.exists(directory):
            os.makedirs(directory)
            logger.warning(f"Created directory {directory}. Please add Bengali literary text files.")
            return []
        
        sentences = []
        for filename in os.listdir(directory):
            if filename.endswith(('.txt', '.md')):
                try:
                    file_path = os.path.join(directory, filename)
                    with open(file_path, 'r', encoding='utf-8') as file:
                        content = file.read()
                        file_sentences = self.extract_sentences(content)
                        sentences.extend(file_sentences)
                        logger.info(f"Processed {filename}: {len(file_sentences)} sentences")
                except Exception as e:
                    logger.warning(f"Error reading {filename}: {e}")
        
        logger.info(f"Extracted {len(sentences)} sentences from literary works")
        return sentences
    
    def generate_synthetic_sentences(self, base_sentences: List[str], target_count: int) -> List[str]:
        """Generate synthetic sentences by modifying existing ones"""
        if len(base_sentences) >= target_count:
            return base_sentences
        
        synthetic_sentences = list(base_sentences)
        needed = target_count - len(base_sentences)
        
        logger.info(f"Generating {needed} synthetic sentences...")
        
        for _ in tqdm(range(needed), desc="Generating synthetic sentences"):
            # Pick a random base sentence
            base_sentence = random.choice(base_sentences)
            
            # Apply simple transformations
            words = base_sentence.split()
            
            if len(words) > 3:
                # Word reordering (simple)
                if random.random() < 0.3:
                    # Swap two adjacent words
                    idx = random.randint(0, len(words) - 2)
                    words[idx], words[idx + 1] = words[idx + 1], words[idx]
                
                # Word substitution with similar words (placeholder)
                if random.random() < 0.2:
                    # This is a simplified approach - in practice, you'd use word embeddings
                    substitutions = {
                        'আমি': 'আমরা',
                        'তুমি': 'তোমরা',
                        'সে': 'তারা',
                        'এটি': 'এগুলি',
                        'ভালো': 'চমৎকার',
                        'খারাপ': 'মন্দ'
                    }
                    
                    for i, word in enumerate(words):
                        if word in substitutions and random.random() < 0.5:
                            words[i] = substitutions[word]
            
            synthetic_sentence = ' '.join(words)
            
            # Ensure it still has proper punctuation
            if not any(punct in synthetic_sentence for punct in ['।', '?', '!', ',', ';', ':']):
                synthetic_sentence += '।'
            
            synthetic_sentences.append(synthetic_sentence)
        
        return synthetic_sentences
    
    def filter_and_deduplicate_sentences(self, sentences: List[str]) -> List[str]:
        """
        Apply quality filtering and deduplication to sentences
        
        Args:
            sentences: List of sentences to filter
            
        Returns:
            Filtered and deduplicated sentences
        """
        logger.info("Applying quality filtering and deduplication...")
        
        # Remove duplicates while preserving order
        unique_sentences = []
        seen = set()
        
        for sentence in sentences:
            # Normalize for comparison (remove extra spaces, etc.)
            normalized = re.sub(r'\s+', ' ', sentence).strip().lower()
            
            if normalized not in seen:
                seen.add(normalized)
                
                # Quality checks
                words = sentence.split()
                word_count = len(words)
                
                # Check length
                if not (self.dataset_config["min_sentence_length"] <= word_count <= 
                       self.dataset_config["max_sentence_length"]):
                    continue
                
                # Check for sufficient Bangla content
                bangla_chars = sum(1 for c in sentence if '\u0980' <= c <= '\u09FF')
                total_chars = sum(1 for c in sentence if c.isalpha())
                
                if total_chars == 0 or (bangla_chars / total_chars) < 0.6:
                    continue
                
                # Check for proper punctuation
                if not any(punct in sentence for punct in ['।', '?', '!', ',', ';', ':']):
                    continue
                
                # Avoid sentences with too many numbers/URLs/emails
                if (len(re.findall(r'\d+', sentence)) > 3 or
                    'http' in sentence.lower() or
                    '@' in sentence):
                    continue
                
                unique_sentences.append(sentence)
        
        logger.info(f"Filtered: {len(sentences)} -> {len(unique_sentences)} sentences")
        logger.info(f"Removed {len(sentences) - len(unique_sentences)} duplicates/low-quality sentences")
        
        return unique_sentences
    
    def create_dataset_splits(self, sentences: List[str]) -> DatasetDict:
        """Create train/validation/test splits from sentences"""
        # Remove duplicates and shuffle
        sentences = list(set(sentences))
        random.shuffle(sentences)
        
        # Filter by length
        filtered_sentences = []
        for sentence in sentences:
            word_count = len(sentence.split())
            if (self.dataset_config["min_sentence_length"] <= word_count <= 
                self.dataset_config["max_sentence_length"]):
                filtered_sentences.append(sentence)
        
        sentences = filtered_sentences
        logger.info(f"After filtering: {len(sentences)} sentences")
        
        # Create punctuated and unpunctuated pairs
        punctuated_texts = sentences
        unpunctuated_texts = [self.remove_punctuation(s) for s in sentences]
        
        # Create splits
        total_count = len(sentences)
        train_count = int(total_count * self.dataset_config["train_ratio"])
        val_count = int(total_count * self.dataset_config["validation_ratio"])
        
        dataset_dict = DatasetDict({
            'train': Dataset.from_dict({
                'unpunctuated_text': unpunctuated_texts[:train_count],
                'punctuated_text': punctuated_texts[:train_count]
            }),
            'validation': Dataset.from_dict({
                'unpunctuated_text': unpunctuated_texts[train_count:train_count+val_count],
                'punctuated_text': punctuated_texts[train_count:train_count+val_count]
            }),
            'test': Dataset.from_dict({
                'unpunctuated_text': unpunctuated_texts[train_count+val_count:],
                'punctuated_text': punctuated_texts[train_count+val_count:]
            })
        })
        
        return dataset_dict
    
    def generate_dataset(self) -> DatasetDict:
        """Generate a complete dataset from all sources"""
        logger.info("Starting comprehensive dataset generation...")
        
        # Collect sentences from different sources
        all_sentences = []
        
        # 1. Comprehensive Internet Scraping (NEW - Most comprehensive)
        try:
            internet_sentences = self.scrape_comprehensive_internet_data()
            all_sentences.extend(internet_sentences)
            logger.info(f"Internet scraping: {len(internet_sentences)} sentences")
        except Exception as e:
            logger.error(f"Error in comprehensive internet scraping: {e}")
        
        # 2. Wikipedia (traditional method as backup)
        try:
            wiki_sentences = self.scrape_wikipedia_articles()
            # Filter out duplicates from internet scraping
            new_wiki = [s for s in wiki_sentences if s not in all_sentences]
            all_sentences.extend(new_wiki)
            logger.info(f"Wikipedia (additional): {len(new_wiki)} sentences")
        except Exception as e:
            logger.error(f"Error scraping Wikipedia: {e}")
        
        # 3. Literary works (local files)
        try:
            literary_sentences = self.load_literary_works()
            all_sentences.extend(literary_sentences)
            logger.info(f"Literary works: {len(literary_sentences)} sentences")
        except Exception as e:
            logger.error(f"Error loading literary works: {e}")
        
        # 4. Social media content (placeholder for future implementation)
        try:
            social_sentences = self.scrape_social_media_content()
            all_sentences.extend(social_sentences)
            logger.info(f"Social media: {len(social_sentences)} sentences")
        except Exception as e:
            logger.error(f"Error scraping social media: {e}")
        
        # 5. Academic content (placeholder for future implementation)
        try:
            academic_sentences = self.scrape_academic_content()
            all_sentences.extend(academic_sentences)
            logger.info(f"Academic content: {len(academic_sentences)} sentences")
        except Exception as e:
            logger.error(f"Error scraping academic content: {e}")
        
        logger.info(f"Collected {len(all_sentences)} sentences from all sources")
        
        # 6. Generate synthetic sentences if needed
        min_sentences = self.config["min_sentences"]
        if len(all_sentences) < min_sentences:
            logger.info(f"Need {min_sentences - len(all_sentences)} more sentences, generating synthetic ones...")
            all_sentences = self.generate_synthetic_sentences(all_sentences, min_sentences)
        
        # 7. Quality filtering and deduplication
        logger.info("Applying quality filtering and deduplication...")
        filtered_sentences = self.filter_and_deduplicate_sentences(all_sentences)
        
        # Create dataset splits
        dataset = self.create_dataset_splits(filtered_sentences)
        
        logger.info("Dataset generation completed!")
        logger.info(f"Train: {len(dataset['train'])} examples")
        logger.info(f"Validation: {len(dataset['validation'])} examples")
        logger.info(f"Test: {len(dataset['test'])} examples")
        
        return dataset
    
    def save_dataset_locally(self, dataset: DatasetDict, output_dir: str = "data/generated_dataset") -> bool:
        """Save dataset to local directory"""
        try:
            os.makedirs(output_dir, exist_ok=True)
            
            for split_name, split_data in dataset.items():
                file_path = os.path.join(output_dir, f"{split_name}.json")
                with open(file_path, 'w', encoding='utf-8') as f:
                    json.dump(split_data.to_dict(), f, ensure_ascii=False, indent=2)
                logger.info(f"Saved {len(split_data)} examples to {file_path}")
            
            return True
            
        except Exception as e:
            logger.error(f"Error saving dataset to {output_dir}: {e}")
            return False
    
    def upload_to_huggingface(self, dataset: DatasetDict, dataset_name: Optional[str] = None) -> bool:
        """Upload dataset to Hugging Face Hub"""
        if not self.hf_token:
            logger.error("No Hugging Face token found. Please set the HUGGINGFACE_TOKEN environment variable.")
            return False
        
        if dataset_name is None:
            dataset_name = self.dataset_config["generated_dataset_name"]
        
        try:
            # Login to Hugging Face
            login(token=self.hf_token)
            
            # Push to Hugging Face
            dataset.push_to_hub(
                dataset_name,
                token=self.hf_token,
                private=HF_CONFIG["private_repo"]
            )
            
            logger.info(f"Successfully uploaded dataset to Hugging Face: {dataset_name}")
            return True
            
        except Exception as e:
            logger.error(f"Error uploading to Hugging Face: {e}")
            return False

# Example usage
if __name__ == "__main__":
    # Initialize generator
    generator = BanglaDatasetGenerator()
    
    # Generate dataset
    dataset = generator.generate_dataset()
    
    # Save locally
    generator.save_dataset_locally(dataset)
    
    # Upload to Hugging Face (optional)
    generator.upload_to_huggingface(dataset)
