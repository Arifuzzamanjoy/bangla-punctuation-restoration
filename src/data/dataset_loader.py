#!/usr/bin/env python3
"""
Dataset loader for Bangla punctuation restoration
"""

import os
import json
from typing import Optional, Dict, Any, Union
from datasets import load_dataset, Dataset, DatasetDict
from huggingface_hub import login
import logging

# Handle config import for different execution contexts
try:
    from config import DATASET_CONFIG, HF_CONFIG
except ImportError:
    import sys
    import os
    sys.path.append(os.path.join(os.path.dirname(__file__), '..', '..'))
    from config import DATASET_CONFIG, HF_CONFIG

# Set up logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class BanglaDatasetLoader:
    """
    A class to load and preprocess Bangla punctuation datasets
    """
    
    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize the dataset loader
        
        Args:
            config: Configuration dictionary for dataset loading
        """
        self.config = config or DATASET_CONFIG
        self.hf_token = os.getenv(HF_CONFIG["token_env_var"])
        
        # Authenticate with Hugging Face if token is available
        if self.hf_token:
            try:
                login(token=self.hf_token)
                logger.info("Successfully authenticated with Hugging Face")
            except Exception as e:
                logger.warning(f"Failed to authenticate with Hugging Face: {e}")
    
    def load_original_dataset(self) -> Optional[DatasetDict]:
        """
        Load the original hishab/hishab-pr-bn-v1 dataset
        
        Returns:
            DatasetDict containing the loaded dataset or None if failed
        """
        dataset_name = self.config["original_dataset"]
        
        try:
            logger.info(f"Loading dataset: {dataset_name}")
            
            # Try to load with authentication first
            if self.hf_token:
                dataset = load_dataset(dataset_name, use_auth_token=self.hf_token)
            else:
                # Try to load without authentication (if public)
                dataset = load_dataset(dataset_name)
            
            logger.info(f"Successfully loaded dataset: {dataset_name}")
            logger.info(f"Dataset structure: {dataset}")
            
            # Print basic info about the dataset
            for split_name, split_data in dataset.items():
                logger.info(f"Split '{split_name}': {len(split_data)} examples")
                if len(split_data) > 0:
                    logger.info(f"Features: {list(split_data.features.keys())}")
                    logger.info(f"First example: {split_data[0]}")
            
            # Convert hishab format to expected format
            if "conversations" in dataset[list(dataset.keys())[0]].features:
                logger.info("Converting hishab conversation format to punctuation format...")
                dataset = self.convert_hishab_format(dataset)
            
            return dataset
            
        except Exception as e:
            logger.error(f"Error loading dataset {dataset_name}: {e}")
            logger.error("Make sure you have the correct permissions and are logged in to Hugging Face")
            return None
    
    def load_generated_dataset(self, dataset_name: Optional[str] = None) -> Optional[DatasetDict]:
        """
        Load a generated dataset from Hugging Face or local path
        
        Args:
            dataset_name: Name of the dataset to load
            
        Returns:
            DatasetDict containing the loaded dataset or None if failed
        """
        if dataset_name is None:
            dataset_name = self.config["generated_dataset_name"]
        
        try:
            logger.info(f"Loading generated dataset: {dataset_name}")
            
            # Try to load from Hugging Face first
            if self.hf_token:
                dataset = load_dataset(dataset_name, use_auth_token=self.hf_token)
            else:
                dataset = load_dataset(dataset_name)
            
            logger.info(f"Successfully loaded generated dataset: {dataset_name}")
            return dataset
            
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            
            # Try to load from local path
            try:
                local_path = os.path.join("data", "generated_dataset")
                if os.path.exists(local_path):
                    logger.info(f"Loading from local path: {local_path}")
                    dataset_dict = {}
                    
                    for split in ['train', 'validation', 'test']:
                        file_path = os.path.join(local_path, f"{split}.json")
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                dataset_dict[split] = Dataset.from_dict(data)
                    
                    if dataset_dict:
                        return DatasetDict(dataset_dict)
                    
            except Exception as e:
                logger.error(f"Failed to load from local path: {e}")
            
            return None
    
    def load_adversarial_dataset(self, dataset_name: Optional[str] = None) -> Optional[DatasetDict]:
        """
        Load an adversarial dataset from Hugging Face or local path
        
        Args:
            dataset_name: Name of the adversarial dataset to load
            
        Returns:
            DatasetDict containing the loaded dataset or None if failed
        """
        if dataset_name is None:
            dataset_name = self.config["adversarial_dataset_name"]
        
        try:
            logger.info(f"Loading adversarial dataset: {dataset_name}")
            
            # Try to load from Hugging Face first
            if self.hf_token:
                dataset = load_dataset(dataset_name, use_auth_token=self.hf_token)
            else:
                dataset = load_dataset(dataset_name)
            
            logger.info(f"Successfully loaded adversarial dataset: {dataset_name}")
            return dataset
            
        except Exception as e:
            logger.warning(f"Failed to load from Hugging Face: {e}")
            
            # Try to load from local path
            try:
                local_path = os.path.join("data", "adversarial_dataset")
                if os.path.exists(local_path):
                    logger.info(f"Loading from local path: {local_path}")
                    dataset_dict = {}
                    
                    for split in ['train', 'validation', 'test']:
                        file_path = os.path.join(local_path, f"{split}.json")
                        if os.path.exists(file_path):
                            with open(file_path, 'r', encoding='utf-8') as f:
                                data = json.load(f)
                                dataset_dict[split] = Dataset.from_dict(data)
                    
                    if dataset_dict:
                        return DatasetDict(dataset_dict)
                    
            except Exception as e:
                logger.error(f"Failed to load from local path: {e}")
            
            return None
    
    def load_dataset_from_path(self, path: str) -> Optional[DatasetDict]:
        """
        Load dataset from a local path
        
        Args:
            path: Path to the dataset directory
            
        Returns:
            DatasetDict containing the loaded dataset or None if failed
        """
        try:
            logger.info(f"Loading dataset from path: {path}")
            
            if os.path.isdir(path):
                # Load from directory with split files
                dataset_dict = {}
                for split in ['train', 'validation', 'test']:
                    file_path = os.path.join(path, f"{split}.json")
                    if os.path.exists(file_path):
                        with open(file_path, 'r', encoding='utf-8') as f:
                            data = json.load(f)
                            dataset_dict[split] = Dataset.from_dict(data)
                
                if dataset_dict:
                    return DatasetDict(dataset_dict)
                else:
                    logger.error(f"No valid split files found in {path}")
                    return None
            
            elif os.path.isfile(path):
                # Load from single file
                with open(path, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    return Dataset.from_dict(data)
            
            else:
                logger.error(f"Path does not exist: {path}")
                return None
                
        except Exception as e:
            logger.error(f"Error loading dataset from {path}: {e}")
            return None
    
    def save_dataset_locally(self, dataset: DatasetDict, output_dir: str) -> bool:
        """
        Save dataset to local directory
        
        Args:
            dataset: Dataset to save
            output_dir: Output directory path
            
        Returns:
            True if successful, False otherwise
        """
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
    
    def get_dataset_info(self, dataset: Union[Dataset, DatasetDict]) -> Dict[str, Any]:
        """
        Get information about a dataset
        
        Args:
            dataset: Dataset to analyze
            
        Returns:
            Dictionary containing dataset information
        """
        info = {}
        
        if isinstance(dataset, DatasetDict):
            for split_name, split_data in dataset.items():
                info[split_name] = {
                    "num_examples": len(split_data),
                    "features": list(split_data.features.keys()),
                    "feature_types": {name: str(feature) for name, feature in split_data.features.items()}
                }
        else:
            info = {
                "num_examples": len(dataset),
                "features": list(dataset.features.keys()),
                "feature_types": {name: str(feature) for name, feature in dataset.features.items()}
            }
        
        return info
    
    def convert_hishab_format(self, dataset: DatasetDict) -> DatasetDict:
        """
        Convert hishab dataset format to expected unpunctuated/punctuated format
        
        Args:
            dataset: Raw hishab dataset
            
        Returns:
            Converted dataset with unpunctuated_text and punctuated_text columns
        """
        converted_dataset = {}
        
        for split_name, split_data in dataset.items():
            unpunctuated_texts = []
            punctuated_texts = []
            
            for example in split_data:
                conversations = example["conversations"]
                unpunctuated_text = None
                punctuated_text = None
                
                # Extract human (unpunctuated) and gpt (punctuated) texts
                for conv in conversations:
                    if conv["from"] == "human":
                        unpunctuated_text = conv["value"]
                    elif conv["from"] == "gpt":
                        punctuated_text = conv["value"]
                
                # Only add if we have both unpunctuated and punctuated versions
                if unpunctuated_text and punctuated_text:
                    unpunctuated_texts.append(unpunctuated_text)
                    punctuated_texts.append(punctuated_text)
            
            # Create new dataset split
            converted_dataset[split_name] = Dataset.from_dict({
                "unpunctuated_text": unpunctuated_texts,
                "punctuated_text": punctuated_texts
            })
            
            logger.info(f"Converted {split_name}: {len(converted_dataset[split_name])} examples")
        
        return DatasetDict(converted_dataset)

# Example usage
if __name__ == "__main__":
    # Initialize loader
    loader = BanglaDatasetLoader()
    
    # Load original dataset
    original_dataset = loader.load_original_dataset()
    
    if original_dataset:
        print("\nOriginal dataset loaded successfully!")
        print("Dataset info:", loader.get_dataset_info(original_dataset))
        
        # Save locally for backup
        loader.save_dataset_locally(original_dataset, "data/original_dataset")
    else:
        print("\nFailed to load original dataset.")
    
    # Try to load generated dataset
    generated_dataset = loader.load_generated_dataset()
    if generated_dataset:
        print("\nGenerated dataset loaded successfully!")
        print("Dataset info:", loader.get_dataset_info(generated_dataset))
    
    # Try to load adversarial dataset
    adversarial_dataset = loader.load_adversarial_dataset()
    if adversarial_dataset:
        print("\nAdversarial dataset loaded successfully!")
        print("Dataset info:", loader.get_dataset_info(adversarial_dataset))
