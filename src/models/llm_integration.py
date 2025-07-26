#!/usr/bin/env python3
"""
LLM-Based Punctuation Restoration with Modern Techniques
"""

import torch
import torch.nn as nn
from transformers import (
    AutoTokenizer, AutoModelForCausalLM, AutoConfig,
    LlamaTokenizer, LlamaForCausalLM,
    GenerationConfig, BitsAndBytesConfig
)
from peft import (
    LoraConfig, get_peft_model, TaskType,
    PeftModel, prepare_model_for_kbit_training
)
import json
from typing import List, Dict, Any, Optional
import logging
from datetime import datetime

logger = logging.getLogger(__name__)

class LLMPunctuationRestorer:
    """Modern LLM-based punctuation restoration with LoRA fine-tuning"""
    
    def __init__(self, 
                 model_name: str = "microsoft/DialoGPT-medium",
                 use_quantization: bool = True,
                 use_lora: bool = True):
        self.model_name = model_name
        self.use_quantization = use_quantization
        self.use_lora = use_lora
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Initialize model and tokenizer
        self._setup_model()
    
    def _setup_model(self):
        """Setup model with modern optimizations"""
        # Quantization config for memory efficiency
        if self.use_quantization:
            bnb_config = BitsAndBytesConfig(
                load_in_4bit=True,
                bnb_4bit_use_double_quant=True,
                bnb_4bit_quant_type="nf4",
                bnb_4bit_compute_dtype=torch.bfloat16
            )
        else:
            bnb_config = None
        
        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            self.model_name,
            quantization_config=bnb_config if self.use_quantization else None,
            device_map="auto" if torch.cuda.is_available() else None,
            torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32,
            trust_remote_code=True
        )
        
        # Prepare for LoRA if quantized
        if self.use_quantization:
            self.model = prepare_model_for_kbit_training(self.model)
        
        # Setup LoRA
        if self.use_lora:
            self._setup_lora()
    
    def _setup_lora(self):
        """Setup LoRA configuration"""
        lora_config = LoraConfig(
            task_type=TaskType.CAUSAL_LM,
            inference_mode=False,
            r=16,  # Rank
            lora_alpha=32,
            lora_dropout=0.1,
            target_modules=["q_proj", "k_proj", "v_proj", "o_proj", 
                          "gate_proj", "up_proj", "down_proj"] if "llama" in self.model_name.lower() 
                         else ["c_attn", "c_proj", "c_fc"]
        )
        
        self.model = get_peft_model(self.model, lora_config)
        self.model.print_trainable_parameters()
    
    def create_prompt(self, unpunctuated_text: str) -> str:
        """Create instruction-following prompt"""
        prompt = f"""### Instruction:
You are an expert in Bangla language. Add appropriate punctuation marks to the following unpunctuated Bangla text. Use these punctuation marks: comma (,), period (।), question mark (?), exclamation mark (!), semicolon (;), colon (:), and hyphen (-).

### Input:
{unpunctuated_text}

### Output:
"""
        return prompt
    
    def generate_punctuated_text(self, 
                                unpunctuated_text: str, 
                                max_length: int = 512,
                                temperature: float = 0.1,
                                do_sample: bool = True) -> str:
        """Generate punctuated text using the LLM"""
        prompt = self.create_prompt(unpunctuated_text)
        
        # Tokenize
        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True,
            max_length=max_length,
            padding=True
        ).to(self.device)
        
        # Generation config
        generation_config = GenerationConfig(
            max_new_tokens=len(unpunctuated_text.split()) + 50,
            temperature=temperature,
            do_sample=do_sample,
            top_p=0.9,
            top_k=50,
            repetition_penalty=1.1,
            pad_token_id=self.tokenizer.pad_token_id,
            eos_token_id=self.tokenizer.eos_token_id
        )
        
        # Generate
        with torch.no_grad():
            outputs = self.model.generate(
                **inputs,
                generation_config=generation_config
            )
        
        # Decode
        generated_text = self.tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        # Extract the punctuated text (after "### Output:")
        try:
            result = generated_text.split("### Output:")[-1].strip()
            return result
        except:
            return unpunctuated_text  # Fallback
    
    def fine_tune_with_lora(self, 
                           train_dataset: List[Dict],
                           validation_dataset: List[Dict],
                           output_dir: str = "models/llm_lora",
                           num_epochs: int = 3,
                           batch_size: int = 4):
        """Fine-tune the model using LoRA"""
        from transformers import Trainer, TrainingArguments, DataCollatorForLanguageModeling
        from datasets import Dataset
        
        # Prepare datasets
        def preprocess_function(examples):
            prompts = [self.create_prompt(ex['unpunctuated_text']) + ex['punctuated_text'] 
                      for ex in examples]
            
            tokenized = self.tokenizer(
                prompts,
                truncation=True,
                padding=True,
                max_length=512,
                return_tensors="pt"
            )
            
            # For causal LM, labels are the same as input_ids
            tokenized["labels"] = tokenized["input_ids"].clone()
            return tokenized
        
        # Convert to datasets
        train_ds = Dataset.from_list(train_dataset).map(
            lambda x: preprocess_function([x]), 
            remove_columns=train_dataset[0].keys() if train_dataset else []
        )
        
        val_ds = Dataset.from_list(validation_dataset).map(
            lambda x: preprocess_function([x]),
            remove_columns=validation_dataset[0].keys() if validation_dataset else []
        )
        
        # Training arguments
        training_args = TrainingArguments(
            output_dir=output_dir,
            num_train_epochs=num_epochs,
            per_device_train_batch_size=batch_size,
            per_device_eval_batch_size=batch_size,
            gradient_accumulation_steps=4,
            warmup_ratio=0.1,
            learning_rate=2e-4,
            fp16=True,
            logging_steps=10,
            evaluation_strategy="steps",
            eval_steps=100,
            save_strategy="steps",
            save_steps=100,
            load_best_model_at_end=True,
            metric_for_best_model="eval_loss",
            greater_is_better=False,
            dataloader_pin_memory=False,
            remove_unused_columns=False,
            report_to="none"
        )
        
        # Data collator
        data_collator = DataCollatorForLanguageModeling(
            tokenizer=self.tokenizer,
            mlm=False
        )
        
        # Trainer
        trainer = Trainer(
            model=self.model,
            args=training_args,
            train_dataset=train_ds,
            eval_dataset=val_ds,
            data_collator=data_collator,
            tokenizer=self.tokenizer
        )
        
        # Train
        trainer.train()
        
        # Save
        trainer.save_model()
        self.tokenizer.save_pretrained(output_dir)
        
        return output_dir

class ModernPromptEngineering:
    """Advanced prompt engineering techniques"""
    
    @staticmethod
    def few_shot_prompt(unpunctuated_text: str, examples: List[Dict]) -> str:
        """Create few-shot learning prompt"""
        prompt = "### Task: Add punctuation to Bangla text\n\n"
        
        # Add examples
        for i, example in enumerate(examples[:3], 1):  # Use 3 examples
            prompt += f"### Example {i}:\n"
            prompt += f"Input: {example['unpunctuated_text']}\n"
            prompt += f"Output: {example['punctuated_text']}\n\n"
        
        # Add current task
        prompt += "### Current Task:\n"
        prompt += f"Input: {unpunctuated_text}\n"
        prompt += "Output: "
        
        return prompt
    
    @staticmethod
    def chain_of_thought_prompt(unpunctuated_text: str) -> str:
        """Create chain-of-thought reasoning prompt"""
        prompt = f"""### Task: Add punctuation to Bangla text with reasoning

### Input: {unpunctuated_text}

### Instructions:
1. Read the text carefully
2. Identify sentence boundaries
3. Determine appropriate punctuation marks based on context
4. Consider the meaning and emotional tone
5. Apply punctuation rules for Bangla language

### Step-by-step reasoning:
Let me analyze this text step by step:

1. First, I'll identify the main clauses and sentences
2. Then I'll look for question words (কি, কেন, কীভাবে, etc.) to add question marks
3. I'll check for emotional expressions that need exclamation marks
4. I'll add commas for natural pauses and list items
5. I'll use periods (।) for sentence endings

### Final punctuated text:
"""
        return prompt

class MultiModalPunctuationModel:
    """Multi-modal model that can use context from multiple sources"""
    
    def __init__(self):
        self.text_model = LLMPunctuationRestorer()
        self.context_analyzer = ContextAnalyzer()
    
    def restore_with_context(self, 
                           unpunctuated_text: str, 
                           context: Dict[str, Any] = None) -> str:
        """Restore punctuation with additional context"""
        
        # Analyze context
        if context:
            context_info = self.context_analyzer.analyze(context)
            enhanced_prompt = self._create_contextual_prompt(
                unpunctuated_text, context_info
            )
        else:
            enhanced_prompt = unpunctuated_text
        
        return self.text_model.generate_punctuated_text(enhanced_prompt)
    
    def _create_contextual_prompt(self, text: str, context_info: Dict) -> str:
        """Create prompt with context information"""
        context_prompt = f"""### Context Information:
- Domain: {context_info.get('domain', 'general')}
- Style: {context_info.get('style', 'formal')}
- Audience: {context_info.get('audience', 'general')}
- Emotion: {context_info.get('emotion', 'neutral')}

### Text to punctuate:
{text}

### Instructions:
Consider the context information above when adding punctuation. Adjust the punctuation style based on the domain, formality level, and emotional tone.

### Punctuated text:
"""
        return context_prompt

class ContextAnalyzer:
    """Analyze context for better punctuation decisions"""
    
    def analyze(self, context: Dict[str, Any]) -> Dict[str, str]:
        """Analyze context and provide insights"""
        analysis = {
            'domain': self._detect_domain(context),
            'style': self._detect_style(context),
            'audience': self._detect_audience(context),
            'emotion': self._detect_emotion(context)
        }
        return analysis
    
    def _detect_domain(self, context: Dict) -> str:
        """Detect domain from context"""
        # Simple domain detection logic
        text = context.get('text', '').lower()
        if any(word in text for word in ['আদালত', 'আইন', 'বিচার']):
            return 'legal'
        elif any(word in text for word in ['চিকিৎসা', 'ডাক্তার', 'হাসপাতাল']):
            return 'medical'
        elif any(word in text for word in ['শিক্ষা', 'বিদ্যালয়', 'ছাত্র']):
            return 'education'
        else:
            return 'general'
    
    def _detect_style(self, context: Dict) -> str:
        """Detect writing style"""
        # Implement style detection logic
        return context.get('style', 'formal')
    
    def _detect_audience(self, context: Dict) -> str:
        """Detect target audience"""
        return context.get('audience', 'general')
    
    def _detect_emotion(self, context: Dict) -> str:
        """Detect emotional tone"""
        return context.get('emotion', 'neutral')

class AdvancedEvaluationMetrics:
    """Modern evaluation metrics for punctuation restoration"""
    
    @staticmethod
    def compute_semantic_similarity(pred_text: str, ref_text: str) -> float:
        """Compute semantic similarity using sentence transformers"""
        try:
            from sentence_transformers import SentenceTransformer
            model = SentenceTransformer('all-MiniLM-L6-v2')
            
            pred_embedding = model.encode([pred_text])
            ref_embedding = model.encode([ref_text])
            
            similarity = torch.cosine_similarity(
                torch.tensor(pred_embedding), 
                torch.tensor(ref_embedding)
            ).item()
            
            return similarity
        except ImportError:
            logger.warning("sentence-transformers not installed")
            return 0.0
    
    @staticmethod
    def compute_readability_score(text: str) -> Dict[str, float]:
        """Compute readability metrics"""
        sentences = text.split('।')
        words = text.split()
        
        avg_sentence_length = len(words) / max(len(sentences), 1)
        
        # Simple complexity measures
        complex_punctuation = sum(1 for char in text if char in ';:')
        punctuation_density = sum(1 for char in text if char in '।?!,;:-') / len(text)
        
        return {
            'avg_sentence_length': avg_sentence_length,
            'punctuation_density': punctuation_density,
            'complex_punctuation_count': complex_punctuation
        }
    
    @staticmethod
    def compute_fluency_score(text: str) -> float:
        """Compute fluency score using language model perplexity"""
        # This would use a pre-trained Bangla language model
        # For now, return a placeholder
        return 0.8  # Placeholder

# Integration example
class HybridPunctuationSystem:
    """Hybrid system combining multiple approaches"""
    
    def __init__(self):
        self.llm_model = LLMPunctuationRestorer()
        self.transformer_model = None  # Your existing transformer model
        self.rule_based_model = None   # Rule-based fallback
        
    def restore_punctuation(self, text: str, method: str = "ensemble") -> str:
        """Restore punctuation using multiple methods"""
        
        if method == "llm":
            return self.llm_model.generate_punctuated_text(text)
        
        elif method == "ensemble":
            # Get predictions from multiple models
            llm_pred = self.llm_model.generate_punctuated_text(text)
            
            # You would add other model predictions here
            predictions = [llm_pred]
            
            # Simple ensemble: return most common prediction
            return predictions[0]  # Simplified for demo
        
        else:
            raise ValueError(f"Unknown method: {method}")

# Usage example
if __name__ == "__main__":
    # Initialize LLM-based model
    llm_restorer = LLMPunctuationRestorer(
        model_name="microsoft/DialoGPT-medium",
        use_quantization=True,
        use_lora=True
    )
    
    # Example text
    test_text = "আমি কেমন আছি তুমি কি জানতে চাও"
    
    # Generate punctuation
    result = llm_restorer.generate_punctuated_text(test_text)
    print(f"Original: {test_text}")
    print(f"Punctuated: {result}")
