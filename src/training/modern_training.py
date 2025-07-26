#!/usr/bin/env python3
"""
Modern Training Techniques for Bangla Punctuation Restoration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import OneCycleLR, CosineAnnealingWarmRestarts
import numpy as np
from typing import Dict, Any, List, Optional, Tuple, Callable
import logging
from dataclasses import dataclass
import math
import random
from transformers import (
    Trainer, TrainingArguments, 
    get_linear_schedule_with_warmup,
    get_cosine_schedule_with_warmup
)
from torch.utils.data import DataLoader
import wandb
from sklearn.metrics import classification_report, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import json
from datetime import datetime

logger = logging.getLogger(__name__)

@dataclass
class ModernTrainingConfig:
    """Configuration for modern training techniques"""
    
    # Optimization
    learning_rate: float = 3e-5
    weight_decay: float = 0.01
    warmup_ratio: float = 0.1
    max_grad_norm: float = 1.0
    
    # Training dynamics
    use_amp: bool = True  # Automatic Mixed Precision
    gradient_accumulation_steps: int = 4
    dataloader_num_workers: int = 4
    
    # Regularization
    dropout_rate: float = 0.1
    label_smoothing: float = 0.1
    mixup_alpha: float = 0.2
    cutmix_alpha: float = 1.0
    
    # Advanced techniques
    use_swa: bool = True  # Stochastic Weight Averaging
    use_ema: bool = True  # Exponential Moving Average
    use_lookahead: bool = True
    use_sam: bool = True  # Sharpness Aware Minimization
    
    # Curriculum learning
    use_curriculum: bool = True
    curriculum_strategy: str = "length_based"
    
    # Multi-task learning
    use_multitask: bool = False
    auxiliary_tasks: List[str] = None
    
    # Meta learning
    use_meta_learning: bool = False
    meta_learning_lr: float = 1e-3

class SharpnessAwareMinimizer(torch.optim.Optimizer):
    """Sharpness Aware Minimization (SAM) optimizer"""
    
    def __init__(self, params, base_optimizer, rho=0.05, adaptive=False, **kwargs):
        assert rho >= 0.0, f"Invalid rho, should be non-negative: {rho}"
        
        defaults = dict(rho=rho, adaptive=adaptive, **kwargs)
        super().__init__(params, defaults)
        
        self.base_optimizer = base_optimizer(self.param_groups, **kwargs)
        self.param_groups = self.base_optimizer.param_groups
    
    @torch.no_grad()
    def first_step(self, zero_grad=False):
        grad_norm = self._grad_norm()
        for group in self.param_groups:
            scale = group["rho"] / (grad_norm + 1e-12)
            
            for p in group["params"]:
                if p.grad is None:
                    continue
                self.state[p]["old_p"] = p.data.clone()
                e_w = (torch.pow(p, 2) if group["adaptive"] else 1.0) * p.grad * scale.to(p)
                p.add_(e_w)  # climb to the local maximum "w + e(w)"
        
        if zero_grad:
            self.zero_grad()
    
    @torch.no_grad()
    def second_step(self, zero_grad=False):
        for group in self.param_groups:
            for p in group["params"]:
                if p.grad is None:
                    continue
                p.data = self.state[p]["old_p"]  # get back to "w" from "w + e(w)"
        
        self.base_optimizer.step()  # do the actual "sharpness-aware" update
        
        if zero_grad:
            self.zero_grad()
    
    def step(self, closure=None):
        assert closure is not None, "Sharpness Aware Minimization requires closure, but it was not provided"
        closure = torch.enable_grad()(closure)  # the closure should do a full forward-backward pass
        
        self.first_step(zero_grad=True)
        closure()
        self.second_step()
    
    def _grad_norm(self):
        shared_device = self.param_groups[0]["params"][0].device  # put everything on the same device, in case of model parallelism
        norm = torch.norm(
            torch.stack([
                ((torch.abs(p) if group["adaptive"] else 1.0) * p.grad).norm(dtype=torch.float32).to(shared_device)
                for group in self.param_groups for p in group["params"]
                if p.grad is not None
            ]),
            dtype=torch.float32
        )
        return norm

class ExponentialMovingAverage:
    """Exponential Moving Average for model parameters"""
    
    def __init__(self, model: nn.Module, decay: float = 0.999):
        self.model = model
        self.decay = decay
        self.shadow = {}
        self.backup = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad:
                self.shadow[name] = param.data.clone()
    
    def update(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                new_average = (1.0 - self.decay) * param.data + self.decay * self.shadow[name]
                self.shadow[name] = new_average.clone()
    
    def apply_shadow(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.shadow
                self.backup[name] = param.data.clone()
                param.data = self.shadow[name]
    
    def restore(self):
        for name, param in self.model.named_parameters():
            if param.requires_grad:
                assert name in self.backup
                param.data = self.backup[name]
        self.backup = {}

class MixupLoss(nn.Module):
    """Mixup augmentation loss"""
    
    def __init__(self, alpha: float = 0.2):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, y_a, y_b, lam):
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class CutMixLoss(nn.Module):
    """CutMix augmentation loss"""
    
    def __init__(self, alpha: float = 1.0):
        super().__init__()
        self.alpha = alpha
    
    def forward(self, pred, y_a, y_b, lam):
        criterion = nn.CrossEntropyLoss(ignore_index=-100)
        return lam * criterion(pred, y_a) + (1 - lam) * criterion(pred, y_b)

class FocalLoss(nn.Module):
    """Focal Loss for handling class imbalance"""
    
    def __init__(self, alpha=1, gamma=2, ignore_index=-100):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.ignore_index = ignore_index
    
    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, ignore_index=self.ignore_index, reduction='none')
        pt = torch.exp(-ce_loss)
        focal_loss = self.alpha * (1-pt)**self.gamma * ce_loss
        return focal_loss.mean()

class CurriculumLearning:
    """Curriculum learning for progressive training"""
    
    def __init__(self, strategy: str = "length_based", total_epochs: int = 10):
        self.strategy = strategy
        self.total_epochs = total_epochs
        self.current_epoch = 0
    
    def get_curriculum_data(self, dataset: List[Dict], epoch: int) -> List[Dict]:
        """Get curriculum data based on current epoch"""
        self.current_epoch = epoch
        progress = epoch / self.total_epochs
        
        if self.strategy == "length_based":
            return self._length_based_curriculum(dataset, progress)
        elif self.strategy == "complexity_based":
            return self._complexity_based_curriculum(dataset, progress)
        else:
            return dataset
    
    def _length_based_curriculum(self, dataset: List[Dict], progress: float) -> List[Dict]:
        """Start with shorter texts, gradually include longer ones"""
        
        # Sort by text length
        sorted_data = sorted(dataset, key=lambda x: len(x['unpunctuated_text'].split()))
        
        # Gradually increase the maximum length
        max_length = int(len(sorted_data) * min(0.5 + 0.5 * progress, 1.0))
        
        return sorted_data[:max_length]
    
    def _complexity_based_curriculum(self, dataset: List[Dict], progress: float) -> List[Dict]:
        """Start with simple punctuation, gradually include complex patterns"""
        
        def get_complexity_score(item):
            text = item['punctuated_text']
            # Simple complexity: number of different punctuation marks
            unique_puncts = set(char for char in text if char in '।?!,;:-')
            return len(unique_puncts)
        
        # Sort by complexity
        sorted_data = sorted(dataset, key=get_complexity_score)
        
        # Gradually increase complexity
        max_items = int(len(sorted_data) * min(0.3 + 0.7 * progress, 1.0))
        
        return sorted_data[:max_items]

class MetaLearningTrainer:
    """Model-Agnostic Meta-Learning (MAML) for few-shot adaptation"""
    
    def __init__(self, model: nn.Module, meta_lr: float = 1e-3, inner_lr: float = 1e-2):
        self.model = model
        self.meta_lr = meta_lr
        self.inner_lr = inner_lr
        self.meta_optimizer = AdamW(model.parameters(), lr=meta_lr)
    
    def meta_train_step(self, support_batches: List[torch.Tensor], 
                       query_batches: List[torch.Tensor]) -> float:
        """Single meta-training step"""
        
        meta_loss = 0.0
        
        for support_batch, query_batch in zip(support_batches, query_batches):
            # Inner loop: adapt to support set
            adapted_params = self._inner_loop(support_batch)
            
            # Outer loop: evaluate on query set
            query_loss = self._outer_loop(query_batch, adapted_params)
            meta_loss += query_loss
        
        # Meta-update
        meta_loss = meta_loss / len(support_batches)
        self.meta_optimizer.zero_grad()
        meta_loss.backward()
        self.meta_optimizer.step()
        
        return meta_loss.item()
    
    def _inner_loop(self, support_batch: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Inner loop adaptation"""
        # Create a copy of model parameters
        adapted_params = {}
        for name, param in self.model.named_parameters():
            adapted_params[name] = param.clone()
        
        # Compute gradients on support set
        support_loss = self._compute_loss(support_batch, adapted_params)
        grads = torch.autograd.grad(support_loss, adapted_params.values(), create_graph=True)
        
        # Update parameters
        for (name, param), grad in zip(adapted_params.items(), grads):
            adapted_params[name] = param - self.inner_lr * grad
        
        return adapted_params
    
    def _outer_loop(self, query_batch: torch.Tensor, 
                   adapted_params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Outer loop evaluation"""
        return self._compute_loss(query_batch, adapted_params)
    
    def _compute_loss(self, batch: torch.Tensor, 
                     params: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss with given parameters"""
        # This would depend on your specific model architecture
        # Placeholder implementation
        return torch.tensor(0.0, requires_grad=True)

class ModernTrainer:
    """Advanced trainer with modern techniques"""
    
    def __init__(self, 
                 model: nn.Module,
                 config: ModernTrainingConfig,
                 train_dataset: torch.utils.data.Dataset,
                 eval_dataset: torch.utils.data.Dataset = None):
        
        self.model = model
        self.config = config
        self.train_dataset = train_dataset
        self.eval_dataset = eval_dataset
        
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        
        # Initialize components
        self._setup_optimizer()
        self._setup_scheduler()
        self._setup_regularization()
        self._setup_augmentation()
        
        # Initialize tracking
        self.global_step = 0
        self.epoch = 0
        self.best_metric = float('-inf')
        
        # Initialize wandb if available
        try:
            self.use_wandb = True
            wandb.init(project="bangla-punctuation-modern")
        except:
            self.use_wandb = False
    
    def _setup_optimizer(self):
        """Setup modern optimizers"""
        
        # Parameter groups with different learning rates
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if not any(nd in n for nd in no_decay)],
                "weight_decay": self.config.weight_decay,
            },
            {
                "params": [p for n, p in self.model.named_parameters() 
                          if any(nd in n for nd in no_decay)],
                "weight_decay": 0.0,
            },
        ]
        
        # Base optimizer
        if self.config.use_sam:
            self.optimizer = SharpnessAwareMinimizer(
                optimizer_grouped_parameters,
                AdamW,
                lr=self.config.learning_rate,
                rho=0.05
            )
        else:
            self.optimizer = AdamW(
                optimizer_grouped_parameters,
                lr=self.config.learning_rate
            )
        
        # Lookahead wrapper
        if self.config.use_lookahead:
            from torch_optimizer import Lookahead
            self.optimizer = Lookahead(self.optimizer, k=5, alpha=0.5)
    
    def _setup_scheduler(self):
        """Setup learning rate scheduler"""
        num_training_steps = len(self.train_dataset) // self.config.gradient_accumulation_steps
        
        self.scheduler = get_cosine_schedule_with_warmup(
            self.optimizer,
            num_warmup_steps=int(num_training_steps * self.config.warmup_ratio),
            num_training_steps=num_training_steps
        )
    
    def _setup_regularization(self):
        """Setup regularization techniques"""
        
        # Exponential Moving Average
        if self.config.use_ema:
            self.ema = ExponentialMovingAverage(self.model, decay=0.999)
        
        # Loss functions
        self.focal_loss = FocalLoss(gamma=2.0)
        self.mixup_loss = MixupLoss(alpha=self.config.mixup_alpha)
        self.cutmix_loss = CutMixLoss(alpha=self.config.cutmix_alpha)
        
        # Curriculum learning
        if self.config.use_curriculum:
            self.curriculum = CurriculumLearning(
                strategy=self.config.curriculum_strategy,
                total_epochs=10  # This should be configurable
            )
    
    def _setup_augmentation(self):
        """Setup data augmentation techniques"""
        self.mixup_prob = 0.5
        self.cutmix_prob = 0.5
    
    def mixup_data(self, x, y, alpha=1.0):
        """Apply mixup augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        mixed_x = lam * x + (1 - lam) * x[index, :]
        y_a, y_b = y, y[index]
        
        return mixed_x, y_a, y_b, lam
    
    def cutmix_data(self, x, y, alpha=1.0):
        """Apply cutmix augmentation"""
        if alpha > 0:
            lam = np.random.beta(alpha, alpha)
        else:
            lam = 1
        
        batch_size = x.size(0)
        index = torch.randperm(batch_size).to(x.device)
        
        y_a, y_b = y, y[index]
        
        # For text data, we'll do token-level cutmix
        seq_len = x.size(1)
        cut_len = int(seq_len * (1 - lam))
        cut_start = np.random.randint(seq_len - cut_len + 1)
        
        mixed_x = x.clone()
        mixed_x[:, cut_start:cut_start + cut_len] = x[index, cut_start:cut_start + cut_len]
        
        return mixed_x, y_a, y_b, lam
    
    def train_epoch(self, epoch: int) -> Dict[str, float]:
        """Train for one epoch with modern techniques"""
        
        self.model.train()
        self.epoch = epoch
        
        # Curriculum learning
        if self.config.use_curriculum:
            curriculum_data = self.curriculum.get_curriculum_data(
                self.train_dataset, epoch
            )
            dataloader = DataLoader(
                curriculum_data, 
                batch_size=32,  # This should be configurable
                shuffle=True,
                num_workers=self.config.dataloader_num_workers
            )
        else:
            dataloader = DataLoader(
                self.train_dataset,
                batch_size=32,
                shuffle=True,
                num_workers=self.config.dataloader_num_workers
            )
        
        total_loss = 0.0
        num_batches = 0
        
        # Mixed precision scaler
        if self.config.use_amp:
            scaler = torch.cuda.amp.GradScaler()
        
        for batch_idx, batch in enumerate(dataloader):
            
            # Move batch to device
            batch = {k: v.to(self.device) for k, v in batch.items()}
            
            # Apply augmentation
            if random.random() < self.mixup_prob:
                inputs, targets_a, targets_b, lam = self.mixup_data(
                    batch['input_ids'], batch['labels'], self.config.mixup_alpha
                )
                batch['input_ids'] = inputs
                use_mixup = True
            elif random.random() < self.cutmix_prob:
                inputs, targets_a, targets_b, lam = self.cutmix_data(
                    batch['input_ids'], batch['labels'], self.config.cutmix_alpha
                )
                batch['input_ids'] = inputs
                use_cutmix = True
            else:
                use_mixup = use_cutmix = False
            
            # Forward pass with mixed precision
            if self.config.use_amp:
                with torch.cuda.amp.autocast():
                    outputs = self.model(**batch)
                    
                    if use_mixup:
                        loss = self.mixup_loss(outputs.logits, targets_a, targets_b, lam)
                    elif use_cutmix:
                        loss = self.cutmix_loss(outputs.logits, targets_a, targets_b, lam)
                    else:
                        loss = self.focal_loss(outputs.logits.view(-1, outputs.logits.size(-1)), 
                                             batch['labels'].view(-1))
            else:
                outputs = self.model(**batch)
                loss = outputs.loss
            
            # Scale loss for gradient accumulation
            loss = loss / self.config.gradient_accumulation_steps
            
            # Backward pass
            if self.config.use_amp:
                scaler.scale(loss).backward()
            else:
                loss.backward()
            
            # Gradient accumulation
            if (batch_idx + 1) % self.config.gradient_accumulation_steps == 0:
                
                # Gradient clipping
                if self.config.use_amp:
                    scaler.unscale_(self.optimizer)
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.config.max_grad_norm)
                
                # Optimizer step
                if self.config.use_sam:
                    # SAM requires a closure
                    def closure():
                        loss_closure = self.model(**batch).loss
                        return loss_closure
                    
                    if self.config.use_amp:
                        scaler.step(lambda: self.optimizer.step(closure))
                        scaler.update()
                    else:
                        self.optimizer.step(closure)
                else:
                    if self.config.use_amp:
                        scaler.step(self.optimizer)
                        scaler.update()
                    else:
                        self.optimizer.step()
                
                # Update learning rate
                self.scheduler.step()
                
                # Update EMA
                if self.config.use_ema:
                    self.ema.update()
                
                # Zero gradients
                self.optimizer.zero_grad()
                
                self.global_step += 1
            
            total_loss += loss.item()
            num_batches += 1
            
            # Logging
            if batch_idx % 100 == 0:
                current_lr = self.scheduler.get_last_lr()[0]
                logger.info(f"Epoch {epoch}, Batch {batch_idx}, Loss: {loss.item():.4f}, LR: {current_lr:.2e}")
                
                if self.use_wandb:
                    wandb.log({
                        "train/loss": loss.item(),
                        "train/learning_rate": current_lr,
                        "train/epoch": epoch,
                        "train/global_step": self.global_step
                    })
        
        avg_loss = total_loss / num_batches
        return {"train_loss": avg_loss}
    
    def evaluate(self) -> Dict[str, float]:
        """Evaluate model with EMA weights if available"""
        
        # Apply EMA weights for evaluation
        if self.config.use_ema:
            self.ema.apply_shadow()
        
        self.model.eval()
        
        if self.eval_dataset is None:
            return {}
        
        eval_dataloader = DataLoader(
            self.eval_dataset,
            batch_size=32,  # This should be configurable
            shuffle=False,
            num_workers=self.config.dataloader_num_workers
        )
        
        total_loss = 0.0
        all_predictions = []
        all_labels = []
        
        with torch.no_grad():
            for batch in eval_dataloader:
                batch = {k: v.to(self.device) for k, v in batch.items()}
                
                outputs = self.model(**batch)
                loss = outputs.loss
                
                total_loss += loss.item()
                
                # Collect predictions and labels
                predictions = torch.argmax(outputs.logits, dim=-1)
                all_predictions.extend(predictions.cpu().numpy().flatten())
                all_labels.extend(batch['labels'].cpu().numpy().flatten())
        
        # Filter out ignored labels
        filtered_preds = []
        filtered_labels = []
        for pred, label in zip(all_predictions, all_labels):
            if label != -100:
                filtered_preds.append(pred)
                filtered_labels.append(label)
        
        # Compute metrics
        from sklearn.metrics import accuracy_score, precision_recall_fscore_support
        
        accuracy = accuracy_score(filtered_labels, filtered_preds)
        precision, recall, f1, _ = precision_recall_fscore_support(
            filtered_labels, filtered_preds, average='macro'
        )
        
        metrics = {
            "eval_loss": total_loss / len(eval_dataloader),
            "eval_accuracy": accuracy,
            "eval_precision": precision,
            "eval_recall": recall,
            "eval_f1": f1
        }
        
        # Restore original weights
        if self.config.use_ema:
            self.ema.restore()
        
        return metrics
    
    def train(self, num_epochs: int = 10, output_dir: str = "models/modern") -> str:
        """Complete training with modern techniques"""
        
        logger.info(f"Starting modern training for {num_epochs} epochs")
        
        best_f1 = 0.0
        
        for epoch in range(num_epochs):
            # Training
            train_metrics = self.train_epoch(epoch)
            
            # Evaluation
            eval_metrics = self.evaluate()
            
            # Logging
            logger.info(f"Epoch {epoch}: {train_metrics} {eval_metrics}")
            
            if self.use_wandb:
                wandb.log({**train_metrics, **eval_metrics, "epoch": epoch})
            
            # Save best model
            current_f1 = eval_metrics.get('eval_f1', 0.0)
            if current_f1 > best_f1:
                best_f1 = current_f1
                self.save_model(f"{output_dir}/best_model")
        
        # Final save
        final_model_path = f"{output_dir}/final_model"
        self.save_model(final_model_path)
        
        logger.info(f"Training completed. Best F1: {best_f1:.4f}")
        
        return final_model_path
    
    def save_model(self, path: str):
        """Save model with all components"""
        Path(path).mkdir(parents=True, exist_ok=True)
        
        # Save model state
        if self.config.use_ema:
            self.ema.apply_shadow()
        
        torch.save({
            'model_state_dict': self.model.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
            'config': self.config,
            'epoch': self.epoch,
            'global_step': self.global_step
        }, f"{path}/pytorch_model.bin")
        
        if self.config.use_ema:
            self.ema.restore()
        
        # Save config
        with open(f"{path}/training_config.json", 'w') as f:
            json.dump(self.config.__dict__, f, indent=2)
        
        logger.info(f"Model saved to {path}")

# Usage example
if __name__ == "__main__":
    from transformers import AutoModel, AutoTokenizer
    
    # Initialize model (placeholder)
    model = AutoModel.from_pretrained("ai4bharat/indic-bert")
    
    # Create config
    config = ModernTrainingConfig(
        learning_rate=3e-5,
        use_amp=True,
        use_ema=True,
        use_sam=True,
        use_curriculum=True
    )
    
    # Initialize trainer
    trainer = ModernTrainer(
        model=model,
        config=config,
        train_dataset=None,  # Your dataset here
        eval_dataset=None    # Your dataset here
    )
    
    # Train
    model_path = trainer.train(num_epochs=10)
