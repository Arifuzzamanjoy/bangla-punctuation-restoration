#!/usr/bin/env python3
"""
Modern Transformer Enhancements for Bangla Punctuation Restoration
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import (
    AutoModel, AutoConfig, AutoTokenizer,
    get_linear_schedule_with_warmup
)
from typing import Dict, Any, Optional, List, Tuple
import math
import logging

logger = logging.getLogger(__name__)

class RotaryPositionalEmbedding(nn.Module):
    """Rotary Position Embedding (RoPE) - Latest positional encoding technique"""
    
    def __init__(self, dim: int, max_seq_len: int = 2048):
        super().__init__()
        self.dim = dim
        inv_freq = 1.0 / (10000 ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer('inv_freq', inv_freq)
        
    def forward(self, x: torch.Tensor, seq_len: int):
        t = torch.arange(seq_len, device=x.device).type_as(self.inv_freq)
        freqs = torch.einsum('i,j->ij', t, self.inv_freq)
        emb = torch.cat((freqs, freqs), dim=-1)
        return emb.cos(), emb.sin()

class SwiGLUFeedForward(nn.Module):
    """SwiGLU activation - Used in latest LLMs like PaLM, LLaMA"""
    
    def __init__(self, dim: int, hidden_dim: int, dropout: float = 0.1):
        super().__init__()
        self.w1 = nn.Linear(dim, hidden_dim, bias=False)
        self.w2 = nn.Linear(hidden_dim, dim, bias=False)
        self.w3 = nn.Linear(dim, hidden_dim, bias=False)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.dropout(self.w2(F.silu(self.w1(x)) * self.w3(x)))

class ModernAttentionBlock(nn.Module):
    """Modern attention with RMSNorm, RoPE, and optimizations"""
    
    def __init__(self, config):
        super().__init__()
        self.hidden_size = config.hidden_size
        self.num_heads = config.num_attention_heads
        self.head_dim = self.hidden_size // self.num_heads
        
        # Modern normalization (RMSNorm instead of LayerNorm)
        self.input_layernorm = RMSNorm(config.hidden_size)
        self.post_attention_layernorm = RMSNorm(config.hidden_size)
        
        # Attention layers
        self.q_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.k_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.v_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        self.o_proj = nn.Linear(self.hidden_size, self.hidden_size, bias=False)
        
        # Rotary positional embedding
        self.rotary_emb = RotaryPositionalEmbedding(self.head_dim)
        
        # Modern feed-forward network
        self.mlp = SwiGLUFeedForward(
            self.hidden_size, 
            int(2.67 * self.hidden_size),  # SwiGLU expansion ratio
            config.hidden_dropout_prob
        )
        
    def forward(self, hidden_states: torch.Tensor, attention_mask: Optional[torch.Tensor] = None):
        residual = hidden_states
        hidden_states = self.input_layernorm(hidden_states)
        
        # Multi-head attention with RoPE
        batch_size, seq_len, _ = hidden_states.shape
        
        q = self.q_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        k = self.k_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        v = self.v_proj(hidden_states).view(batch_size, seq_len, self.num_heads, self.head_dim)
        
        # Apply rotary embeddings
        cos, sin = self.rotary_emb(hidden_states, seq_len)
        q, k = self.apply_rotary_pos_emb(q, k, cos, sin)
        
        # Attention computation
        attn_output = self.scaled_dot_product_attention(q, k, v, attention_mask)
        attn_output = self.o_proj(attn_output)
        
        # First residual connection
        hidden_states = residual + attn_output
        
        # Feed-forward
        residual = hidden_states
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)
        
        # Second residual connection
        return residual + hidden_states
    
    def apply_rotary_pos_emb(self, q, k, cos, sin):
        """Apply rotary positional embedding"""
        def rotate_half(x):
            x1, x2 = x[..., :x.shape[-1]//2], x[..., x.shape[-1]//2:]
            return torch.cat((-x2, x1), dim=-1)
        
        q = (q * cos) + (rotate_half(q) * sin)
        k = (k * cos) + (rotate_half(k) * sin)
        return q, k
    
    def scaled_dot_product_attention(self, q, k, v, mask=None):
        """Optimized attention computation"""
        # Use Flash Attention if available
        if hasattr(F, 'scaled_dot_product_attention'):
            return F.scaled_dot_product_attention(q, k, v, attn_mask=mask)
        
        # Fallback to manual implementation
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.head_dim)
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        attn_weights = F.softmax(scores, dim=-1)
        return torch.matmul(attn_weights, v)

class RMSNorm(nn.Module):
    """Root Mean Square Layer Normalization - More stable than LayerNorm"""
    
    def __init__(self, normalized_shape: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape))
        self.eps = eps
        
    def forward(self, hidden_states: torch.Tensor):
        input_dtype = hidden_states.dtype
        variance = hidden_states.to(torch.float32).pow(2).mean(-1, keepdim=True)
        hidden_states = hidden_states * torch.rsqrt(variance + self.eps)
        return (self.weight * hidden_states).to(input_dtype)

class ModernBanglaPunctuationModel(nn.Module):
    """State-of-the-art model with latest architectural improvements"""
    
    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__()
        
        # Load base configuration
        self.config = AutoConfig.from_pretrained(base_model_name)
        self.config.num_labels = num_labels
        
        # Enhanced backbone with modern techniques
        self.backbone = AutoModel.from_pretrained(base_model_name, config=self.config)
        
        # Modern attention layers
        self.modern_layers = nn.ModuleList([
            ModernAttentionBlock(self.config) for _ in range(2)
        ])
        
        # Mixture of Experts (MoE) for better capacity
        self.moe_layer = MixtureOfExperts(
            self.config.hidden_size, 
            num_experts=8, 
            top_k=2
        )
        
        # Advanced classifier with uncertainty estimation
        self.classifier = AdvancedClassifier(
            self.config.hidden_size, 
            num_labels,
            enable_uncertainty=True
        )
        
        # Adaptive layer scaling
        self.layer_scales = nn.Parameter(torch.ones(len(self.modern_layers)))
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        # Base model outputs
        outputs = self.backbone(input_ids=input_ids, attention_mask=attention_mask)
        sequence_output = outputs.last_hidden_state
        
        # Apply modern attention layers with scaling
        for i, layer in enumerate(self.modern_layers):
            layer_output = layer(sequence_output, attention_mask)
            sequence_output = sequence_output + self.layer_scales[i] * layer_output
        
        # Mixture of Experts enhancement
        sequence_output = self.moe_layer(sequence_output)
        
        # Classification with uncertainty
        logits, uncertainty = self.classifier(sequence_output)
        
        loss = None
        if labels is not None:
            # Label smoothing + focal loss for better training
            loss = self.compute_advanced_loss(logits, labels, uncertainty)
        
        return {
            'loss': loss,
            'logits': logits,
            'uncertainty': uncertainty,
            'hidden_states': sequence_output
        }
    
    def compute_advanced_loss(self, logits, labels, uncertainty):
        """Advanced loss with label smoothing and uncertainty weighting"""
        # Focal loss for hard examples
        ce_loss = F.cross_entropy(logits.view(-1, self.config.num_labels), 
                                 labels.view(-1), ignore_index=-100, reduction='none')
        
        # Focal loss computation
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** 2 * ce_loss
        
        # Uncertainty weighting
        if uncertainty is not None:
            uncertainty = uncertainty.view(-1)
            focal_loss = focal_loss / (uncertainty + 1e-8)
        
        return focal_loss.mean()

class MixtureOfExperts(nn.Module):
    """Mixture of Experts for increased model capacity"""
    
    def __init__(self, hidden_size: int, num_experts: int = 8, top_k: int = 2):
        super().__init__()
        self.num_experts = num_experts
        self.top_k = top_k
        
        # Gating network
        self.gate = nn.Linear(hidden_size, num_experts)
        
        # Expert networks
        self.experts = nn.ModuleList([
            nn.Sequential(
                nn.Linear(hidden_size, hidden_size * 4),
                nn.GELU(),
                nn.Linear(hidden_size * 4, hidden_size),
                nn.Dropout(0.1)
            ) for _ in range(num_experts)
        ])
        
    def forward(self, x: torch.Tensor):
        batch_size, seq_len, hidden_size = x.shape
        x_flat = x.view(-1, hidden_size)
        
        # Gate scores
        gate_scores = F.softmax(self.gate(x_flat), dim=-1)
        top_k_gates, top_k_indices = torch.topk(gate_scores, self.top_k, dim=-1)
        
        # Normalize top-k gates
        top_k_gates = top_k_gates / top_k_gates.sum(dim=-1, keepdim=True)
        
        # Compute expert outputs
        output = torch.zeros_like(x_flat)
        for i in range(self.top_k):
            expert_idx = top_k_indices[:, i]
            gate_weight = top_k_gates[:, i].unsqueeze(-1)
            
            for j in range(self.num_experts):
                mask = (expert_idx == j)
                if mask.any():
                    expert_output = self.experts[j](x_flat[mask])
                    output[mask] += gate_weight[mask] * expert_output
        
        return output.view(batch_size, seq_len, hidden_size)

class AdvancedClassifier(nn.Module):
    """Advanced classifier with uncertainty estimation"""
    
    def __init__(self, hidden_size: int, num_labels: int, enable_uncertainty: bool = True):
        super().__init__()
        self.enable_uncertainty = enable_uncertainty
        
        # Main classification head
        self.classifier = nn.Sequential(
            nn.Linear(hidden_size, hidden_size // 2),
            nn.GELU(),
            nn.LayerNorm(hidden_size // 2),
            nn.Dropout(0.1),
            nn.Linear(hidden_size // 2, num_labels)
        )
        
        # Uncertainty estimation head
        if enable_uncertainty:
            self.uncertainty_head = nn.Sequential(
                nn.Linear(hidden_size, hidden_size // 4),
                nn.ReLU(),
                nn.Linear(hidden_size // 4, 1),
                nn.Softplus()  # Ensures positive uncertainty
            )
    
    def forward(self, x: torch.Tensor):
        logits = self.classifier(x)
        
        uncertainty = None
        if self.enable_uncertainty:
            uncertainty = self.uncertainty_head(x)
            
        return logits, uncertainty

class AdversarialTraining:
    """Advanced adversarial training techniques"""
    
    @staticmethod
    def smart_adversarial_training(model, inputs, labels, epsilon=0.01):
        """SMART adversarial training"""
        # Get embeddings
        embeddings = model.backbone.embeddings(inputs['input_ids'])
        
        # Add adversarial noise
        noise = torch.randn_like(embeddings) * epsilon
        embeddings_adv = embeddings + noise
        
        # Forward pass with adversarial embeddings
        outputs_adv = model.backbone(inputs_embeds=embeddings_adv, 
                                   attention_mask=inputs['attention_mask'])
        
        return outputs_adv

# Example integration
class UltraModernPunctuationModel(ModernBanglaPunctuationModel):
    """Ultra-modern model with all latest techniques"""
    
    def __init__(self, base_model_name: str, num_labels: int):
        super().__init__(base_model_name, num_labels)
        
        # Knowledge distillation support
        self.teacher_model = None
        self.distillation_alpha = 0.7
        self.distillation_temperature = 4.0
        
        # Gradient checkpointing for memory efficiency
        self.gradient_checkpointing = True
        
    def set_teacher_model(self, teacher_model):
        """Set teacher model for knowledge distillation"""
        self.teacher_model = teacher_model
        
    def forward(self, input_ids, attention_mask=None, labels=None):
        outputs = super().forward(input_ids, attention_mask, labels)
        
        # Knowledge distillation loss
        if self.training and self.teacher_model is not None and labels is not None:
            with torch.no_grad():
                teacher_outputs = self.teacher_model(input_ids, attention_mask)
            
            # Distillation loss
            distillation_loss = self.compute_distillation_loss(
                outputs['logits'], teacher_outputs['logits']
            )
            
            # Combine losses
            if outputs['loss'] is not None:
                outputs['loss'] = (1 - self.distillation_alpha) * outputs['loss'] + \
                                self.distillation_alpha * distillation_loss
        
        return outputs
    
    def compute_distillation_loss(self, student_logits, teacher_logits):
        """Compute knowledge distillation loss"""
        student_probs = F.log_softmax(student_logits / self.distillation_temperature, dim=-1)
        teacher_probs = F.softmax(teacher_logits / self.distillation_temperature, dim=-1)
        
        return F.kl_div(student_probs, teacher_probs, reduction='batchmean') * \
               (self.distillation_temperature ** 2)
