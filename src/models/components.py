"""
Shared model components for Aurl.ai music generation.

This module provides reusable building blocks for transformer-based music models.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any
import math

from src.models.attention import create_attention_layer, MusicalPositionalEncoding


class MusicEmbedding(nn.Module):
    """
    Embedding layer for musical tokens with optional learned embeddings.
    
    Handles the 774-token vocabulary discovered in our data analysis.
    """
    
    def __init__(self,
                 vocab_size: int = 774,
                 d_model: int = 512,
                 dropout: float = 0.1,
                 max_sequence_length: int = 2048):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        
        # Token embeddings
        self.token_embedding = nn.Embedding(vocab_size, d_model)
        
        # Positional encoding
        self.positional_encoding = MusicalPositionalEncoding(
            d_model, max_sequence_length, musical_features=True
        )
        
        self.dropout = nn.Dropout(dropout)
        
        # Initialize embeddings
        nn.init.normal_(self.token_embedding.weight, mean=0.0, std=0.02)
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Forward pass of embedding layer.
        
        Args:
            tokens: Token indices [batch_size, seq_len]
            
        Returns:
            Embedded tokens [batch_size, seq_len, d_model]
        """
        # Token embeddings
        embeddings = self.token_embedding(tokens) * math.sqrt(self.d_model)
        
        # Add positional encoding
        embeddings = self.positional_encoding(embeddings)
        
        return self.dropout(embeddings)


class FeedForward(nn.Module):
    """
    Position-wise feed-forward network.
    """
    
    def __init__(self,
                 d_model: int,
                 d_ff: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        if d_ff is None:
            d_ff = 4 * d_model
        
        self.linear1 = nn.Linear(d_model, d_ff)
        self.linear2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.linear2(self.dropout(F.gelu(self.linear1(x))))


class TransformerBlock(nn.Module):
    """
    Single transformer block with configurable attention mechanism.
    """
    
    def __init__(self,
                 d_model: int,
                 attention_config: Dict[str, Any],
                 d_ff: Optional[int] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        self.attention = create_attention_layer(attention_config)
        self.feed_forward = FeedForward(d_model, d_ff, dropout)
        
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass of transformer block.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        # Self-attention with residual connection
        if isinstance(self.attention, nn.MultiheadAttention):
            attn_output, _ = self.attention(x, x, x, attn_mask=mask)
            x = self.norm1(x + self.dropout(attn_output))
        else:
            x = self.norm1(x + self.dropout(self.attention(x, mask)))
        
        # Feed-forward with residual connection
        x = self.norm2(x + self.dropout(self.feed_forward(x)))
        
        return x


class OutputHead(nn.Module):
    """
    Output head for music generation with optional temperature scaling.
    """
    
    def __init__(self,
                 d_model: int,
                 vocab_size: int = 774,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        self.dropout = nn.Dropout(dropout)
        self.output_projection = nn.Linear(d_model, vocab_size)
        
        # Initialize output projection
        nn.init.normal_(self.output_projection.weight, mean=0.0, std=0.02)
        
    def forward(self, 
                x: torch.Tensor, 
                temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass of output head.
        
        Args:
            x: Hidden states [batch_size, seq_len, d_model]
            temperature: Temperature for softmax scaling
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        x = self.dropout(x)
        logits = self.output_projection(x)
        
        if temperature != 1.0:
            logits = logits / temperature
            
        return logits


class BaselineTransformer(nn.Module):
    """
    Simple baseline transformer for integration testing.
    
    This model tests the integration between:
    - Data pipeline (dataset.py)
    - Attention mechanisms (attention.py) 
    - Model components (this file)
    
    Designed to handle the 9,433 average token sequences we discovered.
    """
    
    def __init__(self,
                 vocab_size: int = 774,
                 d_model: int = 512,
                 n_layers: int = 6,
                 n_heads: int = 8,
                 d_ff: Optional[int] = None,
                 max_sequence_length: int = 2048,
                 dropout: float = 0.1,
                 attention_type: str = "hierarchical"):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_sequence_length = max_sequence_length
        
        # Embedding layer
        self.embedding = MusicEmbedding(
            vocab_size, d_model, dropout, max_sequence_length
        )
        
        # Attention configuration
        attention_config = {
            'type': attention_type,
            'd_model': d_model,
            'n_heads': n_heads,
            'dropout': dropout,
            'local_window_size': 256,
            'global_window_size': 64,
            'window_size': 512
        }
        
        # Transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, attention_config, d_ff, dropout)
            for _ in range(n_layers)
        ])
        
        # Output head
        self.output_head = OutputHead(d_model, vocab_size, dropout)
        
        # Final layer norm
        self.final_norm = nn.LayerNorm(d_model)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        """Initialize model weights."""
        if isinstance(module, nn.Linear):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                torch.nn.init.zeros_(module.bias)
        elif isinstance(module, nn.LayerNorm):
            torch.nn.init.zeros_(module.bias)
            torch.nn.init.ones_(module.weight)
        elif isinstance(module, nn.Embedding):
            torch.nn.init.normal_(module.weight, mean=0.0, std=0.02)
    
    def forward(self, 
                tokens: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0) -> torch.Tensor:
        """
        Forward pass of baseline transformer.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            mask: Optional attention mask
            temperature: Temperature for output scaling
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        # Embeddings
        x = self.embedding(tokens)
        
        # Transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        # Final layer norm
        x = self.final_norm(x)
        
        # Output projection
        logits = self.output_head(x, temperature)
        
        return logits
    
    def generate(self,
                 prompt_tokens: torch.Tensor,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None) -> torch.Tensor:
        """
        Generate music tokens autoregressively.
        
        Args:
            prompt_tokens: Initial tokens [batch_size, prompt_len]
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            
        Returns:
            Generated tokens [batch_size, prompt_len + max_new_tokens]
        """
        self.eval()
        
        generated = prompt_tokens.clone()
        batch_size = prompt_tokens.size(0)
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get logits for next token
                logits = self(generated, temperature=temperature)
                next_token_logits = logits[:, -1, :]  # [batch_size, vocab_size]
                
                # Apply top-k filtering
                if top_k is not None:
                    k = min(top_k, next_token_logits.size(-1))  # Ensure k <= vocab_size
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                # Apply top-p filtering
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    # Remove tokens with cumulative probability above threshold
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove, float('-inf'))
                
                # Sample next token
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                # Append to generated sequence
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if maximum sequence length reached
                if generated.size(1) >= self.max_sequence_length:
                    break
        
        return generated
    
    def get_model_size(self) -> Dict[str, int]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),  # Assuming fp32
            'layers': self.n_layers,
            'embedding_size': self.d_model,
            'vocab_size': self.vocab_size
        }


def create_baseline_model(config: Dict[str, Any]) -> BaselineTransformer:
    """
    Factory function to create baseline transformer from config.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured BaselineTransformer
    """
    return BaselineTransformer(
        vocab_size=config.get('vocab_size', 774),
        d_model=config.get('d_model', 512),
        n_layers=config.get('n_layers', 6),
        n_heads=config.get('n_heads', 8),
        d_ff=config.get('d_ff'),
        max_sequence_length=config.get('max_sequence_length', 2048),
        dropout=config.get('dropout', 0.1),
        attention_type=config.get('attention_type', 'hierarchical')
    )