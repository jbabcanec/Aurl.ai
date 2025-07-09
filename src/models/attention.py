"""
Custom attention mechanisms for Aurl.ai music generation.

This module implements hierarchical and efficient attention patterns designed 
specifically for long musical sequences (9,433+ tokens average).
"""

import math
import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Tuple, Dict, Any
import logging

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class MusicalPositionalEncoding(nn.Module):
    """
    Musical-aware positional encoding with relative timing information.
    
    Designed for music where timing relationships are crucial for understanding
    structure, rhythm, and harmonic progressions.
    """
    
    def __init__(self, 
                 d_model: int, 
                 max_len: int = 12000,
                 musical_features: bool = True):
        super().__init__()
        self.d_model = d_model
        self.musical_features = musical_features
        
        # Standard sinusoidal positional encoding
        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len, dtype=torch.float).unsqueeze(1)
        
        div_term = torch.exp(torch.arange(0, d_model, 2).float() * 
                           (-math.log(10000.0) / d_model))
        
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        
        if musical_features:
            # Add musical timing features
            # Beat-level encoding (assuming 4/4 time, 16th note resolution)
            beat_positions = (position % 16) / 16.0  # Position within beat
            measure_positions = (position % 64) / 64.0  # Position within measure
            
            # Use part of embedding space for musical timing
            musical_dim = d_model // 4
            pe[:, :musical_dim//2] += torch.sin(beat_positions * 2 * math.pi)
            pe[:, musical_dim//2:musical_dim] += torch.cos(measure_positions * 2 * math.pi)
        
        self.register_buffer('pe', pe.unsqueeze(0))
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Add positional encoding to input embeddings."""
        seq_len = x.size(1)
        return x + self.pe[:, :seq_len]


class HierarchicalAttention(nn.Module):
    """
    Hierarchical attention mechanism for long musical sequences.
    
    Uses a two-level approach:
    1. Local attention: Fine-grained attention within windows
    2. Global attention: Coarse-grained attention across windows
    
    This reduces complexity from O(n²) to O(n*w + n/w*g) where:
    - n = sequence length
    - w = local window size  
    - g = global window size
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 local_window_size: int = 256,
                 global_window_size: int = 64,
                 dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.local_window_size = local_window_size
        self.global_window_size = global_window_size
        
        # Local attention projections
        self.local_q = nn.Linear(d_model, d_model)
        self.local_k = nn.Linear(d_model, d_model)
        self.local_v = nn.Linear(d_model, d_model)
        
        # Global attention projections
        self.global_q = nn.Linear(d_model, d_model)
        self.global_k = nn.Linear(d_model, d_model)
        self.global_v = nn.Linear(d_model, d_model)
        
        # Output projection - takes concatenated local+global attention
        self.output_proj = nn.Linear(d_model * 2, d_model)
        
        self.dropout = nn.Dropout(dropout)
        self.layer_norm = nn.LayerNorm(d_model)
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with hierarchical attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Store residual connection
        residual = x
        
        # Apply local attention within windows
        local_output = self._apply_local_attention(x, mask)
        
        # Apply global attention across windows  
        global_output = self._apply_global_attention(x, mask)
        
        # Combine local and global attention
        # Use learned combination weights
        combined_features = torch.cat([local_output, global_output], dim=-1)
        combined = self.output_proj(combined_features)
        
        # Residual connection and layer norm
        output = self.layer_norm(residual + self.dropout(combined))
        
        return output
    
    def _apply_local_attention(self, 
                             x: torch.Tensor, 
                             mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention within local windows."""
        batch_size, seq_len, d_model = x.shape
        window_size = self.local_window_size
        
        # Pad sequence to be divisible by window size
        padding = (window_size - seq_len % window_size) % window_size
        if padding > 0:
            x_padded = F.pad(x, (0, 0, 0, padding))
            padded_len = seq_len + padding
        else:
            x_padded = x
            padded_len = seq_len
        
        # Reshape into windows [batch_size, n_windows, window_size, d_model]
        n_windows = padded_len // window_size
        x_windowed = x_padded.view(batch_size, n_windows, window_size, d_model)
        
        # Apply attention within each window
        outputs = []
        for i in range(n_windows):
            window = x_windowed[:, i]  # [batch_size, window_size, d_model]
            
            # Standard scaled dot-product attention
            q = self.local_q(window).view(batch_size, window_size, self.n_heads, self.d_k).transpose(1, 2)
            k = self.local_k(window).view(batch_size, window_size, self.n_heads, self.d_k).transpose(1, 2)
            v = self.local_v(window).view(batch_size, window_size, self.n_heads, self.d_k).transpose(1, 2)
            
            # Attention scores
            scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
            
            # Apply mask if provided (adapt to window)
            if mask is not None:
                window_start = i * window_size
                window_end = min(window_start + window_size, seq_len)
                if window_end > window_start:
                    window_mask = mask[:, window_start:window_end, window_start:window_end]
                    if window_mask.size(-1) < window_size:
                        # Pad mask for this window
                        pad_size = window_size - window_mask.size(-1)
                        window_mask = F.pad(window_mask, (0, pad_size, 0, pad_size), value=float('-inf'))
                    scores = scores + window_mask.unsqueeze(1)
            
            # Softmax and apply to values
            attn_weights = F.softmax(scores, dim=-1)
            attn_weights = self.dropout(attn_weights)
            
            output = torch.matmul(attn_weights, v)
            output = output.transpose(1, 2).contiguous().view(batch_size, window_size, d_model)
            outputs.append(output)
        
        # Concatenate windows and remove padding
        local_output = torch.cat(outputs, dim=1)
        if padding > 0:
            local_output = local_output[:, :seq_len]
        
        return local_output
    
    def _apply_global_attention(self, 
                              x: torch.Tensor, 
                              mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Apply attention across downsampled global context."""
        batch_size, seq_len, d_model = x.shape
        stride = self.global_window_size
        
        # Downsample sequence by taking every stride-th token
        # This creates a coarse representation of the sequence
        global_indices = torch.arange(0, seq_len, stride, device=x.device)
        global_context = x[:, global_indices]  # [batch_size, global_len, d_model]
        global_len = global_context.size(1)
        
        # Apply standard attention on downsampled sequence
        q = self.global_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.global_k(global_context).view(batch_size, global_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.global_v(global_context).view(batch_size, global_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention scores [batch_size, n_heads, seq_len, global_len]
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        # Apply global mask if provided
        if mask is not None:
            # Create global mask by downsampling
            global_mask = mask[:, :, global_indices]
            scores = scores + global_mask.unsqueeze(1)
        
        # Softmax and apply to values
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        global_output = torch.matmul(attn_weights, v)
        global_output = global_output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return global_output


class SlidingWindowAttention(nn.Module):
    """
    Sliding window attention with configurable window size.
    
    More memory-efficient alternative to full attention for very long sequences.
    Each token attends to a fixed-size window around its position.
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 window_size: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        assert d_model % n_heads == 0
        
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        self.window_size = window_size
        
        self.q_proj = nn.Linear(d_model, d_model)
        self.k_proj = nn.Linear(d_model, d_model)
        self.v_proj = nn.Linear(d_model, d_model)
        self.output_proj = nn.Linear(d_model, d_model)
        
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with sliding window attention.
        
        Args:
            x: Input tensor [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Output tensor [batch_size, seq_len, d_model]
        """
        batch_size, seq_len, d_model = x.shape
        
        # Project to q, k, v
        q = self.q_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.k_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.v_proj(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Create sliding window mask
        window_mask = self._create_sliding_window_mask(seq_len, self.window_size, x.device)
        
        # Combine with provided mask
        if mask is not None:
            window_mask = window_mask + mask
        
        # Standard scaled dot-product attention with sliding window mask
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        scores = scores + window_mask.unsqueeze(1)  # Broadcast across heads
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        output = torch.matmul(attn_weights, v)
        output = output.transpose(1, 2).contiguous().view(batch_size, seq_len, d_model)
        
        return self.output_proj(output)
    
    def _create_sliding_window_mask(self, 
                                  seq_len: int, 
                                  window_size: int, 
                                  device: torch.device) -> torch.Tensor:
        """Create sliding window attention mask."""
        mask = torch.full((seq_len, seq_len), float('-inf'), device=device)
        
        half_window = window_size // 2
        for i in range(seq_len):
            start = max(0, i - half_window)
            end = min(seq_len, i + half_window + 1)
            mask[i, start:end] = 0.0
        
        return mask


class MultiScaleAttention(nn.Module):
    """
    Multi-scale attention that combines multiple attention mechanisms.
    
    Designed for musical sequences where both local patterns (motifs, phrases)
    and global structure (form, harmonic progressions) are important.
    """
    
    def __init__(self,
                 d_model: int,
                 n_heads: int = 8,
                 use_hierarchical: bool = True,
                 use_sliding_window: bool = True,
                 local_window_size: int = 256,
                 global_window_size: int = 64,
                 sliding_window_size: int = 512,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.use_hierarchical = use_hierarchical
        self.use_sliding_window = use_sliding_window
        
        attention_modules = []
        
        if use_hierarchical:
            attention_modules.append(HierarchicalAttention(
                d_model, n_heads, local_window_size, global_window_size, dropout
            ))
        
        if use_sliding_window:
            attention_modules.append(SlidingWindowAttention(
                d_model, n_heads, sliding_window_size, dropout
            ))
        
        if not attention_modules:
            # Fallback to standard attention
            attention_modules.append(nn.MultiheadAttention(
                d_model, n_heads, dropout=dropout, batch_first=True
            ))
        
        self.attention_modules = nn.ModuleList(attention_modules)
        
        # Combination layer
        self.combination_proj = nn.Linear(
            d_model * len(attention_modules), d_model
        )
        
        self.layer_norm = nn.LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """Forward pass with multi-scale attention."""
        residual = x
        
        # Apply each attention mechanism
        attention_outputs = []
        for attention_module in self.attention_modules:
            if isinstance(attention_module, nn.MultiheadAttention):
                # Standard PyTorch MultiheadAttention
                attn_output, _ = attention_module(x, x, x, attn_mask=mask)
                attention_outputs.append(attn_output)
            else:
                # Custom attention modules
                attn_output = attention_module(x, mask)
                attention_outputs.append(attn_output)
        
        # Combine attention outputs
        if len(attention_outputs) == 1:
            combined = attention_outputs[0]
        else:
            combined = torch.cat(attention_outputs, dim=-1)
            combined = self.combination_proj(combined)
        
        # Residual connection and layer norm
        output = self.layer_norm(residual + self.dropout(combined))
        
        return output


def create_attention_layer(config: Dict[str, Any]) -> nn.Module:
    """
    Factory function to create attention layer based on configuration.
    
    Args:
        config: Dictionary with attention configuration
        
    Returns:
        Configured attention module
    """
    attention_type = config.get('type', 'hierarchical')
    d_model = config['d_model']
    n_heads = config.get('n_heads', 8)
    dropout = config.get('dropout', 0.1)
    
    if attention_type == 'hierarchical':
        return HierarchicalAttention(
            d_model=d_model,
            n_heads=n_heads,
            local_window_size=config.get('local_window_size', 256),
            global_window_size=config.get('global_window_size', 64),
            dropout=dropout
        )
    
    elif attention_type == 'sliding_window':
        return SlidingWindowAttention(
            d_model=d_model,
            n_heads=n_heads,
            window_size=config.get('window_size', 512),
            dropout=dropout
        )
    
    elif attention_type == 'multi_scale':
        return MultiScaleAttention(
            d_model=d_model,
            n_heads=n_heads,
            use_hierarchical=config.get('use_hierarchical', True),
            use_sliding_window=config.get('use_sliding_window', True),
            local_window_size=config.get('local_window_size', 256),
            global_window_size=config.get('global_window_size', 64),
            sliding_window_size=config.get('sliding_window_size', 512),
            dropout=dropout
        )
    
    elif attention_type == 'standard':
        return nn.MultiheadAttention(
            embed_dim=d_model,
            num_heads=n_heads,
            dropout=dropout,
            batch_first=True
        )
    
    else:
        raise ValueError(f"Unknown attention type: {attention_type}")