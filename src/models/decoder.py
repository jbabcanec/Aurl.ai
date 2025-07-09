"""
Enhanced VAE decoder with hierarchical conditioning and skip connections.

This module implements an improved decoder that prevents posterior collapse
and supports hierarchical latent variables.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple
import math

from src.models.components import TransformerBlock, OutputHead
from src.models.attention import create_attention_layer
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class EnhancedMusicDecoder(nn.Module):
    """
    Enhanced decoder for VAE with hierarchical conditioning.
    
    Features:
    - Hierarchical latent conditioning (global → local → fine)
    - Skip connections to prevent posterior collapse
    - Adaptive conditioning based on sequence position
    - Memory-efficient generation
    """
    
    def __init__(self,
                 d_model: int,
                 latent_dim: int,
                 vocab_size: int,
                 n_layers: int = 6,
                 attention_config: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.1,
                 hierarchical: bool = True,
                 use_skip_connection: bool = True,
                 condition_every_layer: bool = False):
        """
        Initialize enhanced decoder.
        
        Args:
            d_model: Model dimension
            latent_dim: Total latent dimension
            vocab_size: Output vocabulary size (774)
            n_layers: Number of transformer layers
            attention_config: Attention configuration
            dropout: Dropout rate
            hierarchical: Use hierarchical conditioning
            use_skip_connection: Add skip from encoder features
            condition_every_layer: Condition at every transformer layer
        """
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        self.hierarchical = hierarchical
        self.use_skip_connection = use_skip_connection
        self.condition_every_layer = condition_every_layer
        
        # Default attention config
        if attention_config is None:
            attention_config = {
                'type': 'hierarchical',
                'd_model': d_model,
                'n_heads': 8,
                'dropout': dropout
            }
        
        # Latent to initial hidden state
        if hierarchical:
            latent_per_level = latent_dim // 3
            
            # Hierarchical projections
            self.global_proj = nn.Linear(latent_per_level, d_model)
            self.local_proj = nn.Linear(latent_per_level, d_model)
            self.fine_proj = nn.Linear(latent_per_level, d_model)
            
            # Adaptive weighting for different levels
            self.level_weights = nn.Parameter(torch.ones(3) / 3)
            
            # Position-aware conditioning
            self.position_mlp = nn.Sequential(
                nn.Linear(1, 64),
                nn.ReLU(),
                nn.Linear(64, 3),
                nn.Softmax(dim=-1)
            )
        else:
            self.latent_projection = nn.Linear(latent_dim, d_model)
        
        # Skip connection projection
        if use_skip_connection:
            self.skip_proj = nn.Linear(d_model * 2, d_model)
            self.skip_norm = nn.LayerNorm(d_model)
        
        # Decoder transformer blocks
        self.blocks = nn.ModuleList()
        for i in range(n_layers):
            block = TransformerBlock(d_model, attention_config, dropout=dropout)
            self.blocks.append(block)
            
            # Add conditioning at each layer if specified
            if condition_every_layer and i < n_layers - 1:
                if hierarchical:
                    self.blocks.append(nn.ModuleList([
                        nn.Linear(latent_per_level, d_model),
                        nn.Linear(latent_per_level, d_model),
                        nn.Linear(latent_per_level, d_model),
                        nn.LayerNorm(d_model)
                    ]))
                else:
                    self.blocks.append(nn.Sequential(
                        nn.Linear(latent_dim, d_model),
                        nn.LayerNorm(d_model)
                    ))
        
        self.norm = nn.LayerNorm(d_model)
        self.output_head = OutputHead(d_model, vocab_size, dropout)
        
        # Memory bank for caching during generation
        self.register_buffer('memory_bank', None)
        
        logger.info(f"Initialized EnhancedMusicDecoder with hierarchical={hierarchical}, "
                   f"skip_connection={use_skip_connection}")
    
    def forward(self,
                latent: torch.Tensor,
                target_embeddings: torch.Tensor,
                mask: Optional[torch.Tensor] = None,
                encoder_features: Optional[torch.Tensor] = None,
                return_hidden: bool = False) -> torch.Tensor:
        """
        Decode latent representation to musical sequence.
        
        Args:
            latent: Latent vector [batch_size, latent_dim]
            target_embeddings: Target sequence embeddings [batch_size, seq_len, d_model]
            mask: Optional attention mask
            encoder_features: Optional encoder hidden states for skip connection
            return_hidden: Return hidden states for analysis
            
        Returns:
            Logits [batch_size, seq_len, vocab_size] or dict if return_hidden
        """
        batch_size, seq_len, _ = target_embeddings.shape
        
        # Initialize hidden states with latent conditioning
        if self.hierarchical:
            x = self._hierarchical_condition(latent, target_embeddings, seq_len)
        else:
            x = self._standard_condition(latent, target_embeddings)
        
        # Apply skip connection if available
        if self.use_skip_connection and encoder_features is not None:
            # Combine decoder input with encoder features
            skip_features = torch.cat([x, encoder_features], dim=-1)
            skip_cond = self.skip_proj(skip_features)
            x = x + self.skip_norm(skip_cond)
        
        # Store intermediate hidden states if requested
        hidden_states = [] if return_hidden else None
        
        # Pass through decoder blocks
        for i, block in enumerate(self.blocks):
            if isinstance(block, TransformerBlock):
                x = block(x, mask)
                
                if hidden_states is not None:
                    hidden_states.append(x.clone())
                
                # Apply intermediate conditioning if enabled
                if self.condition_every_layer and i < len(self.blocks) - 1:
                    next_block = self.blocks[i + 1]
                    if self.hierarchical and isinstance(next_block, nn.ModuleList):
                        x = self._apply_intermediate_conditioning(x, latent, next_block, seq_len)
                    elif not self.hierarchical and isinstance(next_block, nn.Sequential):
                        latent_cond = next_block(latent).unsqueeze(1).expand(-1, seq_len, -1)
                        x = x + latent_cond
        
        x = self.norm(x)
        
        # Generate output logits
        logits = self.output_head(x)
        
        if return_hidden:
            return {
                'logits': logits,
                'hidden_states': hidden_states,
                'final_hidden': x
            }
        else:
            return logits
    
    def _standard_condition(self, 
                          latent: torch.Tensor, 
                          target_embeddings: torch.Tensor) -> torch.Tensor:
        """Standard latent conditioning."""
        batch_size, seq_len, d_model = target_embeddings.shape
        
        # Project latent to model dimension
        latent_proj = self.latent_projection(latent)  # [batch_size, d_model]
        
        # Broadcast latent across sequence
        latent_expanded = latent_proj.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Add to target embeddings
        return target_embeddings + latent_expanded
    
    def _hierarchical_condition(self,
                              latent: torch.Tensor,
                              target_embeddings: torch.Tensor,
                              seq_len: int) -> torch.Tensor:
        """Hierarchical latent conditioning with position-aware weighting."""
        batch_size = latent.shape[0]
        latent_per_level = self.latent_dim // 3
        
        # Split latent into hierarchical components
        global_latent = latent[:, :latent_per_level]
        local_latent = latent[:, latent_per_level:2*latent_per_level]
        fine_latent = latent[:, 2*latent_per_level:]
        
        # Project each level
        global_proj = self.global_proj(global_latent)  # [batch, d_model]
        local_proj = self.local_proj(local_latent)     # [batch, d_model]
        fine_proj = self.fine_proj(fine_latent)        # [batch, d_model]
        
        # Create position encodings for adaptive weighting
        positions = torch.linspace(0, 1, seq_len, device=latent.device)
        positions = positions.unsqueeze(0).unsqueeze(-1).expand(batch_size, -1, -1)
        
        # Get position-dependent weights for each level
        position_weights = self.position_mlp(positions)  # [batch, seq_len, 3]
        
        # Expand projections
        global_exp = global_proj.unsqueeze(1).expand(-1, seq_len, -1)
        local_exp = local_proj.unsqueeze(1).expand(-1, seq_len, -1)
        fine_exp = fine_proj.unsqueeze(1).expand(-1, seq_len, -1)
        
        # Apply position-dependent weighting
        weighted_global = global_exp * position_weights[:, :, 0:1]
        weighted_local = local_exp * position_weights[:, :, 1:2]
        weighted_fine = fine_exp * position_weights[:, :, 2:3]
        
        # Combine all levels
        latent_conditioning = weighted_global + weighted_local + weighted_fine
        
        return target_embeddings + latent_conditioning
    
    def _apply_intermediate_conditioning(self,
                                       x: torch.Tensor,
                                       latent: torch.Tensor,
                                       cond_layers: nn.ModuleList,
                                       seq_len: int) -> torch.Tensor:
        """Apply conditioning at intermediate layers."""
        latent_per_level = self.latent_dim // 3
        
        # Split latent
        global_latent = latent[:, :latent_per_level]
        local_latent = latent[:, latent_per_level:2*latent_per_level]
        fine_latent = latent[:, 2*latent_per_level:]
        
        # Project and condition
        global_cond = cond_layers[0](global_latent).unsqueeze(1).expand(-1, seq_len, -1)
        local_cond = cond_layers[1](local_latent).unsqueeze(1).expand(-1, seq_len, -1)
        fine_cond = cond_layers[2](fine_latent).unsqueeze(1).expand(-1, seq_len, -1)
        
        # Combine with adaptive weights
        combined = (global_cond + local_cond + fine_cond) / 3
        normalized = cond_layers[3](combined)
        
        return x + normalized
    
    def generate_from_latent(self,
                           latent: torch.Tensor,
                           max_length: int = 512,
                           temperature: float = 1.0,
                           top_k: Optional[int] = None,
                           top_p: Optional[float] = None,
                           start_token: int = 0) -> torch.Tensor:
        """
        Generate music from latent code.
        
        Args:
            latent: Latent vector [batch_size, latent_dim]
            max_length: Maximum sequence length
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p filtering
            start_token: Starting token ID
            
        Returns:
            Generated token sequence [batch_size, seq_len]
        """
        batch_size = latent.shape[0]
        device = latent.device
        
        # Initialize with start token
        generated = torch.full((batch_size, 1), start_token, dtype=torch.long, device=device)
        
        # Cache for autoregressive generation
        cache = None
        
        for _ in range(max_length - 1):
            # Get embeddings (would come from main model)
            # For now, create dummy embeddings
            curr_embeddings = torch.randn(batch_size, generated.shape[1], self.d_model, device=device)
            
            # Decode
            with torch.no_grad():
                logits = self(latent, curr_embeddings)
                next_token_logits = logits[:, -1, :] / temperature
                
                # Apply filtering
                if top_k is not None:
                    k = min(top_k, next_token_logits.size(-1))
                    indices_to_remove = next_token_logits < torch.topk(next_token_logits, k)[0][..., -1, None]
                    next_token_logits[indices_to_remove] = float('-inf')
                
                if top_p is not None:
                    sorted_logits, sorted_indices = torch.sort(next_token_logits, descending=True)
                    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                    
                    sorted_indices_to_remove = cumulative_probs > top_p
                    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
                    sorted_indices_to_remove[..., 0] = False
                    
                    indices_to_remove = sorted_indices[sorted_indices_to_remove]
                    next_token_logits.scatter_(1, indices_to_remove, float('-inf'))
                
                # Sample
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop on END token (1)
                if (next_token == 1).all():
                    break
        
        return generated
    
    def get_conditioning_info(self) -> Dict[str, Any]:
        """Get information about conditioning setup."""
        info = {
            'hierarchical': self.hierarchical,
            'use_skip_connection': self.use_skip_connection,
            'condition_every_layer': self.condition_every_layer,
            'n_layers': len([b for b in self.blocks if isinstance(b, TransformerBlock)])
        }
        
        if self.hierarchical:
            info['level_weights'] = self.level_weights.detach().cpu().numpy().tolist()
        
        return info