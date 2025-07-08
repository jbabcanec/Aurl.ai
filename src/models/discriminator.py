"""
Enhanced GAN discriminator for musical sequence generation.

This module implements a multi-scale discriminator architecture designed specifically
for music generation with the following features:

1. Multi-scale discrimination (local, phrase, global levels)
2. Spectral normalization for training stability
3. Music-specific feature extraction (rhythm, harmony, melody)
4. Progressive training support
5. Comprehensive regularization techniques

Designed to work with Aurl.ai's 774-token vocabulary and hierarchical VAE structure.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import math
import numpy as np

from src.models.components import TransformerBlock
from src.models.attention import create_attention_layer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class SpectralNorm(nn.Module):
    """
    Spectral normalization for improved GAN training stability.
    
    Implements the spectral normalization technique from "Spectral Normalization for GANs"
    which constrains the Lipschitz constant of the discriminator.
    """
    
    def __init__(self, module, name='weight', n_power_iterations=1):
        super().__init__()
        self.module = module
        self.name = name
        self.n_power_iterations = n_power_iterations
        
        # Get weight tensor
        w = getattr(self.module, self.name)
        
        # Create parameter-like buffers
        self.register_buffer(f'{name}_u', F.normalize(torch.randn(w.size(0)), dim=0))
        self.register_buffer(f'{name}_v', F.normalize(torch.randn(w.size(1)), dim=0))
        self.register_buffer(f'{name}_sigma', torch.ones(1))
        
    def forward(self, *args, **kwargs):
        """Apply spectral normalization and forward pass."""
        self._compute_spectral_norm()
        return self.module(*args, **kwargs)
    
    def _compute_spectral_norm(self):
        """Compute spectral norm using power iteration."""
        w = getattr(self.module, self.name)
        u = getattr(self, f'{self.name}_u')
        v = getattr(self, f'{self.name}_v')
        
        # Reshape weight for matrix operations
        w_mat = w.view(w.size(0), -1)
        
        # Power iteration
        for _ in range(self.n_power_iterations):
            v_new = F.normalize(torch.mv(w_mat.t(), u), dim=0)
            u_new = F.normalize(torch.mv(w_mat, v_new), dim=0)
            u = u_new
            v = v_new
        
        # Compute spectral norm
        sigma = torch.dot(u, torch.mv(w_mat, v))
        
        # Apply normalization directly to weight data
        w_normalized = w / sigma.clamp(min=1e-8)
        
        # Update the module weight
        with torch.no_grad():
            getattr(self.module, self.name).copy_(w_normalized)
        
        # Update buffers
        getattr(self, f'{self.name}_u').copy_(u)
        getattr(self, f'{self.name}_v').copy_(v)
        getattr(self, f'{self.name}_sigma').copy_(sigma)


def spectral_norm(module, name='weight', n_power_iterations=1):
    """Convenience function to apply spectral normalization."""
    return SpectralNorm(module, name, n_power_iterations)


class MusicalFeatureExtractor(nn.Module):
    """
    Extract music-specific features for discriminator analysis.
    
    Analyzes musical sequences to extract meaningful features:
    - Rhythmic patterns (timing, note durations)
    - Harmonic content (chord progressions, intervals)
    - Melodic contour (pitch patterns, direction)
    - Dynamics (velocity patterns, accents)
    """
    
    def __init__(self, d_model: int, vocab_size: int = 774):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        
        # Musical token ranges (based on our 774 vocab)
        self.note_on_start = 0      # NOTE_ON tokens: 0-127
        self.note_off_start = 128   # NOTE_OFF tokens: 128-255
        self.time_shift_start = 256 # TIME_SHIFT tokens: 256-767
        self.velocity_start = 768   # VELOCITY_CHANGE tokens: 768-773
        
        # Feature extraction layers
        self.rhythm_analyzer = nn.Sequential(
            nn.Conv1d(1, 32, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.Conv1d(32, 64, kernel_size=4, stride=2, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(d_model // 4)
        )
        
        self.harmony_analyzer = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=12, stride=1, padding=6),  # Chord analysis
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=8, stride=2, padding=3),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(d_model // 4)
        )
        
        self.melody_analyzer = nn.Sequential(
            nn.Conv1d(128, 64, kernel_size=5, stride=1, padding=2),   # Melodic patterns
            nn.ReLU(),
            nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(d_model // 4)
        )
        
        self.dynamics_analyzer = nn.Sequential(
            nn.Conv1d(6, 16, kernel_size=4, stride=1, padding=2),     # Velocity patterns
            nn.ReLU(),
            nn.Conv1d(16, 16, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.AdaptiveAvgPool1d(d_model // 4)
        )
        
        # Combine features
        self.feature_combiner = nn.Linear(d_model, d_model)
        
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """
        Extract musical features from token sequences.
        
        Args:
            tokens: Token sequence [batch_size, seq_len]
            
        Returns:
            Musical features [batch_size, seq_len, d_model]
        """
        batch_size, seq_len = tokens.shape
        device = tokens.device
        
        # Extract different musical aspects
        rhythm_features = self._extract_rhythm_features(tokens)
        harmony_features = self._extract_harmony_features(tokens)
        melody_features = self._extract_melody_features(tokens)
        dynamics_features = self._extract_dynamics_features(tokens)
        
        # Combine all features - ensure they sum to d_model
        total_features = rhythm_features.size(1) + harmony_features.size(1) + melody_features.size(1) + dynamics_features.size(1)
        
        if total_features != self.d_model:
            # Adjust to match d_model exactly
            rhythm_features = F.adaptive_avg_pool1d(rhythm_features.unsqueeze(1), self.d_model // 4).squeeze(1)
            harmony_features = F.adaptive_avg_pool1d(harmony_features.unsqueeze(1), self.d_model // 4).squeeze(1)
            melody_features = F.adaptive_avg_pool1d(melody_features.unsqueeze(1), self.d_model // 4).squeeze(1)
            dynamics_features = F.adaptive_avg_pool1d(dynamics_features.unsqueeze(1), self.d_model // 4).squeeze(1)
        
        combined = torch.cat([
            rhythm_features,
            harmony_features, 
            melody_features,
            dynamics_features
        ], dim=1)  # [batch_size, d_model]
        
        # Project and expand to sequence length
        features = self.feature_combiner(combined)
        features = features.unsqueeze(1).expand(-1, seq_len, -1)
        
        return features
    
    def _extract_rhythm_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Extract rhythmic timing patterns."""
        batch_size, seq_len = tokens.shape
        
        # Extract time shift tokens (256-767)
        time_shifts = ((tokens >= self.time_shift_start) & 
                      (tokens < self.velocity_start)).float()
        
        # Analyze timing patterns
        time_shifts = time_shifts.unsqueeze(1)  # [batch, 1, seq_len]
        rhythm_features = self.rhythm_analyzer(time_shifts)
        
        return rhythm_features.view(batch_size, -1)
    
    def _extract_harmony_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Extract harmonic content (simultaneous notes)."""
        batch_size, seq_len = tokens.shape
        
        # Create pitch class histogram over time windows
        pitch_classes = torch.zeros(batch_size, 128, seq_len, device=tokens.device)
        
        # Mark active notes (NOTE_ON: 0-127)
        note_on_mask = tokens < self.note_off_start
        note_pitches = tokens * note_on_mask.long()
        
        for i in range(batch_size):
            for t in range(seq_len):
                if note_on_mask[i, t]:
                    pitch = note_pitches[i, t] % 128
                    pitch_classes[i, pitch, t] = 1.0
        
        # Analyze harmonic patterns
        harmony_features = self.harmony_analyzer(pitch_classes)
        
        return harmony_features.view(batch_size, -1)
    
    def _extract_melody_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Extract melodic contour and patterns."""
        batch_size, seq_len = tokens.shape
        
        # Extract note sequences and compute intervals
        note_sequence = torch.zeros(batch_size, 128, seq_len, device=tokens.device)
        
        # Mark melodic notes (highest active note at each timestep)
        note_on_mask = tokens < self.note_off_start
        note_pitches = tokens * note_on_mask.long()
        
        for i in range(batch_size):
            for t in range(seq_len):
                if note_on_mask[i, t]:
                    pitch = note_pitches[i, t] % 128
                    note_sequence[i, pitch, t] = 1.0
        
        # Analyze melodic patterns
        melody_features = self.melody_analyzer(note_sequence)
        
        return melody_features.view(batch_size, -1)
    
    def _extract_dynamics_features(self, tokens: torch.Tensor) -> torch.Tensor:
        """Extract velocity/dynamics patterns."""
        batch_size, seq_len = tokens.shape
        
        # Extract velocity tokens (768-773)
        velocity_mask = tokens >= self.velocity_start
        velocity_values = (tokens - self.velocity_start) * velocity_mask.long()
        
        # Create velocity pattern representation
        velocity_patterns = torch.zeros(batch_size, 6, seq_len, device=tokens.device)
        
        for i in range(batch_size):
            for t in range(seq_len):
                if velocity_mask[i, t]:
                    vel_idx = velocity_values[i, t].clamp(0, 5)
                    velocity_patterns[i, vel_idx, t] = 1.0
        
        # Analyze dynamics patterns
        dynamics_features = self.dynamics_analyzer(velocity_patterns)
        
        return dynamics_features.view(batch_size, -1)


class SimpleAttention(nn.Module):
    """Simple multi-head attention for discriminator."""
    
    def __init__(self, d_model: int, n_heads: int = 8, dropout: float = 0.1):
        super().__init__()
        self.d_model = d_model
        self.n_heads = n_heads
        self.d_k = d_model // n_heads
        
        self.w_q = nn.Linear(d_model, d_model)
        self.w_k = nn.Linear(d_model, d_model)
        self.w_v = nn.Linear(d_model, d_model)
        self.w_o = nn.Linear(d_model, d_model)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        batch_size, seq_len, _ = x.shape
        
        # Linear projections
        q = self.w_q(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        k = self.w_k(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        v = self.w_v(x).view(batch_size, seq_len, self.n_heads, self.d_k).transpose(1, 2)
        
        # Attention
        scores = torch.matmul(q, k.transpose(-2, -1)) / math.sqrt(self.d_k)
        
        if mask is not None:
            scores = scores.masked_fill(mask == 0, -1e9)
        
        attn_weights = F.softmax(scores, dim=-1)
        attn_weights = self.dropout(attn_weights)
        
        # Apply attention
        attn_output = torch.matmul(attn_weights, v)
        attn_output = attn_output.transpose(1, 2).contiguous().view(batch_size, seq_len, self.d_model)
        
        return self.w_o(attn_output)


class DiscriminatorBlock(nn.Module):
    """
    Enhanced discriminator block with musical awareness.
    
    Combines transformer attention with musical feature analysis
    and spectral normalization for stability.
    """
    
    def __init__(self,
                 d_model: int,
                 attention_config: Dict[str, Any],
                 dropout: float = 0.1,
                 use_spectral_norm: bool = True):
        super().__init__()
        
        self.d_model = d_model
        
        # Use simple attention for discriminator
        n_heads = attention_config.get('n_heads', 8)
        self.attention = SimpleAttention(d_model, n_heads, dropout)
        
        # Feed forward with spectral normalization
        self.ff = nn.Sequential(
            nn.Linear(d_model, d_model * 4),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model * 4, d_model),
            nn.Dropout(dropout)
        )
        
        if use_spectral_norm:
            self.ff[0] = spectral_norm(self.ff[0])
            self.ff[3] = spectral_norm(self.ff[3])
        
        # Layer norms
        self.norm1 = nn.LayerNorm(d_model)
        self.norm2 = nn.LayerNorm(d_model)
        
        # Musical feature integration
        self.feature_gate = nn.Sequential(
            nn.Linear(d_model * 2, d_model),
            nn.Sigmoid()
        )
        
        if use_spectral_norm:
            self.feature_gate[0] = spectral_norm(self.feature_gate[0])
    
    def forward(self,
                x: torch.Tensor,
                musical_features: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Forward pass with musical feature integration.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            musical_features: Musical features [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Enhanced representations [batch_size, seq_len, d_model]
        """
        # Self-attention
        attn_out = self.attention(x, mask)
        x = self.norm1(x + attn_out)
        
        # Integrate musical features
        combined = torch.cat([x, musical_features], dim=-1)
        gate = self.feature_gate(combined)
        x = x * gate + musical_features * (1 - gate)
        
        # Feed forward
        ff_out = self.ff(x)
        x = self.norm2(x + ff_out)
        
        return x


class MultiScaleDiscriminator(nn.Module):
    """
    Multi-scale discriminator for musical sequences.
    
    Analyzes music at multiple temporal scales:
    - Local (individual notes, short patterns)
    - Phrase (musical phrases, measures)
    - Global (entire piece structure)
    
    Each scale provides different discrimination signals.
    """
    
    def __init__(self,
                 d_model: int,
                 vocab_size: int = 774,
                 n_layers: int = 6,
                 attention_config: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.1,
                 use_spectral_norm: bool = True,
                 progressive_training: bool = False):
        super().__init__()
        
        self.d_model = d_model
        self.vocab_size = vocab_size
        self.progressive_training = progressive_training
        
        # Default attention config
        if attention_config is None:
            attention_config = {
                'type': 'sliding_window',
                'd_model': d_model,
                'n_heads': 8,
                'dropout': dropout,
                'window_size': 64
            }
        
        # Input embedding
        self.embedding = nn.Embedding(vocab_size, d_model)
        self.pos_embedding = nn.Parameter(torch.randn(1, 2048, d_model) * 0.02)
        
        # Musical feature extractor
        self.feature_extractor = MusicalFeatureExtractor(d_model, vocab_size)
        
        # Multi-scale processing paths
        self.local_discriminator = self._build_scale_discriminator(
            n_layers // 3, attention_config, dropout, use_spectral_norm, "local"
        )
        
        self.phrase_discriminator = self._build_scale_discriminator(
            n_layers // 3, attention_config, dropout, use_spectral_norm, "phrase"
        )
        
        self.global_discriminator = self._build_scale_discriminator(
            n_layers - 2 * (n_layers // 3), attention_config, dropout, use_spectral_norm, "global"
        )
        
        # Scale-specific pooling
        self.local_pool = nn.AdaptiveAvgPool1d(1)
        self.phrase_pool = nn.AdaptiveAvgPool1d(8)
        self.global_pool = nn.AdaptiveAvgPool1d(1)
        
        # Classification heads
        self.local_head = self._build_classification_head(d_model, use_spectral_norm, "local")
        self.phrase_head = self._build_classification_head(d_model * 8, use_spectral_norm, "phrase")
        self.global_head = self._build_classification_head(d_model, use_spectral_norm, "global")
        
        # Progressive training stages
        if progressive_training:
            self.register_buffer('training_stage', torch.tensor(0))  # 0=local, 1=phrase, 2=global
        
        logger.info(f"Initialized MultiScaleDiscriminator with {n_layers} layers")
    
    def _build_scale_discriminator(self,
                                 n_layers: int,
                                 attention_config: Dict[str, Any],
                                 dropout: float,
                                 use_spectral_norm: bool,
                                 scale_name: str) -> nn.ModuleList:
        """Build discriminator blocks for a specific scale."""
        # Use standard attention for discriminator to avoid mask issues
        scale_config = attention_config.copy()
        scale_config['type'] = 'standard'  # Force standard attention
        scale_config['n_heads'] = min(scale_config.get('n_heads', 8), 8)  # Limit heads
        
        blocks = nn.ModuleList([
            DiscriminatorBlock(self.d_model, scale_config, dropout, use_spectral_norm)
            for _ in range(n_layers)
        ])
        
        return blocks
    
    def _build_classification_head(self,
                                 input_dim: int,
                                 use_spectral_norm: bool,
                                 scale_name: str) -> nn.Module:
        """Build classification head for a specific scale."""
        head = nn.Sequential(
            nn.Linear(input_dim, input_dim // 2),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 2, input_dim // 4),
            nn.LeakyReLU(0.2),
            nn.Dropout(0.3),
            nn.Linear(input_dim // 4, 1)
        )
        
        if use_spectral_norm:
            head[0] = spectral_norm(head[0])
            head[3] = spectral_norm(head[3])
            head[6] = spectral_norm(head[6])
        
        return head
    
    def forward(self,
                tokens: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Multi-scale discrimination of musical sequences.
        
        Args:
            tokens: Token sequence [batch_size, seq_len]
            mask: Optional attention mask
            
        Returns:
            Dictionary containing:
            - local_logits: Local discrimination [batch_size, 1]
            - phrase_logits: Phrase discrimination [batch_size, 1] 
            - global_logits: Global discrimination [batch_size, 1]
            - combined_logits: Combined discrimination [batch_size, 1]
            - features: Intermediate features for feature matching
        """
        batch_size, seq_len = tokens.shape
        
        # Input embeddings
        x = self.embedding(tokens)
        
        # Add positional embeddings
        if seq_len <= self.pos_embedding.size(1):
            x = x + self.pos_embedding[:, :seq_len, :]
        else:
            # Handle longer sequences
            pos_emb = self.pos_embedding.repeat(1, (seq_len // 2048) + 1, 1)
            x = x + pos_emb[:, :seq_len, :]
        
        # Extract musical features
        musical_features = self.feature_extractor(tokens)
        
        # Store intermediate features for feature matching
        features = {'input': x.clone()}
        
        # Multi-scale processing (disable masks for simplicity in discriminator)
        if not self.progressive_training or self.training_stage >= 0:
            # Local scale (fine-grained patterns)
            local_x = x
            for i, block in enumerate(self.local_discriminator):
                local_x = block(local_x, musical_features, None)  # No mask
                features[f'local_{i}'] = local_x.clone()
            
            # Pool and classify
            local_pooled = self.local_pool(local_x.transpose(1, 2)).squeeze(-1)
            local_logits = self.local_head(local_pooled)
        else:
            local_logits = torch.zeros(batch_size, 1, device=tokens.device)
        
        if not self.progressive_training or self.training_stage >= 1:
            # Phrase scale (medium-term patterns)
            phrase_x = x
            for i, block in enumerate(self.phrase_discriminator):
                phrase_x = block(phrase_x, musical_features, None)  # No mask
                features[f'phrase_{i}'] = phrase_x.clone()
            
            # Pool and classify
            phrase_pooled = self.phrase_pool(phrase_x.transpose(1, 2))
            phrase_pooled = phrase_pooled.reshape(batch_size, -1)
            phrase_logits = self.phrase_head(phrase_pooled)
        else:
            phrase_logits = torch.zeros(batch_size, 1, device=tokens.device)
        
        if not self.progressive_training or self.training_stage >= 2:
            # Global scale (long-term structure)
            global_x = x
            for i, block in enumerate(self.global_discriminator):
                global_x = block(global_x, musical_features, None)  # No mask
                features[f'global_{i}'] = global_x.clone()
            
            # Pool and classify
            global_pooled = self.global_pool(global_x.transpose(1, 2)).squeeze(-1)
            global_logits = self.global_head(global_pooled)
        else:
            global_logits = torch.zeros(batch_size, 1, device=tokens.device)
        
        # Combine predictions (weighted average)
        if self.progressive_training:
            if self.training_stage == 0:
                combined_logits = local_logits
            elif self.training_stage == 1:
                combined_logits = 0.6 * local_logits + 0.4 * phrase_logits
            else:
                combined_logits = 0.4 * local_logits + 0.4 * phrase_logits + 0.2 * global_logits
        else:
            combined_logits = 0.4 * local_logits + 0.4 * phrase_logits + 0.2 * global_logits
        
        return {
            'local_logits': local_logits,
            'phrase_logits': phrase_logits,
            'global_logits': global_logits,
            'combined_logits': combined_logits,
            'features': features
        }
    
    def advance_training_stage(self):
        """Advance to next progressive training stage."""
        if self.progressive_training and hasattr(self, 'training_stage'):
            self.training_stage = min(self.training_stage + 1, 2)
            logger.info(f"Advanced discriminator training to stage {self.training_stage}")
    
    def get_feature_maps(self, layer_name: str) -> torch.Tensor:
        """Get intermediate feature maps for analysis."""
        if hasattr(self, '_last_features') and layer_name in self._last_features:
            return self._last_features[layer_name]
        return None


# For backward compatibility with existing code
MusicDiscriminator = MultiScaleDiscriminator