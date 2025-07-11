"""
MusicTransformerVAEGAN: Configurable music generation model for Aurl.ai.

This module implements a highly configurable architecture that can operate in multiple modes:
- transformer: Pure transformer for autoregressive generation
- vae: Variational autoencoder for latent space music generation  
- vae_gan: Full VAE-GAN with adversarial training for enhanced realism

Designed to handle the long sequences (9,433+ tokens) discovered in our data analysis.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, Union
import math
import logging

from src.models.components import (
    MusicEmbedding, TransformerBlock, OutputHead, 
    FeedForward, create_baseline_model
)
from src.models.attention import create_attention_layer
from src.models.discriminator import MultiScaleDiscriminator
from src.models.encoder import EnhancedMusicEncoder
from src.models.decoder import EnhancedMusicDecoder
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class MusicEncoder(nn.Module):
    """
    Encoder component for VAE mode.
    
    Encodes musical sequences into a latent representation.
    """
    
    def __init__(self,
                 d_model: int,
                 latent_dim: int,
                 n_layers: int = 6,
                 attention_config: Dict[str, Any] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        
        # Default attention config
        if attention_config is None:
            attention_config = {
                'type': 'hierarchical',
                'd_model': d_model,
                'n_heads': 8,
                'dropout': dropout
            }
        
        # Encoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, attention_config, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Latent space projections
        self.mu_projection = nn.Linear(d_model, latent_dim)
        self.logvar_projection = nn.Linear(d_model, latent_dim)
        
        # Global pooling for sequence-level latent
        self.pooling = nn.AdaptiveAvgPool1d(1)
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> Tuple[torch.Tensor, torch.Tensor]:
        """
        Encode musical sequence to latent space.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            mu: Mean of latent distribution [batch_size, latent_dim]
            logvar: Log variance of latent distribution [batch_size, latent_dim]
        """
        # Pass through encoder blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        # Global pooling to get sequence-level representation
        # [batch_size, seq_len, d_model] -> [batch_size, d_model]
        x_pooled = self.pooling(x.transpose(1, 2)).squeeze(-1)
        
        # Project to latent space parameters
        mu = self.mu_projection(x_pooled)
        logvar = self.logvar_projection(x_pooled)
        
        return mu, logvar


class MusicDecoder(nn.Module):
    """
    Decoder component for VAE mode.
    
    Decodes latent representations back to musical sequences.
    """
    
    def __init__(self,
                 d_model: int,
                 latent_dim: int,
                 vocab_size: int,
                 n_layers: int = 6,
                 attention_config: Dict[str, Any] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.vocab_size = vocab_size
        
        # Default attention config
        if attention_config is None:
            attention_config = {
                'type': 'hierarchical',
                'd_model': d_model,
                'n_heads': 8,
                'dropout': dropout
            }
        
        # Latent to sequence projection
        self.latent_projection = nn.Linear(latent_dim, d_model)
        
        # Decoder blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, attention_config, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        self.output_head = OutputHead(d_model, vocab_size, dropout)
        
    def forward(self, 
                latent: torch.Tensor, 
                target_embeddings: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Decode latent representation to musical sequence.
        
        Args:
            latent: Latent vector [batch_size, latent_dim]
            target_embeddings: Target sequence embeddings [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Logits [batch_size, seq_len, vocab_size]
        """
        batch_size, seq_len, d_model = target_embeddings.shape
        
        # Project latent to model dimension
        latent_proj = self.latent_projection(latent)  # [batch_size, d_model]
        
        # Broadcast latent across sequence length and add to target embeddings
        latent_expanded = latent_proj.unsqueeze(1).expand(-1, seq_len, -1)
        x = target_embeddings + latent_expanded
        
        # Pass through decoder blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        # Generate output logits
        logits = self.output_head(x)
        
        return logits


class MusicDiscriminator(nn.Module):
    """
    Discriminator component for GAN mode.
    
    Distinguishes between real and generated musical sequences.
    """
    
    def __init__(self,
                 d_model: int,
                 n_layers: int = 4,
                 attention_config: Dict[str, Any] = None,
                 dropout: float = 0.1):
        super().__init__()
        
        self.d_model = d_model
        
        # Default attention config (simpler for discriminator)
        if attention_config is None:
            attention_config = {
                'type': 'sliding_window',
                'd_model': d_model,
                'n_heads': 4,
                'window_size': 256,
                'dropout': dropout
            }
        
        # Discriminator blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, attention_config, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Classification head
        self.pooling = nn.AdaptiveAvgPool1d(1)
        self.classifier = nn.Sequential(
            nn.Linear(d_model, d_model // 2),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(d_model // 2, 1)
        )
        
    def forward(self, x: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Classify sequence as real or fake.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Classification logits [batch_size, 1]
        """
        # Pass through discriminator blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        # Global pooling and classification
        x_pooled = self.pooling(x.transpose(1, 2)).squeeze(-1)
        logits = self.classifier(x_pooled)
        
        return logits


class MusicTransformerVAEGAN(nn.Module):
    """
    Configurable music generation model that can operate in multiple modes.
    
    Modes:
    - transformer: Pure autoregressive transformer
    - vae: Variational autoencoder for latent-based generation
    - vae_gan: Full VAE-GAN with adversarial training
    
    Designed for the Aurl.ai project to handle long musical sequences efficiently.
    """
    
    def __init__(self,
                 vocab_size: int = 774,
                 d_model: int = 512,
                 n_layers: int = 8,
                 n_heads: int = 8,
                 d_ff: Optional[int] = None,
                 max_sequence_length: int = 2048,
                 dropout: float = 0.1,
                 mode: str = "transformer",
                 
                 # VAE-specific parameters
                 latent_dim: int = 128,
                 encoder_layers: int = 6,
                 decoder_layers: int = 6,
                 beta: float = 1.0,
                 
                 # GAN-specific parameters  
                 discriminator_layers: int = 4,
                 
                 # Attention configuration
                 attention_type: str = "hierarchical",
                 local_window_size: int = 256,
                 global_window_size: int = 64,
                 sliding_window_size: int = 512):
        
        super().__init__()
        
        # Validate mode
        if mode not in ["transformer", "vae", "vae_gan"]:
            raise ValueError(f"Invalid mode: {mode}. Must be one of ['transformer', 'vae', 'vae_gan']")
        
        self.mode = mode
        self.vocab_size = vocab_size
        self.d_model = d_model
        self.n_layers = n_layers
        self.max_sequence_length = max_sequence_length
        self.latent_dim = latent_dim
        self.beta = beta
        
        # Shared components
        self.embedding = MusicEmbedding(
            vocab_size, d_model, dropout, max_sequence_length
        )
        
        # Attention configuration
        attention_config = {
            'type': attention_type,
            'd_model': d_model,
            'n_heads': n_heads,
            'dropout': dropout,
            'local_window_size': local_window_size,
            'global_window_size': global_window_size,
            'window_size': sliding_window_size
        }
        
        if mode == "transformer":
            # Pure transformer mode
            self.blocks = nn.ModuleList([
                TransformerBlock(d_model, attention_config, d_ff, dropout)
                for _ in range(n_layers)
            ])
            self.norm = nn.LayerNorm(d_model)
            self.output_head = OutputHead(d_model, vocab_size, dropout)
            
        elif mode in ["vae", "vae_gan"]:
            # Enhanced VAE components
            self.encoder = EnhancedMusicEncoder(
                d_model=d_model,
                latent_dim=latent_dim,
                n_layers=encoder_layers,
                attention_config=attention_config,
                dropout=dropout,
                beta=1.0,
                hierarchical=True,
                free_bits=0.1
            )
            self.decoder = EnhancedMusicDecoder(
                d_model=d_model,
                latent_dim=latent_dim,
                vocab_size=vocab_size,
                n_layers=decoder_layers,
                attention_config=attention_config,
                dropout=dropout,
                hierarchical=True,
                use_skip_connection=True
            )
            
            if mode == "vae_gan":
                # GAN component
                discriminator_attention_config = attention_config.copy()
                discriminator_attention_config['type'] = 'sliding_window'
                discriminator_attention_config['n_heads'] = 4
                
                self.discriminator = MultiScaleDiscriminator(
                    d_model=d_model,
                    vocab_size=vocab_size,
                    n_layers=discriminator_layers,
                    attention_config=discriminator_attention_config,
                    dropout=dropout,
                    use_spectral_norm=True,
                    progressive_training=False
                )
        
        self.apply(self._init_weights)
        
        logger.info(f"Initialized MusicTransformerVAEGAN in {mode} mode")
        logger.info(f"Model size: {self.get_model_size()}")
    
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
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Reparameterization trick for VAE.
        
        Args:
            mu: Mean of latent distribution
            logvar: Log variance of latent distribution
            
        Returns:
            Sampled latent vector
        """
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std
    
    def forward(self, 
                tokens: torch.Tensor, 
                mask: Optional[torch.Tensor] = None,
                temperature: float = 1.0,
                return_latent: bool = False) -> Union[torch.Tensor, Dict[str, torch.Tensor]]:
        """
        Forward pass - behavior depends on model mode.
        
        Args:
            tokens: Input token indices [batch_size, seq_len]
            mask: Optional attention mask
            temperature: Temperature for output scaling
            return_latent: Whether to return latent variables (VAE modes only)
            
        Returns:
            In transformer mode: logits [batch_size, seq_len, vocab_size]
            In VAE modes: dict with 'logits', 'mu', 'logvar', 'z'
        """
        embeddings = self.embedding(tokens)
        
        if self.mode == "transformer":
            # Pure transformer forward pass
            x = embeddings
            for block in self.blocks:
                x = block(x, mask)
            x = self.norm(x)
            logits = self.output_head(x, temperature)
            return logits
        
        elif self.mode in ["vae", "vae_gan"]:
            # VAE forward pass
            # Encode
            encoder_output = self.encoder(embeddings, mask)
            mu = encoder_output['mu']
            logvar = encoder_output['logvar']
            z = encoder_output['z']
            kl_loss = encoder_output['kl_loss']
            
            # Decode
            logits = self.decoder(z, embeddings, mask)
            
            if return_latent:
                return {
                    'reconstruction': logits,  # Tests expect 'reconstruction' key
                    'logits': logits,
                    'mu': mu,
                    'logvar': logvar,
                    'z': z,
                    'kl_loss': kl_loss
                }
            else:
                return logits
    
    def discriminate(self, embeddings: torch.Tensor, mask: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Discriminator forward pass (GAN mode only).
        
        Args:
            embeddings: Input embeddings [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Classification logits [batch_size, 1]
        """
        if self.mode != "vae_gan":
            raise ValueError("Discriminator only available in vae_gan mode")
        
        return self.discriminator(embeddings, mask)
    
    def compute_loss(self, 
                     tokens: torch.Tensor, 
                     mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute loss based on model mode.
        
        Args:
            tokens: Target tokens [batch_size, seq_len]
            mask: Optional mask
            
        Returns:
            Dictionary of losses
        """
        if self.mode == "transformer":
            # Standard cross-entropy loss
            logits = self(tokens, mask)
            loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                tokens.view(-1),
                ignore_index=2  # Ignore PAD tokens
            )
            return {'total_loss': loss, 'reconstruction_loss': loss}
        
        elif self.mode in ["vae", "vae_gan"]:
            # VAE losses
            outputs = self(tokens, mask, return_latent=True)
            logits, mu, logvar, z = outputs['logits'], outputs['mu'], outputs['logvar'], outputs['z']
            
            # Reconstruction loss
            recon_loss = F.cross_entropy(
                logits.view(-1, self.vocab_size),
                tokens.view(-1),
                ignore_index=2,
                reduction='mean'
            )
            
            # KL divergence loss (already computed in encoder)
            kl_loss = outputs.get('kl_loss', -0.5 * torch.sum(1 + logvar - mu.pow(2) - logvar.exp()) / mu.size(0))
            
            # Total VAE loss
            vae_loss = recon_loss + self.beta * kl_loss
            
            losses = {
                'total_loss': vae_loss,
                'reconstruction_loss': recon_loss,
                'kl_loss': kl_loss
            }
            
            if self.mode == "vae_gan":
                # Add basic GAN losses for compatibility
                # More sophisticated GAN training happens in the loss framework
                with torch.no_grad():
                    fake_tokens = torch.multinomial(
                        F.softmax(logits.view(-1, self.vocab_size), dim=-1),
                        num_samples=1
                    ).view(tokens.shape)
                    
                embeddings = self.embedding(tokens)
                real_score = self.discriminator(embeddings, mask).mean()
                fake_embeddings = self.embedding(fake_tokens)
                fake_score = self.discriminator(fake_embeddings, mask).mean()
                
                # Simple GAN losses
                generator_loss = -fake_score  # Generator wants high fake score
                discriminator_loss = real_score - fake_score  # Discriminator wants real > fake
                
                losses.update({
                    'generator_loss': generator_loss,
                    'discriminator_loss': discriminator_loss
                })
            
            return losses
    
    def generate(self,
                 prompt_tokens: Optional[torch.Tensor] = None,
                 max_new_tokens: int = 100,
                 temperature: float = 1.0,
                 top_k: Optional[int] = None,
                 top_p: Optional[float] = None,
                 latent: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Generate music tokens.
        
        Args:
            prompt_tokens: Initial tokens [batch_size, prompt_len] (transformer mode)
            max_new_tokens: Maximum number of tokens to generate
            temperature: Sampling temperature
            top_k: Top-k filtering
            top_p: Top-p (nucleus) filtering
            latent: Latent vector for generation [batch_size, latent_dim] (VAE modes)
            
        Returns:
            Generated tokens
        """
        self.eval()
        
        if self.mode == "transformer":
            if prompt_tokens is None:
                raise ValueError("prompt_tokens required for transformer mode")
            
            # Use baseline transformer generation logic
            return self._generate_autoregressive(
                prompt_tokens, max_new_tokens, temperature, top_k, top_p
            )
        
        elif self.mode in ["vae", "vae_gan"]:
            if latent is None:
                # Sample from prior
                batch_size = 1 if prompt_tokens is None else prompt_tokens.size(0)
                latent = torch.randn(batch_size, self.latent_dim, device=self.embedding.token_embedding.weight.device)
            
            # Generate from latent
            return self._generate_from_latent(latent, max_new_tokens, temperature, top_k, top_p)
    
    def _generate_autoregressive(self,
                               prompt_tokens: torch.Tensor,
                               max_new_tokens: int,
                               temperature: float,
                               top_k: Optional[int],
                               top_p: Optional[float]) -> torch.Tensor:
        """Autoregressive generation for transformer mode."""
        generated = prompt_tokens.clone()
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                logits = self(generated, temperature=temperature)
                next_token_logits = logits[:, -1, :]
                
                # Apply filtering and sampling
                next_token_logits = self._apply_sampling_filters(
                    next_token_logits, temperature, top_k, top_p
                )
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                if generated.size(1) >= self.max_sequence_length:
                    break
        
        return generated
    
    def _generate_from_latent(self,
                            latent: torch.Tensor,
                            max_new_tokens: int,
                            temperature: float,
                            top_k: Optional[int],
                            top_p: Optional[float]) -> torch.Tensor:
        """Generation from latent vector for VAE modes."""
        batch_size = latent.size(0)
        device = latent.device
        
        # Start with START token
        generated = torch.zeros(batch_size, 1, dtype=torch.long, device=device)  # START token = 0
        
        with torch.no_grad():
            for _ in range(max_new_tokens):
                # Get embeddings for current sequence
                embeddings = self.embedding(generated)
                
                # Decode from latent
                logits = self.decoder(latent, embeddings)
                next_token_logits = logits[:, -1, :]
                
                # Apply filtering and sampling
                next_token_logits = self._apply_sampling_filters(
                    next_token_logits, temperature, top_k, top_p
                )
                
                probs = F.softmax(next_token_logits, dim=-1)
                next_token = torch.multinomial(probs, num_samples=1)
                
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop on END token or max length
                if next_token.item() == 1 or generated.size(1) >= self.max_sequence_length:  # END token = 1
                    break
        
        return generated
    
    def _apply_sampling_filters(self,
                              logits: torch.Tensor,
                              temperature: float,
                              top_k: Optional[int],
                              top_p: Optional[float]) -> torch.Tensor:
        """Apply temperature, top-k, and top-p filtering to logits."""
        logits = logits / temperature
        
        if top_k is not None:
            indices_to_remove = logits < torch.topk(logits, top_k)[0][..., -1, None]
            logits[indices_to_remove] = float('-inf')
        
        if top_p is not None:
            sorted_logits, sorted_indices = torch.sort(logits, descending=True)
            cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
            
            sorted_indices_to_remove = cumulative_probs > top_p
            sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
            sorted_indices_to_remove[..., 0] = False
            
            indices_to_remove = sorted_indices[sorted_indices_to_remove]
            logits.scatter_(1, indices_to_remove, float('-inf'))
        
        return logits
    
    def get_model_size(self) -> Dict[str, Any]:
        """Get model size statistics."""
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)
        
        return {
            'total_parameters': total_params,
            'trainable_parameters': trainable_params,
            'model_size_mb': total_params * 4 / (1024 * 1024),
            'mode': self.mode,
            'layers': self.n_layers,
            'embedding_size': self.d_model,
            'vocab_size': self.vocab_size,
            'max_sequence_length': self.max_sequence_length
        }


def create_model_from_config(config: Dict[str, Any]) -> MusicTransformerVAEGAN:
    """
    Factory function to create model from configuration.
    
    Args:
        config: Model configuration dictionary
        
    Returns:
        Configured MusicTransformerVAEGAN
    """
    return MusicTransformerVAEGAN(**config)