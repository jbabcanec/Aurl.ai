"""
Enhanced VAE encoder with β-VAE support and musical enhancements.

This module implements an improved encoder for the Aurl.ai VAE architecture
with disentanglement control and hierarchical latent variables.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, Tuple, List
import math

from src.models.components import TransformerBlock
from src.models.attention import create_attention_layer
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class EnhancedMusicEncoder(nn.Module):
    """
    Enhanced encoder for VAE with β-VAE support and hierarchical latents.
    
    Features:
    - β-VAE for disentanglement control
    - Hierarchical latent variables (global, local, fine)
    - Posterior collapse prevention
    - Structured latent space for interpretability
    """
    
    def __init__(self,
                 d_model: int,
                 latent_dim: int,
                 n_layers: int = 6,
                 attention_config: Optional[Dict[str, Any]] = None,
                 dropout: float = 0.1,
                 beta: float = 1.0,
                 hierarchical: bool = True,
                 use_batch_norm: bool = True,
                 free_bits: float = 0.0):
        """
        Initialize enhanced encoder.
        
        Args:
            d_model: Model dimension
            latent_dim: Total latent dimension
            n_layers: Number of transformer layers
            attention_config: Attention configuration
            dropout: Dropout rate
            beta: β-VAE parameter (1.0 = standard VAE)
            hierarchical: Use hierarchical latent structure
            use_batch_norm: Apply batch norm to latents
            free_bits: Minimum KL per dimension
        """
        super().__init__()
        
        self.d_model = d_model
        self.latent_dim = latent_dim
        self.beta = beta
        self.hierarchical = hierarchical
        self.use_batch_norm = use_batch_norm
        self.free_bits = free_bits
        
        # Validate latent dimension for hierarchical mode
        if hierarchical and latent_dim % 3 != 0:
            raise ValueError(f"For hierarchical mode, latent_dim must be divisible by 3, got {latent_dim}")
        
        # Default attention config
        if attention_config is None:
            attention_config = {
                'type': 'hierarchical',
                'd_model': d_model,
                'n_heads': 8,
                'dropout': dropout
            }
        
        # Encoder transformer blocks
        self.blocks = nn.ModuleList([
            TransformerBlock(d_model, attention_config, dropout=dropout)
            for _ in range(n_layers)
        ])
        
        self.norm = nn.LayerNorm(d_model)
        
        # Pooling strategies for different levels
        if hierarchical:
            # Different pooling for different hierarchy levels
            self.global_pool = nn.AdaptiveAvgPool1d(1)  # Full sequence
            self.local_pool = nn.AdaptiveMaxPool1d(8)   # 8 local regions
            self.fine_pool = nn.AdaptiveMaxPool1d(32)   # 32 fine regions
            
            # Hierarchical latent projections
            latent_per_level = latent_dim // 3
            
            # Global latent (piece-level)
            self.global_mu = nn.Linear(d_model, latent_per_level)
            self.global_logvar = nn.Linear(d_model, latent_per_level)
            
            # Local latent (phrase-level)
            self.local_mu = nn.Linear(d_model * 8, latent_per_level)
            self.local_logvar = nn.Linear(d_model * 8, latent_per_level)
            
            # Fine latent (measure/bar-level)
            self.fine_mu = nn.Linear(d_model * 32, latent_per_level)
            self.fine_logvar = nn.Linear(d_model * 32, latent_per_level)
            
        else:
            # Standard single-level latent
            self.pooling = nn.AdaptiveAvgPool1d(1)
            self.mu_projection = nn.Linear(d_model, latent_dim)
            self.logvar_projection = nn.Linear(d_model, latent_dim)
        
        # Batch normalization for latent regularization
        if use_batch_norm:
            self.latent_bn_mu = nn.BatchNorm1d(latent_dim)
            self.latent_bn_logvar = nn.BatchNorm1d(latent_dim)
        
        # Initialize projections
        self._init_latent_projections()
        
        logger.info(f"Initialized EnhancedMusicEncoder with beta={beta}, "
                   f"hierarchical={hierarchical}, latent_dim={latent_dim}")
    
    def _init_latent_projections(self):
        """Initialize latent projections with small values for stability."""
        # Initialize mu projections with small random values
        for name, param in self.named_parameters():
            if 'mu' in name and 'weight' in name and param.dim() >= 2:
                nn.init.xavier_normal_(param, gain=0.1)
            elif 'mu' in name and 'bias' in name:
                nn.init.zeros_(param)
            # Initialize logvar projections to output small variances initially
            elif 'logvar' in name and 'weight' in name and param.dim() >= 2:
                nn.init.xavier_normal_(param, gain=0.01)
            elif 'logvar' in name and 'bias' in name:
                nn.init.constant_(param, -2.0)  # Start with small variance
    
    def forward(self, 
                x: torch.Tensor, 
                mask: Optional[torch.Tensor] = None
                ) -> Dict[str, torch.Tensor]:
        """
        Encode musical sequence to latent space.
        
        Args:
            x: Input embeddings [batch_size, seq_len, d_model]
            mask: Optional attention mask
            
        Returns:
            Dictionary containing:
            - mu: Mean of latent distribution
            - logvar: Log variance of latent distribution
            - z: Sampled latent (if training)
            - kl_loss: KL divergence loss
            - latent_info: Additional information for analysis
        """
        batch_size, seq_len, _ = x.shape
        
        # Pass through encoder transformer blocks
        for block in self.blocks:
            x = block(x, mask)
        
        x = self.norm(x)
        
        if self.hierarchical:
            return self._hierarchical_encode(x, batch_size)
        else:
            return self._standard_encode(x, batch_size)
    
    def _standard_encode(self, x: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """Standard VAE encoding."""
        # Global pooling to get sequence-level representation
        x_pooled = self.pooling(x.transpose(1, 2)).squeeze(-1)
        
        # Project to latent space parameters
        mu = self.mu_projection(x_pooled)
        logvar = self.logvar_projection(x_pooled)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm and batch_size > 1:
            mu = self.latent_bn_mu(mu)
            logvar = self.latent_bn_logvar(logvar)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Compute KL loss with free bits
        kl_loss = self._compute_kl_loss(mu, logvar)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'kl_loss': kl_loss,
            'latent_info': {
                'active_dims': (kl_loss > 0.1).float().mean().item(),
                'mean_kl': kl_loss.mean().item()
            }
        }
    
    def _hierarchical_encode(self, x: torch.Tensor, batch_size: int) -> Dict[str, torch.Tensor]:
        """Hierarchical VAE encoding with multiple latent levels."""
        # Get representations at different scales
        x_transposed = x.transpose(1, 2)  # [batch, d_model, seq_len]
        
        # Global representation (entire sequence)
        global_repr = self.global_pool(x_transposed).squeeze(-1)  # [batch, d_model]
        
        # Local representations (8 regions)
        local_repr = self.local_pool(x_transposed)  # [batch, d_model, 8]
        local_repr = local_repr.reshape(batch_size, -1)  # [batch, d_model * 8]
        
        # Fine representations (32 regions)  
        fine_repr = self.fine_pool(x_transposed)  # [batch, d_model, 32]
        fine_repr = fine_repr.reshape(batch_size, -1)  # [batch, d_model * 32]
        
        # Project each level to latent parameters
        global_mu = self.global_mu(global_repr)
        global_logvar = self.global_logvar(global_repr)
        
        local_mu = self.local_mu(local_repr)
        local_logvar = self.local_logvar(local_repr)
        
        fine_mu = self.fine_mu(fine_repr)
        fine_logvar = self.fine_logvar(fine_repr)
        
        # Concatenate all levels
        mu = torch.cat([global_mu, local_mu, fine_mu], dim=1)
        logvar = torch.cat([global_logvar, local_logvar, fine_logvar], dim=1)
        
        # Apply batch normalization if enabled
        if self.use_batch_norm and batch_size > 1:
            mu = self.latent_bn_mu(mu)
            logvar = self.latent_bn_logvar(logvar)
        
        # Sample latent
        z = self.reparameterize(mu, logvar)
        
        # Compute KL loss for each level
        global_kl = self._compute_kl_loss(global_mu, global_logvar)
        local_kl = self._compute_kl_loss(local_mu, local_logvar)
        fine_kl = self._compute_kl_loss(fine_mu, fine_logvar)
        
        kl_loss = torch.cat([global_kl, local_kl, fine_kl], dim=1)
        
        return {
            'mu': mu,
            'logvar': logvar,
            'z': z,
            'kl_loss': kl_loss,
            'latent_info': {
                'global_mu': global_mu,
                'local_mu': local_mu,
                'fine_mu': fine_mu,
                'active_dims': (kl_loss > 0.1).float().mean().item(),
                'mean_kl': kl_loss.mean().item(),
                'global_kl': global_kl.mean().item(),
                'local_kl': local_kl.mean().item(),
                'fine_kl': fine_kl.mean().item()
            }
        }
    
    def reparameterize(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """Reparameterization trick for VAE."""
        if self.training:
            std = torch.exp(0.5 * logvar)
            eps = torch.randn_like(std)
            return mu + eps * std
        else:
            return mu
    
    def _compute_kl_loss(self, mu: torch.Tensor, logvar: torch.Tensor) -> torch.Tensor:
        """
        Compute KL divergence with free bits.
        
        Returns per-dimension KL (not summed).
        """
        # KL divergence per dimension
        kl = -0.5 * (1 + logvar - mu.pow(2) - logvar.exp())
        
        # Apply free bits if specified
        if self.free_bits > 0:
            kl = F.relu(kl - self.free_bits) + self.free_bits
        
        return kl
    
    def get_latent_statistics(self) -> Dict[str, float]:
        """Get statistics about latent space usage."""
        stats = {
            'beta': self.beta,
            'hierarchical': self.hierarchical,
            'latent_dim': self.latent_dim,
            'free_bits': self.free_bits
        }
        
        if self.hierarchical:
            stats['levels'] = 3
            stats['dims_per_level'] = self.latent_dim // 3
        
        return stats


class LatentRegularizer(nn.Module):
    """
    Additional regularization for latent space.
    
    Implements various regularization techniques:
    - Mutual information penalty
    - Orthogonality constraints
    - Sparsity encouragement
    """
    
    def __init__(self,
                 latent_dim: int,
                 mi_penalty: float = 0.1,
                 ortho_penalty: float = 0.01,
                 sparsity_penalty: float = 0.01):
        super().__init__()
        
        self.latent_dim = latent_dim
        self.mi_penalty = mi_penalty
        self.ortho_penalty = ortho_penalty
        self.sparsity_penalty = sparsity_penalty
    
    def forward(self, z: torch.Tensor) -> Dict[str, torch.Tensor]:
        """
        Compute regularization losses.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            
        Returns:
            Dictionary of regularization losses
        """
        losses = {}
        
        # Mutual information penalty (encourage independence)
        if self.mi_penalty > 0:
            # Compute correlation matrix
            z_norm = (z - z.mean(dim=0, keepdim=True)) / (z.std(dim=0, keepdim=True) + 1e-8)
            corr = torch.matmul(z_norm.T, z_norm) / z.shape[0]
            
            # Penalty for off-diagonal elements
            latent_dim = z.shape[1]
            off_diag = corr - torch.eye(latent_dim, device=z.device)
            losses['mi_loss'] = self.mi_penalty * (off_diag ** 2).sum()
        
        # Orthogonality penalty (for structured latents)
        if self.ortho_penalty > 0 and z.shape[1] >= 32:  # Use actual latent dim
            # Reshape to groups (e.g., rhythm, pitch, harmony, dynamics)
            if z.shape[1] % 4 == 0:
                z_groups = z.view(z.shape[0], 4, -1)
                
                # Compute group means across batch
                group_means = z_groups.mean(dim=0).mean(dim=1)  # [4] - average activation per group
                
                # Compute correlation between groups across dimensions
                z_groups_flat = z_groups.view(z.shape[0], 4, -1).permute(0, 2, 1)  # [batch, dims_per_group, 4]
                z_groups_flat = z_groups_flat.reshape(-1, 4)  # [batch * dims_per_group, 4]
                
                # Correlation matrix between the 4 groups
                z_centered = z_groups_flat - z_groups_flat.mean(dim=0, keepdim=True)
                cov = torch.matmul(z_centered.T, z_centered) / z_centered.shape[0]
                std = torch.sqrt(torch.diag(cov))
                corr = cov / (std.unsqueeze(0) * std.unsqueeze(1) + 1e-8)
                
                # Penalty for high correlation between groups
                identity = torch.eye(4, device=z.device)
                ortho_penalty = (corr - identity) ** 2
                losses['ortho_loss'] = self.ortho_penalty * ortho_penalty.sum()
            else:
                # Skip orthogonality if dimensions don't divide evenly
                losses['ortho_loss'] = torch.tensor(0.0, device=z.device)
        
        # Sparsity penalty (encourage sparse activation)
        if self.sparsity_penalty > 0:
            losses['sparsity_loss'] = self.sparsity_penalty * torch.abs(z).mean()
        
        return losses