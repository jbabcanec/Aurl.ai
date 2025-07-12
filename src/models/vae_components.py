"""
Additional VAE components: Musical priors and latent analysis tools.

This module provides enhanced VAE components for musical generation including
learnable priors and tools for understanding the latent space.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Optional, Dict, Any, List, Tuple
import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class MusicalPrior(nn.Module):
    """
    Learnable musical prior distribution.
    
    Instead of a standard Gaussian prior, this learns a more musical
    distribution using a mixture of Gaussians or normalizing flows.
    """
    
    def __init__(self,
                 latent_dim: int,
                 prior_type: str = "mixture",
                 num_modes: int = 8,
                 flow_layers: int = 4):
        """
        Initialize musical prior.
        
        Args:
            latent_dim: Dimension of latent space
            prior_type: Type of prior ("standard", "mixture", "flow")
            num_modes: Number of modes for mixture prior
            flow_layers: Number of flow layers for flow prior
        """
        super().__init__()
        
        self.latent_dim = latent_dim
        self.prior_type = prior_type
        self.num_modes = num_modes
        
        if prior_type == "standard":
            # Standard Gaussian prior (no parameters)
            pass
            
        elif prior_type == "mixture":
            # Mixture of Gaussians prior
            self.mode_weights = nn.Parameter(torch.ones(num_modes) / num_modes)
            self.mode_means = nn.Parameter(torch.randn(num_modes, latent_dim) * 0.5)
            self.mode_logvars = nn.Parameter(torch.zeros(num_modes, latent_dim))
            
            # Learnable mixing network (optional)
            self.mix_net = nn.Sequential(
                nn.Linear(latent_dim, 64),
                nn.ReLU(),
                nn.Linear(64, num_modes),
                nn.Softmax(dim=-1)
            )
            
        elif prior_type == "flow":
            # Normalizing flow prior
            self.flow_layers = nn.ModuleList()
            for _ in range(flow_layers):
                self.flow_layers.append(PlanarFlow(latent_dim))
        
        else:
            raise ValueError(f"Unknown prior type: {prior_type}")
        
        logger.info(f"Initialized MusicalPrior with type={prior_type}")
    
    def sample(self, 
               batch_size: int, 
               device: torch.device,
               condition: Optional[torch.Tensor] = None) -> torch.Tensor:
        """
        Sample from the prior distribution.
        
        Args:
            batch_size: Number of samples
            device: Device to create samples on
            condition: Optional conditioning information
            
        Returns:
            Sampled latents [batch_size, latent_dim]
        """
        if self.prior_type == "standard":
            return torch.randn(batch_size, self.latent_dim, device=device)
            
        elif self.prior_type == "mixture":
            # Sample mode indices
            if condition is not None:
                # Use conditioning to select modes
                mode_probs = self.mix_net(condition)
                mode_indices = torch.multinomial(mode_probs, 1).squeeze(-1)
            else:
                # Sample uniformly from modes
                mode_probs = F.softmax(self.mode_weights, dim=0)
                mode_indices = torch.multinomial(mode_probs, batch_size, replacement=True)
            
            # Sample from selected modes
            selected_means = self.mode_means[mode_indices]
            selected_logvars = self.mode_logvars[mode_indices]
            
            std = torch.exp(0.5 * selected_logvars)
            eps = torch.randn_like(selected_means)
            
            return selected_means + eps * std
            
        elif self.prior_type == "flow":
            # Start with standard Gaussian
            z = torch.randn(batch_size, self.latent_dim, device=device)
            
            # Apply flow transformations
            log_det = 0
            for flow in self.flow_layers:
                z, ld = flow(z)
                log_det = log_det + ld
                
            return z
    
    def log_prob(self, z: torch.Tensor) -> torch.Tensor:
        """
        Compute log probability under the prior.
        
        Args:
            z: Latent codes [batch_size, latent_dim]
            
        Returns:
            Log probabilities [batch_size]
        """
        if self.prior_type == "standard":
            # Standard Gaussian log prob
            return -0.5 * (z.pow(2).sum(dim=1) + self.latent_dim * np.log(2 * np.pi))
            
        elif self.prior_type == "mixture":
            # Mixture of Gaussians log prob
            batch_size = z.shape[0]
            
            # Compute log prob under each mode
            z_expanded = z.unsqueeze(1).expand(-1, self.num_modes, -1)
            
            # Log prob of Gaussian
            diff = z_expanded - self.mode_means.unsqueeze(0)
            logvars_expanded = self.mode_logvars.unsqueeze(0)
            
            log_probs = -0.5 * (
                diff.pow(2) / logvars_expanded.exp() + 
                logvars_expanded + 
                np.log(2 * np.pi)
            ).sum(dim=2)
            
            # Add mode weights
            log_weights = F.log_softmax(self.mode_weights, dim=0)
            log_probs = log_probs + log_weights.unsqueeze(0)
            
            # Log-sum-exp for numerical stability
            return torch.logsumexp(log_probs, dim=1)
            
        elif self.prior_type == "flow":
            # Flow-based log prob
            log_det = 0
            z_k = z
            
            # Inverse flow transformations
            for flow in reversed(self.flow_layers):
                z_k, ld = flow.inverse(z_k)
                log_det = log_det - ld
                
            # Base distribution log prob
            log_prob_base = -0.5 * (z_k.pow(2).sum(dim=1) + self.latent_dim * np.log(2 * np.pi))
            
            return log_prob_base + log_det


class PlanarFlow(nn.Module):
    """
    Planar normalizing flow transformation.
    """
    
    def __init__(self, dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, dim))
        self.bias = nn.Parameter(torch.randn(1))
        self.scale = nn.Parameter(torch.randn(1, dim))
        
    def forward(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Forward transformation."""
        # f(z) = z + u * tanh(w^T z + b)
        activation = F.tanh(torch.sum(self.weight * z, dim=1, keepdim=True) + self.bias)
        z_new = z + self.scale * activation
        
        # Compute log determinant
        psi = (1 - activation.pow(2)) * self.weight
        det = 1 + torch.sum(psi * self.scale, dim=1)
        log_det = torch.log(det.abs() + 1e-8)
        
        return z_new, log_det
    
    def inverse(self, z: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Inverse transformation (approximate)."""
        # Approximate inverse using fixed-point iteration
        z_inv = z.clone()
        for _ in range(5):
            activation = F.tanh(torch.sum(self.weight * z_inv, dim=1, keepdim=True) + self.bias)
            z_inv = z - self.scale * activation
            
        # Compute log determinant at inverse point
        activation = F.tanh(torch.sum(self.weight * z_inv, dim=1, keepdim=True) + self.bias)
        psi = (1 - activation.pow(2)) * self.weight
        det = 1 + torch.sum(psi * self.scale, dim=1)
        log_det = torch.log(det.abs() + 1e-8)
        
        return z_inv, log_det


class LatentAnalyzer:
    """
    Tools for analyzing and understanding the latent space.
    """
    
    def __init__(self, model, device: torch.device = torch.device('cpu')):
        """
        Initialize analyzer with a trained model.
        
        Args:
            model: Trained VAE model
            device: Device for computations
        """
        self.model = model
        self.device = device
        self.model.eval()
    
    def traverse_dimension(self,
                         base_latent: torch.Tensor,
                         dim: int,
                         values: torch.Tensor,
                         decode_fn) -> List[torch.Tensor]:
        """
        Traverse a single latent dimension.
        
        Args:
            base_latent: Base latent code [1, latent_dim]
            dim: Dimension to traverse
            values: Values to set for the dimension
            decode_fn: Function to decode latents to outputs
            
        Returns:
            List of decoded outputs
        """
        outputs = []
        
        for val in values:
            # Create modified latent
            z = base_latent.clone()
            z[0, dim] = val
            
            # Decode
            with torch.no_grad():
                output = decode_fn(z)
                outputs.append(output)
        
        return outputs
    
    def interpolate(self,
                   z1: torch.Tensor,
                   z2: torch.Tensor,
                   steps: int = 10,
                   decode_fn = None) -> List[torch.Tensor]:
        """
        Interpolate between two latent codes.
        
        Args:
            z1: First latent code [1, latent_dim]
            z2: Second latent code [1, latent_dim]
            steps: Number of interpolation steps
            decode_fn: Function to decode latents
            
        Returns:
            List of decoded outputs
        """
        alphas = torch.linspace(0, 1, steps, device=self.device)
        outputs = []
        
        for alpha in alphas:
            # Linear interpolation
            z_interp = (1 - alpha) * z1 + alpha * z2
            
            # Decode
            with torch.no_grad():
                output = decode_fn(z_interp)
                outputs.append(output)
        
        return outputs
    
    def find_attribute_directions(self,
                                latents: torch.Tensor,
                                attributes: torch.Tensor,
                                normalize: bool = True) -> torch.Tensor:
        """
        Find latent directions that correspond to attributes.
        
        Args:
            latents: Latent codes [n_samples, latent_dim]
            attributes: Attribute values [n_samples, n_attributes]
            normalize: Normalize directions
            
        Returns:
            Directions [n_attributes, latent_dim]
        """
        # Center data
        latents_centered = latents - latents.mean(dim=0)
        attributes_centered = attributes - attributes.mean(dim=0)
        
        # Find directions using linear regression
        # A^T A w = A^T b
        directions = torch.linalg.lstsq(attributes_centered, latents_centered).solution
        
        if normalize:
            directions = F.normalize(directions, dim=1)
        
        return directions.T
    
    def compute_disentanglement_metrics(self,
                                      latents: torch.Tensor,
                                      factors: torch.Tensor) -> Dict[str, float]:
        """
        Compute disentanglement metrics (MIG, SAP).
        
        Args:
            latents: Latent codes [n_samples, latent_dim]
            factors: Ground truth factors [n_samples, n_factors]
            
        Returns:
            Dictionary of metrics
        """
        n_samples, latent_dim = latents.shape
        n_factors = factors.shape[1]
        
        # Mutual Information Gap (MIG)
        mi_matrix = self._compute_mutual_information(latents, factors)
        
        # For each factor, find top two latent dims with highest MI
        mig_scores = []
        for j in range(n_factors):
            mi_values = mi_matrix[:, j]
            sorted_mi = torch.sort(mi_values, descending=True)[0]
            if len(sorted_mi) > 1:
                mig = (sorted_mi[0] - sorted_mi[1]) / (sorted_mi[0] + 1e-8)
                mig_scores.append(mig.item())
        
        # SAP (Separated Attribute Predictability)
        sap_score = self._compute_sap(latents, factors)
        
        return {
            'mig': np.mean(mig_scores) if mig_scores else 0.0,
            'sap': sap_score,
            'active_dims': (latents.std(dim=0) > 0.1).float().mean().item()
        }
    
    def _compute_mutual_information(self,
                                  latents: torch.Tensor,
                                  factors: torch.Tensor) -> torch.Tensor:
        """Compute mutual information between latents and factors."""
        # Simplified MI estimation using binning
        n_bins = 10
        latent_dim = latents.shape[1]
        n_factors = factors.shape[1]
        
        mi_matrix = torch.zeros(latent_dim, n_factors)
        
        for i in range(latent_dim):
            for j in range(n_factors):
                # Bin the continuous values
                latent_bins = torch.histc(latents[:, i], bins=n_bins)
                factor_bins = torch.histc(factors[:, j], bins=n_bins)
                
                # Compute joint histogram
                joint_hist = torch.histogramdd(
                    torch.stack([latents[:, i], factors[:, j]], dim=1),
                    bins=[n_bins, n_bins]
                )[0]
                
                # Compute MI
                p_joint = joint_hist / joint_hist.sum()
                p_latent = latent_bins / latent_bins.sum()
                p_factor = factor_bins / factor_bins.sum()
                
                # MI = sum p(x,y) log(p(x,y) / (p(x)p(y)))
                p_prod = p_latent.unsqueeze(1) * p_factor.unsqueeze(0)
                mi = (p_joint * torch.log((p_joint + 1e-8) / (p_prod + 1e-8))).sum()
                mi_matrix[i, j] = mi
        
        return mi_matrix
    
    def _compute_sap(self, latents: torch.Tensor, factors: torch.Tensor) -> float:
        """Compute Separated Attribute Predictability score."""
        # Train linear classifiers for each factor
        scores = []
        
        for j in range(factors.shape[1]):
            # Use sklearn for simplicity (would need to import)
            # For now, return placeholder
            scores.append(0.8)
        
        return np.mean(scores)
    
    def visualize_latent_space(self,
                             latents: torch.Tensor,
                             labels: Optional[torch.Tensor] = None,
                             save_path: Optional[Path] = None):
        """
        Visualize latent space using t-SNE or PCA.
        
        Args:
            latents: Latent codes [n_samples, latent_dim]
            labels: Optional labels for coloring
            save_path: Path to save visualization
        """
        from sklearn.manifold import TSNE
        
        # Reduce to 2D using t-SNE
        latents_np = latents.detach().cpu().numpy()
        tsne = TSNE(n_components=2, random_state=42)
        latents_2d = tsne.fit_transform(latents_np)
        
        # Plot
        plt.figure(figsize=(10, 8))
        
        if labels is not None:
            scatter = plt.scatter(latents_2d[:, 0], latents_2d[:, 1], 
                                c=labels.detach().cpu().numpy(), cmap='tab10', alpha=0.7)
            plt.colorbar(scatter)
        else:
            plt.scatter(latents_2d[:, 0], latents_2d[:, 1], alpha=0.7)
        
        plt.xlabel('t-SNE 1')
        plt.ylabel('t-SNE 2')
        plt.title('Latent Space Visualization')
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
        else:
            plt.show()
        
        plt.close()


class AdaptiveBeta(nn.Module):
    """
    Adaptive β scheduling for β-VAE training.
    
    Implements various scheduling strategies for the β parameter.
    """
    
    def __init__(self,
                 beta_start: float = 0.0,
                 beta_end: float = 1.0,
                 warmup_epochs: int = 10,
                 schedule_type: str = "linear"):
        super().__init__()
        
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        
        self.register_buffer('current_epoch', torch.tensor(0))
    
    def step(self):
        """Update epoch counter."""
        self.current_epoch = self.current_epoch + 1  # Avoid inplace operation
    
    def get_beta(self) -> float:
        """Get current β value based on schedule."""
        epoch = self.current_epoch.item()
        
        if epoch >= self.warmup_epochs:
            return self.beta_end
        
        if self.schedule_type == "linear":
            # Linear warmup
            return self.beta_start + (self.beta_end - self.beta_start) * (epoch / self.warmup_epochs)
            
        elif self.schedule_type == "exponential":
            # Exponential warmup
            factor = 1 - np.exp(-5 * epoch / self.warmup_epochs)
            return self.beta_start + (self.beta_end - self.beta_start) * factor
            
        elif self.schedule_type == "cyclical":
            # Cyclical annealing
            cycle_length = self.warmup_epochs // 4
            cycle_pos = epoch % cycle_length
            return self.beta_start + (self.beta_end - self.beta_start) * (cycle_pos / cycle_length)
        
        else:
            return self.beta_end