"""
Loss landscape visualization and monitoring tools for Aurl.ai training.

This module provides comprehensive tools for visualizing and analyzing
the loss landscape during VAE-GAN training including:

1. Real-time loss monitoring and plotting
2. Loss landscape visualization and analysis
3. Gradient flow monitoring
4. Training stability metrics
5. Multi-objective loss balance tracking
6. Musical generation quality metrics

Designed to provide deep insights into training dynamics and help
optimize the complex VAE-GAN loss landscape.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union
from pathlib import Path
import json
from datetime import datetime
import warnings

from src.utils.logger import setup_logger

logger = setup_logger(__name__)

# Suppress matplotlib warnings for cleaner output
warnings.filterwarnings('ignore', category=UserWarning, module='matplotlib')


class LossMonitor:
    """
    Real-time loss monitoring and tracking system.
    
    Maintains rolling histories of all loss components and provides
    real-time visualization and analysis capabilities.
    """
    
    def __init__(self,
                 history_length: int = 1000,
                 save_dir: Optional[Path] = None,
                 plot_frequency: int = 100):
        self.history_length = history_length
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/loss_monitoring")
        self.plot_frequency = plot_frequency
        
        # Create save directory
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Loss histories
        self.loss_histories = {}
        self.step_counter = 0
        self.epoch_counter = 0
        
        # Current session metadata
        self.session_start = datetime.now()
        self.session_id = self.session_start.strftime("%Y%m%d_%H%M%S")
        
        logger.info(f"Initialized LossMonitor with session_id={self.session_id}")
    
    def update(self, losses: Dict[str, Union[torch.Tensor, float]], step: Optional[int] = None):
        """Update loss histories with new values."""
        if step is None:
            step = self.step_counter
        
        self.step_counter = max(self.step_counter + 1, step + 1)
        
        # Convert tensors to scalars and update histories
        for loss_name, loss_value in losses.items():
            if isinstance(loss_value, torch.Tensor):
                loss_value = loss_value.item()
            
            if loss_name not in self.loss_histories:
                self.loss_histories[loss_name] = []
            
            self.loss_histories[loss_name].append((step, loss_value))
            
            # Maintain history length
            if len(self.loss_histories[loss_name]) > self.history_length:
                self.loss_histories[loss_name].pop(0)
    
    def plot_losses(self,
                   loss_groups: Optional[Dict[str, List[str]]] = None,
                   figsize: Tuple[int, int] = (15, 10),
                   save_plot: bool = True) -> None:
        """Plot loss histories grouped by category."""
        if not self.loss_histories:
            logger.warning("No loss history to plot")
            return
        
        # Default loss groupings
        if loss_groups is None:
            loss_groups = {
                'Reconstruction': ['recon_total', 'recon_perceptual_reconstruction', 'recon_structure_emphasis'],
                'VAE': ['kl_loss', 'kl_raw', 'beta'],
                'Adversarial': ['gen_total', 'disc_total', 'gen_balanced', 'disc_balanced'],
                'Features': ['gen_feature_matching', 'gen_perceptual'],
                'Constraints': ['constraint_rhythm_constraint', 'constraint_harmony_constraint', 'constraint_total'],
                'Metrics': ['reconstruction_accuracy', 'perplexity', 'active_latent_dims']
            }
        
        # Filter groups to only include losses we have data for
        available_groups = {}
        for group_name, loss_names in loss_groups.items():
            available_losses = [name for name in loss_names if name in self.loss_histories]
            if available_losses:
                available_groups[group_name] = available_losses
        
        if not available_groups:
            logger.warning("No matching losses found for plotting")
            return
        
        # Create subplots
        n_groups = len(available_groups)
        n_cols = min(3, n_groups)
        n_rows = (n_groups + n_cols - 1) // n_cols
        
        fig, axes = plt.subplots(n_rows, n_cols, figsize=figsize)
        if n_groups == 1:
            axes = [axes]
        elif n_rows == 1:
            axes = [axes] if n_cols == 1 else axes
        else:
            axes = axes.flatten() if n_groups > 1 else [axes]
        
        # Plot each group
        for idx, (group_name, loss_names) in enumerate(available_groups.items()):
            ax = axes[idx] if idx < len(axes) else axes[-1]
            
            for loss_name in loss_names:
                if loss_name in self.loss_histories:
                    steps, values = zip(*self.loss_histories[loss_name])
                    ax.plot(steps, values, label=loss_name.replace('_', ' '), alpha=0.8)
            
            ax.set_title(f'{group_name} Losses')
            ax.set_xlabel('Training Step')
            ax.set_ylabel('Loss Value')
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
            ax.grid(True, alpha=0.3)
            
            # Use log scale for better visualization if values vary widely
            values_range = []
            for loss_name in loss_names:
                if loss_name in self.loss_histories and self.loss_histories[loss_name]:
                    _, values = zip(*self.loss_histories[loss_name])
                    values_range.extend(values)
            
            if values_range and max(values_range) / min([v for v in values_range if v > 0] + [1]) > 100:
                ax.set_yscale('log')
        
        # Remove unused subplots
        for idx in range(len(available_groups), len(axes)):
            fig.delaxes(axes[idx])
        
        plt.tight_layout()
        
        if save_plot:
            plot_path = self.save_dir / f"loss_plot_{self.session_id}_step_{self.step_counter}.png"
            plt.savefig(plot_path, dpi=150, bbox_inches='tight')
            logger.info(f"Saved loss plot to {plot_path}")
        
        plt.show()
        plt.close()
    
    def save_histories(self, filename: Optional[str] = None) -> Path:
        """Save loss histories to JSON file."""
        if filename is None:
            filename = f"loss_histories_{self.session_id}.json"
        
        save_path = self.save_dir / filename
        
        # Convert histories to serializable format
        serializable_histories = {}
        for loss_name, history in self.loss_histories.items():
            serializable_histories[loss_name] = {
                'steps': [step for step, _ in history],
                'values': [value for _, value in history]
            }
        
        metadata = {
            'session_id': self.session_id,
            'session_start': self.session_start.isoformat(),
            'total_steps': self.step_counter,
            'total_epochs': self.epoch_counter,
            'history_length': self.history_length
        }
        
        data = {
            'metadata': metadata,
            'histories': serializable_histories
        }
        
        with open(save_path, 'w') as f:
            json.dump(data, f, indent=2)
        
        logger.info(f"Saved loss histories to {save_path}")
        return save_path
    
    def get_recent_stats(self, window_size: int = 100) -> Dict[str, Dict[str, float]]:
        """Get statistics for recent loss values."""
        stats = {}
        
        for loss_name, history in self.loss_histories.items():
            if len(history) > 0:
                recent_values = [value for _, value in history[-window_size:]]
                
                stats[loss_name] = {
                    'mean': np.mean(recent_values),
                    'std': np.std(recent_values),
                    'min': np.min(recent_values),
                    'max': np.max(recent_values),
                    'latest': recent_values[-1],
                    'trend': 'stable'
                }
                
                # Simple trend detection
                if len(recent_values) >= 10:
                    first_half = np.mean(recent_values[:len(recent_values)//2])
                    second_half = np.mean(recent_values[len(recent_values)//2:])
                    
                    relative_change = (second_half - first_half) / (abs(first_half) + 1e-8)
                    
                    if relative_change > 0.1:
                        stats[loss_name]['trend'] = 'increasing'
                    elif relative_change < -0.1:
                        stats[loss_name]['trend'] = 'decreasing'
        
        return stats


class LossLandscapeAnalyzer:
    """
    Advanced loss landscape analysis and visualization.
    
    Provides tools for understanding the geometry and properties
    of the loss landscape during training.
    """
    
    def __init__(self,
                 model: nn.Module,
                 loss_fn: nn.Module,
                 save_dir: Optional[Path] = None):
        self.model = model
        self.loss_fn = loss_fn
        self.save_dir = Path(save_dir) if save_dir else Path("outputs/loss_landscape")
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Store original parameters for restoration
        self.original_params = None
        
    def save_checkpoint(self):
        """Save current model parameters."""
        self.original_params = {}
        for name, param in self.model.named_parameters():
            self.original_params[name] = param.data.clone()
    
    def restore_checkpoint(self):
        """Restore model to saved parameters."""
        if self.original_params is None:
            logger.warning("No checkpoint saved, cannot restore")
            return
        
        for name, param in self.model.named_parameters():
            if name in self.original_params:
                param.data.copy_(self.original_params[name])
    
    def compute_gradient_norm(self,
                            data_batch: Dict[str, torch.Tensor]) -> float:
        """Compute gradient norm at current model state."""
        self.model.zero_grad()
        
        # Forward pass and loss computation
        # This would need to be adapted to your specific data format
        loss = self._compute_loss(data_batch)
        loss.backward()
        
        # Compute gradient norm
        total_norm = 0.0
        for param in self.model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        
        return total_norm ** 0.5
    
    def analyze_sharpness(self,
                         data_batch: Dict[str, torch.Tensor],
                         epsilon: float = 0.01,
                         num_directions: int = 10) -> Dict[str, float]:
        """Analyze loss landscape sharpness around current point."""
        self.save_checkpoint()
        
        original_loss = self._compute_loss(data_batch).item()
        
        sharpness_metrics = {
            'original_loss': original_loss,
            'max_increase': 0.0,
            'avg_increase': 0.0,
            'sharpness_score': 0.0
        }
        
        loss_increases = []
        
        try:
            for _ in range(num_directions):
                # Generate random direction
                direction = {}
                for name, param in self.model.named_parameters():
                    direction[name] = torch.randn_like(param) * epsilon
                
                # Move in positive direction
                for name, param in self.model.named_parameters():
                    param.data.add_(direction[name])
                
                pos_loss = self._compute_loss(data_batch).item()
                
                # Move in negative direction (from original point)
                self.restore_checkpoint()
                for name, param in self.model.named_parameters():
                    param.data.sub_(direction[name])
                
                neg_loss = self._compute_loss(data_batch).item()
                
                # Compute increase
                max_increase = max(pos_loss - original_loss, neg_loss - original_loss)
                loss_increases.append(max_increase)
                
                # Restore for next iteration
                self.restore_checkpoint()
            
            # Compute metrics
            sharpness_metrics['max_increase'] = max(loss_increases)
            sharpness_metrics['avg_increase'] = np.mean(loss_increases)
            sharpness_metrics['sharpness_score'] = np.mean(loss_increases) / (original_loss + 1e-8)
            
        finally:
            self.restore_checkpoint()
        
        return sharpness_metrics
    
    def visualize_loss_surface_2d(self,
                                 data_batch: Dict[str, torch.Tensor],
                                 direction1: Optional[Dict[str, torch.Tensor]] = None,
                                 direction2: Optional[Dict[str, torch.Tensor]] = None,
                                 alpha_range: Tuple[float, float] = (-1.0, 1.0),
                                 beta_range: Tuple[float, float] = (-1.0, 1.0),
                                 resolution: int = 20) -> np.ndarray:
        """Visualize 2D slice of loss surface."""
        self.save_checkpoint()
        
        # Generate random directions if not provided
        if direction1 is None:
            direction1 = {}
            for name, param in self.model.named_parameters():
                direction1[name] = torch.randn_like(param)
                # Normalize
                norm = direction1[name].norm()
                direction1[name] = direction1[name] / (norm + 1e-8)
        
        if direction2 is None:
            direction2 = {}
            for name, param in self.model.named_parameters():
                direction2[name] = torch.randn_like(param)
                # Make orthogonal to direction1
                dot_product = sum((direction1[name] * direction2[name]).sum() for name in direction1)
                for name in direction2:
                    direction2[name] = direction2[name] - dot_product * direction1[name]
                # Normalize
                norm = sum(direction2[name].norm()**2 for name in direction2)**0.5
                for name in direction2:
                    direction2[name] = direction2[name] / (norm + 1e-8)
        
        # Create grid
        alphas = np.linspace(alpha_range[0], alpha_range[1], resolution)
        betas = np.linspace(beta_range[0], beta_range[1], resolution)
        
        loss_surface = np.zeros((resolution, resolution))
        
        try:
            for i, alpha in enumerate(alphas):
                for j, beta in enumerate(betas):
                    # Move to point (alpha, beta)
                    self.restore_checkpoint()
                    for name, param in self.model.named_parameters():
                        param.data.add_(alpha * direction1[name] + beta * direction2[name])
                    
                    # Compute loss
                    loss = self._compute_loss(data_batch).item()
                    loss_surface[i, j] = loss
        
        finally:
            self.restore_checkpoint()
        
        # Plot surface
        plt.figure(figsize=(10, 8))
        plt.contour(betas, alphas, loss_surface, levels=20)
        plt.colorbar(label='Loss Value')
        plt.xlabel('Direction 2')
        plt.ylabel('Direction 1')
        plt.title('Loss Landscape 2D Slice')
        
        # Mark current position
        plt.plot(0, 0, 'r*', markersize=15, label='Current Position')
        plt.legend()
        
        save_path = self.save_dir / f"loss_surface_2d_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        logger.info(f"Saved loss surface plot to {save_path}")
        
        plt.show()
        plt.close()
        
        return loss_surface
    
    def _compute_loss(self, data_batch: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Compute loss for given data batch. Should be implemented based on your model."""
        # This is a placeholder - you'll need to implement this based on your model architecture
        # For now, return a dummy loss
        dummy_input = torch.randn(2, 100, device=next(self.model.parameters()).device)
        dummy_target = torch.randint(0, 774, (2, 100), device=next(self.model.parameters()).device)
        
        try:
            if hasattr(self.model, 'forward'):
                output = self.model(dummy_input)
                if isinstance(output, dict):
                    logits = output.get('logits', output.get('reconstruction_logits', dummy_input))
                else:
                    logits = output
                
                if logits.dim() == 3:
                    loss = F.cross_entropy(logits.view(-1, logits.size(-1)), dummy_target.view(-1))
                else:
                    loss = F.mse_loss(logits, dummy_input)
                return loss
            else:
                return torch.tensor(1.0, device=next(self.model.parameters()).device)
        except Exception as e:
            logger.warning(f"Error computing loss in landscape analyzer: {e}")
            return torch.tensor(1.0, device=next(self.model.parameters()).device)


class TrainingStabilityMonitor:
    """
    Monitor training stability and detect potential issues.
    
    Tracks various stability metrics and provides early warnings
    for common training problems.
    """
    
    def __init__(self,
                 window_size: int = 100,
                 stability_threshold: float = 0.1):
        self.window_size = window_size
        self.stability_threshold = stability_threshold
        
        # Metric histories
        self.gradient_norms = []
        self.loss_values = []
        self.learning_rates = []
        
        # Stability flags
        self.stability_warnings = []
    
    def update(self,
               gradient_norm: float,
               loss_value: float,
               learning_rate: float):
        """Update stability metrics."""
        self.gradient_norms.append(gradient_norm)
        self.loss_values.append(loss_value)
        self.learning_rates.append(learning_rate)
        
        # Maintain window size
        if len(self.gradient_norms) > self.window_size:
            self.gradient_norms.pop(0)
            self.loss_values.pop(0)
            self.learning_rates.pop(0)
        
        # Check for stability issues
        self._check_stability()
    
    def _check_stability(self):
        """Check for training stability issues."""
        if len(self.gradient_norms) < 10:
            return
        
        # Check for exploding gradients
        recent_grad_norm = np.mean(self.gradient_norms[-10:])
        if recent_grad_norm > 10.0:
            self.stability_warnings.append({
                'type': 'exploding_gradients',
                'severity': 'high',
                'message': f'High gradient norm detected: {recent_grad_norm:.2f}',
                'step': len(self.gradient_norms)
            })
        
        # Check for vanishing gradients
        if recent_grad_norm < 1e-6:
            self.stability_warnings.append({
                'type': 'vanishing_gradients',
                'severity': 'medium',
                'message': f'Very low gradient norm: {recent_grad_norm:.2e}',
                'step': len(self.gradient_norms)
            })
        
        # Check for loss instability
        if len(self.loss_values) >= 20:
            recent_loss_std = np.std(self.loss_values[-20:])
            recent_loss_mean = np.mean(self.loss_values[-20:])
            
            if recent_loss_std / (recent_loss_mean + 1e-8) > self.stability_threshold:
                self.stability_warnings.append({
                    'type': 'loss_instability',
                    'severity': 'medium',
                    'message': f'High loss variance detected: {recent_loss_std:.3f}',
                    'step': len(self.loss_values)
                })
    
    def get_stability_report(self) -> Dict[str, Any]:
        """Generate stability report."""
        if not self.gradient_norms:
            return {'status': 'no_data'}
        
        # Convert tensors to numpy safely (handle different devices)
        gradient_norms_np = []
        loss_values_np = []
        
        for norm in self.gradient_norms:
            if isinstance(norm, torch.Tensor):
                gradient_norms_np.append(norm.detach().cpu().numpy())
            else:
                gradient_norms_np.append(norm)
        
        for loss in self.loss_values:
            if isinstance(loss, torch.Tensor):
                loss_values_np.append(loss.detach().cpu().numpy())
            else:
                loss_values_np.append(loss)
        
        report = {
            'status': 'stable',
            'metrics': {
                'avg_gradient_norm': np.mean(gradient_norms_np) if gradient_norms_np else 0.0,
                'gradient_norm_std': np.std(gradient_norms_np) if gradient_norms_np else 0.0,
                'avg_loss': np.mean(loss_values_np) if loss_values_np else 0.0,
                'loss_std': np.std(loss_values_np) if loss_values_np else 0.0,
                'recent_warnings': len([w for w in self.stability_warnings if 
                                      len(self.gradient_norms) - w['step'] < 50])
            },
            'warnings': self.stability_warnings[-10:],  # Recent warnings
            'recommendations': []
        }
        
        # Add recommendations based on detected issues
        recent_warnings = [w for w in self.stability_warnings if 
                          len(self.gradient_norms) - w['step'] < 50]
        
        if any(w['type'] == 'exploding_gradients' for w in recent_warnings):
            report['status'] = 'unstable'
            report['recommendations'].append('Consider reducing learning rate or adding gradient clipping')
        
        if any(w['type'] == 'vanishing_gradients' for w in recent_warnings):
            report['status'] = 'concerning'
            report['recommendations'].append('Consider increasing learning rate or checking model architecture')
        
        if any(w['type'] == 'loss_instability' for w in recent_warnings):
            report['status'] = 'concerning'
            report['recommendations'].append('Consider reducing learning rate or adjusting loss weights')
        
        return report