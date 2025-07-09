"""
Advanced Regularization System for Aurl.ai Music Generation.

This module provides comprehensive regularization techniques:
- Enhanced dropout strategies (adaptive, scheduled, musical)
- Advanced gradient clipping (adaptive, per-layer, norm-based)
- Weight decay strategies (adaptive, layer-specific)
- Musical domain-specific regularization
- Activation regularization
- Spectral regularization for stable training

Integrates with existing model architecture and training systems.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass
from enum import Enum
import math

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class DropoutStrategy(Enum):
    """Different dropout strategies."""
    STANDARD = "standard"
    ADAPTIVE = "adaptive"
    SCHEDULED = "scheduled"
    MUSICAL = "musical"
    LAYERWISE = "layerwise"


class GradientClipStrategy(Enum):
    """Gradient clipping strategies."""
    NORM = "norm"
    VALUE = "value"
    ADAPTIVE = "adaptive"
    PER_LAYER = "per_layer"
    PERCENTILE = "percentile"


@dataclass
class RegularizationConfig:
    """Configuration for regularization strategies."""
    
    # Dropout configuration
    dropout_strategy: DropoutStrategy = DropoutStrategy.ADAPTIVE
    base_dropout: float = 0.1
    max_dropout: float = 0.3
    dropout_schedule_epochs: int = 20
    
    # Layer-specific dropout rates
    embedding_dropout: float = 0.1
    attention_dropout: float = 0.1
    ffn_dropout: float = 0.15
    output_dropout: float = 0.1
    
    # Gradient clipping
    gradient_clip_strategy: GradientClipStrategy = GradientClipStrategy.ADAPTIVE
    max_grad_norm: float = 1.0
    adaptive_clip_percentile: float = 0.95
    per_layer_clip: bool = True
    
    # Weight decay
    weight_decay: float = 1e-4
    adaptive_weight_decay: bool = True
    layer_specific_decay: Dict[str, float] = None
    
    # Advanced regularization
    activation_regularization: bool = True
    activation_reg_strength: float = 1e-5
    spectral_regularization: bool = False
    spectral_reg_strength: float = 1e-4
    
    # Musical domain regularization
    musical_consistency_reg: bool = True
    musical_reg_strength: float = 1e-3
    temporal_smoothness_reg: bool = True
    temporal_reg_strength: float = 1e-4


class AdaptiveDropout(nn.Module):
    """
    Adaptive dropout that adjusts dropout rate based on training progress,
    model performance, or musical coherence metrics.
    """
    
    def __init__(self, 
                 base_dropout: float = 0.1,
                 max_dropout: float = 0.3,
                 strategy: DropoutStrategy = DropoutStrategy.ADAPTIVE):
        super().__init__()
        
        self.base_dropout = base_dropout
        self.max_dropout = max_dropout
        self.strategy = strategy
        self.current_dropout = base_dropout
        
        # Tracking variables
        self.training_step = 0
        self.performance_history = []
        
        logger.debug(f"Initialized AdaptiveDropout: {strategy.value}, "
                    f"range=[{base_dropout}, {max_dropout}]")
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """Apply adaptive dropout."""
        if not self.training:
            return x
        
        dropout_rate = self._get_current_dropout_rate()
        return F.dropout(x, p=dropout_rate, training=True)
    
    def update(self, 
               step: int = None, 
               performance_metric: float = None,
               epoch: int = None,
               total_epochs: int = None):
        """Update dropout rate based on training progress."""
        
        if step is not None:
            self.training_step = step
        
        if performance_metric is not None:
            self.performance_history.append(performance_metric)
        
        if self.strategy == DropoutStrategy.ADAPTIVE:
            self._update_adaptive_dropout(performance_metric)
        elif self.strategy == DropoutStrategy.SCHEDULED:
            self._update_scheduled_dropout(epoch, total_epochs)
        elif self.strategy == DropoutStrategy.MUSICAL:
            self._update_musical_dropout(performance_metric)
    
    def _get_current_dropout_rate(self) -> float:
        """Get current dropout rate."""
        return self.current_dropout
    
    def _update_adaptive_dropout(self, performance_metric: float = None):
        """Update dropout based on performance trends."""
        if len(self.performance_history) < 5:
            return
        
        # Calculate performance trend
        recent_performance = np.mean(self.performance_history[-5:])
        older_performance = np.mean(self.performance_history[-10:-5]) if len(self.performance_history) >= 10 else recent_performance
        
        if recent_performance > older_performance:
            # Performance degrading - increase dropout
            self.current_dropout = min(self.current_dropout * 1.05, self.max_dropout)
        else:
            # Performance improving - slightly decrease dropout
            self.current_dropout = max(self.current_dropout * 0.98, self.base_dropout)
    
    def _update_scheduled_dropout(self, epoch: int, total_epochs: int):
        """Update dropout on a schedule."""
        if epoch is None or total_epochs is None:
            return
        
        # Linear increase from base to max over first half of training
        mid_point = total_epochs // 2
        if epoch <= mid_point:
            progress = epoch / mid_point
            self.current_dropout = self.base_dropout + progress * (self.max_dropout - self.base_dropout)
        else:
            # Decrease back to base in second half
            progress = (epoch - mid_point) / (total_epochs - mid_point)
            self.current_dropout = self.max_dropout - progress * (self.max_dropout - self.base_dropout)
    
    def _update_musical_dropout(self, musical_quality: float = None):
        """Update dropout based on musical quality metrics."""
        if musical_quality is None:
            return
        
        # Higher dropout when musical quality is low
        if musical_quality < 0.3:
            self.current_dropout = min(self.current_dropout * 1.1, self.max_dropout)
        elif musical_quality > 0.7:
            self.current_dropout = max(self.current_dropout * 0.95, self.base_dropout)


class GradientClipper:
    """
    Advanced gradient clipping with multiple strategies.
    """
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.strategy = config.gradient_clip_strategy
        self.max_norm = config.max_grad_norm
        
        # Tracking for adaptive clipping
        self.gradient_history = []
        self.clip_history = []
        
        logger.info(f"Initialized GradientClipper: {self.strategy.value}, "
                   f"max_norm={self.max_norm}")
    
    def clip_gradients(self, 
                      model: nn.Module, 
                      return_norm: bool = False) -> Optional[float]:
        """
        Clip gradients according to configured strategy.
        
        Returns:
            Gradient norm if return_norm=True
        """
        
        if self.strategy == GradientClipStrategy.NORM:
            return self._clip_by_norm(model, return_norm)
        elif self.strategy == GradientClipStrategy.VALUE:
            return self._clip_by_value(model, return_norm)
        elif self.strategy == GradientClipStrategy.ADAPTIVE:
            return self._clip_adaptive(model, return_norm)
        elif self.strategy == GradientClipStrategy.PER_LAYER:
            return self._clip_per_layer(model, return_norm)
        elif self.strategy == GradientClipStrategy.PERCENTILE:
            return self._clip_by_percentile(model, return_norm)
        else:
            return self._clip_by_norm(model, return_norm)
    
    def _clip_by_norm(self, model: nn.Module, return_norm: bool) -> Optional[float]:
        """Standard gradient norm clipping."""
        total_norm = torch.nn.utils.clip_grad_norm_(
            model.parameters(), 
            self.max_norm
        ).item()
        
        return total_norm if return_norm else None
    
    def _clip_by_value(self, model: nn.Module, return_norm: bool) -> Optional[float]:
        """Clip gradients by absolute value."""
        total_norm = 0.0
        
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
                param.grad.data.clamp_(-self.max_norm, self.max_norm)
        
        total_norm = total_norm ** 0.5
        return total_norm if return_norm else None
    
    def _clip_adaptive(self, model: nn.Module, return_norm: bool) -> Optional[float]:
        """Adaptive gradient clipping based on gradient history."""
        # Calculate current gradient norm
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                total_norm += param.grad.data.norm(2).item() ** 2
        total_norm = total_norm ** 0.5
        
        self.gradient_history.append(total_norm)
        
        # Adaptive clip value based on percentile of recent gradients
        if len(self.gradient_history) > 10:
            recent_norms = self.gradient_history[-50:]  # Last 50 updates
            adaptive_clip = np.percentile(recent_norms, self.config.adaptive_clip_percentile * 100)
            clip_value = min(adaptive_clip, self.max_norm)
        else:
            clip_value = self.max_norm
        
        # Apply clipping
        if total_norm > clip_value:
            clip_factor = clip_value / total_norm
            for param in model.parameters():
                if param.grad is not None:
                    param.grad.data.mul_(clip_factor)
        
        return total_norm if return_norm else None
    
    def _clip_per_layer(self, model: nn.Module, return_norm: bool) -> Optional[float]:
        """Clip gradients per layer."""
        total_norm = 0.0
        layer_norms = {}
        
        # Group parameters by layer
        for name, param in model.named_parameters():
            if param.grad is not None:
                layer_name = name.split('.')[0]  # Get top-level layer
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = []
                layer_norms[layer_name].append(param)
        
        # Clip each layer separately
        for layer_name, params in layer_norms.items():
            layer_norm = torch.nn.utils.clip_grad_norm_(
                params, 
                self.max_norm
            ).item()
            total_norm += layer_norm ** 2
        
        total_norm = total_norm ** 0.5
        return total_norm if return_norm else None
    
    def _clip_by_percentile(self, model: nn.Module, return_norm: bool) -> Optional[float]:
        """Clip using percentile-based threshold."""
        # Collect all gradient values
        all_grads = []
        for param in model.parameters():
            if param.grad is not None:
                all_grads.extend(param.grad.data.abs().flatten().tolist())
        
        if not all_grads:
            return 0.0 if return_norm else None
        
        # Calculate percentile threshold
        threshold = np.percentile(all_grads, self.config.adaptive_clip_percentile * 100)
        clip_value = min(threshold, self.max_norm)
        
        # Apply clipping
        total_norm = 0.0
        for param in model.parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                if param_norm > clip_value:
                    param.grad.data.mul_(clip_value / param_norm)
        
        total_norm = total_norm ** 0.5
        return total_norm if return_norm else None


class WeightRegularizer:
    """
    Advanced weight decay and regularization strategies.
    """
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.base_weight_decay = config.weight_decay
        self.adaptive_decay = config.adaptive_weight_decay
        self.layer_specific = config.layer_specific_decay or {}
        
        # Tracking for adaptive weight decay
        self.weight_norms_history = []
        
    def get_weight_decay_for_param(self, param_name: str, current_weight_norm: float = None) -> float:
        """Get weight decay value for specific parameter."""
        
        # Check for layer-specific decay rates
        for layer_pattern, decay_rate in self.layer_specific.items():
            if layer_pattern in param_name:
                return decay_rate
        
        # Adaptive weight decay based on weight magnitudes
        if self.adaptive_decay and current_weight_norm is not None:
            if current_weight_norm > 1.0:
                # Increase decay for large weights
                adaptive_factor = min(current_weight_norm, 2.0)
                return self.base_weight_decay * adaptive_factor
            else:
                # Reduce decay for small weights
                return self.base_weight_decay * max(current_weight_norm, 0.5)
        
        return self.base_weight_decay
    
    def apply_weight_regularization(self, model: nn.Module) -> Dict[str, float]:
        """Apply weight regularization and return statistics."""
        
        total_reg_loss = 0.0
        param_stats = {}
        
        for name, param in model.named_parameters():
            if param.requires_grad and param.grad is not None:
                
                # Calculate weight norm
                weight_norm = param.data.norm(2).item()
                
                # Get appropriate weight decay
                weight_decay = self.get_weight_decay_for_param(name, weight_norm)
                
                # Apply L2 regularization (weight decay)
                reg_loss = 0.5 * weight_decay * (param.data ** 2).sum()
                total_reg_loss += reg_loss.item()
                
                # Add regularization to gradients
                param.grad.data.add_(param.data, alpha=weight_decay)
                
                param_stats[name] = {
                    'weight_norm': weight_norm,
                    'weight_decay': weight_decay,
                    'reg_loss': reg_loss.item()
                }
        
        return {
            'total_regularization_loss': total_reg_loss,
            'param_stats': param_stats
        }


class MusicalRegularizer:
    """
    Musical domain-specific regularization techniques.
    """
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        self.consistency_strength = config.musical_reg_strength
        self.temporal_strength = config.temporal_reg_strength
        
    def apply_musical_regularization(self, 
                                   outputs: torch.Tensor,
                                   targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Apply musical regularization losses.
        
        Args:
            outputs: Model outputs [batch, seq_len, vocab_size]
            targets: Target sequences [batch, seq_len] (optional)
            
        Returns:
            Dictionary of regularization losses
        """
        losses = {}
        
        if self.config.musical_consistency_reg:
            losses['musical_consistency'] = self._musical_consistency_loss(outputs)
        
        if self.config.temporal_smoothness_reg:
            losses['temporal_smoothness'] = self._temporal_smoothness_loss(outputs)
        
        return losses
    
    def _musical_consistency_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Regularization to encourage musical consistency.
        Penalizes rapid changes in musical content.
        """
        # Calculate differences between consecutive predictions
        diff = torch.diff(outputs, dim=1)  # [batch, seq_len-1, vocab_size]
        
        # L2 norm of differences (penalize large jumps)
        consistency_loss = torch.mean(torch.norm(diff, dim=-1) ** 2)
        
        return self.consistency_strength * consistency_loss
    
    def _temporal_smoothness_loss(self, outputs: torch.Tensor) -> torch.Tensor:
        """
        Regularization to encourage temporal smoothness in musical generation.
        """
        # Second-order differences (acceleration in musical space)
        first_diff = torch.diff(outputs, dim=1)
        second_diff = torch.diff(first_diff, dim=1)
        
        # Penalize large second-order changes
        smoothness_loss = torch.mean(torch.norm(second_diff, dim=-1) ** 2)
        
        return self.temporal_strength * smoothness_loss


class ComprehensiveRegularizer:
    """
    Unified regularization system combining all techniques.
    """
    
    def __init__(self, config: RegularizationConfig):
        self.config = config
        
        # Initialize sub-regularizers
        self.gradient_clipper = GradientClipper(config)
        self.weight_regularizer = WeightRegularizer(config)
        self.musical_regularizer = MusicalRegularizer(config)
        
        # Create adaptive dropout layers for different components
        self.dropout_layers = {
            'embedding': AdaptiveDropout(config.embedding_dropout, strategy=config.dropout_strategy),
            'attention': AdaptiveDropout(config.attention_dropout, strategy=config.dropout_strategy),
            'ffn': AdaptiveDropout(config.ffn_dropout, strategy=config.dropout_strategy),
            'output': AdaptiveDropout(config.output_dropout, strategy=config.dropout_strategy)
        }
        
        logger.info(f"Initialized ComprehensiveRegularizer with {len(self.dropout_layers)} dropout layers")
    
    def apply_regularization(self, 
                           model: nn.Module,
                           outputs: torch.Tensor = None,
                           targets: torch.Tensor = None,
                           step: int = None,
                           epoch: int = None,
                           metrics: Dict[str, float] = None) -> Dict[str, Any]:
        """
        Apply comprehensive regularization and return statistics.
        """
        results = {}
        
        # Update adaptive dropout rates
        self._update_dropout_rates(step, epoch, metrics)
        
        # Apply gradient clipping
        grad_norm = self.gradient_clipper.clip_gradients(model, return_norm=True)
        results['gradient_norm'] = grad_norm
        
        # Apply weight regularization
        weight_reg_stats = self.weight_regularizer.apply_weight_regularization(model)
        results.update(weight_reg_stats)
        
        # Apply musical regularization if outputs provided
        if outputs is not None:
            musical_losses = self.musical_regularizer.apply_musical_regularization(outputs, targets)
            results['musical_regularization'] = musical_losses
        
        return results
    
    def _update_dropout_rates(self, step: int, epoch: int, metrics: Dict[str, float]):
        """Update all adaptive dropout rates."""
        musical_quality = metrics.get('musical_quality') if metrics else None
        
        for name, dropout_layer in self.dropout_layers.items():
            dropout_layer.update(
                step=step,
                epoch=epoch,
                performance_metric=musical_quality
            )
    
    def get_dropout_layer(self, layer_type: str) -> AdaptiveDropout:
        """Get specific dropout layer."""
        return self.dropout_layers.get(layer_type, self.dropout_layers['embedding'])
    
    def get_regularization_summary(self) -> Dict[str, Any]:
        """Get summary of current regularization state."""
        return {
            'dropout_rates': {
                name: layer.current_dropout 
                for name, layer in self.dropout_layers.items()
            },
            'gradient_clip_strategy': self.config.gradient_clip_strategy.value,
            'weight_decay': self.config.weight_decay,
            'musical_regularization': {
                'consistency': self.config.musical_consistency_reg,
                'temporal_smoothness': self.config.temporal_smoothness_reg
            }
        }


def create_regularization_config(**kwargs) -> RegularizationConfig:
    """Create regularization configuration with musical defaults."""
    
    musical_defaults = {
        "dropout_strategy": DropoutStrategy.ADAPTIVE,
        "base_dropout": 0.1,
        "max_dropout": 0.25,
        "gradient_clip_strategy": GradientClipStrategy.ADAPTIVE,
        "max_grad_norm": 1.0,
        "weight_decay": 1e-4,
        "adaptive_weight_decay": True,
        "musical_consistency_reg": True,
        "musical_reg_strength": 1e-3,
        "temporal_smoothness_reg": True,
        "temporal_reg_strength": 5e-4
    }
    
    # Override with provided kwargs
    musical_defaults.update(kwargs)
    
    return RegularizationConfig(**musical_defaults)