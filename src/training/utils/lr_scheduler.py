"""
Advanced Learning Rate Scheduling for Aurl.ai Music Generation.

This module implements sophisticated learning rate scheduling strategies:
- Warmup strategies (linear, exponential, cosine)
- Decay strategies (linear, cosine, exponential, polynomial)
- Adaptive scheduling based on metrics
- Plateau-based reduction
- Cyclical learning rates for better convergence
- Musical domain-specific scheduling patterns

Designed to work with our early stopping system and training infrastructure
for optimal training dynamics.
"""

import torch
import torch.optim as optim
from torch.optim.lr_scheduler import _LRScheduler
import math
import numpy as np
from typing import Dict, List, Optional, Union, Callable, Any
from dataclasses import dataclass
from enum import Enum
import warnings

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class WarmupStrategy(Enum):
    """Warmup strategies for learning rate."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    CONSTANT = "constant"


class DecayStrategy(Enum):
    """Decay strategies for learning rate."""
    LINEAR = "linear"
    COSINE = "cosine"
    EXPONENTIAL = "exponential"
    POLYNOMIAL = "polynomial"
    STEP = "step"
    MULTISTEP = "multistep"
    PLATEAU = "plateau"
    CYCLICAL = "cyclical"


@dataclass
class LRSchedulerConfig:
    """Configuration for learning rate scheduling."""
    
    # Base learning rate settings
    base_lr: float = 1e-4
    max_lr: float = 1e-3
    min_lr: float = 1e-7
    
    # Warmup configuration
    warmup_strategy: WarmupStrategy = WarmupStrategy.LINEAR
    warmup_steps: int = 1000
    warmup_epochs: int = 0  # If > 0, use epochs instead of steps
    
    # Main decay configuration
    decay_strategy: DecayStrategy = DecayStrategy.COSINE
    total_steps: int = 100000
    total_epochs: int = 100
    
    # Strategy-specific parameters
    decay_rate: float = 0.95  # For exponential decay
    decay_steps: int = 5000   # For step decay
    polynomial_power: float = 2.0  # For polynomial decay
    
    # Cyclical parameters
    cycle_length: int = 10000  # Steps per cycle
    cycle_decay: float = 0.9   # Decay factor per cycle
    
    # Plateau-based reduction
    plateau_patience: int = 5
    plateau_factor: float = 0.5
    plateau_threshold: float = 1e-4
    plateau_cooldown: int = 2
    
    # Advanced features
    use_adaptive: bool = True
    metric_based_reduction: bool = True
    target_metric: str = "total_loss"
    
    # Musical domain-specific
    musical_warmup: bool = True  # Gentle warmup for musical convergence
    stability_periods: List[int] = None  # Epochs to maintain stable LR


class WarmupScheduler:
    """Handles various warmup strategies."""
    
    def __init__(self, 
                 strategy: WarmupStrategy,
                 warmup_steps: int,
                 base_lr: float,
                 target_lr: float):
        self.strategy = strategy
        self.warmup_steps = warmup_steps
        self.base_lr = base_lr
        self.target_lr = target_lr
        
        # For cosine warmup
        self.warmup_cosine_start = 0.1 * target_lr
    
    def get_lr(self, step: int) -> float:
        """Get learning rate for current step during warmup."""
        if step >= self.warmup_steps:
            return self.target_lr
        
        progress = step / self.warmup_steps
        
        if self.strategy == WarmupStrategy.LINEAR:
            return self.base_lr + progress * (self.target_lr - self.base_lr)
        
        elif self.strategy == WarmupStrategy.EXPONENTIAL:
            # Exponential interpolation
            ratio = self.target_lr / self.base_lr
            return self.base_lr * (ratio ** progress)
        
        elif self.strategy == WarmupStrategy.COSINE:
            # Cosine warmup - smoother start
            cosine_progress = (1 - math.cos(progress * math.pi)) / 2
            return self.warmup_cosine_start + cosine_progress * (self.target_lr - self.warmup_cosine_start)
        
        elif self.strategy == WarmupStrategy.CONSTANT:
            return self.base_lr
        
        else:
            return self.target_lr


class AdaptiveLRScheduler(_LRScheduler):
    """
    Advanced learning rate scheduler with multiple strategies and adaptive features.
    
    Combines warmup, main decay, and adaptive reduction based on training metrics.
    """
    
    def __init__(self, 
                 optimizer: optim.Optimizer,
                 config: LRSchedulerConfig,
                 last_epoch: int = -1):
        self.config = config
        self.warmup_scheduler = WarmupScheduler(
            config.warmup_strategy,
            config.warmup_steps,
            config.min_lr,
            config.base_lr
        )
        
        # Tracking variables
        self.current_step = 0
        self.current_epoch = 0
        self.plateau_count = 0
        self.best_metric = float('inf')
        self.cooldown_counter = 0
        self.cycle_number = 0
        
        # Metric history for adaptive decisions
        self.metric_history = []
        self.lr_history = []
        
        # Initialize base class
        super().__init__(optimizer, last_epoch)
        
        logger.info(f"Initialized AdaptiveLRScheduler: "
                   f"warmup={config.warmup_steps} steps, "
                   f"strategy={config.decay_strategy.value}, "
                   f"base_lr={config.base_lr}")
    
    def step(self, metrics: Dict[str, float] = None, epoch: int = None):
        """Step the scheduler with optional metrics for adaptive behavior."""
        
        if epoch is not None:
            self.current_epoch = epoch
        else:
            self.current_epoch += 1
        
        self.current_step += 1
        
        # Track metrics for adaptive scheduling
        if metrics and self.config.use_adaptive:
            self._update_metrics(metrics)
        
        # Call parent step method
        super().step()
        
        # Log learning rate changes
        current_lrs = [group['lr'] for group in self.optimizer.param_groups]
        self.lr_history.append(current_lrs[0])
        
        if self.current_step % 1000 == 0:
            logger.debug(f"Step {self.current_step}: LR = {current_lrs[0]:.2e}")
    
    def get_lr(self):
        """Calculate learning rates for all parameter groups."""
        if not self._get_lr_called_within_step:
            warnings.warn("To get the last learning rate computed by the scheduler, "
                         "please use `get_last_lr()`.", UserWarning)
        
        lrs = []
        for base_lr in self.base_lrs:
            # Calculate learning rate based on current phase
            if self.current_step < self.config.warmup_steps:
                lr = self._get_warmup_lr(base_lr)
            else:
                lr = self._get_main_lr(base_lr)
            
            # Apply minimum learning rate constraint
            lr = max(lr, self.config.min_lr)
            lrs.append(lr)
        
        return lrs
    
    def _get_warmup_lr(self, base_lr: float) -> float:
        """Get learning rate during warmup phase."""
        return self.warmup_scheduler.get_lr(self.current_step)
    
    def _get_main_lr(self, base_lr: float) -> float:
        """Get learning rate during main training phase."""
        # Adjust step to account for warmup
        adjusted_step = self.current_step - self.config.warmup_steps
        adjusted_epoch = max(0, self.current_epoch - 
                           (self.config.warmup_epochs if self.config.warmup_epochs > 0 
                            else self.config.warmup_steps // 1000))
        
        if self.config.decay_strategy == DecayStrategy.LINEAR:
            return self._linear_decay(base_lr, adjusted_step)
        
        elif self.config.decay_strategy == DecayStrategy.COSINE:
            return self._cosine_decay(base_lr, adjusted_step)
        
        elif self.config.decay_strategy == DecayStrategy.EXPONENTIAL:
            return self._exponential_decay(base_lr, adjusted_step)
        
        elif self.config.decay_strategy == DecayStrategy.POLYNOMIAL:
            return self._polynomial_decay(base_lr, adjusted_step)
        
        elif self.config.decay_strategy == DecayStrategy.STEP:
            return self._step_decay(base_lr, adjusted_step)
        
        elif self.config.decay_strategy == DecayStrategy.CYCLICAL:
            return self._cyclical_decay(base_lr, adjusted_step)
        
        else:
            return base_lr
    
    def _linear_decay(self, base_lr: float, step: int) -> float:
        """Linear decay from base_lr to min_lr."""
        progress = min(step / self.config.total_steps, 1.0)
        return base_lr - progress * (base_lr - self.config.min_lr)
    
    def _cosine_decay(self, base_lr: float, step: int) -> float:
        """Cosine annealing decay."""
        progress = min(step / self.config.total_steps, 1.0)
        cosine_decay = 0.5 * (1 + math.cos(math.pi * progress))
        return self.config.min_lr + (base_lr - self.config.min_lr) * cosine_decay
    
    def _exponential_decay(self, base_lr: float, step: int) -> float:
        """Exponential decay."""
        decay_factor = self.config.decay_rate ** (step // self.config.decay_steps)
        return max(base_lr * decay_factor, self.config.min_lr)
    
    def _polynomial_decay(self, base_lr: float, step: int) -> float:
        """Polynomial decay."""
        progress = min(step / self.config.total_steps, 1.0)
        decay_factor = (1 - progress) ** self.config.polynomial_power
        return self.config.min_lr + (base_lr - self.config.min_lr) * decay_factor
    
    def _step_decay(self, base_lr: float, step: int) -> float:
        """Step decay - reduce LR at specific intervals."""
        decay_factor = self.config.decay_rate ** (step // self.config.decay_steps)
        return max(base_lr * decay_factor, self.config.min_lr)
    
    def _cyclical_decay(self, base_lr: float, step: int) -> float:
        """Cyclical learning rate with decay."""
        # Current position in cycle
        cycle_progress = (step % self.config.cycle_length) / self.config.cycle_length
        
        # Which cycle we're in
        cycle_num = step // self.config.cycle_length
        
        # Decay the maximum LR for this cycle
        cycle_max_lr = base_lr * (self.config.cycle_decay ** cycle_num)
        cycle_min_lr = max(cycle_max_lr * 0.1, self.config.min_lr)
        
        # Cosine within the cycle
        cosine_factor = 0.5 * (1 + math.cos(math.pi * cycle_progress))
        return cycle_min_lr + (cycle_max_lr - cycle_min_lr) * cosine_factor
    
    def _update_metrics(self, metrics: Dict[str, float]):
        """Update metrics for adaptive scheduling decisions."""
        target_metric = metrics.get(self.config.target_metric)
        if target_metric is None:
            return
        
        self.metric_history.append(target_metric)
        
        # Check for plateau if metric-based reduction is enabled
        if self.config.metric_based_reduction and self.cooldown_counter <= 0:
            if target_metric < self.best_metric - self.config.plateau_threshold:
                self.best_metric = target_metric
                self.plateau_count = 0
            else:
                self.plateau_count += 1
                
                if self.plateau_count >= self.config.plateau_patience:
                    self._reduce_lr_on_plateau()
                    self.plateau_count = 0
                    self.cooldown_counter = self.config.plateau_cooldown
        
        if self.cooldown_counter > 0:
            self.cooldown_counter -= 1
    
    def _reduce_lr_on_plateau(self):
        """Reduce learning rate when plateau is detected."""
        for param_group in self.optimizer.param_groups:
            old_lr = param_group['lr']
            new_lr = max(old_lr * self.config.plateau_factor, self.config.min_lr)
            param_group['lr'] = new_lr
            
        logger.info(f"Plateau detected - reducing LR by factor {self.config.plateau_factor}")
    
    def get_lr_statistics(self) -> Dict[str, Any]:
        """Get comprehensive statistics about learning rate evolution."""
        if not self.lr_history:
            return {}
        
        return {
            "current_lr": self.lr_history[-1] if self.lr_history else 0,
            "max_lr": max(self.lr_history),
            "min_lr": min(self.lr_history),
            "mean_lr": np.mean(self.lr_history),
            "lr_reductions": sum(1 for i in range(1, len(self.lr_history)) 
                               if self.lr_history[i] < self.lr_history[i-1] * 0.95),
            "plateau_count": self.plateau_count,
            "cycle_number": self.cycle_number,
            "step": self.current_step,
            "epoch": self.current_epoch
        }
    
    def state_dict(self):
        """Return state dictionary for checkpointing."""
        state = super().state_dict()
        state.update({
            'current_step': self.current_step,
            'current_epoch': self.current_epoch,
            'plateau_count': self.plateau_count,
            'best_metric': self.best_metric,
            'cooldown_counter': self.cooldown_counter,
            'cycle_number': self.cycle_number,
            'metric_history': self.metric_history[-100:],  # Keep last 100
            'lr_history': self.lr_history[-1000:]  # Keep last 1000
        })
        return state
    
    def load_state_dict(self, state_dict):
        """Load state dictionary from checkpoint."""
        # Load scheduler state
        self.current_step = state_dict.pop('current_step', 0)
        self.current_epoch = state_dict.pop('current_epoch', 0)
        self.plateau_count = state_dict.pop('plateau_count', 0)
        self.best_metric = state_dict.pop('best_metric', float('inf'))
        self.cooldown_counter = state_dict.pop('cooldown_counter', 0)
        self.cycle_number = state_dict.pop('cycle_number', 0)
        self.metric_history = state_dict.pop('metric_history', [])
        self.lr_history = state_dict.pop('lr_history', [])
        
        # Load base scheduler state
        super().load_state_dict(state_dict)


class StochasticWeightAveraging:
    """
    Stochastic Weight Averaging implementation for better generalization.
    
    Averages model weights over the last few epochs of training to achieve
    better convergence and generalization.
    """
    
    def __init__(self, 
                 model: torch.nn.Module,
                 swa_start_epoch: int = 10,
                 swa_lr: float = 1e-5,
                 swa_freq: int = 1,
                 device: torch.device = None):
        
        self.model = model
        self.swa_start_epoch = swa_start_epoch
        self.swa_lr = swa_lr
        self.swa_freq = swa_freq
        self.device = device or next(model.parameters()).device
        
        # Initialize SWA model
        self.swa_model = self._create_swa_model()
        self.swa_count = 0
        self.is_active = False
        
        logger.info(f"Initialized SWA: start_epoch={swa_start_epoch}, "
                   f"lr={swa_lr}, freq={swa_freq}")
    
    def _create_swa_model(self):
        """Create a copy of the model for SWA averaging."""
        import copy
        
        # Create a deep copy of the model
        swa_model = copy.deepcopy(self.model)
        swa_model.to(self.device)
        
        return swa_model
    
    def update(self, epoch: int, model: torch.nn.Module = None):
        """Update SWA model if conditions are met."""
        if epoch < self.swa_start_epoch:
            return
        
        if not self.is_active:
            self.is_active = True
            logger.info(f"Starting SWA at epoch {epoch}")
        
        if (epoch - self.swa_start_epoch) % self.swa_freq == 0:
            self._update_swa_weights(model or self.model)
            self.swa_count += 1
            
            if self.swa_count % 5 == 0:
                logger.debug(f"SWA update #{self.swa_count} at epoch {epoch}")
    
    def _update_swa_weights(self, model: torch.nn.Module):
        """Update SWA weights with current model weights."""
        with torch.no_grad():
            for swa_param, model_param in zip(self.swa_model.parameters(), model.parameters()):
                if self.swa_count == 0:
                    # First update - just copy
                    swa_param.data.copy_(model_param.data)
                else:
                    # Running average: new_avg = old_avg + (new_val - old_avg) / count
                    swa_param.data.add_(
                        (model_param.data - swa_param.data) / (self.swa_count + 1)
                    )
    
    def get_swa_model(self) -> torch.nn.Module:
        """Get the SWA-averaged model."""
        if not self.is_active or self.swa_count == 0:
            logger.warning("SWA not active or no updates performed")
            return self.model
        
        return self.swa_model
    
    def finalize(self, model: torch.nn.Module = None) -> torch.nn.Module:
        """Finalize SWA and return the averaged model."""
        if not self.is_active:
            return model or self.model
        
        logger.info(f"Finalizing SWA with {self.swa_count} accumulated updates")
        return self.swa_model
    
    def state_dict(self) -> Dict[str, Any]:
        """Get state dict for checkpointing."""
        return {
            'swa_model_state': self.swa_model.state_dict() if self.is_active else None,
            'swa_count': self.swa_count,
            'is_active': self.is_active,
            'swa_start_epoch': self.swa_start_epoch
        }
    
    def load_state_dict(self, state_dict: Dict[str, Any]):
        """Load state dict from checkpoint."""
        self.swa_count = state_dict.get('swa_count', 0)
        self.is_active = state_dict.get('is_active', False)
        self.swa_start_epoch = state_dict.get('swa_start_epoch', self.swa_start_epoch)
        
        if self.is_active and state_dict.get('swa_model_state'):
            self.swa_model.load_state_dict(state_dict['swa_model_state'])


def create_lr_scheduler(optimizer: optim.Optimizer, 
                       config: Dict[str, Any]) -> AdaptiveLRScheduler:
    """
    Create learning rate scheduler with configuration.
    
    Args:
        optimizer: PyTorch optimizer
        config: Configuration dictionary
        
    Returns:
        Configured AdaptiveLRScheduler
    """
    
    # Convert config dict to LRSchedulerConfig
    scheduler_config = LRSchedulerConfig(**config)
    
    return AdaptiveLRScheduler(optimizer, scheduler_config)


def create_musical_lr_config(**kwargs) -> LRSchedulerConfig:
    """Create LR scheduler config optimized for musical training."""
    
    musical_defaults = {
        "base_lr": 1e-4,
        "max_lr": 5e-4,
        "min_lr": 1e-7,
        "warmup_strategy": WarmupStrategy.COSINE,
        "warmup_steps": 2000,
        "decay_strategy": DecayStrategy.COSINE,
        "total_steps": 100000,
        "plateau_patience": 8,
        "plateau_factor": 0.6,
        "use_adaptive": True,
        "metric_based_reduction": True,
        "target_metric": "musical_quality",
        "musical_warmup": True
    }
    
    # Override with provided kwargs
    musical_defaults.update(kwargs)
    
    return LRSchedulerConfig(**musical_defaults)