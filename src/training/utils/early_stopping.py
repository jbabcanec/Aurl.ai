"""
Early Stopping and Regularization System for Aurl.ai Music Generation.

This module implements comprehensive early stopping mechanisms with:
- Patience-based early stopping
- Multi-metric early stopping (reconstruction + musical quality)
- Plateau detection with automatic learning rate reduction
- Training instability detection and recovery
- Musical coherence-based stopping criteria
- Advanced regularization techniques

Designed to integrate seamlessly with our existing training infrastructure
and provide sophisticated training control for optimal model convergence.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
import time
import math
from collections import deque, defaultdict
from pathlib import Path
import json
from enum import Enum

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class StoppingCriterion(Enum):
    """Enumeration of early stopping criteria."""
    LOSS = "loss"
    MUSICAL_QUALITY = "musical_quality"
    RECONSTRUCTION = "reconstruction"
    KL_DIVERGENCE = "kl_divergence"
    ADVERSARIAL = "adversarial"
    COMBINED = "combined"


class InstabilityType(Enum):
    """Types of training instabilities that can be detected."""
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    LOSS_DIVERGENCE = "loss_divergence"
    NAN_VALUES = "nan_values"
    OSCILLATING_LOSS = "oscillating_loss"
    PLATEAU_EXCEEDED = "plateau_exceeded"
    MEMORY_OVERFLOW = "memory_overflow"


@dataclass
class EarlyStoppingConfig:
    """Configuration for early stopping behavior."""
    
    # Basic early stopping
    patience: int = 10
    min_delta: float = 1e-4
    restore_best_weights: bool = True
    
    # Multi-metric stopping
    primary_metric: str = "total_loss"
    secondary_metrics: List[str] = field(default_factory=lambda: ["reconstruction_loss", "musical_quality"])
    metric_weights: Dict[str, float] = field(default_factory=lambda: {
        "total_loss": 0.4,
        "reconstruction_loss": 0.3,
        "musical_quality": 0.3
    })
    
    # Plateau detection
    plateau_patience: int = 5
    plateau_threshold: float = 1e-3
    lr_reduction_factor: float = 0.5
    min_lr: float = 1e-7
    
    # Instability detection
    detect_instability: bool = True
    gradient_threshold: float = 10.0
    loss_spike_threshold: float = 2.0
    nan_tolerance: int = 3
    
    # Musical quality thresholds
    min_musical_quality: float = 0.3
    musical_quality_patience: int = 20
    
    # Recovery mechanisms
    enable_recovery: bool = True
    max_recovery_attempts: int = 3
    recovery_lr_factor: float = 0.1


@dataclass
class TrainingState:
    """Current state of training for early stopping decisions."""
    
    epoch: int = 0
    step: int = 0
    best_metric: float = float('inf')
    best_epoch: int = 0
    patience_counter: int = 0
    plateau_counter: int = 0
    instability_count: int = 0
    recovery_attempts: int = 0
    
    # Metric history
    metric_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    gradient_norms: deque = field(default_factory=lambda: deque(maxlen=100))
    loss_history: deque = field(default_factory=lambda: deque(maxlen=50))
    
    # State flags
    should_stop: bool = False
    should_reduce_lr: bool = False
    needs_recovery: bool = False
    instability_type: Optional[InstabilityType] = None


class MetricTracker:
    """Tracks and analyzes training metrics for early stopping decisions."""
    
    def __init__(self, window_size: int = 10):
        self.window_size = window_size
        self.metrics = defaultdict(lambda: deque(maxlen=window_size))
        self.best_values = {}
        self.timestamps = deque(maxlen=window_size)
    
    def update(self, metrics: Dict[str, float], timestamp: float = None):
        """Update metrics with new values."""
        if timestamp is None:
            timestamp = time.time()
        
        self.timestamps.append(timestamp)
        
        for name, value in metrics.items():
            if not math.isnan(value) and not math.isinf(value):
                self.metrics[name].append(value)
                
                # Track best values (assuming lower is better)
                if name not in self.best_values or value < self.best_values[name]:
                    self.best_values[name] = value
            else:
                logger.warning(f"Invalid metric value for {name}: {value}")
    
    def get_trend(self, metric_name: str, window: int = None) -> str:
        """Analyze trend for a specific metric."""
        if metric_name not in self.metrics:
            return "unknown"
        
        values = list(self.metrics[metric_name])
        if len(values) < 3:
            return "insufficient_data"
        
        window = window or min(len(values), self.window_size // 2)
        recent = values[-window:]
        
        if len(recent) < 2:
            return "insufficient_data"
        
        # Simple linear trend analysis
        x = np.arange(len(recent))
        trend = np.polyfit(x, recent, 1)[0]
        
        if abs(trend) < 1e-6:
            return "stable"
        elif trend > 0:
            return "increasing"
        else:
            return "decreasing"
    
    def is_plateaued(self, metric_name: str, threshold: float = 1e-3, window: int = 5) -> bool:
        """Check if metric has plateaued."""
        if metric_name not in self.metrics or len(self.metrics[metric_name]) < window:
            return False
        
        recent_values = list(self.metrics[metric_name])[-window:]
        max_val = max(recent_values)
        min_val = min(recent_values)
        
        return (max_val - min_val) < threshold
    
    def get_volatility(self, metric_name: str, window: int = None) -> float:
        """Calculate volatility (standard deviation) of recent metric values."""
        if metric_name not in self.metrics:
            return 0.0
        
        values = list(self.metrics[metric_name])
        window = window or min(len(values), self.window_size)
        
        if len(values) < 2:
            return 0.0
        
        recent = values[-window:]
        return float(np.std(recent))


class GradientMonitor:
    """Monitors gradient norms and detects gradient-related issues."""
    
    def __init__(self, history_size: int = 100):
        self.gradient_norms = deque(maxlen=history_size)
        self.layer_norms = defaultdict(lambda: deque(maxlen=history_size))
        self.timestamps = deque(maxlen=history_size)
    
    def update(self, model: nn.Module, timestamp: float = None):
        """Update gradient norms from model parameters."""
        if timestamp is None:
            timestamp = time.time()
        
        total_norm = 0.0
        layer_norms = {}
        
        for name, param in model.named_parameters():
            if param.grad is not None:
                param_norm = param.grad.data.norm(2).item()
                total_norm += param_norm ** 2
                
                # Track layer-specific norms
                layer_name = name.split('.')[0]  # Get top-level layer name
                if layer_name not in layer_norms:
                    layer_norms[layer_name] = 0.0
                layer_norms[layer_name] += param_norm ** 2
        
        total_norm = total_norm ** 0.5
        
        self.gradient_norms.append(total_norm)
        self.timestamps.append(timestamp)
        
        for layer_name, norm_squared in layer_norms.items():
            self.layer_norms[layer_name].append(norm_squared ** 0.5)
    
    def detect_gradient_issues(self, 
                             explosion_threshold: float = 10.0,
                             vanishing_threshold: float = 1e-6) -> Optional[InstabilityType]:
        """Detect gradient explosion or vanishing."""
        if len(self.gradient_norms) < 3:
            return None
        
        recent_norm = self.gradient_norms[-1]
        avg_norm = np.mean(list(self.gradient_norms)[-10:])
        
        if recent_norm > explosion_threshold or recent_norm > 3 * avg_norm:
            return InstabilityType.GRADIENT_EXPLOSION
        elif recent_norm < vanishing_threshold:
            return InstabilityType.GRADIENT_VANISHING
        
        return None
    
    def get_gradient_stats(self) -> Dict[str, float]:
        """Get statistics about recent gradient norms."""
        if len(self.gradient_norms) == 0:
            return {}
        
        norms = list(self.gradient_norms)
        return {
            "current_norm": norms[-1],
            "mean_norm": np.mean(norms),
            "std_norm": np.std(norms),
            "max_norm": np.max(norms),
            "min_norm": np.min(norms)
        }


class InstabilityDetector:
    """Detects various types of training instabilities."""
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.loss_history = deque(maxlen=50)
        self.nan_count = 0
        self.spike_count = 0
    
    def update(self, losses: Dict[str, float], gradient_norm: float = None):
        """Update with current training state."""
        # Check for NaN values
        for name, loss in losses.items():
            if math.isnan(loss) or math.isinf(loss):
                self.nan_count += 1
                logger.warning(f"NaN/Inf detected in {name}: {loss}")
                return InstabilityType.NAN_VALUES
        
        # Reset NaN count if no issues
        self.nan_count = 0
        
        # Track total loss for spike detection
        total_loss = losses.get('total_loss', sum(losses.values()))
        self.loss_history.append(total_loss)
        
        if len(self.loss_history) >= 3:
            # Detect loss spikes
            recent_losses = list(self.loss_history)[-3:]
            if recent_losses[-1] > self.config.loss_spike_threshold * np.mean(recent_losses[:-1]):
                self.spike_count += 1
                if self.spike_count >= 3:
                    return InstabilityType.LOSS_DIVERGENCE
            else:
                self.spike_count = 0
        
        # Check for oscillating loss
        if len(self.loss_history) >= 10:
            recent = list(self.loss_history)[-10:]
            volatility = np.std(recent) / (np.mean(recent) + 1e-8)
            if volatility > 0.5:  # High volatility indicates oscillation
                return InstabilityType.OSCILLATING_LOSS
        
        return None
    
    def should_trigger_recovery(self, instability_type: InstabilityType) -> bool:
        """Determine if recovery mechanisms should be triggered."""
        critical_types = {
            InstabilityType.GRADIENT_EXPLOSION,
            InstabilityType.NAN_VALUES,
            InstabilityType.LOSS_DIVERGENCE
        }
        return instability_type in critical_types


class EarlyStopping:
    """
    Comprehensive early stopping system with multiple criteria and recovery mechanisms.
    
    Features:
    - Multi-metric early stopping
    - Plateau detection with LR reduction
    - Training instability detection
    - Musical quality-based stopping
    - Recovery mechanisms for training issues
    """
    
    def __init__(self, config: EarlyStoppingConfig):
        self.config = config
        self.state = TrainingState()
        self.metric_tracker = MetricTracker()
        self.gradient_monitor = GradientMonitor()
        self.instability_detector = InstabilityDetector(config)
        
        logger.info(f"Initialized EarlyStopping with patience={config.patience}, "
                   f"primary_metric={config.primary_metric}")
    
    def update(self, 
               metrics: Dict[str, float], 
               model: nn.Module = None,
               epoch: int = None,
               step: int = None) -> Dict[str, Any]:
        """
        Update early stopping state with new metrics.
        
        Returns:
            Dict containing stopping decision and recommendations
        """
        # Update state
        if epoch is not None:
            self.state.epoch = epoch
        if step is not None:
            self.state.step = step
        
        # Update metric tracking
        self.metric_tracker.update(metrics)
        
        # Update gradient monitoring
        if model is not None:
            self.gradient_monitor.update(model)
            gradient_stats = self.gradient_monitor.get_gradient_stats()
            if gradient_stats:
                self.state.gradient_norms.append(gradient_stats["current_norm"])
        
        # Check for instabilities
        instability = self.instability_detector.update(metrics)
        if instability:
            self.state.instability_count += 1
            self.state.instability_type = instability
            logger.warning(f"Detected training instability: {instability}")
            
            if self.instability_detector.should_trigger_recovery(instability):
                self.state.needs_recovery = True
        
        # Check gradient issues
        if model is not None:
            gradient_issue = self.gradient_monitor.detect_gradient_issues(
                self.config.gradient_threshold
            )
            if gradient_issue:
                self.state.instability_type = gradient_issue
                self.state.needs_recovery = True
        
        # Evaluate stopping criteria
        result = self._evaluate_stopping_criteria(metrics)
        
        # Update metric history for state tracking
        for name, value in metrics.items():
            self.state.metric_history[name].append(value)
        
        return result
    
    def _evaluate_stopping_criteria(self, metrics: Dict[str, float]) -> Dict[str, Any]:
        """Evaluate all stopping criteria and return decision."""
        result = {
            "should_stop": False,
            "should_reduce_lr": False,
            "reason": None,
            "best_metric": self.state.best_metric,
            "patience_counter": self.state.patience_counter,
            "recommendations": []
        }
        
        # Calculate combined metric if using multi-metric stopping
        if self.config.primary_metric == "combined":
            combined_metric = self._calculate_combined_metric(metrics)
            current_metric = combined_metric
        else:
            current_metric = metrics.get(self.config.primary_metric, float('inf'))
        
        # Check for improvement
        improved = current_metric < (self.state.best_metric - self.config.min_delta)
        
        if improved:
            self.state.best_metric = current_metric
            self.state.best_epoch = self.state.epoch
            self.state.patience_counter = 0
            self.state.plateau_counter = 0
            result["recommendations"].append("Performance improved - continuing training")
        else:
            self.state.patience_counter += 1
            self.state.plateau_counter += 1
        
        # Check main early stopping criterion
        if self.state.patience_counter >= self.config.patience:
            self.state.should_stop = True
            result["should_stop"] = True
            result["reason"] = f"No improvement for {self.config.patience} epochs"
        
        # Check plateau detection for LR reduction
        if (self.state.plateau_counter >= self.config.plateau_patience and 
            not result["should_stop"]):
            result["should_reduce_lr"] = True
            self.state.plateau_counter = 0  # Reset after LR reduction
            result["recommendations"].append("Plateau detected - reducing learning rate")
        
        # Check musical quality stopping criteria
        musical_quality = metrics.get("musical_quality", 0.0)
        if musical_quality < self.config.min_musical_quality:
            musical_patience = getattr(self.state, 'musical_quality_patience_counter', 0)
            musical_patience += 1
            self.state.musical_quality_patience_counter = musical_patience
            
            if musical_patience >= self.config.musical_quality_patience:
                self.state.should_stop = True
                result["should_stop"] = True
                result["reason"] = "Musical quality below minimum threshold"
        else:
            self.state.musical_quality_patience_counter = 0
        
        # Check for critical instabilities
        if self.state.needs_recovery and self.config.enable_recovery:
            if self.state.recovery_attempts < self.config.max_recovery_attempts:
                result["recommendations"].append(
                    f"Training instability detected: {self.state.instability_type}. "
                    f"Consider recovery measures."
                )
                self.state.recovery_attempts += 1
                self.state.needs_recovery = False
            else:
                self.state.should_stop = True
                result["should_stop"] = True
                result["reason"] = "Max recovery attempts exceeded"
        
        # Provide additional recommendations
        if self.metric_tracker.get_volatility(self.config.primary_metric) > 0.3:
            result["recommendations"].append("High metric volatility - consider regularization")
        
        if len(self.state.gradient_norms) > 0 and self.state.gradient_norms[-1] > 5.0:
            result["recommendations"].append("Large gradient norms detected - consider gradient clipping")
        
        result["best_metric"] = self.state.best_metric
        result["current_metric"] = current_metric
        result["patience_counter"] = self.state.patience_counter
        
        return result
    
    def _calculate_combined_metric(self, metrics: Dict[str, float]) -> float:
        """Calculate weighted combination of multiple metrics."""
        combined = 0.0
        total_weight = 0.0
        
        for metric_name, weight in self.config.metric_weights.items():
            if metric_name in metrics:
                combined += weight * metrics[metric_name]
                total_weight += weight
        
        return combined / max(total_weight, 1e-8)
    
    def get_state_summary(self) -> Dict[str, Any]:
        """Get comprehensive summary of current early stopping state."""
        return {
            "epoch": self.state.epoch,
            "best_metric": self.state.best_metric,
            "best_epoch": self.state.best_epoch,
            "patience_counter": self.state.patience_counter,
            "plateau_counter": self.state.plateau_counter,
            "instability_count": self.state.instability_count,
            "recovery_attempts": self.state.recovery_attempts,
            "should_stop": self.state.should_stop,
            "needs_recovery": self.state.needs_recovery,
            "instability_type": self.state.instability_type.value if self.state.instability_type else None,
            "gradient_stats": self.gradient_monitor.get_gradient_stats(),
            "metric_trends": {
                name: self.metric_tracker.get_trend(name) 
                for name in ["total_loss", "reconstruction_loss", "musical_quality"]
                if name in self.metric_tracker.metrics
            }
        }
    
    def save_state(self, filepath: Path):
        """Save early stopping state to file."""
        state_dict = {
            "config": {
                "patience": self.config.patience,
                "min_delta": self.config.min_delta,
                "primary_metric": self.config.primary_metric,
                "metric_weights": self.config.metric_weights
            },
            "state": self.get_state_summary(),
            "metric_history": {
                name: list(values) 
                for name, values in self.state.metric_history.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2, default=str)
        
        logger.info(f"Saved early stopping state to {filepath}")
    
    def load_state(self, filepath: Path):
        """Load early stopping state from file."""
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Restore critical state information
        saved_state = state_dict.get("state", {})
        self.state.best_metric = saved_state.get("best_metric", float('inf'))
        self.state.best_epoch = saved_state.get("best_epoch", 0)
        self.state.patience_counter = saved_state.get("patience_counter", 0)
        self.state.recovery_attempts = saved_state.get("recovery_attempts", 0)
        
        logger.info(f"Loaded early stopping state from {filepath}")


def create_early_stopping_config(**kwargs) -> EarlyStoppingConfig:
    """Create early stopping configuration with sensible defaults for music generation."""
    
    defaults = {
        "patience": 15,
        "min_delta": 1e-4,
        "primary_metric": "combined",
        "secondary_metrics": ["reconstruction_loss", "musical_quality", "kl_divergence"],
        "metric_weights": {
            "total_loss": 0.3,
            "reconstruction_loss": 0.3,
            "musical_quality": 0.2,
            "kl_divergence": 0.2
        },
        "plateau_patience": 7,
        "lr_reduction_factor": 0.6,
        "min_musical_quality": 0.3,
        "musical_quality_patience": 25,
        "enable_recovery": True,
        "max_recovery_attempts": 2
    }
    
    # Override with provided kwargs
    defaults.update(kwargs)
    
    return EarlyStoppingConfig(**defaults)