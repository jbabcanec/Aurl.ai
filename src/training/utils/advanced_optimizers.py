"""
Advanced Optimization Techniques for Aurl.ai Music Generation.

This module implements state-of-the-art optimizers and optimization strategies:
- Lion optimizer implementation
- Enhanced AdamW variants
- Sophia optimizer (second-order)
- AdaFactor for memory-efficient optimization
- Musical domain-specific optimization schedules
- Adaptive optimization strategy selection
- Optimizer ensemble techniques

Designed for optimal music generation model convergence and stability.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import Optimizer
import numpy as np
import math
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
from collections import defaultdict, deque

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class OptimizerType(Enum):
    """Available optimizer types."""
    LION = "lion"
    ADAMW = "adamw"
    ADAMW_ENHANCED = "adamw_enhanced"
    SOPHIA = "sophia"
    ADAFACTOR = "adafactor"
    LAMB = "lamb"
    MUSICAL_ADAPTIVE = "musical_adaptive"


class OptimizationStrategy(Enum):
    """Optimization strategies."""
    STANDARD = "standard"
    LAYERWISE_ADAPTIVE = "layerwise_adaptive"
    MUSICAL_DOMAIN = "musical_domain"
    ADAPTIVE_ENSEMBLE = "adaptive_ensemble"
    PROGRESSIVE = "progressive"


@dataclass
class OptimizerConfig:
    """Configuration for advanced optimizers."""
    
    # Optimizer selection
    optimizer_type: OptimizerType = OptimizerType.LION
    strategy: OptimizationStrategy = OptimizationStrategy.MUSICAL_DOMAIN
    
    # Learning rate configuration
    learning_rate: float = 1e-4
    min_lr: float = 1e-7
    max_lr: float = 1e-3
    
    # Lion parameters
    lion_beta1: float = 0.9
    lion_beta2: float = 0.99
    lion_weight_decay: float = 0.01
    
    # AdamW parameters
    adamw_beta1: float = 0.9
    adamw_beta2: float = 0.999
    adamw_eps: float = 1e-8
    adamw_weight_decay: float = 0.01
    adamw_amsgrad: bool = False
    
    # Enhanced AdamW parameters
    enhanced_rectify: bool = True
    enhanced_lookahead: bool = True
    lookahead_k: int = 5
    lookahead_alpha: float = 0.5
    
    # Sophia parameters (second-order)
    sophia_beta1: float = 0.965
    sophia_beta2: float = 0.99
    sophia_eps: float = 1e-4
    sophia_clip_threshold: float = 1.0
    sophia_update_period: int = 10
    
    # AdaFactor parameters
    adafactor_eps2: float = 1e-30
    adafactor_decay_rate: float = -0.8
    adafactor_beta1: Optional[float] = None
    adafactor_weight_decay: float = 0.0
    adafactor_scale_parameter: bool = True
    adafactor_relative_step: bool = True
    
    # Musical domain optimization
    musical_lr_scaling: Dict[str, float] = field(default_factory=lambda: {
        "encoder": 1.0,
        "decoder": 1.2,
        "attention": 0.8,
        "embedding": 0.5,
        "output": 1.5
    })
    
    # Layerwise adaptation
    enable_layerwise_adaptation: bool = True
    layer_decay_factor: float = 0.8
    
    # Adaptive ensemble
    enable_optimizer_ensemble: bool = False
    ensemble_optimizers: List[OptimizerType] = field(default_factory=lambda: [
        OptimizerType.LION, OptimizerType.ADAMW_ENHANCED
    ])
    ensemble_weights: List[float] = field(default_factory=lambda: [0.7, 0.3])
    
    # Progressive optimization
    enable_progressive_optimization: bool = True
    progressive_stages: List[Dict[str, Any]] = field(default_factory=lambda: [
        {"epochs": 20, "optimizer": OptimizerType.ADAMW, "lr_factor": 1.0},
        {"epochs": 30, "optimizer": OptimizerType.LION, "lr_factor": 0.8},
        {"epochs": -1, "optimizer": OptimizerType.ADAMW_ENHANCED, "lr_factor": 0.6}
    ])
    
    # Gradient processing
    gradient_clipping: float = 1.0
    gradient_centralization: bool = True
    gradient_standardization: bool = False
    
    # Warmup and scheduling
    warmup_steps: int = 2000
    warmup_type: str = "linear"  # linear, cosine, exponential
    lr_schedule: str = "cosine"  # linear, cosine, polynomial, onecycle


class LionOptimizer(Optimizer):
    """
    Lion (EvoLved Sign Momentum) Optimizer.
    
    Reference: https://arxiv.org/abs/2302.06675
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.9, 0.99), weight_decay=0.01):
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, weight_decay=weight_decay)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg = state['exp_avg']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Lion update
                update = exp_avg.mul(beta1).add(grad, alpha=1 - beta1)
                p.add_(torch.sign(update), alpha=-group['lr'])
                
                # Update momentum
                exp_avg.mul_(beta2).add_(grad, alpha=1 - beta2)
                
                # Weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])
        
        return loss


class AdamWEnhanced(Optimizer):
    """
    Enhanced AdamW with Rectified Adam and Lookahead.
    """
    
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8,
                 weight_decay=1e-2, amsgrad=False, rectify=True,
                 lookahead=True, lookahead_k=5, lookahead_alpha=0.5):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        if not 0.0 <= weight_decay:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       amsgrad=amsgrad, rectify=rectify, lookahead=lookahead,
                       lookahead_k=lookahead_k, lookahead_alpha=lookahead_alpha)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['amsgrad']:
                        state['max_exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    if group['lookahead']:
                        state['slow_weights'] = p.data.clone()
                        state['lookahead_step'] = 0
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                if group['amsgrad']:
                    max_exp_avg_sq = state['max_exp_avg_sq']
                
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])
                
                # Exponential moving average of gradient values
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                exp_avg_sq.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                if group['amsgrad']:
                    torch.maximum(max_exp_avg_sq, exp_avg_sq, out=max_exp_avg_sq)
                    denom = (max_exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                else:
                    denom = (exp_avg_sq.sqrt() / math.sqrt(1 - beta2 ** state['step'])).add_(group['eps'])
                
                bias_correction1 = 1 - beta1 ** state['step']
                
                # Rectified Adam
                if group['rectify']:
                    # Calculate the maximum length of the approximated SMA
                    rho_inf = 2 / (1 - beta2) - 1
                    rho_t = rho_inf - 2 * state['step'] * beta2 ** state['step'] / (1 - beta2 ** state['step'])
                    
                    if rho_t > 5:
                        # Variance rectification term
                        rect_term = math.sqrt((rho_t - 4) * (rho_t - 2) * rho_inf / ((rho_inf - 4) * (rho_inf - 2) * rho_t))
                        step_size = group['lr'] * rect_term / bias_correction1
                    else:
                        step_size = group['lr'] / bias_correction1
                else:
                    step_size = group['lr'] / bias_correction1
                
                p.addcdiv_(exp_avg, denom, value=-step_size)
                
                # Lookahead
                if group['lookahead']:
                    state['lookahead_step'] += 1
                    if state['lookahead_step'] % group['lookahead_k'] == 0:
                        # Update slow weights
                        alpha = group['lookahead_alpha']
                        state['slow_weights'].add_(p.data - state['slow_weights'], alpha=alpha)
                        p.data.copy_(state['slow_weights'])
        
        return loss


class SophiaOptimizer(Optimizer):
    """
    Sophia: A Scalable Stochastic Second-order Optimizer.
    
    Reference: https://arxiv.org/abs/2305.14342
    """
    
    def __init__(self, params, lr=1e-4, betas=(0.965, 0.99), eps=1e-4,
                 weight_decay=1e-1, clip_threshold=1.0, update_period=10):
        
        if not 0.0 <= lr:
            raise ValueError(f"Invalid learning rate: {lr}")
        if not 0.0 <= eps:
            raise ValueError(f"Invalid epsilon value: {eps}")
        if not 0.0 <= betas[0] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 0: {betas[0]}")
        if not 0.0 <= betas[1] < 1.0:
            raise ValueError(f"Invalid beta parameter at index 1: {betas[1]}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, weight_decay=weight_decay,
                       clip_threshold=clip_threshold, update_period=update_period)
        super().__init__(params, defaults)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                state = self.state[p]
                
                # State initialization
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p)
                    state['hessian_diag'] = torch.zeros_like(p)
                
                exp_avg, hessian_diag = state['exp_avg'], state['hessian_diag']
                beta1, beta2 = group['betas']
                state['step'] += 1
                
                # Update biased first moment estimate
                exp_avg.mul_(beta1).add_(grad, alpha=1 - beta1)
                
                # Update Hessian diagonal estimate periodically
                if state['step'] % group['update_period'] == 0:
                    # Approximate diagonal Hessian with gradient squared
                    hessian_diag.mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
                
                # Bias correction
                bias_correction1 = 1 - beta1 ** state['step']
                bias_correction2 = 1 - beta2 ** state['step']
                
                # Compute update
                corrected_exp_avg = exp_avg / bias_correction1
                corrected_hessian = hessian_diag / bias_correction2
                
                # Clipped update
                update = corrected_exp_avg / (corrected_hessian.sqrt() + group['eps'])
                update = torch.clamp(update, -group['clip_threshold'], group['clip_threshold'])
                
                # Apply weight decay
                if group['weight_decay'] > 0:
                    p.add_(p, alpha=-group['weight_decay'] * group['lr'])
                
                # Apply update
                p.add_(update, alpha=-group['lr'])
        
        return loss


class AdaFactorOptimizer(Optimizer):
    """
    AdaFactor: Adaptive Learning Rates with Sublinear Memory Cost.
    
    Reference: https://arxiv.org/abs/1804.04235
    """
    
    def __init__(self, params, lr=None, eps2=1e-30, decay_rate=-0.8,
                 beta1=None, weight_decay=0.0, scale_parameter=True, relative_step=True):
        
        if lr is not None and lr <= 0.0:
            raise ValueError(f"Invalid learning rate: {lr}")
        if weight_decay < 0.0:
            raise ValueError(f"Invalid weight_decay value: {weight_decay}")
        
        defaults = dict(lr=lr, eps2=eps2, decay_rate=decay_rate, beta1=beta1,
                       weight_decay=weight_decay, scale_parameter=scale_parameter,
                       relative_step=relative_step)
        super().__init__(params, defaults)
    
    def _get_lr(self, param_group, param_state):
        """Compute learning rate for given parameter group and state."""
        min_step = 1e-6 * param_state['step'] if param_group['scale_parameter'] else 1e-2
        rel_step_sz = min(min_step, 1.0 / math.sqrt(param_state['step']))
        param_scale = 1.0
        if param_group['scale_parameter']:
            param_scale = max(param_group['eps2'], param_state['RMS'])
        return param_scale * rel_step_sz
    
    def _get_options(self, param_group, param_shape):
        """Determine factorization options."""
        factored = len(param_shape) >= 2
        use_first_moment = param_group['beta1'] is not None
        return factored, use_first_moment
    
    def _rms(self, tensor):
        """Root mean square."""
        return tensor.norm(2) / (tensor.numel() ** 0.5)
    
    def _approx_sq_grad(self, exp_avg_sq_row, exp_avg_sq_col):
        """Approximation of exponential moving average of square of gradient."""
        r_factor = ((exp_avg_sq_row / exp_avg_sq_row.mean(dim=-1, keepdim=True))
                   .rsqrt_().unsqueeze(-1).clamp_(0, math.inf))
        c_factor = (exp_avg_sq_col.rsqrt()).unsqueeze(0).clamp_(0, math.inf)
        return torch.mul(r_factor, c_factor)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            for p in group['params']:
                if p.grad is None:
                    continue
                
                grad = p.grad
                if grad.dtype in {torch.float16, torch.bfloat16}:
                    grad = grad.float()
                
                state = self.state[p]
                grad_shape = grad.shape
                
                factored, use_first_moment = self._get_options(group, grad_shape)
                
                # State Initialization
                if len(state) == 0:
                    state['step'] = 0
                    
                    if use_first_moment:
                        state['exp_avg'] = torch.zeros_like(grad)
                    if factored:
                        state['exp_avg_sq_row'] = torch.zeros(grad_shape[:-1])
                        state['exp_avg_sq_col'] = torch.zeros(grad_shape[:-2] + grad_shape[-1:])
                    else:
                        state['exp_avg_sq'] = torch.zeros_like(grad)
                    
                    state['RMS'] = 0
                
                state['step'] += 1
                state['RMS'] = self._rms(p.data)
                
                lr = group['lr']
                if group['lr'] is None:
                    lr = self._get_lr(group, state)
                
                beta2t = 1.0 - math.pow(state['step'], group['decay_rate'])
                update = grad**2 + group['eps2']
                
                if factored:
                    exp_avg_sq_r = state['exp_avg_sq_row']
                    exp_avg_sq_c = state['exp_avg_sq_col']
                    
                    exp_avg_sq_r.mul_(beta2t).add_(update.mean(dim=-1), alpha=1.0 - beta2t)
                    exp_avg_sq_c.mul_(beta2t).add_(update.mean(dim=-2), alpha=1.0 - beta2t)
                    update = self._approx_sq_grad(exp_avg_sq_r, exp_avg_sq_c)
                    update.mul_(grad)
                else:
                    exp_avg_sq = state['exp_avg_sq']
                    exp_avg_sq.mul_(beta2t).add_(update, alpha=1.0 - beta2t)
                    update = exp_avg_sq.rsqrt().mul_(grad)
                
                update.div_((self._rms(update) / max(1.0, self._rms(p.data))))
                
                if use_first_moment:
                    exp_avg = state['exp_avg']
                    exp_avg.mul_(group['beta1']).add_(update, alpha=1 - group['beta1'])
                    update = exp_avg
                
                if group['weight_decay'] > 0:
                    p.data.mul_(1 - group['weight_decay'] * lr)
                
                p.data.add_(update, alpha=-lr)
        
        return loss


class MusicalAdaptiveOptimizer:
    """
    Musical domain-specific adaptive optimizer that switches between optimizers
    based on training phase and musical metrics.
    """
    
    def __init__(self, model_params, config: OptimizerConfig):
        self.config = config
        self.current_optimizer = None
        self.optimizer_history = []
        self.performance_history = deque(maxlen=100)
        self.step_count = 0
        
        # Initialize optimizers
        self.optimizers = self._create_optimizers(model_params)
        self.current_optimizer_type = config.optimizer_type
        self.current_optimizer = self.optimizers[self.current_optimizer_type]
        
        logger.info(f"Initialized musical adaptive optimizer with {len(self.optimizers)} optimizers")
    
    def _create_optimizers(self, model_params) -> Dict[OptimizerType, Optimizer]:
        """Create all available optimizers."""
        
        optimizers = {}
        
        # Lion optimizer
        if OptimizerType.LION in [self.config.optimizer_type] + self.config.ensemble_optimizers:
            optimizers[OptimizerType.LION] = LionOptimizer(
                model_params,
                lr=self.config.learning_rate,
                betas=(self.config.lion_beta1, self.config.lion_beta2),
                weight_decay=self.config.lion_weight_decay
            )
        
        # Standard AdamW
        if OptimizerType.ADAMW in [self.config.optimizer_type] + self.config.ensemble_optimizers:
            optimizers[OptimizerType.ADAMW] = torch.optim.AdamW(
                model_params,
                lr=self.config.learning_rate,
                betas=(self.config.adamw_beta1, self.config.adamw_beta2),
                eps=self.config.adamw_eps,
                weight_decay=self.config.adamw_weight_decay,
                amsgrad=self.config.adamw_amsgrad
            )
        
        # Enhanced AdamW
        if OptimizerType.ADAMW_ENHANCED in [self.config.optimizer_type] + self.config.ensemble_optimizers:
            optimizers[OptimizerType.ADAMW_ENHANCED] = AdamWEnhanced(
                model_params,
                lr=self.config.learning_rate,
                betas=(self.config.adamw_beta1, self.config.adamw_beta2),
                eps=self.config.adamw_eps,
                weight_decay=self.config.adamw_weight_decay,
                rectify=self.config.enhanced_rectify,
                lookahead=self.config.enhanced_lookahead,
                lookahead_k=self.config.lookahead_k,
                lookahead_alpha=self.config.lookahead_alpha
            )
        
        # Sophia optimizer
        if OptimizerType.SOPHIA in [self.config.optimizer_type] + self.config.ensemble_optimizers:
            optimizers[OptimizerType.SOPHIA] = SophiaOptimizer(
                model_params,
                lr=self.config.learning_rate,
                betas=(self.config.sophia_beta1, self.config.sophia_beta2),
                eps=self.config.sophia_eps,
                clip_threshold=self.config.sophia_clip_threshold,
                update_period=self.config.sophia_update_period
            )
        
        # AdaFactor optimizer
        if OptimizerType.ADAFACTOR in [self.config.optimizer_type] + self.config.ensemble_optimizers:
            optimizers[OptimizerType.ADAFACTOR] = AdaFactorOptimizer(
                model_params,
                lr=self.config.learning_rate if not self.config.adafactor_relative_step else None,
                eps2=self.config.adafactor_eps2,
                decay_rate=self.config.adafactor_decay_rate,
                beta1=self.config.adafactor_beta1,
                weight_decay=self.config.adafactor_weight_decay,
                scale_parameter=self.config.adafactor_scale_parameter,
                relative_step=self.config.adafactor_relative_step
            )
        
        return optimizers
    
    def step(self, closure=None):
        """Perform optimization step."""
        
        self.step_count += 1
        
        # Ensemble optimization
        if self.config.enable_optimizer_ensemble:
            return self._ensemble_step(closure)
        else:
            return self.current_optimizer.step(closure)
    
    def _ensemble_step(self, closure=None):
        """Perform ensemble optimization step."""
        
        # Store original parameters
        original_params = {}
        for name, param in self.current_optimizer.param_groups[0]['params']:
            if hasattr(param, 'data'):
                original_params[id(param)] = param.data.clone()
        
        losses = []
        param_updates = {}
        
        # Get updates from each optimizer in ensemble
        for opt_type, weight in zip(self.config.ensemble_optimizers, self.config.ensemble_weights):
            if opt_type in self.optimizers:
                optimizer = self.optimizers[opt_type]
                
                # Reset parameters to original
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        if id(param) in original_params:
                            param.data.copy_(original_params[id(param)])
                
                # Perform step
                loss = optimizer.step(closure)
                losses.append(loss)
                
                # Store parameter updates
                for param_group in optimizer.param_groups:
                    for param in param_group['params']:
                        param_id = id(param)
                        if param_id not in param_updates:
                            param_updates[param_id] = []
                        
                        update = param.data - original_params[param_id]
                        param_updates[param_id].append((update, weight))
        
        # Apply weighted combination of updates
        for param_group in self.current_optimizer.param_groups:
            for param in param_group['params']:
                param_id = id(param)
                if param_id in param_updates and param_id in original_params:
                    # Reset to original
                    param.data.copy_(original_params[param_id])
                    
                    # Apply weighted updates
                    for update, weight in param_updates[param_id]:
                        param.data.add_(update, alpha=weight)
        
        return losses[0] if losses else None
    
    def adapt_optimizer(self, performance_metrics: Dict[str, float], epoch: int):
        """Adapt optimizer based on performance metrics."""
        
        # Track performance
        primary_metric = performance_metrics.get("total_loss", 0.0)
        self.performance_history.append(primary_metric)
        
        # Progressive optimization
        if self.config.enable_progressive_optimization:
            self._update_progressive_optimizer(epoch)
        
        # Adaptive switching based on performance
        if len(self.performance_history) >= 20:
            self._evaluate_optimizer_performance(performance_metrics)
    
    def _update_progressive_optimizer(self, epoch: int):
        """Update optimizer based on progressive stages."""
        
        cumulative_epochs = 0
        for stage in self.config.progressive_stages:
            stage_epochs = stage["epochs"]
            if stage_epochs == -1:  # Final stage
                target_optimizer = stage["optimizer"]
                break
            
            if epoch < cumulative_epochs + stage_epochs:
                target_optimizer = stage["optimizer"]
                break
            
            cumulative_epochs += stage_epochs
        
        # Switch optimizer if needed
        if target_optimizer != self.current_optimizer_type:
            self._switch_optimizer(target_optimizer)
    
    def _evaluate_optimizer_performance(self, performance_metrics: Dict[str, float]):
        """Evaluate and potentially switch optimizer based on performance."""
        
        if len(self.performance_history) < 20:
            return
        
        # Calculate performance trend
        recent_performance = list(self.performance_history)[-10:]
        older_performance = list(self.performance_history)[-20:-10]
        
        recent_avg = np.mean(recent_performance)
        older_avg = np.mean(older_performance)
        
        # If performance is degrading, consider switching
        if recent_avg > older_avg * 1.05:  # 5% degradation threshold
            self._consider_optimizer_switch(performance_metrics)
    
    def _consider_optimizer_switch(self, performance_metrics: Dict[str, float]):
        """Consider switching to a different optimizer."""
        
        # Simple heuristic for optimizer selection
        musical_quality = performance_metrics.get("musical_quality", 0.5)
        gradient_norm = performance_metrics.get("gradient_norm", 1.0)
        
        # Choose optimizer based on current training characteristics
        if gradient_norm > 5.0:  # High gradients
            target_optimizer = OptimizerType.LION  # Better gradient handling
        elif musical_quality < 0.3:  # Poor musical quality
            target_optimizer = OptimizerType.ADAMW_ENHANCED  # More stable
        elif len(self.performance_history) > 50:  # Late in training
            target_optimizer = OptimizerType.SOPHIA  # Second-order for fine-tuning
        else:
            return  # Keep current optimizer
        
        if target_optimizer != self.current_optimizer_type and target_optimizer in self.optimizers:
            self._switch_optimizer(target_optimizer)
    
    def _switch_optimizer(self, new_optimizer_type: OptimizerType):
        """Switch to a different optimizer."""
        
        logger.info(f"Switching optimizer from {self.current_optimizer_type.value} to {new_optimizer_type.value}")
        
        # Record switch
        self.optimizer_history.append({
            "step": self.step_count,
            "from": self.current_optimizer_type.value,
            "to": new_optimizer_type.value,
            "performance": list(self.performance_history)[-5:] if self.performance_history else []
        })
        
        # Switch optimizers
        self.current_optimizer_type = new_optimizer_type
        self.current_optimizer = self.optimizers[new_optimizer_type]
    
    def zero_grad(self):
        """Zero gradients for current optimizer."""
        self.current_optimizer.zero_grad()
    
    def state_dict(self):
        """Get state dictionary."""
        return {
            "current_optimizer_type": self.current_optimizer_type.value,
            "optimizer_states": {
                opt_type.value: optimizer.state_dict() 
                for opt_type, optimizer in self.optimizers.items()
            },
            "step_count": self.step_count,
            "optimizer_history": self.optimizer_history
        }
    
    def load_state_dict(self, state_dict):
        """Load state dictionary."""
        self.step_count = state_dict.get("step_count", 0)
        self.optimizer_history = state_dict.get("optimizer_history", [])
        
        # Load optimizer states
        optimizer_states = state_dict.get("optimizer_states", {})
        for opt_type_str, opt_state in optimizer_states.items():
            opt_type = OptimizerType(opt_type_str)
            if opt_type in self.optimizers:
                self.optimizers[opt_type].load_state_dict(opt_state)
        
        # Set current optimizer
        current_type_str = state_dict.get("current_optimizer_type", self.config.optimizer_type.value)
        self.current_optimizer_type = OptimizerType(current_type_str)
        self.current_optimizer = self.optimizers[self.current_optimizer_type]


def create_musical_optimizer_config(**kwargs) -> OptimizerConfig:
    """Create optimizer configuration optimized for music generation."""
    
    defaults = {
        "optimizer_type": OptimizerType.LION,
        "strategy": OptimizationStrategy.MUSICAL_DOMAIN,
        "learning_rate": 1e-4,
        "min_lr": 1e-7,
        "max_lr": 5e-4,
        
        # Lion optimized for music
        "lion_beta1": 0.9,
        "lion_beta2": 0.99,
        "lion_weight_decay": 0.01,
        
        # Musical domain scaling
        "musical_lr_scaling": {
            "encoder": 1.0,
            "decoder": 1.2,
            "attention": 0.8,
            "embedding": 0.5,
            "output": 1.5
        },
        
        "enable_layerwise_adaptation": True,
        "layer_decay_factor": 0.85,
        
        "enable_progressive_optimization": True,
        "progressive_stages": [
            {"epochs": 15, "optimizer": OptimizerType.ADAMW, "lr_factor": 1.0},
            {"epochs": 25, "optimizer": OptimizerType.LION, "lr_factor": 0.8},
            {"epochs": -1, "optimizer": OptimizerType.ADAMW_ENHANCED, "lr_factor": 0.6}
        ],
        
        "gradient_clipping": 1.0,
        "gradient_centralization": True,
        "warmup_steps": 2000,
        "warmup_type": "cosine",
        "lr_schedule": "cosine"
    }
    
    # Override with provided kwargs
    defaults.update(kwargs)
    
    return OptimizerConfig(**defaults)


def create_advanced_optimizer(model_parameters,
                            config: OptimizerConfig) -> Union[Optimizer, MusicalAdaptiveOptimizer]:
    """
    Create advanced optimizer based on configuration.
    
    Args:
        model_parameters: Model parameters to optimize
        config: Optimizer configuration
        
    Returns:
        Configured optimizer
    """
    
    if config.strategy == OptimizationStrategy.MUSICAL_DOMAIN:
        return MusicalAdaptiveOptimizer(model_parameters, config)
    else:
        # Create single optimizer
        if config.optimizer_type == OptimizerType.LION:
            return LionOptimizer(
                model_parameters,
                lr=config.learning_rate,
                betas=(config.lion_beta1, config.lion_beta2),
                weight_decay=config.lion_weight_decay
            )
        elif config.optimizer_type == OptimizerType.ADAMW_ENHANCED:
            return AdamWEnhanced(
                model_parameters,
                lr=config.learning_rate,
                betas=(config.adamw_beta1, config.adamw_beta2),
                eps=config.adamw_eps,
                weight_decay=config.adamw_weight_decay,
                rectify=config.enhanced_rectify,
                lookahead=config.enhanced_lookahead,
                lookahead_k=config.lookahead_k,
                lookahead_alpha=config.lookahead_alpha
            )
        elif config.optimizer_type == OptimizerType.SOPHIA:
            return SophiaOptimizer(
                model_parameters,
                lr=config.learning_rate,
                betas=(config.sophia_beta1, config.sophia_beta2),
                eps=config.sophia_eps,
                clip_threshold=config.sophia_clip_threshold,
                update_period=config.sophia_update_period
            )
        elif config.optimizer_type == OptimizerType.ADAFACTOR:
            return AdaFactorOptimizer(
                model_parameters,
                lr=config.learning_rate if not config.adafactor_relative_step else None,
                eps2=config.adafactor_eps2,
                decay_rate=config.adafactor_decay_rate,
                beta1=config.adafactor_beta1,
                weight_decay=config.adafactor_weight_decay,
                scale_parameter=config.adafactor_scale_parameter,
                relative_step=config.adafactor_relative_step
            )
        else:  # Default to AdamW
            return torch.optim.AdamW(
                model_parameters,
                lr=config.learning_rate,
                betas=(config.adamw_beta1, config.adamw_beta2),
                eps=config.adamw_eps,
                weight_decay=config.adamw_weight_decay,
                amsgrad=config.adamw_amsgrad
            )