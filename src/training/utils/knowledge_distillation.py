"""
Teacher-Student Knowledge Distillation for Aurl.ai Music Generation.

This module implements sophisticated knowledge distillation strategies:
- Progressive knowledge transfer from teacher to student models
- Musical domain-specific distillation techniques
- Multi-level distillation (attention, embeddings, outputs)
- Adaptive temperature scheduling
- Musical quality preservation during distillation
- Ensemble teacher distillation

Designed for efficient music generation model compression and knowledge transfer.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import math
from collections import defaultdict

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class DistillationStrategy(Enum):
    """Available knowledge distillation strategies."""
    OUTPUT_DISTILLATION = "output_distillation"
    FEATURE_DISTILLATION = "feature_distillation"
    ATTENTION_DISTILLATION = "attention_distillation"
    PROGRESSIVE_DISTILLATION = "progressive_distillation"
    MUSICAL_DISTILLATION = "musical_distillation"
    ENSEMBLE_DISTILLATION = "ensemble_distillation"


class DistillationLevel(Enum):
    """Levels at which distillation can be applied."""
    OUTPUT_ONLY = "output_only"
    INTERMEDIATE_FEATURES = "intermediate_features"
    ATTENTION_MAPS = "attention_maps"
    EMBEDDINGS = "embeddings"
    FULL_HIERARCHY = "full_hierarchy"


@dataclass
class DistillationConfig:
    """Configuration for knowledge distillation."""
    
    # Strategy configuration
    strategy: DistillationStrategy = DistillationStrategy.MUSICAL_DISTILLATION
    distillation_levels: List[DistillationLevel] = field(default_factory=lambda: [
        DistillationLevel.OUTPUT_ONLY,
        DistillationLevel.ATTENTION_MAPS
    ])
    
    # Temperature scheduling
    temperature: float = 4.0
    min_temperature: float = 1.0
    max_temperature: float = 8.0
    temperature_schedule: str = "cosine"  # linear, cosine, exponential, adaptive
    temperature_epochs: int = 50
    
    # Loss weighting
    distillation_weight: float = 0.7
    student_loss_weight: float = 0.3
    feature_loss_weight: float = 0.5
    attention_loss_weight: float = 0.3
    
    # Musical distillation parameters
    musical_quality_weight: float = 0.4
    harmonic_distillation_weight: float = 0.3
    rhythmic_distillation_weight: float = 0.3
    
    # Progressive distillation
    enable_progressive: bool = True
    progressive_stages: int = 3
    stage_epochs: int = 20
    
    # Feature matching
    feature_matching_layers: List[str] = field(default_factory=lambda: [
        "encoder.layers.2", "encoder.layers.4", "decoder.layers.2"
    ])
    
    # Attention distillation
    attention_layer_names: List[str] = field(default_factory=lambda: [
        "encoder.attention", "decoder.self_attention", "decoder.cross_attention"
    ])
    
    # Ensemble distillation
    enable_ensemble_teachers: bool = False
    teacher_ensemble_size: int = 3
    ensemble_weighting: str = "equal"  # equal, performance, adaptive
    
    # Adaptive parameters
    enable_adaptive_weighting: bool = True
    adaptation_frequency: int = 10  # epochs
    performance_threshold: float = 0.05


@dataclass
class DistillationState:
    """Current state of knowledge distillation."""
    
    epoch: int = 0
    stage: int = 0
    current_temperature: float = 4.0
    current_weights: Dict[str, float] = field(default_factory=lambda: {
        "distillation": 0.7, "student": 0.3, "feature": 0.5, "attention": 0.3
    })
    
    # Performance tracking
    teacher_performance: List[float] = field(default_factory=list)
    student_performance: List[float] = field(default_factory=list)
    distillation_gap: List[float] = field(default_factory=list)
    
    # Progressive distillation state
    active_levels: List[DistillationLevel] = field(default_factory=lambda: [DistillationLevel.OUTPUT_ONLY])
    stage_start_epoch: int = 0
    
    # Adaptation history
    adaptations: List[Dict[str, Any]] = field(default_factory=list)
    last_adaptation_epoch: int = 0


class MusicalDistillationLoss:
    """Musical domain-specific distillation loss functions."""
    
    def __init__(self, config: DistillationConfig):
        self.config = config
        
    def compute_musical_distillation_loss(self,
                                        teacher_output: Dict[str, torch.Tensor],
                                        student_output: Dict[str, torch.Tensor],
                                        targets: torch.Tensor,
                                        temperature: float) -> Dict[str, torch.Tensor]:
        """Compute musical domain-specific distillation losses."""
        
        losses = {}
        
        # Standard output distillation
        if "logits" in teacher_output and "logits" in student_output:
            output_loss = self._compute_output_distillation(
                teacher_output["logits"], student_output["logits"], temperature
            )
            losses["output_distillation"] = output_loss
        
        # Harmonic distillation (focus on harmonic relationships)
        if "harmonic_features" in teacher_output and "harmonic_features" in student_output:
            harmonic_loss = self._compute_harmonic_distillation(
                teacher_output["harmonic_features"], student_output["harmonic_features"]
            )
            losses["harmonic_distillation"] = harmonic_loss
        
        # Rhythmic distillation (focus on temporal patterns)
        if "rhythmic_features" in teacher_output and "rhythmic_features" in student_output:
            rhythmic_loss = self._compute_rhythmic_distillation(
                teacher_output["rhythmic_features"], student_output["rhythmic_features"]
            )
            losses["rhythmic_distillation"] = rhythmic_loss
        
        # Musical structure distillation
        if "structure_features" in teacher_output and "structure_features" in student_output:
            structure_loss = self._compute_structure_distillation(
                teacher_output["structure_features"], student_output["structure_features"]
            )
            losses["structure_distillation"] = structure_loss
        
        # Combine losses
        total_loss = torch.tensor(0.0, device=student_output["logits"].device)
        
        if "output_distillation" in losses:
            total_loss += losses["output_distillation"]
        
        if "harmonic_distillation" in losses:
            total_loss += self.config.harmonic_distillation_weight * losses["harmonic_distillation"]
        
        if "rhythmic_distillation" in losses:
            total_loss += self.config.rhythmic_distillation_weight * losses["rhythmic_distillation"]
        
        losses["total_musical_distillation"] = total_loss
        
        return losses
    
    def _compute_output_distillation(self,
                                   teacher_logits: torch.Tensor,
                                   student_logits: torch.Tensor,
                                   temperature: float) -> torch.Tensor:
        """Compute standard output distillation loss."""
        
        # Apply temperature scaling
        teacher_probs = F.softmax(teacher_logits / temperature, dim=-1)
        student_log_probs = F.log_softmax(student_logits / temperature, dim=-1)
        
        # KL divergence loss
        kl_loss = F.kl_div(student_log_probs, teacher_probs, reduction='batchmean')
        
        # Scale by temperature^2 (standard practice)
        return kl_loss * (temperature ** 2)
    
    def _compute_harmonic_distillation(self,
                                     teacher_features: torch.Tensor,
                                     student_features: torch.Tensor) -> torch.Tensor:
        """Compute harmonic-focused distillation loss."""
        
        # Focus on harmonic relationships - use cosine similarity loss
        teacher_norm = F.normalize(teacher_features, p=2, dim=-1)
        student_norm = F.normalize(student_features, p=2, dim=-1)
        
        # Cosine similarity loss (1 - cosine_similarity)
        cosine_sim = F.cosine_similarity(teacher_norm, student_norm, dim=-1)
        harmonic_loss = 1.0 - cosine_sim.mean()
        
        return harmonic_loss
    
    def _compute_rhythmic_distillation(self,
                                     teacher_features: torch.Tensor,
                                     student_features: torch.Tensor) -> torch.Tensor:
        """Compute rhythm-focused distillation loss."""
        
        # Focus on temporal patterns - use MSE on sequence features
        batch_size, seq_len, feature_dim = teacher_features.shape
        
        # Calculate temporal differences (rhythm patterns)
        teacher_temporal = teacher_features[:, 1:] - teacher_features[:, :-1]
        student_temporal = student_features[:, 1:] - student_features[:, :-1]
        
        # MSE loss on temporal differences
        rhythmic_loss = F.mse_loss(student_temporal, teacher_temporal)
        
        return rhythmic_loss
    
    def _compute_structure_distillation(self,
                                      teacher_features: torch.Tensor,
                                      student_features: torch.Tensor) -> torch.Tensor:
        """Compute musical structure distillation loss."""
        
        # Focus on long-range dependencies and structure
        # Use attention-like mechanism to capture structural relationships
        
        batch_size, seq_len, feature_dim = teacher_features.shape
        
        # Compute self-attention patterns for both teacher and student
        teacher_attn = torch.bmm(teacher_features, teacher_features.transpose(1, 2))
        student_attn = torch.bmm(student_features, student_features.transpose(1, 2))
        
        # Normalize attention patterns
        teacher_attn = F.softmax(teacher_attn / math.sqrt(feature_dim), dim=-1)
        student_attn = F.softmax(student_attn / math.sqrt(feature_dim), dim=-1)
        
        # MSE loss on attention patterns
        structure_loss = F.mse_loss(student_attn, teacher_attn)
        
        return structure_loss


class FeatureDistillationLoss:
    """Feature-level distillation loss functions."""
    
    def __init__(self):
        pass
    
    def compute_feature_distillation_loss(self,
                                        teacher_features: Dict[str, torch.Tensor],
                                        student_features: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute feature-level distillation losses."""
        
        losses = {}
        
        for layer_name in teacher_features:
            if layer_name in student_features:
                teacher_feat = teacher_features[layer_name]
                student_feat = student_features[layer_name]
                
                # Adapt dimensions if necessary
                student_feat_adapted = self._adapt_feature_dimensions(
                    student_feat, teacher_feat.shape
                )
                
                # Compute feature matching loss
                feature_loss = F.mse_loss(student_feat_adapted, teacher_feat)
                losses[f"feature_{layer_name}"] = feature_loss
        
        # Combine feature losses
        if losses:
            total_feature_loss = sum(losses.values()) / len(losses)
            losses["total_feature_distillation"] = total_feature_loss
        
        return losses
    
    def _adapt_feature_dimensions(self,
                                student_features: torch.Tensor,
                                target_shape: torch.Size) -> torch.Tensor:
        """Adapt student feature dimensions to match teacher."""
        
        if student_features.shape == target_shape:
            return student_features
        
        # Handle different feature dimensions through linear projection
        if len(student_features.shape) == 3:  # [batch, seq, dim]
            batch_size, seq_len, student_dim = student_features.shape
            target_dim = target_shape[-1]
            
            if student_dim != target_dim:
                # Project to target dimension
                projection = nn.Linear(student_dim, target_dim).to(student_features.device)
                student_features = projection(student_features)
        
        return student_features


class AttentionDistillationLoss:
    """Attention-level distillation loss functions."""
    
    def __init__(self):
        pass
    
    def compute_attention_distillation_loss(self,
                                          teacher_attention: Dict[str, torch.Tensor],
                                          student_attention: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute attention-level distillation losses."""
        
        losses = {}
        
        for layer_name in teacher_attention:
            if layer_name in student_attention:
                teacher_attn = teacher_attention[layer_name]
                student_attn = student_attention[layer_name]
                
                # Adapt attention dimensions if necessary
                student_attn_adapted = self._adapt_attention_dimensions(
                    student_attn, teacher_attn.shape
                )
                
                # Compute attention matching loss
                attention_loss = self._compute_attention_matching_loss(
                    teacher_attn, student_attn_adapted
                )
                losses[f"attention_{layer_name}"] = attention_loss
        
        # Combine attention losses
        if losses:
            total_attention_loss = sum(losses.values()) / len(losses)
            losses["total_attention_distillation"] = total_attention_loss
        
        return losses
    
    def _adapt_attention_dimensions(self,
                                  student_attention: torch.Tensor,
                                  target_shape: torch.Size) -> torch.Tensor:
        """Adapt student attention dimensions to match teacher."""
        
        if student_attention.shape == target_shape:
            return student_attention
        
        # Handle different attention head dimensions
        if len(student_attention.shape) == 4:  # [batch, heads, seq, seq]
            batch_size, student_heads, seq_len, _ = student_attention.shape
            target_heads = target_shape[1]
            
            if student_heads != target_heads:
                # Average or interpolate heads
                if student_heads > target_heads:
                    # Average groups of heads
                    heads_per_group = student_heads // target_heads
                    student_attention = student_attention.view(
                        batch_size, target_heads, heads_per_group, seq_len, seq_len
                    ).mean(dim=2)
                else:
                    # Repeat heads
                    repeat_factor = target_heads // student_heads
                    student_attention = student_attention.repeat(1, repeat_factor, 1, 1)
        
        return student_attention
    
    def _compute_attention_matching_loss(self,
                                       teacher_attention: torch.Tensor,
                                       student_attention: torch.Tensor) -> torch.Tensor:
        """Compute attention matching loss."""
        
        # Use MSE loss on attention patterns
        attention_loss = F.mse_loss(student_attention, teacher_attention)
        
        return attention_loss


class KnowledgeDistillationTrainer:
    """
    Comprehensive knowledge distillation trainer for music generation models.
    
    Features:
    - Multiple distillation strategies
    - Progressive knowledge transfer
    - Musical domain-specific distillation
    - Adaptive temperature and weight scheduling
    - Multi-level distillation (output, feature, attention)
    """
    
    def __init__(self,
                 teacher_model: nn.Module,
                 student_model: nn.Module,
                 config: DistillationConfig):
        """
        Initialize knowledge distillation trainer.
        
        Args:
            teacher_model: Pre-trained teacher model
            student_model: Student model to train
            config: Distillation configuration
        """
        
        self.teacher_model = teacher_model
        self.student_model = student_model
        self.config = config
        self.state = DistillationState()
        
        # Initialize loss functions
        self.musical_loss_fn = MusicalDistillationLoss(config)
        self.feature_loss_fn = FeatureDistillationLoss()
        self.attention_loss_fn = AttentionDistillationLoss()
        
        # Set initial state
        self.state.current_temperature = config.temperature
        self.state.active_levels = [DistillationLevel.OUTPUT_ONLY]
        
        # Prepare teacher model
        self.teacher_model.eval()
        for param in self.teacher_model.parameters():
            param.requires_grad = False
        
        logger.info(f"Initialized knowledge distillation with strategy: {config.strategy}")
        logger.info(f"Teacher model parameters: {sum(p.numel() for p in teacher_model.parameters()):,}")
        logger.info(f"Student model parameters: {sum(p.numel() for p in student_model.parameters()):,}")
    
    def distillation_forward(self,
                           batch: Dict[str, torch.Tensor],
                           return_features: bool = True) -> Dict[str, Any]:
        """
        Forward pass through both teacher and student models.
        
        Args:
            batch: Input batch
            return_features: Whether to return intermediate features
            
        Returns:
            Dictionary containing teacher and student outputs and features
        """
        
        # Teacher forward pass (no gradients)
        with torch.no_grad():
            teacher_outputs = self._forward_with_features(
                self.teacher_model, batch, return_features
            )
        
        # Student forward pass (with gradients)
        student_outputs = self._forward_with_features(
            self.student_model, batch, return_features
        )
        
        return {
            "teacher": teacher_outputs,
            "student": student_outputs,
            "inputs": batch
        }
    
    def compute_distillation_loss(self,
                                forward_outputs: Dict[str, Any],
                                targets: torch.Tensor = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive distillation loss.
        
        Args:
            forward_outputs: Outputs from distillation_forward
            targets: Ground truth targets
            
        Returns:
            Dictionary of computed losses
        """
        
        teacher_outputs = forward_outputs["teacher"]
        student_outputs = forward_outputs["student"]
        
        losses = {}
        
        # Musical distillation loss
        if self.config.strategy == DistillationStrategy.MUSICAL_DISTILLATION:
            musical_losses = self.musical_loss_fn.compute_musical_distillation_loss(
                teacher_outputs, student_outputs, targets, self.state.current_temperature
            )
            losses.update(musical_losses)
        
        # Standard output distillation
        elif DistillationLevel.OUTPUT_ONLY in self.state.active_levels:
            if "logits" in teacher_outputs and "logits" in student_outputs:
                output_loss = self.musical_loss_fn._compute_output_distillation(
                    teacher_outputs["logits"], student_outputs["logits"], 
                    self.state.current_temperature
                )
                losses["output_distillation"] = output_loss
        
        # Feature distillation
        if DistillationLevel.INTERMEDIATE_FEATURES in self.state.active_levels:
            if "features" in teacher_outputs and "features" in student_outputs:
                feature_losses = self.feature_loss_fn.compute_feature_distillation_loss(
                    teacher_outputs["features"], student_outputs["features"]
                )
                losses.update(feature_losses)
        
        # Attention distillation
        if DistillationLevel.ATTENTION_MAPS in self.state.active_levels:
            if "attention" in teacher_outputs and "attention" in student_outputs:
                attention_losses = self.attention_loss_fn.compute_attention_distillation_loss(
                    teacher_outputs["attention"], student_outputs["attention"]
                )
                losses.update(attention_losses)
        
        # Student task loss (if targets provided)
        if targets is not None and "logits" in student_outputs:
            student_task_loss = F.cross_entropy(
                student_outputs["logits"].view(-1, student_outputs["logits"].size(-1)),
                targets.view(-1),
                ignore_index=-1
            )
            losses["student_task_loss"] = student_task_loss
        
        # Combine losses with current weights
        total_loss = self._combine_losses(losses)
        losses["total_distillation_loss"] = total_loss
        
        return losses
    
    def update_distillation_schedule(self, epoch: int, performance_metrics: Dict[str, float] = None):
        """Update distillation schedule (temperature, weights, active levels)."""
        
        self.state.epoch = epoch
        
        # Update temperature
        self._update_temperature_schedule(epoch)
        
        # Update progressive distillation levels
        if self.config.enable_progressive:
            self._update_progressive_distillation(epoch)
        
        # Update adaptive weights
        if self.config.enable_adaptive_weighting and performance_metrics:
            self._update_adaptive_weights(epoch, performance_metrics)
        
        # Log updates
        self._log_distillation_update()
    
    def _forward_with_features(self,
                             model: nn.Module,
                             batch: Dict[str, torch.Tensor],
                             return_features: bool) -> Dict[str, torch.Tensor]:
        """Forward pass that captures intermediate features."""
        
        # This is a simplified version - in practice, you'd need to modify
        # your model to return intermediate features
        
        if hasattr(model, 'forward_with_features') and return_features:
            return model.forward_with_features(batch)
        else:
            # Standard forward pass
            outputs = model(batch)
            
            # Convert to expected format
            if isinstance(outputs, torch.Tensor):
                return {"logits": outputs}
            elif isinstance(outputs, dict):
                return outputs
            else:
                return {"logits": outputs[0] if isinstance(outputs, tuple) else outputs}
    
    def _update_temperature_schedule(self, epoch: int):
        """Update temperature based on schedule."""
        
        if self.config.temperature_schedule == "linear":
            progress = min(epoch / self.config.temperature_epochs, 1.0)
            self.state.current_temperature = (
                self.config.temperature - progress * 
                (self.config.temperature - self.config.min_temperature)
            )
        
        elif self.config.temperature_schedule == "cosine":
            progress = min(epoch / self.config.temperature_epochs, 1.0)
            cosine_progress = 0.5 * (1 + math.cos(math.pi * progress))
            self.state.current_temperature = (
                self.config.min_temperature + 
                cosine_progress * (self.config.temperature - self.config.min_temperature)
            )
        
        elif self.config.temperature_schedule == "exponential":
            progress = min(epoch / self.config.temperature_epochs, 1.0)
            exp_progress = math.exp(-3 * progress)
            self.state.current_temperature = (
                self.config.min_temperature + 
                exp_progress * (self.config.temperature - self.config.min_temperature)
            )
        
        # Ensure temperature is within bounds
        self.state.current_temperature = max(
            self.config.min_temperature, 
            min(self.config.max_temperature, self.state.current_temperature)
        )
    
    def _update_progressive_distillation(self, epoch: int):
        """Update progressive distillation levels."""
        
        stage = min(epoch // self.config.stage_epochs, self.config.progressive_stages - 1)
        
        if stage != self.state.stage:
            self.state.stage = stage
            self.state.stage_start_epoch = epoch
            
            # Update active distillation levels based on stage
            if stage == 0:
                self.state.active_levels = [DistillationLevel.OUTPUT_ONLY]
            elif stage == 1:
                self.state.active_levels = [
                    DistillationLevel.OUTPUT_ONLY,
                    DistillationLevel.ATTENTION_MAPS
                ]
            else:  # stage >= 2
                self.state.active_levels = [
                    DistillationLevel.OUTPUT_ONLY,
                    DistillationLevel.ATTENTION_MAPS,
                    DistillationLevel.INTERMEDIATE_FEATURES
                ]
            
            logger.info(f"Advanced to distillation stage {stage}, active levels: {[l.value for l in self.state.active_levels]}")
    
    def _update_adaptive_weights(self, epoch: int, performance_metrics: Dict[str, float]):
        """Update loss weights adaptively based on performance."""
        
        # Track performance
        teacher_perf = performance_metrics.get("teacher_loss", 0.0)
        student_perf = performance_metrics.get("student_loss", 0.0)
        
        self.state.teacher_performance.append(teacher_perf)
        self.state.student_performance.append(student_perf)
        
        # Calculate performance gap
        if teacher_perf > 0:
            gap = (student_perf - teacher_perf) / teacher_perf
            self.state.distillation_gap.append(gap)
        
        # Adapt weights if needed
        if (epoch - self.state.last_adaptation_epoch >= self.config.adaptation_frequency and
            len(self.state.distillation_gap) >= 5):
            
            recent_gap = np.mean(self.state.distillation_gap[-5:])
            
            if recent_gap > self.config.performance_threshold:
                # Student is lagging, increase distillation weight
                self.state.current_weights["distillation"] = min(0.9, 
                    self.state.current_weights["distillation"] + 0.1)
                self.state.current_weights["student"] = 1.0 - self.state.current_weights["distillation"]
                
                self.state.last_adaptation_epoch = epoch
                
                adaptation_info = {
                    "epoch": epoch,
                    "action": "increase_distillation_weight",
                    "gap": recent_gap,
                    "new_weights": self.state.current_weights.copy()
                }
                self.state.adaptations.append(adaptation_info)
                
                logger.info(f"Adapted distillation weights at epoch {epoch}: {self.state.current_weights}")
    
    def _combine_losses(self, losses: Dict[str, torch.Tensor]) -> torch.Tensor:
        """Combine individual losses into total loss."""
        
        total_loss = torch.tensor(0.0, device=next(iter(losses.values())).device)
        
        # Primary distillation losses
        if "total_musical_distillation" in losses:
            total_loss += self.state.current_weights["distillation"] * losses["total_musical_distillation"]
        elif "output_distillation" in losses:
            total_loss += self.state.current_weights["distillation"] * losses["output_distillation"]
        
        # Student task loss
        if "student_task_loss" in losses:
            total_loss += self.state.current_weights["student"] * losses["student_task_loss"]
        
        # Feature distillation
        if "total_feature_distillation" in losses:
            total_loss += self.state.current_weights["feature"] * losses["total_feature_distillation"]
        
        # Attention distillation
        if "total_attention_distillation" in losses:
            total_loss += self.state.current_weights["attention"] * losses["total_attention_distillation"]
        
        return total_loss
    
    def _log_distillation_update(self):
        """Log current distillation state."""
        
        if self.state.epoch % 10 == 0:  # Log every 10 epochs
            logger.info(f"Distillation Update - Epoch {self.state.epoch}:")
            logger.info(f"  Temperature: {self.state.current_temperature:.3f}")
            logger.info(f"  Active Levels: {[l.value for l in self.state.active_levels]}")
            logger.info(f"  Loss Weights: {self.state.current_weights}")
            logger.info(f"  Stage: {self.state.stage}")
    
    def save_distillation_state(self, filepath: Path):
        """Save distillation state to file."""
        
        state_dict = {
            "config": {
                "strategy": self.config.strategy.value,
                "temperature": self.config.temperature,
                "distillation_weight": self.config.distillation_weight,
                "student_loss_weight": self.config.student_loss_weight
            },
            "state": {
                "epoch": self.state.epoch,
                "stage": self.state.stage,
                "current_temperature": self.state.current_temperature,
                "current_weights": self.state.current_weights,
                "active_levels": [l.value for l in self.state.active_levels],
                "stage_start_epoch": self.state.stage_start_epoch,
                "last_adaptation_epoch": self.state.last_adaptation_epoch,
                "adaptations": self.state.adaptations,
                "performance_gap": self.state.distillation_gap[-20:]  # Save last 20
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info(f"Saved distillation state to {filepath}")
    
    def get_distillation_summary(self) -> Dict[str, Any]:
        """Get comprehensive distillation state summary."""
        
        return {
            "epoch": self.state.epoch,
            "stage": self.state.stage,
            "strategy": self.config.strategy.value,
            "current_temperature": self.state.current_temperature,
            "active_levels": [l.value for l in self.state.active_levels],
            "current_weights": self.state.current_weights,
            "performance_gap": np.mean(self.state.distillation_gap[-10:]) if self.state.distillation_gap else 0.0,
            "adaptations_count": len(self.state.adaptations),
            "last_adaptation": self.state.last_adaptation_epoch
        }


def create_musical_distillation_config(**kwargs) -> DistillationConfig:
    """Create distillation configuration optimized for music generation."""
    
    defaults = {
        "strategy": DistillationStrategy.MUSICAL_DISTILLATION,
        "distillation_levels": [
            DistillationLevel.OUTPUT_ONLY,
            DistillationLevel.ATTENTION_MAPS
        ],
        "temperature": 4.0,
        "min_temperature": 1.5,
        "temperature_schedule": "cosine",
        "temperature_epochs": 40,
        "distillation_weight": 0.7,
        "student_loss_weight": 0.3,
        "musical_quality_weight": 0.4,
        "harmonic_distillation_weight": 0.3,
        "rhythmic_distillation_weight": 0.3,
        "enable_progressive": True,
        "progressive_stages": 3,
        "stage_epochs": 15,
        "enable_adaptive_weighting": True,
        "adaptation_frequency": 10
    }
    
    # Override with provided kwargs
    defaults.update(kwargs)
    
    return DistillationConfig(**defaults)