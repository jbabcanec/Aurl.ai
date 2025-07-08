"""
Comprehensive loss functions for Aurl.ai music generation training.

This module implements the complete loss framework for training the enhanced
VAE-GAN architecture with:

1. Perceptual reconstruction loss with musical weighting
2. Adaptive KL divergence scheduling and annealing
3. Adversarial loss with stability techniques
4. Musical constraint losses for rhythm and harmony
5. Multi-objective loss balancing with automatic weighting
6. Loss landscape monitoring and visualization

Designed to work with the complete Aurl.ai architecture including enhanced VAE,
multi-scale discriminator, and musical intelligence features.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from typing import Dict, List, Optional, Tuple, Any, Union
import math
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt

from src.models.gan_losses import ComprehensiveGANLoss
from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class PerceptualReconstructionLoss(nn.Module):
    """
    Enhanced reconstruction loss with musical perceptual weighting.
    
    Goes beyond simple cross-entropy to weight tokens based on their
    musical importance and perceptual salience.
    """
    
    def __init__(self,
                 vocab_size: int = 774,
                 musical_weighting: bool = True,
                 perceptual_emphasis: float = 2.0,
                 note_weight: float = 3.0,
                 time_weight: float = 2.0,
                 velocity_weight: float = 1.5):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.musical_weighting = musical_weighting
        self.perceptual_emphasis = perceptual_emphasis
        
        # Musical token ranges (based on 774 vocab)
        self.note_on_start = 0       # NOTE_ON: 0-127
        self.note_off_start = 128    # NOTE_OFF: 128-255
        self.time_shift_start = 256  # TIME_SHIFT: 256-767
        self.velocity_start = 768    # VELOCITY_CHANGE: 768-773
        
        # Create perceptual importance weights
        if musical_weighting:
            self.register_buffer('token_weights', self._create_token_weights(
                note_weight, time_weight, velocity_weight
            ))
        else:
            self.register_buffer('token_weights', torch.ones(vocab_size))
        
        logger.info(f"Initialized PerceptualReconstructionLoss with musical_weighting={musical_weighting}")
    
    def _create_token_weights(self,
                            note_weight: float,
                            time_weight: float, 
                            velocity_weight: float) -> torch.Tensor:
        """Create perceptual importance weights for each token."""
        weights = torch.ones(self.vocab_size)
        
        # NOTE_ON tokens (0-127) - most important for melody/harmony
        weights[self.note_on_start:self.note_off_start] = note_weight
        
        # NOTE_OFF tokens (128-255) - important for note duration
        weights[self.note_off_start:self.time_shift_start] = note_weight * 0.7
        
        # TIME_SHIFT tokens (256-767) - crucial for rhythm
        weights[self.time_shift_start:self.velocity_start] = time_weight
        
        # VELOCITY_CHANGE tokens (768-773) - important for dynamics
        weights[self.velocity_start:] = velocity_weight
        
        # Emphasize musically critical time shifts (common note values)
        # 16th note (15.625ms * 4 = 62.5ms), 8th note (125ms), quarter note (250ms)
        critical_times = [4, 8, 16, 32]  # Relative to 15.625ms base
        for time_val in critical_times:
            if self.time_shift_start + time_val < self.velocity_start:
                weights[self.time_shift_start + time_val] *= 1.5
        
        return weights
    
    def forward(self,
                logits: torch.Tensor,
                targets: torch.Tensor,
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute perceptual reconstruction loss.
        
        Args:
            logits: Model predictions [batch_size, seq_len, vocab_size]
            targets: Target tokens [batch_size, seq_len]
            mask: Optional mask for valid positions
            
        Returns:
            Dictionary containing loss components
        """
        batch_size, seq_len, _ = logits.shape
        
        # Reshape for loss computation
        logits_flat = logits.view(-1, self.vocab_size)
        targets_flat = targets.view(-1)
        
        # Basic cross-entropy loss
        ce_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        ce_loss = ce_loss.view(batch_size, seq_len)
        
        losses = {'base_reconstruction': ce_loss.mean()}
        
        if self.musical_weighting:
            # Apply perceptual weights
            target_weights = self.token_weights[targets_flat].view(batch_size, seq_len)
            weighted_loss = ce_loss * target_weights
            losses['perceptual_reconstruction'] = weighted_loss.mean()
            
            # Musical structure emphasis
            structure_loss = self._compute_structure_loss(logits, targets)
            losses['structure_emphasis'] = structure_loss
        else:
            losses['perceptual_reconstruction'] = losses['base_reconstruction']
            losses['structure_emphasis'] = torch.tensor(0.0, device=logits.device)
        
        # Apply mask if provided
        if mask is not None:
            for key in losses:
                if isinstance(losses[key], torch.Tensor) and losses[key].dim() > 0:
                    mask_expanded = mask.view(batch_size, seq_len)
                    masked_loss = losses[key] * mask_expanded
                    losses[key] = masked_loss.sum() / mask_expanded.sum().clamp(min=1)
        
        # Combined perceptual loss
        total_loss = (losses['perceptual_reconstruction'] + 
                     0.1 * losses['structure_emphasis'])
        losses['total'] = total_loss
        
        return losses
    
    def _compute_structure_loss(self,
                              logits: torch.Tensor,
                              targets: torch.Tensor) -> torch.Tensor:
        """Compute loss that emphasizes musical structure."""
        batch_size, seq_len, vocab_size = logits.shape
        
        # Detect note onset patterns
        note_on_mask = (targets >= self.note_on_start) & (targets < self.note_off_start)
        
        if note_on_mask.sum() == 0:
            return torch.tensor(0.0, device=logits.device)
        
        # Focus loss on note onset positions
        note_positions = note_on_mask.float()
        
        # Get logits at note positions
        logits_flat = logits.view(-1, vocab_size)
        targets_flat = targets.view(-1)
        positions_flat = note_positions.view(-1)
        
        # Emphasize correct note prediction
        note_loss = F.cross_entropy(logits_flat, targets_flat, reduction='none')
        weighted_note_loss = note_loss * positions_flat * self.perceptual_emphasis
        
        return weighted_note_loss.mean()


class AdaptiveKLScheduler(nn.Module):
    """
    Adaptive KL divergence scheduling with multiple annealing strategies.
    
    Implements sophisticated β-VAE scheduling to balance reconstruction
    and disentanglement throughout training.
    """
    
    def __init__(self,
                 beta_start: float = 0.0,
                 beta_end: float = 1.0,
                 warmup_epochs: int = 10,
                 schedule_type: str = "cyclical_linear",
                 cycle_length: int = 20,
                 free_bits: float = 0.1,
                 target_kl: float = 10.0,
                 adaptive_threshold: float = 0.1):
        super().__init__()
        
        self.beta_start = beta_start
        self.beta_end = beta_end
        self.warmup_epochs = warmup_epochs
        self.schedule_type = schedule_type
        self.cycle_length = cycle_length
        self.free_bits = free_bits
        self.target_kl = target_kl
        self.adaptive_threshold = adaptive_threshold
        
        self.register_buffer('current_epoch', torch.tensor(0))
        self.register_buffer('current_beta', torch.tensor(beta_start))
        self.register_buffer('kl_history', torch.zeros(100))  # Rolling history
        self.register_buffer('history_idx', torch.tensor(0))
        
        logger.info(f"Initialized AdaptiveKLScheduler with {schedule_type} schedule")
    
    def step(self, kl_divergence: Optional[float] = None):
        """Update scheduler state."""
        self.current_epoch += 1
        
        # Update KL history for adaptive scheduling
        if kl_divergence is not None:
            idx = self.history_idx % 100
            self.kl_history[idx] = kl_divergence
            self.history_idx += 1
        
        # Update beta based on schedule
        self.current_beta = torch.tensor(self._compute_beta())
    
    def _compute_beta(self) -> float:
        """Compute current beta value based on schedule."""
        epoch = self.current_epoch.item()
        
        if self.schedule_type == "linear":
            if epoch <= self.warmup_epochs:
                if epoch == 0:
                    return self.beta_start
                return self.beta_start + (self.beta_end - self.beta_start) * (epoch / self.warmup_epochs)
            return self.beta_end
            
        elif self.schedule_type == "cyclical_linear":
            # Cyclical annealing with linear segments
            cycle_pos = epoch % self.cycle_length
            cycle_progress = cycle_pos / self.cycle_length
            
            if cycle_progress < 0.5:
                # Ascending phase
                progress = cycle_progress * 2
                return self.beta_start + (self.beta_end - self.beta_start) * progress
            else:
                # Descending phase
                progress = (cycle_progress - 0.5) * 2
                return self.beta_end - (self.beta_end - self.beta_start) * progress
                
        elif self.schedule_type == "adaptive":
            # Adaptive scheduling based on KL divergence
            if self.history_idx < 10:  # Not enough history
                return self.beta_start
            
            recent_kl = self.kl_history[:min(self.history_idx, 100)].mean().item()
            
            if recent_kl < self.target_kl - self.adaptive_threshold:
                # KL too low, increase beta
                return min(self.current_beta.item() * 1.1, self.beta_end)
            elif recent_kl > self.target_kl + self.adaptive_threshold:
                # KL too high, decrease beta
                return max(self.current_beta.item() * 0.9, self.beta_start)
            else:
                # KL in target range
                return self.current_beta.item()
                
        elif self.schedule_type == "cosine":
            # Cosine annealing
            if epoch < self.warmup_epochs:
                return self.beta_start + (self.beta_end - self.beta_start) * (
                    1 - math.cos(math.pi * epoch / self.warmup_epochs)
                ) / 2
            return self.beta_end
            
        else:
            return self.beta_end
    
    def get_kl_loss(self, kl_divergence: torch.Tensor) -> torch.Tensor:
        """Apply current beta and free bits to KL divergence."""
        # Apply free bits
        if self.free_bits > 0:
            kl_divergence = F.relu(kl_divergence - self.free_bits) + self.free_bits
        
        # Apply current beta
        return self.current_beta * kl_divergence.mean()
    
    def get_state(self) -> Dict[str, Any]:
        """Get current scheduler state."""
        return {
            'epoch': self.current_epoch.item(),
            'beta': self.current_beta.item(),
            'schedule_type': self.schedule_type,
            'recent_kl': self.kl_history[:min(self.history_idx, 100)].mean().item() if self.history_idx > 0 else 0.0
        }


class AdversarialStabilizer(nn.Module):
    """
    Adversarial loss stabilization techniques for VAE-GAN training.
    
    Implements gradient balancing, loss scaling, and stability monitoring
    to ensure stable adversarial training.
    """
    
    def __init__(self,
                 generator_loss_scale: float = 1.0,
                 discriminator_loss_scale: float = 1.0,
                 gradient_penalty_weight: float = 10.0,
                 gradient_clip_value: float = 1.0,
                 loss_smoothing: float = 0.1,
                 balance_threshold: float = 0.1):
        super().__init__()
        
        self.generator_loss_scale = generator_loss_scale
        self.discriminator_loss_scale = discriminator_loss_scale
        self.gradient_penalty_weight = gradient_penalty_weight
        self.gradient_clip_value = gradient_clip_value
        self.loss_smoothing = loss_smoothing
        self.balance_threshold = balance_threshold
        
        # Track loss history for balancing
        self.register_buffer('gen_loss_history', torch.zeros(100))
        self.register_buffer('disc_loss_history', torch.zeros(100))
        self.register_buffer('history_idx', torch.tensor(0))
        
    def balance_losses(self,
                      generator_loss: torch.Tensor,
                      discriminator_loss: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor]:
        """Apply adaptive loss balancing."""
        # Update history
        idx = self.history_idx % 100
        self.gen_loss_history[idx] = generator_loss.item()
        self.disc_loss_history[idx] = discriminator_loss.item()
        self.history_idx += 1
        
        # Compute balance ratio
        if self.history_idx > 10:
            recent_gen = self.gen_loss_history[:min(self.history_idx, 100)].mean()
            recent_disc = self.disc_loss_history[:min(self.history_idx, 100)].mean()
            
            if recent_gen > 0 and recent_disc > 0:
                balance_ratio = recent_gen / recent_disc
                
                # Adjust scales based on balance
                if balance_ratio > 1 + self.balance_threshold:
                    # Generator loss too high, reduce scale
                    gen_scale = self.generator_loss_scale * 0.95
                    disc_scale = self.discriminator_loss_scale * 1.05
                elif balance_ratio < 1 - self.balance_threshold:
                    # Discriminator loss too high, reduce scale  
                    gen_scale = self.generator_loss_scale * 1.05
                    disc_scale = self.discriminator_loss_scale * 0.95
                else:
                    gen_scale = self.generator_loss_scale
                    disc_scale = self.discriminator_loss_scale
            else:
                gen_scale = self.generator_loss_scale
                disc_scale = self.discriminator_loss_scale
        else:
            gen_scale = self.generator_loss_scale
            disc_scale = self.discriminator_loss_scale
        
        return generator_loss * gen_scale, discriminator_loss * disc_scale
    
    def stabilize_gradients(self, model: nn.Module) -> float:
        """Apply gradient clipping and return gradient norm."""
        total_norm = 0.0
        for p in model.parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
        total_norm = total_norm ** (1. / 2)
        
        # Clip gradients
        if self.gradient_clip_value > 0:
            torch.nn.utils.clip_grad_norm_(model.parameters(), self.gradient_clip_value)
        
        return total_norm


class MusicalConstraintLoss(nn.Module):
    """
    Musical constraint losses for rhythm and harmony consistency.
    
    Implements soft constraints that encourage musically coherent generation
    without being too restrictive.
    """
    
    def __init__(self,
                 vocab_size: int = 774,
                 rhythm_weight: float = 1.0,
                 harmony_weight: float = 1.0,
                 voice_leading_weight: float = 0.5,
                 enable_constraints: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.rhythm_weight = rhythm_weight
        self.harmony_weight = harmony_weight
        self.voice_leading_weight = voice_leading_weight
        self.enable_constraints = enable_constraints
        
        # Musical token ranges
        self.note_on_start = 0
        self.note_off_start = 128
        self.time_shift_start = 256
        self.velocity_start = 768
        
    def forward(self, generated_tokens: torch.Tensor) -> Dict[str, torch.Tensor]:
        """Compute musical constraint losses."""
        if not self.enable_constraints:
            return {
                'rhythm_constraint': torch.tensor(0.0, device=generated_tokens.device),
                'harmony_constraint': torch.tensor(0.0, device=generated_tokens.device), 
                'voice_leading_constraint': torch.tensor(0.0, device=generated_tokens.device),
                'total': torch.tensor(0.0, device=generated_tokens.device)
            }
        
        losses = {}
        
        # Rhythm constraint: encourage regular timing patterns
        losses['rhythm_constraint'] = self._rhythm_regularity_loss(generated_tokens)
        
        # Harmony constraint: discourage harsh intervals
        losses['harmony_constraint'] = self._harmony_consistency_loss(generated_tokens)
        
        # Voice leading constraint: encourage smooth melodic motion
        losses['voice_leading_constraint'] = self._voice_leading_loss(generated_tokens)
        
        # Combined constraint loss
        total_loss = (self.rhythm_weight * losses['rhythm_constraint'] +
                     self.harmony_weight * losses['harmony_constraint'] +
                     self.voice_leading_weight * losses['voice_leading_constraint'])
        losses['total'] = total_loss
        
        return losses
    
    def _rhythm_regularity_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encourage rhythmic regularity."""
        batch_size, seq_len = tokens.shape
        
        # Extract time shift patterns
        time_mask = (tokens >= self.time_shift_start) & (tokens < self.velocity_start)
        
        if time_mask.sum() == 0:
            return torch.tensor(0.0, device=tokens.device)
        
        # Compute timing intervals
        time_diffs = []
        for b in range(batch_size):
            time_positions = torch.where(time_mask[b])[0]
            if len(time_positions) > 1:
                diffs = torch.diff(time_positions.float())
                time_diffs.append(diffs)
        
        if not time_diffs:
            return torch.tensor(0.0, device=tokens.device)
        
        # Penalize highly irregular timing
        all_diffs = torch.cat(time_diffs)
        timing_variance = torch.var(all_diffs)
        
        return timing_variance * 0.01  # Small penalty for irregularity
    
    def _harmony_consistency_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Discourage harmonically harsh combinations."""
        batch_size, seq_len = tokens.shape
        
        # Find simultaneous notes (simplified)
        note_on_mask = (tokens >= self.note_on_start) & (tokens < self.note_off_start)
        
        penalty = 0.0
        for b in range(batch_size):
            for t in range(seq_len - 1):
                if note_on_mask[b, t] and note_on_mask[b, t + 1]:
                    # Check interval between consecutive notes
                    note1 = tokens[b, t] % 12  # Pitch class
                    note2 = tokens[b, t + 1] % 12
                    interval = abs(note1 - note2)
                    
                    # Penalize tritones (6 semitones) slightly
                    if interval == 6:
                        penalty += 0.1
        
        return torch.tensor(penalty, device=tokens.device) / (batch_size * seq_len)
    
    def _voice_leading_loss(self, tokens: torch.Tensor) -> torch.Tensor:
        """Encourage smooth voice leading."""
        batch_size, seq_len = tokens.shape
        
        # Track melodic intervals
        leap_penalty = 0.0
        note_positions = []
        
        note_on_mask = (tokens >= self.note_on_start) & (tokens < self.note_off_start)
        
        for b in range(batch_size):
            notes = tokens[b][note_on_mask[b]]
            if len(notes) > 1:
                intervals = torch.diff(notes.float())
                # Penalize large leaps (>octave = 12 semitones)
                large_leaps = (torch.abs(intervals) > 12).float()
                leap_penalty += large_leaps.sum()
        
        return torch.tensor(leap_penalty, device=tokens.device, dtype=torch.float32) / batch_size


class MultiObjectiveLossBalancer(nn.Module):
    """
    Automatic multi-objective loss balancing using uncertainty weighting.
    
    Implements the method from "Multi-Task Learning Using Uncertainty to Weigh Losses"
    adapted for music generation with VAE-GAN objectives.
    """
    
    def __init__(self,
                 num_objectives: int = 6,
                 init_log_var: float = 0.0,
                 min_weight: float = 1e-4,
                 max_weight: float = 100.0):
        super().__init__()
        
        self.num_objectives = num_objectives
        self.min_weight = min_weight
        self.max_weight = max_weight
        
        # Learnable uncertainty parameters
        self.log_vars = nn.Parameter(torch.full((num_objectives,), init_log_var))
        
        # Objective names for tracking
        self.objective_names = [
            'reconstruction', 'kl_divergence', 'adversarial_gen', 
            'adversarial_disc', 'feature_matching', 'musical_constraints'
        ]
        
        logger.info(f"Initialized MultiObjectiveLossBalancer for {num_objectives} objectives")
    
    def forward(self, losses: Dict[str, torch.Tensor]) -> Tuple[torch.Tensor, Dict[str, float]]:
        """
        Compute balanced loss using uncertainty weighting.
        
        Args:
            losses: Dictionary of individual loss components
            
        Returns:
            Tuple of (balanced_loss, weights_dict)
        """
        loss_values = []
        weights = {}
        
        # Extract loss values in consistent order
        for i, name in enumerate(self.objective_names[:len(losses)]):
            if name in losses:
                loss_values.append(losses[name])
                # Compute weight from uncertainty
                precision = torch.exp(-self.log_vars[i])
                weight = precision.clamp(self.min_weight, self.max_weight)
                weights[name] = weight.item()
            else:
                loss_values.append(torch.tensor(0.0, device=self.log_vars.device))
                weights[name] = 0.0
        
        # Compute weighted loss
        total_loss = 0.0
        for i, loss in enumerate(loss_values):
            if i < len(self.log_vars):
                precision = torch.exp(-self.log_vars[i])
                precision = precision.clamp(self.min_weight, self.max_weight)
                # Multi-task loss: precision * loss + log_var (regularization)
                weighted_loss = precision * loss + self.log_vars[i]
                total_loss += weighted_loss
        
        return total_loss, weights
    
    def get_weights(self) -> Dict[str, float]:
        """Get current objective weights."""
        weights = {}
        for i, name in enumerate(self.objective_names):
            if i < len(self.log_vars):
                precision = torch.exp(-self.log_vars[i])
                weight = precision.clamp(self.min_weight, self.max_weight)
                weights[name] = weight.item()
        return weights


class ComprehensiveLossFramework(nn.Module):
    """
    Complete loss framework integrating all components for Aurl.ai training.
    
    Coordinates reconstruction, KL, adversarial, constraint, and balancing losses
    for optimal VAE-GAN training.
    """
    
    def __init__(self,
                 vocab_size: int = 774,
                 # Reconstruction loss params
                 musical_weighting: bool = True,
                 perceptual_emphasis: float = 2.0,
                 # KL scheduling params
                 beta_start: float = 0.0,
                 beta_end: float = 1.0,
                 kl_schedule: str = "cyclical_linear",
                 warmup_epochs: int = 10,
                 # Adversarial params
                 adversarial_weight: float = 1.0,
                 feature_matching_weight: float = 10.0,
                 # Constraint params
                 enable_musical_constraints: bool = True,
                 constraint_weight: float = 0.1,
                 # Balancing params
                 use_automatic_balancing: bool = True):
        super().__init__()
        
        self.vocab_size = vocab_size
        self.use_automatic_balancing = use_automatic_balancing
        
        # Initialize loss components
        self.reconstruction_loss = PerceptualReconstructionLoss(
            vocab_size=vocab_size,
            musical_weighting=musical_weighting,
            perceptual_emphasis=perceptual_emphasis
        )
        
        self.kl_scheduler = AdaptiveKLScheduler(
            beta_start=beta_start,
            beta_end=beta_end,
            schedule_type=kl_schedule,
            warmup_epochs=warmup_epochs
        )
        
        self.adversarial_stabilizer = AdversarialStabilizer()
        
        self.gan_loss = ComprehensiveGANLoss(
            feature_matching_weight=feature_matching_weight,
            vocab_size=vocab_size
        )
        
        self.constraint_loss = MusicalConstraintLoss(
            vocab_size=vocab_size,
            enable_constraints=enable_musical_constraints
        )
        
        if use_automatic_balancing:
            self.loss_balancer = MultiObjectiveLossBalancer(num_objectives=6)
        
        # Manual weights (used if automatic balancing disabled)
        self.manual_weights = {
            'reconstruction': 1.0,
            'kl_divergence': 1.0,
            'adversarial_gen': adversarial_weight,
            'adversarial_disc': adversarial_weight,
            'feature_matching': feature_matching_weight,
            'musical_constraints': constraint_weight
        }
        
        logger.info(f"Initialized ComprehensiveLossFramework with auto_balancing={use_automatic_balancing}")
    
    def forward(self,
                # Model outputs
                reconstruction_logits: torch.Tensor,
                target_tokens: torch.Tensor,
                encoder_output: Dict[str, torch.Tensor],
                # Discriminator outputs
                real_discriminator_output: Dict[str, torch.Tensor],
                fake_discriminator_output: Dict[str, torch.Tensor],
                # Models for additional computations
                discriminator: nn.Module,
                generated_tokens: torch.Tensor,
                # Optional
                mask: Optional[torch.Tensor] = None) -> Dict[str, torch.Tensor]:
        """
        Compute comprehensive loss for VAE-GAN training.
        
        Returns:
            Dictionary containing all loss components and metrics
        """
        losses = {}
        
        # 1. Reconstruction Loss
        recon_losses = self.reconstruction_loss(reconstruction_logits, target_tokens, mask)
        losses.update({f'recon_{k}': v for k, v in recon_losses.items()})
        reconstruction_loss = recon_losses['total']
        
        # 2. KL Divergence Loss
        kl_divergence = encoder_output['kl_loss']
        kl_loss = self.kl_scheduler.get_kl_loss(kl_divergence)
        losses['kl_loss'] = kl_loss
        losses['kl_raw'] = kl_divergence.mean()
        losses['beta'] = self.kl_scheduler.current_beta
        
        # 3. Adversarial Losses
        # Generator loss
        gen_losses = self.gan_loss.generator_loss(
            fake_discriminator_output,
            real_discriminator_output['features'],
            fake_discriminator_output['features'],
            generated_tokens
        )
        losses.update({f'gen_{k}': v for k, v in gen_losses.items()})
        
        # Discriminator loss
        disc_losses = self.gan_loss.discriminator_loss(
            real_discriminator_output,
            fake_discriminator_output,
            discriminator,
            target_tokens,
            generated_tokens
        )
        losses.update({f'disc_{k}': v for k, v in disc_losses.items()})
        
        # Balance adversarial losses
        balanced_gen, balanced_disc = self.adversarial_stabilizer.balance_losses(
            gen_losses['total'], disc_losses['total']
        )
        losses['gen_balanced'] = balanced_gen
        losses['disc_balanced'] = balanced_disc
        
        # 4. Musical Constraint Losses
        constraint_losses = self.constraint_loss(generated_tokens)
        losses.update({f'constraint_{k}': v for k, v in constraint_losses.items()})
        
        # 5. Combine losses
        objective_losses = {
            'reconstruction': reconstruction_loss,
            'kl_divergence': kl_loss,
            'adversarial_gen': balanced_gen,
            'adversarial_disc': balanced_disc,
            'feature_matching': gen_losses['feature_matching'],
            'musical_constraints': constraint_losses['total']
        }
        
        if self.use_automatic_balancing:
            # Automatic balancing
            total_loss, weights = self.loss_balancer(objective_losses)
            losses['total_loss'] = total_loss
            losses.update({f'weight_{k}': v for k, v in weights.items()})
        else:
            # Manual balancing
            total_loss = sum(
                self.manual_weights.get(k, 1.0) * v 
                for k, v in objective_losses.items()
            )
            losses['total_loss'] = total_loss
            losses.update({f'weight_{k}': v for k, v in self.manual_weights.items()})
        
        # 6. Additional metrics
        losses.update(self._compute_metrics(reconstruction_logits, target_tokens, encoder_output))
        
        return losses
    
    def _compute_metrics(self,
                        logits: torch.Tensor,
                        targets: torch.Tensor,
                        encoder_output: Dict[str, torch.Tensor]) -> Dict[str, torch.Tensor]:
        """Compute additional training metrics."""
        metrics = {}
        
        # Reconstruction accuracy
        predictions = torch.argmax(logits, dim=-1)
        accuracy = (predictions == targets).float().mean()
        metrics['reconstruction_accuracy'] = accuracy
        
        # Perplexity
        log_probs = F.log_softmax(logits, dim=-1)
        nll = F.nll_loss(log_probs.view(-1, self.vocab_size), targets.view(-1))
        perplexity = torch.exp(nll)
        metrics['perplexity'] = perplexity
        
        # KL metrics
        if 'latent_info' in encoder_output:
            latent_info = encoder_output['latent_info']
            if 'active_dims' in latent_info:
                metrics['active_latent_dims'] = torch.tensor(latent_info['active_dims'])
        
        return metrics
    
    def step_epoch(self, avg_kl: Optional[float] = None):
        """Update schedulers and advance epoch."""
        self.kl_scheduler.step(avg_kl)
        
    def get_state(self) -> Dict[str, Any]:
        """Get current loss framework state."""
        state = {
            'kl_scheduler': self.kl_scheduler.get_state(),
            'adversarial_stabilizer': {
                'gen_loss_avg': self.adversarial_stabilizer.gen_loss_history.mean().item(),
                'disc_loss_avg': self.adversarial_stabilizer.disc_loss_history.mean().item()
            }
        }
        
        if self.use_automatic_balancing:
            state['loss_weights'] = self.loss_balancer.get_weights()
        else:
            state['loss_weights'] = self.manual_weights
        
        return state