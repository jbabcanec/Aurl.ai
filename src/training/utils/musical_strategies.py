"""
Musical Domain-Specific Training Strategies for Aurl.ai Music Generation.

This module implements specialized training strategies for music generation:
- Musical domain-specific loss scheduling
- Genre-aware training protocols
- Style-specific model configurations
- Musical coherence-guided training
- Harmonic progression-aware learning
- Rhythmic pattern optimization
- Musical structure enforcement

Designed to leverage musical domain knowledge for optimal training outcomes.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import math
from collections import defaultdict, deque

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class MusicalDomain(Enum):
    """Musical domains for specialized training."""
    CLASSICAL = "classical"
    JAZZ = "jazz"
    POP = "pop"
    ROCK = "rock"
    FOLK = "folk"
    ELECTRONIC = "electronic"
    WORLD = "world"
    CONTEMPORARY = "contemporary"


class MusicalAspect(Enum):
    """Aspects of music to focus training on."""
    RHYTHM = "rhythm"
    HARMONY = "harmony"
    MELODY = "melody"
    STRUCTURE = "structure"
    DYNAMICS = "dynamics"
    TIMBRE = "timbre"
    STYLE = "style"
    EXPRESSION = "expression"


class TrainingPhase(Enum):
    """Musical training phases."""
    FOUNDATION = "foundation"        # Basic musical understanding
    SPECIALIZATION = "specialization" # Genre/style-specific training
    REFINEMENT = "refinement"        # Quality and expression refinement
    MASTERY = "mastery"             # Advanced musical intelligence


@dataclass
class MusicalTrainingConfig:
    """Configuration for musical domain-specific training."""
    
    # Domain specification
    primary_domain: MusicalDomain = MusicalDomain.CONTEMPORARY
    secondary_domains: List[MusicalDomain] = field(default_factory=list)
    
    # Musical aspects to emphasize
    focus_aspects: List[MusicalAspect] = field(default_factory=lambda: [
        MusicalAspect.RHYTHM, MusicalAspect.HARMONY, MusicalAspect.MELODY
    ])
    
    # Aspect weights for different phases
    phase_aspect_weights: Dict[TrainingPhase, Dict[str, float]] = field(default_factory=lambda: {
        TrainingPhase.FOUNDATION: {
            "rhythm": 0.4, "harmony": 0.3, "melody": 0.3
        },
        TrainingPhase.SPECIALIZATION: {
            "rhythm": 0.25, "harmony": 0.25, "melody": 0.25, "structure": 0.15, "style": 0.1
        },
        TrainingPhase.REFINEMENT: {
            "rhythm": 0.2, "harmony": 0.2, "melody": 0.2, "structure": 0.15, 
            "style": 0.1, "dynamics": 0.1, "expression": 0.05
        },
        TrainingPhase.MASTERY: {
            "rhythm": 0.15, "harmony": 0.15, "melody": 0.15, "structure": 0.15,
            "style": 0.15, "dynamics": 0.1, "expression": 0.1, "timbre": 0.05
        }
    })
    
    # Musical theory integration
    enforce_music_theory: bool = True
    theory_compliance_weight: float = 0.2
    
    # Genre-specific settings
    genre_adaptation: bool = True
    genre_transition_epochs: int = 20
    cross_genre_training: bool = True
    
    # Musical coherence settings
    coherence_validation: bool = True
    coherence_threshold: float = 0.6
    coherence_patience: int = 15
    
    # Harmonic progression settings
    harmonic_progression_guidance: bool = True
    progression_loss_weight: float = 0.3
    
    # Rhythmic pattern settings
    rhythmic_pattern_enforcement: bool = True
    rhythm_loss_weight: float = 0.25
    
    # Structural guidance
    structural_guidance: bool = True
    structure_loss_weight: float = 0.2
    
    # Style consistency
    style_consistency_enforcement: bool = True
    style_loss_weight: float = 0.15
    
    # Adaptive settings
    adaptive_weighting: bool = True
    adaptation_frequency: int = 50
    performance_history_window: int = 100


@dataclass
class MusicalTrainingState:
    """State of musical domain-specific training."""
    
    current_phase: TrainingPhase = TrainingPhase.FOUNDATION
    current_domain: MusicalDomain = MusicalDomain.CONTEMPORARY
    epoch: int = 0
    
    # Aspect performance tracking
    aspect_performance: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    aspect_weights: Dict[str, float] = field(default_factory=dict)
    
    # Musical metrics
    musical_coherence_history: List[float] = field(default_factory=list)
    harmonic_quality_history: List[float] = field(default_factory=list)
    rhythmic_quality_history: List[float] = field(default_factory=list)
    
    # Phase transitions
    phase_transitions: List[Dict[str, Any]] = field(default_factory=list)
    last_phase_transition: int = 0
    
    # Domain adaptation
    domain_adaptations: List[Dict[str, Any]] = field(default_factory=list)
    domain_performance: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    
    # Theory compliance
    theory_compliance_scores: List[float] = field(default_factory=list)
    theory_violations: List[Dict[str, Any]] = field(default_factory=list)


class MusicalTheoryValidator:
    """Validates musical theory compliance during training."""
    
    def __init__(self):
        self.key_signatures = self._load_key_signatures()
        self.chord_progressions = self._load_chord_progressions()
        self.scale_patterns = self._load_scale_patterns()
        
    def validate_harmonic_progression(self, 
                                   chord_sequence: List[List[int]],
                                   key: str = "C_major") -> Dict[str, float]:
        """Validate harmonic progression against music theory."""
        
        if len(chord_sequence) < 2:
            return {"compliance": 0.5, "progression_strength": 0.0}
        
        # Analyze chord progression
        progression_strength = self._analyze_progression_strength(chord_sequence, key)
        voice_leading_quality = self._analyze_voice_leading(chord_sequence)
        harmonic_rhythm = self._analyze_harmonic_rhythm(chord_sequence)
        
        # Overall compliance
        compliance = (progression_strength + voice_leading_quality + harmonic_rhythm) / 3.0
        
        return {
            "compliance": compliance,
            "progression_strength": progression_strength,
            "voice_leading_quality": voice_leading_quality,
            "harmonic_rhythm": harmonic_rhythm
        }
    
    def validate_melodic_line(self, 
                            melody_notes: List[int],
                            key: str = "C_major") -> Dict[str, float]:
        """Validate melodic line against music theory."""
        
        if len(melody_notes) < 3:
            return {"compliance": 0.5, "contour_quality": 0.0}
        
        # Analyze melodic aspects
        scale_compliance = self._check_scale_compliance(melody_notes, key)
        contour_quality = self._analyze_melodic_contour(melody_notes)
        interval_quality = self._analyze_melodic_intervals(melody_notes)
        
        # Overall compliance
        compliance = (scale_compliance + contour_quality + interval_quality) / 3.0
        
        return {
            "compliance": compliance,
            "scale_compliance": scale_compliance,
            "contour_quality": contour_quality,
            "interval_quality": interval_quality
        }
    
    def validate_rhythmic_pattern(self, 
                                rhythm_pattern: List[float],
                                time_signature: Tuple[int, int] = (4, 4)) -> Dict[str, float]:
        """Validate rhythmic pattern against music theory."""
        
        if len(rhythm_pattern) < 2:
            return {"compliance": 0.5, "pattern_strength": 0.0}
        
        # Analyze rhythmic aspects
        meter_compliance = self._check_meter_compliance(rhythm_pattern, time_signature)
        pattern_coherence = self._analyze_rhythmic_coherence(rhythm_pattern)
        syncopation_quality = self._analyze_syncopation(rhythm_pattern, time_signature)
        
        # Overall compliance
        compliance = (meter_compliance + pattern_coherence + syncopation_quality) / 3.0
        
        return {
            "compliance": compliance,
            "meter_compliance": meter_compliance,
            "pattern_coherence": pattern_coherence,
            "syncopation_quality": syncopation_quality
        }
    
    def _load_key_signatures(self) -> Dict[str, List[int]]:
        """Load key signatures and their notes."""
        
        return {
            "C_major": [0, 2, 4, 5, 7, 9, 11],
            "G_major": [0, 2, 4, 6, 7, 9, 11],
            "D_major": [0, 2, 4, 6, 7, 9, 11],
            "A_major": [0, 2, 4, 6, 7, 9, 11],
            "E_major": [0, 2, 4, 6, 7, 9, 11],
            "B_major": [0, 2, 4, 6, 7, 9, 11],
            "F_major": [0, 2, 4, 5, 7, 9, 10],
            "Bb_major": [0, 2, 3, 5, 7, 9, 10],
            "Eb_major": [0, 2, 3, 5, 7, 8, 10],
            "Ab_major": [0, 1, 3, 5, 7, 8, 10],
            "Db_major": [0, 1, 3, 5, 6, 8, 10],
            "Gb_major": [0, 1, 3, 4, 6, 8, 10],
            "A_minor": [0, 2, 3, 5, 7, 8, 10],
            "E_minor": [0, 2, 3, 5, 7, 8, 10],
            "B_minor": [0, 2, 3, 5, 7, 8, 10],
            "F#_minor": [0, 2, 3, 5, 7, 8, 10],
            "C#_minor": [0, 2, 3, 5, 7, 8, 10],
            "G#_minor": [0, 2, 3, 5, 7, 8, 10],
            "D_minor": [0, 2, 3, 5, 7, 8, 10],
            "G_minor": [0, 2, 3, 5, 7, 8, 10],
            "C_minor": [0, 2, 3, 5, 7, 8, 10],
            "F_minor": [0, 2, 3, 5, 7, 8, 10],
            "Bb_minor": [0, 2, 3, 5, 7, 8, 10],
            "Eb_minor": [0, 2, 3, 5, 7, 8, 10]
        }
    
    def _load_chord_progressions(self) -> Dict[str, List[List[int]]]:
        """Load common chord progressions."""
        
        return {
            "I-V-vi-IV": [[0, 4, 7], [7, 11, 2], [9, 0, 4], [5, 9, 0]],
            "ii-V-I": [[2, 5, 9], [7, 11, 2], [0, 4, 7]],
            "vi-IV-I-V": [[9, 0, 4], [5, 9, 0], [0, 4, 7], [7, 11, 2]],
            "I-vi-ii-V": [[0, 4, 7], [9, 0, 4], [2, 5, 9], [7, 11, 2]]
        }
    
    def _load_scale_patterns(self) -> Dict[str, List[int]]:
        """Load scale patterns."""
        
        return {
            "major": [0, 2, 4, 5, 7, 9, 11],
            "minor": [0, 2, 3, 5, 7, 8, 10],
            "dorian": [0, 2, 3, 5, 7, 9, 10],
            "mixolydian": [0, 2, 4, 5, 7, 9, 10],
            "pentatonic": [0, 2, 4, 7, 9]
        }
    
    def _analyze_progression_strength(self, 
                                   chord_sequence: List[List[int]], 
                                   key: str) -> float:
        """Analyze strength of chord progression."""
        
        # Simplified analysis based on root motion
        if len(chord_sequence) < 2:
            return 0.5
        
        strong_progressions = 0
        total_progressions = len(chord_sequence) - 1
        
        for i in range(len(chord_sequence) - 1):
            current_root = chord_sequence[i][0] % 12
            next_root = chord_sequence[i + 1][0] % 12
            
            # Strong progressions: 5th down, 4th up, 2nd down
            interval = (next_root - current_root) % 12
            if interval in [7, 5, 10]:  # 5th down, 4th up, 2nd down
                strong_progressions += 1
            elif interval in [2, 9]:     # 2nd up, 6th down (weaker but good)
                strong_progressions += 0.5
        
        return strong_progressions / max(total_progressions, 1)
    
    def _analyze_voice_leading(self, chord_sequence: List[List[int]]) -> float:
        """Analyze voice leading quality."""
        
        if len(chord_sequence) < 2:
            return 0.5
        
        voice_leading_scores = []
        
        for i in range(len(chord_sequence) - 1):
            current_chord = sorted(chord_sequence[i])
            next_chord = sorted(chord_sequence[i + 1])
            
            # Simple voice leading: minimize movement
            total_movement = 0
            for j in range(min(len(current_chord), len(next_chord))):
                movement = abs(next_chord[j] - current_chord[j])
                total_movement += movement
            
            # Good voice leading has small movements
            avg_movement = total_movement / max(min(len(current_chord), len(next_chord)), 1)
            voice_leading_score = max(0, 1.0 - avg_movement / 12.0)  # Normalize by octave
            voice_leading_scores.append(voice_leading_score)
        
        return np.mean(voice_leading_scores) if voice_leading_scores else 0.5
    
    def _analyze_harmonic_rhythm(self, chord_sequence: List[List[int]]) -> float:
        """Analyze harmonic rhythm quality."""
        
        # Simplified: assume regular harmonic rhythm is good
        # In practice, this would analyze actual timing
        return 0.8  # Assume regular harmonic rhythm
    
    def _check_scale_compliance(self, melody_notes: List[int], key: str) -> float:
        """Check if melody complies with scale."""
        
        if key not in self.key_signatures:
            return 0.5
        
        scale_notes = set(self.key_signatures[key])
        
        compliant_notes = 0
        for note in melody_notes:
            if note % 12 in scale_notes:
                compliant_notes += 1
        
        return compliant_notes / max(len(melody_notes), 1)
    
    def _analyze_melodic_contour(self, melody_notes: List[int]) -> float:
        """Analyze melodic contour quality."""
        
        if len(melody_notes) < 3:
            return 0.5
        
        # Analyze direction changes
        directions = []
        for i in range(len(melody_notes) - 1):
            if melody_notes[i + 1] > melody_notes[i]:
                directions.append(1)
            elif melody_notes[i + 1] < melody_notes[i]:
                directions.append(-1)
            else:
                directions.append(0)
        
        # Good contour has balanced direction changes
        direction_changes = sum(1 for i in range(len(directions) - 1) 
                              if directions[i] != directions[i + 1] and directions[i] != 0)
        
        change_ratio = direction_changes / max(len(directions) - 1, 1)
        
        # Optimal contour has some direction changes
        if 0.2 <= change_ratio <= 0.6:
            return 0.9
        elif change_ratio <= 0.8:
            return 0.6
        else:
            return 0.3
    
    def _analyze_melodic_intervals(self, melody_notes: List[int]) -> float:
        """Analyze quality of melodic intervals."""
        
        if len(melody_notes) < 2:
            return 0.5
        
        intervals = [abs(melody_notes[i + 1] - melody_notes[i]) 
                    for i in range(len(melody_notes) - 1)]
        
        # Good melody has mostly steps with some leaps
        steps = sum(1 for interval in intervals if interval <= 2)
        small_leaps = sum(1 for interval in intervals if 3 <= interval <= 5)
        large_leaps = sum(1 for interval in intervals if interval > 5)
        
        total_intervals = len(intervals)
        
        step_ratio = steps / total_intervals
        leap_ratio = (small_leaps + large_leaps) / total_intervals
        
        # Optimal ratio: mostly steps with some leaps
        if step_ratio >= 0.6 and leap_ratio <= 0.4:
            return 0.9
        elif step_ratio >= 0.4:
            return 0.7
        else:
            return 0.4
    
    def _check_meter_compliance(self, 
                              rhythm_pattern: List[float], 
                              time_signature: Tuple[int, int]) -> float:
        """Check if rhythm complies with time signature."""
        
        beats_per_measure = time_signature[0]
        beat_unit = time_signature[1]
        
        # Simplified: check if pattern fits time signature
        total_duration = sum(rhythm_pattern)
        expected_duration = beats_per_measure * (4.0 / beat_unit)
        
        # Good if close to expected duration
        duration_ratio = min(total_duration, expected_duration) / max(total_duration, expected_duration)
        
        return duration_ratio
    
    def _analyze_rhythmic_coherence(self, rhythm_pattern: List[float]) -> float:
        """Analyze rhythmic coherence."""
        
        if len(rhythm_pattern) < 2:
            return 0.5
        
        # Check for patterns and regularity
        pattern_variance = np.var(rhythm_pattern)
        pattern_mean = np.mean(rhythm_pattern)
        
        # Good rhythm has some regularity but not too much
        if pattern_mean > 0:
            coefficient_of_variation = pattern_variance / pattern_mean
            if 0.2 <= coefficient_of_variation <= 0.8:
                return 0.9
            else:
                return 0.6
        
        return 0.5
    
    def _analyze_syncopation(self, 
                           rhythm_pattern: List[float], 
                           time_signature: Tuple[int, int]) -> float:
        """Analyze syncopation quality."""
        
        # Simplified syncopation analysis
        # Good syncopation adds interest without being chaotic
        
        if len(rhythm_pattern) < 4:
            return 0.5
        
        # Look for off-beat emphasis
        beat_positions = np.cumsum([0] + rhythm_pattern[:-1])
        beat_unit = 4.0 / time_signature[1]
        
        on_beat_count = 0
        off_beat_count = 0
        
        for pos in beat_positions:
            if pos % beat_unit < 0.1:  # On beat
                on_beat_count += 1
            else:  # Off beat
                off_beat_count += 1
        
        total_beats = on_beat_count + off_beat_count
        if total_beats == 0:
            return 0.5
        
        syncopation_ratio = off_beat_count / total_beats
        
        # Good syncopation: some off-beat emphasis
        if 0.2 <= syncopation_ratio <= 0.5:
            return 0.9
        elif syncopation_ratio <= 0.7:
            return 0.6
        else:
            return 0.3


class MusicalDomainTrainer:
    """
    Musical domain-specific trainer that applies music theory and domain knowledge.
    
    Features:
    - Genre-specific training protocols
    - Musical theory-guided loss functions
    - Adaptive musical aspect weighting
    - Coherence-based training validation
    - Style-specific model configurations
    """
    
    def __init__(self, config: MusicalTrainingConfig):
        self.config = config
        self.state = MusicalTrainingState()
        self.theory_validator = MusicalTheoryValidator()
        
        # Initialize aspect weights
        self.state.aspect_weights = self.config.phase_aspect_weights[TrainingPhase.FOUNDATION].copy()
        
        logger.info(f"Initialized musical domain trainer for {config.primary_domain.value}")
        logger.info(f"Focus aspects: {[aspect.value for aspect in config.focus_aspects]}")
    
    def get_musical_loss_weights(self, epoch: int) -> Dict[str, float]:
        """Get current musical loss weights based on training phase and performance."""
        
        # Update training phase if needed
        self._update_training_phase(epoch)
        
        # Get base weights for current phase
        base_weights = self.config.phase_aspect_weights[self.state.current_phase].copy()
        
        # Apply adaptive weighting if enabled
        if self.config.adaptive_weighting and epoch % self.config.adaptation_frequency == 0:
            base_weights = self._apply_adaptive_weighting(base_weights)
        
        # Add domain-specific adjustments
        adjusted_weights = self._apply_domain_adjustments(base_weights)
        
        # Apply musical theory compliance weight
        if self.config.enforce_music_theory:
            adjusted_weights["theory_compliance"] = self.config.theory_compliance_weight
        
        return adjusted_weights
    
    def validate_musical_quality(self, 
                               generated_sequence: torch.Tensor,
                               context: Dict[str, Any] = None) -> Dict[str, float]:
        """Validate musical quality of generated sequence."""
        
        # Convert tokens to musical elements (simplified)
        musical_elements = self._parse_musical_elements(generated_sequence)
        
        validation_results = {}
        
        # Validate different musical aspects
        if "harmony" in musical_elements:
            harmonic_result = self.theory_validator.validate_harmonic_progression(
                musical_elements["harmony"]
            )
            validation_results["harmonic_compliance"] = harmonic_result["compliance"]
        
        if "melody" in musical_elements:
            melodic_result = self.theory_validator.validate_melodic_line(
                musical_elements["melody"]
            )
            validation_results["melodic_compliance"] = melodic_result["compliance"]
        
        if "rhythm" in musical_elements:
            rhythmic_result = self.theory_validator.validate_rhythmic_pattern(
                musical_elements["rhythm"]
            )
            validation_results["rhythmic_compliance"] = rhythmic_result["compliance"]
        
        # Overall musical coherence
        if validation_results:
            overall_coherence = np.mean(list(validation_results.values()))
            validation_results["overall_coherence"] = overall_coherence
        
        return validation_results
    
    def get_domain_specific_config(self, domain: MusicalDomain) -> Dict[str, Any]:
        """Get domain-specific configuration adjustments."""
        
        domain_configs = {
            MusicalDomain.CLASSICAL: {
                "emphasis": ["structure", "harmony", "melody"],
                "loss_weights": {"harmony": 0.4, "structure": 0.3, "melody": 0.3},
                "complexity_preference": "high",
                "theory_strictness": 0.9
            },
            MusicalDomain.JAZZ: {
                "emphasis": ["harmony", "rhythm", "expression"],
                "loss_weights": {"harmony": 0.4, "rhythm": 0.3, "expression": 0.3},
                "complexity_preference": "high",
                "theory_strictness": 0.7
            },
            MusicalDomain.POP: {
                "emphasis": ["melody", "rhythm", "structure"],
                "loss_weights": {"melody": 0.4, "rhythm": 0.3, "structure": 0.3},
                "complexity_preference": "medium",
                "theory_strictness": 0.6
            },
            MusicalDomain.ROCK: {
                "emphasis": ["rhythm", "energy", "structure"],
                "loss_weights": {"rhythm": 0.4, "energy": 0.3, "structure": 0.3},
                "complexity_preference": "medium",
                "theory_strictness": 0.5
            },
            MusicalDomain.FOLK: {
                "emphasis": ["melody", "simplicity", "authenticity"],
                "loss_weights": {"melody": 0.4, "simplicity": 0.3, "authenticity": 0.3},
                "complexity_preference": "low",
                "theory_strictness": 0.8
            },
            MusicalDomain.ELECTRONIC: {
                "emphasis": ["rhythm", "timbre", "innovation"],
                "loss_weights": {"rhythm": 0.4, "timbre": 0.3, "innovation": 0.3},
                "complexity_preference": "high",
                "theory_strictness": 0.4
            }
        }
        
        return domain_configs.get(domain, {
            "emphasis": ["rhythm", "harmony", "melody"],
            "loss_weights": {"rhythm": 0.33, "harmony": 0.33, "melody": 0.34},
            "complexity_preference": "medium",
            "theory_strictness": 0.7
        })
    
    def update_musical_performance(self, 
                                 epoch: int, 
                                 musical_metrics: Dict[str, float]):
        """Update musical performance tracking."""
        
        self.state.epoch = epoch
        
        # Update aspect performance
        for aspect, score in musical_metrics.items():
            self.state.aspect_performance[aspect].append(score)
        
        # Update overall musical metrics
        if "musical_coherence" in musical_metrics:
            self.state.musical_coherence_history.append(musical_metrics["musical_coherence"])
        
        if "harmonic_quality" in musical_metrics:
            self.state.harmonic_quality_history.append(musical_metrics["harmonic_quality"])
        
        if "rhythmic_quality" in musical_metrics:
            self.state.rhythmic_quality_history.append(musical_metrics["rhythmic_quality"])
        
        # Update theory compliance
        if "theory_compliance" in musical_metrics:
            self.state.theory_compliance_scores.append(musical_metrics["theory_compliance"])
        
        # Check for phase transition
        if self._should_transition_phase():
            self._transition_to_next_phase()
    
    def _update_training_phase(self, epoch: int):
        """Update training phase based on epoch and performance."""
        
        # Simple epoch-based phase transitions
        if epoch < 50:
            target_phase = TrainingPhase.FOUNDATION
        elif epoch < 100:
            target_phase = TrainingPhase.SPECIALIZATION
        elif epoch < 150:
            target_phase = TrainingPhase.REFINEMENT
        else:
            target_phase = TrainingPhase.MASTERY
        
        if target_phase != self.state.current_phase:
            self._transition_to_phase(target_phase)
    
    def _apply_adaptive_weighting(self, base_weights: Dict[str, float]) -> Dict[str, float]:
        """Apply adaptive weighting based on performance history."""
        
        adjusted_weights = base_weights.copy()
        
        # Adjust weights based on aspect performance
        for aspect, weight in base_weights.items():
            if aspect in self.state.aspect_performance:
                recent_performance = self.state.aspect_performance[aspect][-10:]
                if len(recent_performance) >= 5:
                    avg_performance = np.mean(recent_performance)
                    
                    # Increase weight for underperforming aspects
                    if avg_performance < 0.5:
                        adjusted_weights[aspect] = min(1.0, weight * 1.2)
                    # Decrease weight for well-performing aspects
                    elif avg_performance > 0.8:
                        adjusted_weights[aspect] = max(0.1, weight * 0.8)
        
        # Normalize weights
        total_weight = sum(adjusted_weights.values())
        if total_weight > 0:
            adjusted_weights = {k: v / total_weight for k, v in adjusted_weights.items()}
        
        return adjusted_weights
    
    def _apply_domain_adjustments(self, weights: Dict[str, float]) -> Dict[str, float]:
        """Apply domain-specific adjustments to weights."""
        
        domain_config = self.get_domain_specific_config(self.state.current_domain)
        domain_weights = domain_config.get("loss_weights", {})
        
        # Blend base weights with domain-specific weights
        adjusted_weights = weights.copy()
        for aspect, domain_weight in domain_weights.items():
            if aspect in adjusted_weights:
                # Weighted average with domain preference
                adjusted_weights[aspect] = 0.7 * adjusted_weights[aspect] + 0.3 * domain_weight
        
        return adjusted_weights
    
    def _parse_musical_elements(self, sequence: torch.Tensor) -> Dict[str, Any]:
        """Parse musical elements from token sequence."""
        
        # Simplified parsing - in practice, this would use the vocabulary
        tokens = sequence.flatten().tolist()
        
        musical_elements = {}
        
        # Extract harmony (simplified: group note tokens)
        harmony_tokens = [t for t in tokens if 21 <= t <= 108]  # Note range
        if harmony_tokens:
            # Group into potential chords
            chords = []
            current_chord = []
            for token in harmony_tokens:
                if not current_chord or abs(token - current_chord[-1]) <= 12:
                    current_chord.append(token)
                else:
                    if len(current_chord) >= 2:
                        chords.append(current_chord)
                    current_chord = [token]
            if len(current_chord) >= 2:
                chords.append(current_chord)
            
            musical_elements["harmony"] = chords
        
        # Extract melody (simplified: take note tokens in sequence)
        melody_tokens = [t for t in tokens if 60 <= t <= 84]  # Melody range
        if melody_tokens:
            musical_elements["melody"] = melody_tokens
        
        # Extract rhythm (simplified: time-related tokens)
        rhythm_tokens = [t for t in tokens if 100 <= t <= 200]  # Time range
        if rhythm_tokens:
            # Convert to duration pattern
            rhythm_pattern = [0.25 + (t - 100) * 0.01 for t in rhythm_tokens]  # Simplified
            musical_elements["rhythm"] = rhythm_pattern
        
        return musical_elements
    
    def _should_transition_phase(self) -> bool:
        """Check if should transition to next phase."""
        
        # Check performance thresholds
        if len(self.state.musical_coherence_history) < 10:
            return False
        
        recent_coherence = np.mean(self.state.musical_coherence_history[-10:])
        
        # Phase-specific transition criteria
        if self.state.current_phase == TrainingPhase.FOUNDATION:
            return recent_coherence >= 0.5
        elif self.state.current_phase == TrainingPhase.SPECIALIZATION:
            return recent_coherence >= 0.65
        elif self.state.current_phase == TrainingPhase.REFINEMENT:
            return recent_coherence >= 0.8
        else:  # MASTERY
            return False  # No further transitions
    
    def _transition_to_next_phase(self):
        """Transition to the next training phase."""
        
        phase_order = [
            TrainingPhase.FOUNDATION,
            TrainingPhase.SPECIALIZATION,
            TrainingPhase.REFINEMENT,
            TrainingPhase.MASTERY
        ]
        
        current_index = phase_order.index(self.state.current_phase)
        if current_index < len(phase_order) - 1:
            next_phase = phase_order[current_index + 1]
            self._transition_to_phase(next_phase)
    
    def _transition_to_phase(self, target_phase: TrainingPhase):
        """Transition to specific training phase."""
        
        previous_phase = self.state.current_phase
        self.state.current_phase = target_phase
        self.state.last_phase_transition = self.state.epoch
        
        # Update aspect weights for new phase
        self.state.aspect_weights = self.config.phase_aspect_weights[target_phase].copy()
        
        # Record transition
        transition_info = {
            "epoch": self.state.epoch,
            "from_phase": previous_phase.value,
            "to_phase": target_phase.value,
            "trigger": "performance_threshold"
        }
        self.state.phase_transitions.append(transition_info)
        
        logger.info(f"Transitioned to {target_phase.value} phase at epoch {self.state.epoch}")
    
    def get_training_recommendations(self) -> List[str]:
        """Get training recommendations based on current state."""
        
        recommendations = []
        
        # Performance-based recommendations
        if len(self.state.musical_coherence_history) >= 10:
            recent_coherence = np.mean(self.state.musical_coherence_history[-10:])
            
            if recent_coherence < 0.4:
                recommendations.append("Focus on basic musical patterns and theory compliance")
            elif recent_coherence < 0.6:
                recommendations.append("Increase musical structure and harmonic guidance")
            elif recent_coherence > 0.8:
                recommendations.append("Consider advancing to more complex musical tasks")
        
        # Aspect-specific recommendations
        for aspect, performance in self.state.aspect_performance.items():
            if len(performance) >= 5:
                recent_avg = np.mean(performance[-5:])
                if recent_avg < 0.4:
                    recommendations.append(f"Strengthen {aspect} training components")
        
        # Theory compliance recommendations
        if len(self.state.theory_compliance_scores) >= 5:
            recent_compliance = np.mean(self.state.theory_compliance_scores[-5:])
            if recent_compliance < 0.6:
                recommendations.append("Increase music theory enforcement weight")
        
        # Phase-specific recommendations
        if self.state.current_phase == TrainingPhase.FOUNDATION:
            recommendations.append("Ensure solid grasp of basic musical concepts")
        elif self.state.current_phase == TrainingPhase.SPECIALIZATION:
            recommendations.append("Focus on genre-specific characteristics")
        elif self.state.current_phase == TrainingPhase.REFINEMENT:
            recommendations.append("Emphasize musical expression and nuance")
        
        return recommendations[:3]  # Limit to top 3
    
    def get_musical_training_summary(self) -> Dict[str, Any]:
        """Get comprehensive musical training summary."""
        
        return {
            "current_phase": self.state.current_phase.value,
            "current_domain": self.state.current_domain.value,
            "epoch": self.state.epoch,
            "aspect_weights": self.state.aspect_weights,
            "musical_coherence": {
                "current": self.state.musical_coherence_history[-1] if self.state.musical_coherence_history else 0.0,
                "trend": self.state.musical_coherence_history[-5:] if len(self.state.musical_coherence_history) >= 5 else [],
                "average": np.mean(self.state.musical_coherence_history) if self.state.musical_coherence_history else 0.0
            },
            "theory_compliance": {
                "current": self.state.theory_compliance_scores[-1] if self.state.theory_compliance_scores else 0.0,
                "average": np.mean(self.state.theory_compliance_scores) if self.state.theory_compliance_scores else 0.0
            },
            "phase_transitions": len(self.state.phase_transitions),
            "domain_adaptations": len(self.state.domain_adaptations),
            "recommendations": self.get_training_recommendations()
        }


def create_classical_training_config() -> MusicalTrainingConfig:
    """Create configuration for classical music training."""
    
    return MusicalTrainingConfig(
        primary_domain=MusicalDomain.CLASSICAL,
        focus_aspects=[
            MusicalAspect.HARMONY,
            MusicalAspect.STRUCTURE,
            MusicalAspect.MELODY
        ],
        enforce_music_theory=True,
        theory_compliance_weight=0.3,
        harmonic_progression_guidance=True,
        progression_loss_weight=0.4,
        structural_guidance=True,
        structure_loss_weight=0.3,
        adaptive_weighting=True
    )


def create_jazz_training_config() -> MusicalTrainingConfig:
    """Create configuration for jazz music training."""
    
    return MusicalTrainingConfig(
        primary_domain=MusicalDomain.JAZZ,
        focus_aspects=[
            MusicalAspect.HARMONY,
            MusicalAspect.RHYTHM,
            MusicalAspect.EXPRESSION
        ],
        enforce_music_theory=True,
        theory_compliance_weight=0.2,
        harmonic_progression_guidance=True,
        progression_loss_weight=0.35,
        rhythmic_pattern_enforcement=True,
        rhythm_loss_weight=0.3,
        style_consistency_enforcement=True,
        style_loss_weight=0.2,
        adaptive_weighting=True
    )


def create_pop_training_config() -> MusicalTrainingConfig:
    """Create configuration for pop music training."""
    
    return MusicalTrainingConfig(
        primary_domain=MusicalDomain.POP,
        focus_aspects=[
            MusicalAspect.MELODY,
            MusicalAspect.RHYTHM,
            MusicalAspect.STRUCTURE
        ],
        enforce_music_theory=True,
        theory_compliance_weight=0.15,
        structural_guidance=True,
        structure_loss_weight=0.25,
        rhythmic_pattern_enforcement=True,
        rhythm_loss_weight=0.25,
        style_consistency_enforcement=True,
        style_loss_weight=0.15,
        adaptive_weighting=True,
        cross_genre_training=True
    )