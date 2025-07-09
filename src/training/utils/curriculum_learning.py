"""
Progressive Training Curriculum System for Aurl.ai Music Generation.

This module implements sophisticated curriculum learning strategies:
- Progressive sequence length training
- Musical complexity progression
- Multi-stage difficulty curriculum
- Adaptive curriculum based on model performance
- Musical domain-specific curriculum strategies
- Curriculum state management and resumability

Designed for optimal music generation model training with gradual complexity increase.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import math
from collections import defaultdict

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class CurriculumStrategy(Enum):
    """Available curriculum learning strategies."""
    LINEAR = "linear"
    EXPONENTIAL = "exponential"
    COSINE = "cosine"
    STEP = "step"
    ADAPTIVE = "adaptive"
    MUSICAL = "musical"


class CurriculumDimension(Enum):
    """Dimensions along which curriculum can progress."""
    SEQUENCE_LENGTH = "sequence_length"
    MUSICAL_COMPLEXITY = "musical_complexity"
    POLYPHONY = "polyphony"
    TEMPO_RANGE = "tempo_range"
    DYNAMIC_RANGE = "dynamic_range"
    HARMONIC_COMPLEXITY = "harmonic_complexity"
    RHYTHMIC_COMPLEXITY = "rhythmic_complexity"


@dataclass
class CurriculumConfig:
    """Configuration for curriculum learning."""
    
    # Strategy configuration
    strategy: CurriculumStrategy = CurriculumStrategy.MUSICAL
    dimensions: List[CurriculumDimension] = field(default_factory=lambda: [
        CurriculumDimension.SEQUENCE_LENGTH,
        CurriculumDimension.MUSICAL_COMPLEXITY
    ])
    
    # Sequence length curriculum
    start_seq_length: int = 256
    end_seq_length: int = 1024
    seq_length_epochs: int = 20
    
    # Musical complexity curriculum
    start_complexity: float = 0.3  # 30% complexity
    end_complexity: float = 1.0    # 100% complexity
    complexity_epochs: int = 30
    
    # Polyphony curriculum
    start_polyphony: int = 2
    end_polyphony: int = 8
    polyphony_epochs: int = 15
    
    # Adaptive curriculum parameters
    performance_threshold: float = 0.1  # Relative improvement threshold
    adaptation_patience: int = 5        # Epochs to wait before adaptation
    min_epoch_per_stage: int = 3       # Minimum epochs per curriculum stage
    
    # Musical domain parameters
    musical_progression_style: str = "conservative"  # conservative, aggressive, balanced
    enable_genre_curriculum: bool = True
    enable_style_progression: bool = True
    
    # State management
    save_curriculum_state: bool = True
    curriculum_state_file: str = "curriculum_state.json"


@dataclass
class CurriculumState:
    """Current state of curriculum learning."""
    
    epoch: int = 0
    stage: int = 0
    current_seq_length: int = 256
    current_complexity: float = 0.3
    current_polyphony: int = 2
    
    # Performance tracking for adaptive curriculum
    recent_performance: List[float] = field(default_factory=list)
    performance_history: Dict[str, List[float]] = field(default_factory=lambda: defaultdict(list))
    stage_start_epoch: int = 0
    
    # Musical curriculum state
    current_genres: List[str] = field(default_factory=lambda: ["simple"])
    current_styles: List[str] = field(default_factory=lambda: ["basic"])
    complexity_weights: Dict[str, float] = field(default_factory=lambda: {
        "rhythm": 0.3, "harmony": 0.3, "melody": 0.3, "structure": 0.1
    })
    
    # Adaptation tracking
    adaptations: List[Dict[str, Any]] = field(default_factory=list)
    last_adaptation_epoch: int = 0


class MusicalComplexityAnalyzer:
    """Analyzes and quantifies musical complexity for curriculum progression."""
    
    def __init__(self):
        self.complexity_metrics = {
            "rhythmic": self._analyze_rhythmic_complexity,
            "harmonic": self._analyze_harmonic_complexity,
            "melodic": self._analyze_melodic_complexity,
            "structural": self._analyze_structural_complexity
        }
    
    def analyze_sample_complexity(self, 
                                midi_data: Any, 
                                tokens: torch.Tensor = None) -> Dict[str, float]:
        """Analyze complexity of a musical sample."""
        
        complexity_scores = {}
        
        # Analyze different complexity dimensions
        for metric_name, analyzer_func in self.complexity_metrics.items():
            try:
                score = analyzer_func(midi_data, tokens)
                complexity_scores[metric_name] = score
            except Exception as e:
                logger.warning(f"Failed to analyze {metric_name} complexity: {e}")
                complexity_scores[metric_name] = 0.5  # Default moderate complexity
        
        # Calculate overall complexity
        weights = {"rhythmic": 0.3, "harmonic": 0.3, "melodic": 0.3, "structural": 0.1}
        overall_complexity = sum(
            weights[metric] * score 
            for metric, score in complexity_scores.items()
            if metric in weights
        )
        
        complexity_scores["overall"] = overall_complexity
        return complexity_scores
    
    def _analyze_rhythmic_complexity(self, midi_data: Any, tokens: torch.Tensor = None) -> float:
        """Analyze rhythmic complexity (note timing patterns, syncopation)."""
        
        if tokens is not None:
            # Analyze from tokens if available
            return self._analyze_token_rhythm_complexity(tokens)
        
        # Fallback to MIDI analysis
        if hasattr(midi_data, 'instruments'):
            note_onsets = []
            for instrument in midi_data.instruments:
                note_onsets.extend([note.start for note in instrument.notes])
            
            if len(note_onsets) < 3:
                return 0.1  # Very simple
            
            # Calculate inter-onset intervals
            note_onsets.sort()
            intervals = np.diff(note_onsets)
            
            # Complexity based on interval variety and regularity
            interval_variety = len(np.unique(np.round(intervals, 2))) / len(intervals)
            interval_regularity = 1.0 - np.std(intervals) / (np.mean(intervals) + 1e-8)
            
            # Higher variety and lower regularity = higher complexity
            complexity = interval_variety * (1.0 - interval_regularity * 0.5)
            return np.clip(complexity, 0.0, 1.0)
        
        return 0.5  # Default moderate complexity
    
    def _analyze_harmonic_complexity(self, midi_data: Any, tokens: torch.Tensor = None) -> float:
        """Analyze harmonic complexity (chord progressions, dissonance)."""
        
        if hasattr(midi_data, 'instruments'):
            # Simple harmony analysis based on simultaneous notes
            all_notes = []
            for instrument in midi_data.instruments:
                for note in instrument.notes:
                    all_notes.append((note.start, note.end, note.pitch))
            
            if len(all_notes) < 2:
                return 0.1  # Monophonic = simple
            
            # Find simultaneous notes (chords)
            time_points = sorted(set([note[0] for note in all_notes] + [note[1] for note in all_notes]))
            chord_complexities = []
            
            for i in range(len(time_points) - 1):
                start_time = time_points[i]
                end_time = time_points[i + 1]
                
                # Find notes active in this time slice
                active_pitches = []
                for note_start, note_end, pitch in all_notes:
                    if note_start <= start_time < note_end:
                        active_pitches.append(pitch)
                
                if len(active_pitches) > 1:
                    # Calculate harmonic complexity based on interval relationships
                    unique_pitches = sorted(set(active_pitches))
                    intervals = []
                    for j in range(len(unique_pitches) - 1):
                        intervals.append(unique_pitches[j + 1] - unique_pitches[j])
                    
                    # Complexity based on number of notes and interval variety
                    note_complexity = min(len(unique_pitches) / 6.0, 1.0)  # Max 6 notes
                    interval_complexity = len(set(intervals)) / max(len(intervals), 1)
                    
                    chord_complexities.append(note_complexity * 0.7 + interval_complexity * 0.3)
            
            return np.mean(chord_complexities) if chord_complexities else 0.3
        
        return 0.5
    
    def _analyze_melodic_complexity(self, midi_data: Any, tokens: torch.Tensor = None) -> float:
        """Analyze melodic complexity (range, contour, leaps)."""
        
        if hasattr(midi_data, 'instruments'):
            # Analyze melody from lead instrument (usually first)
            if not midi_data.instruments:
                return 0.1
            
            lead_instrument = midi_data.instruments[0]
            pitches = [note.pitch for note in lead_instrument.notes]
            
            if len(pitches) < 3:
                return 0.1
            
            # Calculate melodic features
            pitch_range = max(pitches) - min(pitches)
            range_complexity = min(pitch_range / 24.0, 1.0)  # Normalize to 2 octaves
            
            # Melodic intervals (leaps vs steps)
            intervals = [abs(pitches[i+1] - pitches[i]) for i in range(len(pitches)-1)]
            large_leaps = sum(1 for interval in intervals if interval > 4)  # > major third
            leap_complexity = min(large_leaps / len(intervals), 0.5)
            
            # Contour complexity (direction changes)
            directions = [1 if pitches[i+1] > pitches[i] else -1 for i in range(len(pitches)-1)]
            direction_changes = sum(1 for i in range(len(directions)-1) 
                                  if directions[i] != directions[i+1])
            contour_complexity = min(direction_changes / len(directions), 0.5)
            
            # Combine factors
            melodic_complexity = (range_complexity * 0.4 + 
                                leap_complexity * 0.3 + 
                                contour_complexity * 0.3)
            
            return np.clip(melodic_complexity, 0.0, 1.0)
        
        return 0.5
    
    def _analyze_structural_complexity(self, midi_data: Any, tokens: torch.Tensor = None) -> float:
        """Analyze structural complexity (form, repetition patterns)."""
        
        # Simple structural analysis based on piece length and instrument count
        if hasattr(midi_data, 'instruments'):
            duration = midi_data.end_time
            num_instruments = len(midi_data.instruments)
            
            # Longer pieces and more instruments = more complex
            duration_complexity = min(duration / 60.0, 1.0)  # Normalize to 1 minute
            instrument_complexity = min(num_instruments / 4.0, 1.0)  # Normalize to 4 instruments
            
            structural_complexity = duration_complexity * 0.6 + instrument_complexity * 0.4
            return np.clip(structural_complexity, 0.0, 1.0)
        
        return 0.5
    
    def _analyze_token_rhythm_complexity(self, tokens: torch.Tensor) -> float:
        """Analyze rhythmic complexity from token sequence."""
        
        # Simple analysis based on token diversity and patterns
        unique_tokens = len(torch.unique(tokens))
        total_tokens = len(tokens)
        
        if total_tokens == 0:
            return 0.1
        
        # Token diversity as proxy for rhythmic complexity
        diversity = unique_tokens / total_tokens
        
        # Pattern regularity analysis (simplified)
        if total_tokens > 10:
            # Look for repeated patterns
            pattern_complexity = 1.0 - self._calculate_pattern_repetition(tokens)
            complexity = diversity * 0.6 + pattern_complexity * 0.4
        else:
            complexity = diversity
        
        return np.clip(complexity, 0.0, 1.0)
    
    def _calculate_pattern_repetition(self, tokens: torch.Tensor) -> float:
        """Calculate how repetitive the token patterns are."""
        
        tokens_list = tokens.tolist()
        pattern_lengths = [2, 3, 4]  # Look for patterns of these lengths
        
        total_patterns = 0
        repeated_patterns = 0
        
        for pattern_len in pattern_lengths:
            if len(tokens_list) < pattern_len * 2:
                continue
            
            patterns = {}
            for i in range(len(tokens_list) - pattern_len + 1):
                pattern = tuple(tokens_list[i:i + pattern_len])
                patterns[pattern] = patterns.get(pattern, 0) + 1
                total_patterns += 1
            
            # Count repeated patterns
            for count in patterns.values():
                if count > 1:
                    repeated_patterns += count - 1
        
        if total_patterns == 0:
            return 0.0
        
        return repeated_patterns / total_patterns


class ProgressiveCurriculumScheduler:
    """
    Progressive curriculum scheduler that gradually increases training difficulty.
    
    Features:
    - Multiple curriculum dimensions (sequence length, complexity, etc.)
    - Adaptive progression based on model performance
    - Musical domain-specific curriculum strategies
    - State persistence for resumable training
    """
    
    def __init__(self, config: CurriculumConfig):
        self.config = config
        self.state = CurriculumState()
        self.complexity_analyzer = MusicalComplexityAnalyzer()
        
        # Initialize curriculum state
        self.state.current_seq_length = config.start_seq_length
        self.state.current_complexity = config.start_complexity
        self.state.current_polyphony = config.start_polyphony
        
        logger.info(f"Initialized curriculum scheduler with strategy: {config.strategy}")
        logger.info(f"Curriculum dimensions: {[d.value for d in config.dimensions]}")
    
    def get_current_curriculum_params(self) -> Dict[str, Any]:
        """Get current curriculum parameters for training."""
        
        return {
            "sequence_length": self.state.current_seq_length,
            "complexity_threshold": self.state.current_complexity,
            "max_polyphony": self.state.current_polyphony,
            "allowed_genres": self.state.current_genres.copy(),
            "allowed_styles": self.state.current_styles.copy(),
            "complexity_weights": self.state.complexity_weights.copy(),
            "stage": self.state.stage,
            "epoch": self.state.epoch
        }
    
    def should_include_sample(self, sample_info: Dict[str, Any]) -> bool:
        """Determine if a sample should be included based on current curriculum."""
        
        # Check sequence length
        if sample_info.get("sequence_length", 0) > self.state.current_seq_length:
            return False
        
        # Check musical complexity
        sample_complexity = sample_info.get("complexity", 0.5)
        if sample_complexity > self.state.current_complexity:
            return False
        
        # Check polyphony
        sample_polyphony = sample_info.get("max_polyphony", 1)
        if sample_polyphony > self.state.current_polyphony:
            return False
        
        # Check genre/style if enabled
        if self.config.enable_genre_curriculum:
            sample_genre = sample_info.get("genre", "unknown")
            if sample_genre not in self.state.current_genres and "all" not in self.state.current_genres:
                return False
        
        return True
    
    def update_curriculum(self, epoch: int, performance_metrics: Dict[str, float] = None):
        """Update curriculum based on current epoch and performance."""
        
        self.state.epoch = epoch
        
        # Track performance for adaptive curriculum
        if performance_metrics and self.config.strategy == CurriculumStrategy.ADAPTIVE:
            self._update_performance_tracking(performance_metrics)
        
        # Update curriculum parameters based on strategy
        if self.config.strategy == CurriculumStrategy.LINEAR:
            self._update_linear_curriculum(epoch)
        elif self.config.strategy == CurriculumStrategy.EXPONENTIAL:
            self._update_exponential_curriculum(epoch)
        elif self.config.strategy == CurriculumStrategy.COSINE:
            self._update_cosine_curriculum(epoch)
        elif self.config.strategy == CurriculumStrategy.STEP:
            self._update_step_curriculum(epoch)
        elif self.config.strategy == CurriculumStrategy.ADAPTIVE:
            self._update_adaptive_curriculum(epoch, performance_metrics)
        elif self.config.strategy == CurriculumStrategy.MUSICAL:
            self._update_musical_curriculum(epoch, performance_metrics)
        
        # Log curriculum updates
        self._log_curriculum_update()
        
        # Save state if configured
        if self.config.save_curriculum_state:
            self.save_state()
    
    def _update_linear_curriculum(self, epoch: int):
        """Update curriculum using linear progression."""
        
        for dimension in self.config.dimensions:
            if dimension == CurriculumDimension.SEQUENCE_LENGTH:
                progress = min(epoch / self.config.seq_length_epochs, 1.0)
                self.state.current_seq_length = int(
                    self.config.start_seq_length + 
                    progress * (self.config.end_seq_length - self.config.start_seq_length)
                )
            
            elif dimension == CurriculumDimension.MUSICAL_COMPLEXITY:
                progress = min(epoch / self.config.complexity_epochs, 1.0)
                self.state.current_complexity = (
                    self.config.start_complexity + 
                    progress * (self.config.end_complexity - self.config.start_complexity)
                )
            
            elif dimension == CurriculumDimension.POLYPHONY:
                progress = min(epoch / self.config.polyphony_epochs, 1.0)
                self.state.current_polyphony = int(
                    self.config.start_polyphony + 
                    progress * (self.config.end_polyphony - self.config.start_polyphony)
                )
    
    def _update_exponential_curriculum(self, epoch: int):
        """Update curriculum using exponential progression."""
        
        for dimension in self.config.dimensions:
            if dimension == CurriculumDimension.SEQUENCE_LENGTH:
                progress = min(epoch / self.config.seq_length_epochs, 1.0)
                exp_progress = 1.0 - math.exp(-3 * progress)  # Exponential curve
                self.state.current_seq_length = int(
                    self.config.start_seq_length + 
                    exp_progress * (self.config.end_seq_length - self.config.start_seq_length)
                )
            
            elif dimension == CurriculumDimension.MUSICAL_COMPLEXITY:
                progress = min(epoch / self.config.complexity_epochs, 1.0)
                exp_progress = 1.0 - math.exp(-3 * progress)
                self.state.current_complexity = (
                    self.config.start_complexity + 
                    exp_progress * (self.config.end_complexity - self.config.start_complexity)
                )
    
    def _update_cosine_curriculum(self, epoch: int):
        """Update curriculum using cosine progression."""
        
        for dimension in self.config.dimensions:
            if dimension == CurriculumDimension.SEQUENCE_LENGTH:
                progress = min(epoch / self.config.seq_length_epochs, 1.0)
                cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
                self.state.current_seq_length = int(
                    self.config.start_seq_length + 
                    cosine_progress * (self.config.end_seq_length - self.config.start_seq_length)
                )
            
            elif dimension == CurriculumDimension.MUSICAL_COMPLEXITY:
                progress = min(epoch / self.config.complexity_epochs, 1.0)
                cosine_progress = 0.5 * (1 - math.cos(math.pi * progress))
                self.state.current_complexity = (
                    self.config.start_complexity + 
                    cosine_progress * (self.config.end_complexity - self.config.start_complexity)
                )
    
    def _update_step_curriculum(self, epoch: int):
        """Update curriculum using step progression."""
        
        # Define step thresholds
        seq_length_steps = [
            (0, self.config.start_seq_length),
            (self.config.seq_length_epochs // 3, (self.config.start_seq_length + self.config.end_seq_length) // 2),
            (2 * self.config.seq_length_epochs // 3, self.config.end_seq_length)
        ]
        
        complexity_steps = [
            (0, self.config.start_complexity),
            (self.config.complexity_epochs // 3, (self.config.start_complexity + self.config.end_complexity) / 2),
            (2 * self.config.complexity_epochs // 3, self.config.end_complexity)
        ]
        
        # Update sequence length
        if CurriculumDimension.SEQUENCE_LENGTH in self.config.dimensions:
            for step_epoch, seq_length in reversed(seq_length_steps):
                if epoch >= step_epoch:
                    self.state.current_seq_length = seq_length
                    break
        
        # Update complexity
        if CurriculumDimension.MUSICAL_COMPLEXITY in self.config.dimensions:
            for step_epoch, complexity in reversed(complexity_steps):
                if epoch >= step_epoch:
                    self.state.current_complexity = complexity
                    break
    
    def _update_adaptive_curriculum(self, epoch: int, performance_metrics: Dict[str, float]):
        """Update curriculum adaptively based on performance."""
        
        if not performance_metrics or len(self.state.recent_performance) < self.config.adaptation_patience:
            # Not enough data for adaptation, use linear progression
            self._update_linear_curriculum(epoch)
            return
        
        # Check if model is performing well enough to advance
        recent_avg = np.mean(self.state.recent_performance[-self.config.adaptation_patience:])
        older_avg = np.mean(self.state.recent_performance[-2*self.config.adaptation_patience:-self.config.adaptation_patience])
        
        performance_improvement = (older_avg - recent_avg) / max(abs(older_avg), 1e-8)
        
        # Advance curriculum if performance is improving
        if (performance_improvement > self.config.performance_threshold and 
            epoch - self.state.stage_start_epoch >= self.config.min_epoch_per_stage):
            
            self._advance_curriculum_stage()
            
        # Otherwise maintain current level
    
    def _update_musical_curriculum(self, epoch: int, performance_metrics: Dict[str, float]):
        """Update curriculum using musical domain knowledge."""
        
        # Musical progression strategy
        if self.config.musical_progression_style == "conservative":
            # Slow, careful progression focusing on musical quality
            base_progress = min(epoch / (self.config.complexity_epochs * 1.5), 1.0)
        elif self.config.musical_progression_style == "aggressive":
            # Fast progression for quick exploration
            base_progress = min(epoch / (self.config.complexity_epochs * 0.7), 1.0)
        else:  # balanced
            base_progress = min(epoch / self.config.complexity_epochs, 1.0)
        
        # Musical curriculum progression
        self._update_musical_genre_progression(base_progress)
        self._update_musical_complexity_weights(base_progress)
        
        # Standard dimension updates with musical adjustments
        self.state.current_seq_length = int(
            self.config.start_seq_length + 
            base_progress * (self.config.end_seq_length - self.config.start_seq_length)
        )
        
        self.state.current_complexity = (
            self.config.start_complexity + 
            base_progress * (self.config.end_complexity - self.config.start_complexity)
        )
    
    def _update_musical_genre_progression(self, progress: float):
        """Update allowed genres based on musical curriculum progression."""
        
        # Define genre progression from simple to complex
        genre_stages = [
            (0.0, ["simple", "folk"]),
            (0.2, ["simple", "folk", "pop"]),
            (0.4, ["simple", "folk", "pop", "rock"]),
            (0.6, ["simple", "folk", "pop", "rock", "jazz"]),
            (0.8, ["simple", "folk", "pop", "rock", "jazz", "classical"]),
            (1.0, ["all"])  # All genres allowed
        ]
        
        for threshold, genres in reversed(genre_stages):
            if progress >= threshold:
                self.state.current_genres = genres
                break
    
    def _update_musical_complexity_weights(self, progress: float):
        """Update complexity weights for different musical aspects."""
        
        # Start with rhythm emphasis, gradually add harmony and melody complexity
        if progress < 0.3:
            # Focus on rhythm
            self.state.complexity_weights = {
                "rhythm": 0.6, "harmony": 0.2, "melody": 0.2, "structure": 0.0
            }
        elif progress < 0.6:
            # Add harmony
            self.state.complexity_weights = {
                "rhythm": 0.4, "harmony": 0.4, "melody": 0.2, "structure": 0.0
            }
        elif progress < 0.9:
            # Add melody complexity
            self.state.complexity_weights = {
                "rhythm": 0.3, "harmony": 0.3, "melody": 0.3, "structure": 0.1
            }
        else:
            # Full complexity
            self.state.complexity_weights = {
                "rhythm": 0.25, "harmony": 0.25, "melody": 0.25, "structure": 0.25
            }
    
    def _update_performance_tracking(self, performance_metrics: Dict[str, float]):
        """Update performance tracking for adaptive curriculum."""
        
        # Use primary metric (usually loss) for adaptation
        primary_metric = performance_metrics.get("total_loss", 
                                                performance_metrics.get("loss", 0.0))
        
        self.state.recent_performance.append(primary_metric)
        
        # Keep only recent performance history
        if len(self.state.recent_performance) > 2 * self.config.adaptation_patience:
            self.state.recent_performance = self.state.recent_performance[-2 * self.config.adaptation_patience:]
        
        # Track all metrics
        for metric_name, value in performance_metrics.items():
            self.state.performance_history[metric_name].append(value)
    
    def _advance_curriculum_stage(self):
        """Advance to next curriculum stage."""
        
        self.state.stage += 1
        self.state.stage_start_epoch = self.state.epoch
        self.state.last_adaptation_epoch = self.state.epoch
        
        # Record adaptation
        adaptation_info = {
            "epoch": self.state.epoch,
            "stage": self.state.stage,
            "seq_length": self.state.current_seq_length,
            "complexity": self.state.current_complexity,
            "trigger": "performance_improvement"
        }
        self.state.adaptations.append(adaptation_info)
        
        logger.info(f"Advanced curriculum to stage {self.state.stage} at epoch {self.state.epoch}")
    
    def _log_curriculum_update(self):
        """Log current curriculum state."""
        
        if self.state.epoch % 10 == 0:  # Log every 10 epochs
            logger.info(f"Curriculum Update - Epoch {self.state.epoch}:")
            logger.info(f"  Sequence Length: {self.state.current_seq_length}")
            logger.info(f"  Complexity Threshold: {self.state.current_complexity:.3f}")
            logger.info(f"  Max Polyphony: {self.state.current_polyphony}")
            logger.info(f"  Allowed Genres: {self.state.current_genres}")
            logger.info(f"  Stage: {self.state.stage}")
    
    def save_state(self, filepath: Path = None):
        """Save curriculum state to file."""
        
        if filepath is None:
            filepath = Path(self.config.curriculum_state_file)
        
        state_dict = {
            "config": {
                "strategy": self.config.strategy.value,
                "dimensions": [d.value for d in self.config.dimensions],
                "start_seq_length": self.config.start_seq_length,
                "end_seq_length": self.config.end_seq_length,
                "seq_length_epochs": self.config.seq_length_epochs,
                "start_complexity": self.config.start_complexity,
                "end_complexity": self.config.end_complexity,
                "complexity_epochs": self.config.complexity_epochs
            },
            "state": {
                "epoch": self.state.epoch,
                "stage": self.state.stage,
                "current_seq_length": self.state.current_seq_length,
                "current_complexity": self.state.current_complexity,
                "current_polyphony": self.state.current_polyphony,
                "current_genres": self.state.current_genres,
                "current_styles": self.state.current_styles,
                "complexity_weights": self.state.complexity_weights,
                "stage_start_epoch": self.state.stage_start_epoch,
                "last_adaptation_epoch": self.state.last_adaptation_epoch,
                "recent_performance": self.state.recent_performance[-20:],  # Save last 20
                "adaptations": self.state.adaptations
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_dict, f, indent=2)
        
        logger.info(f"Saved curriculum state to {filepath}")
    
    def load_state(self, filepath: Path):
        """Load curriculum state from file."""
        
        with open(filepath, 'r') as f:
            state_dict = json.load(f)
        
        # Restore state
        saved_state = state_dict.get("state", {})
        self.state.epoch = saved_state.get("epoch", 0)
        self.state.stage = saved_state.get("stage", 0)
        self.state.current_seq_length = saved_state.get("current_seq_length", self.config.start_seq_length)
        self.state.current_complexity = saved_state.get("current_complexity", self.config.start_complexity)
        self.state.current_polyphony = saved_state.get("current_polyphony", self.config.start_polyphony)
        self.state.current_genres = saved_state.get("current_genres", ["simple"])
        self.state.current_styles = saved_state.get("current_styles", ["basic"])
        self.state.complexity_weights = saved_state.get("complexity_weights", 
                                                      {"rhythm": 0.3, "harmony": 0.3, "melody": 0.3, "structure": 0.1})
        self.state.stage_start_epoch = saved_state.get("stage_start_epoch", 0)
        self.state.last_adaptation_epoch = saved_state.get("last_adaptation_epoch", 0)
        self.state.recent_performance = saved_state.get("recent_performance", [])
        self.state.adaptations = saved_state.get("adaptations", [])
        
        logger.info(f"Loaded curriculum state from {filepath}")
        logger.info(f"Resumed at epoch {self.state.epoch}, stage {self.state.stage}")
    
    def get_curriculum_summary(self) -> Dict[str, Any]:
        """Get comprehensive curriculum state summary."""
        
        return {
            "epoch": self.state.epoch,
            "stage": self.state.stage,
            "strategy": self.config.strategy.value,
            "dimensions": [d.value for d in self.config.dimensions],
            "current_parameters": self.get_current_curriculum_params(),
            "performance_trend": self._calculate_performance_trend(),
            "adaptations_count": len(self.state.adaptations),
            "last_adaptation": self.state.last_adaptation_epoch
        }
    
    def _calculate_performance_trend(self) -> str:
        """Calculate current performance trend."""
        
        if len(self.state.recent_performance) < 5:
            return "insufficient_data"
        
        recent = self.state.recent_performance[-5:]
        trend = np.polyfit(range(len(recent)), recent, 1)[0]
        
        if trend < -0.001:
            return "improving"
        elif trend > 0.001:
            return "degrading"
        else:
            return "stable"


def create_musical_curriculum_config(**kwargs) -> CurriculumConfig:
    """Create curriculum configuration optimized for music generation."""
    
    defaults = {
        "strategy": CurriculumStrategy.MUSICAL,
        "dimensions": [
            CurriculumDimension.SEQUENCE_LENGTH,
            CurriculumDimension.MUSICAL_COMPLEXITY,
            CurriculumDimension.POLYPHONY
        ],
        "start_seq_length": 256,
        "end_seq_length": 1024,
        "seq_length_epochs": 25,
        "start_complexity": 0.2,
        "end_complexity": 0.9,
        "complexity_epochs": 40,
        "start_polyphony": 2,
        "end_polyphony": 6,
        "polyphony_epochs": 20,
        "musical_progression_style": "balanced",
        "enable_genre_curriculum": True,
        "enable_style_progression": True,
        "save_curriculum_state": True
    }
    
    # Override with provided kwargs
    defaults.update(kwargs)
    
    return CurriculumConfig(**defaults)