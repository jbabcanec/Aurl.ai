"""
Real-Time Musical Sample Quality Evaluation for Aurl.ai Music Generation.

This module implements comprehensive real-time quality assessment:
- Real-time musical quality scoring during training
- Multi-dimensional musical analysis (rhythm, harmony, melody, structure)
- Performance-efficient quality metrics
- Musical Turing test integration
- Adaptive quality thresholds
- Quality-based training feedback

Designed for production training with minimal computational overhead.
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
import time
from collections import defaultdict, deque
import threading
import queue
from concurrent.futures import ThreadPoolExecutor

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class QualityDimension(Enum):
    """Dimensions of musical quality assessment."""
    RHYTHMIC_COHERENCE = "rhythmic_coherence"
    HARMONIC_CONSISTENCY = "harmonic_consistency"
    MELODIC_FLOW = "melodic_flow"
    STRUCTURAL_INTEGRITY = "structural_integrity"
    STYLISTIC_AUTHENTICITY = "stylistic_authenticity"
    TEMPORAL_CONSISTENCY = "temporal_consistency"
    MUSICAL_CREATIVITY = "musical_creativity"
    OVERALL_QUALITY = "overall_quality"


class EvaluationMode(Enum):
    """Modes of quality evaluation."""
    FAST = "fast"                    # Lightweight metrics only
    BALANCED = "balanced"            # Balance of speed and accuracy
    COMPREHENSIVE = "comprehensive"  # Full analysis
    ADAPTIVE = "adaptive"           # Adapt based on training phase


@dataclass
class QualityConfig:
    """Configuration for quality evaluation."""
    
    # Evaluation mode
    mode: EvaluationMode = EvaluationMode.BALANCED
    
    # Quality dimensions to evaluate
    enabled_dimensions: List[QualityDimension] = field(default_factory=lambda: [
        QualityDimension.RHYTHMIC_COHERENCE,
        QualityDimension.HARMONIC_CONSISTENCY,
        QualityDimension.MELODIC_FLOW,
        QualityDimension.OVERALL_QUALITY
    ])
    
    # Dimension weights for overall score
    dimension_weights: Dict[str, float] = field(default_factory=lambda: {
        "rhythmic_coherence": 0.25,
        "harmonic_consistency": 0.25,
        "melodic_flow": 0.25,
        "structural_integrity": 0.1,
        "stylistic_authenticity": 0.1,
        "temporal_consistency": 0.05
    })
    
    # Performance settings
    batch_evaluation: bool = True
    max_batch_size: int = 16
    async_evaluation: bool = True
    max_workers: int = 4
    
    # Sampling settings
    evaluation_frequency: int = 10  # Evaluate every N batches
    samples_per_evaluation: int = 4
    max_sequence_length: int = 512
    
    # Quality thresholds
    quality_thresholds: Dict[str, float] = field(default_factory=lambda: {
        "excellent": 0.8,
        "good": 0.6,
        "acceptable": 0.4,
        "poor": 0.2
    })
    
    # Adaptive settings
    adapt_to_training_phase: bool = True
    adapt_frequency: int = 100
    quality_history_window: int = 50
    
    # Musical domain settings
    genre_specific_evaluation: bool = True
    style_consistency_weight: float = 0.3
    musical_theory_compliance: bool = True
    
    # Efficiency settings
    cache_evaluations: bool = True
    cache_size: int = 1000
    skip_identical_samples: bool = True


@dataclass
class QualityResult:
    """Result of quality evaluation."""
    
    timestamp: float
    sample_id: str
    
    # Dimension scores
    dimension_scores: Dict[str, float] = field(default_factory=dict)
    overall_score: float = 0.0
    
    # Analysis details
    sequence_length: int = 0
    unique_tokens: int = 0
    polyphony_level: float = 0.0
    
    # Musical features
    rhythmic_features: Dict[str, float] = field(default_factory=dict)
    harmonic_features: Dict[str, float] = field(default_factory=dict)
    melodic_features: Dict[str, float] = field(default_factory=dict)
    
    # Performance metrics
    evaluation_time: float = 0.0
    computation_cost: str = "unknown"
    
    # Quality category
    quality_category: str = "unknown"
    
    # Recommendations
    improvement_suggestions: List[str] = field(default_factory=list)


class RhythmicAnalyzer:
    """Analyzes rhythmic coherence and quality."""
    
    def __init__(self):
        self.tempo_tolerance = 0.1
        self.rhythm_pattern_cache = {}
    
    def analyze_rhythmic_coherence(self, 
                                 tokens: torch.Tensor,
                                 token_to_event_fn: Callable = None) -> Dict[str, float]:
        """Analyze rhythmic coherence of musical sequence."""
        
        if tokens.dim() > 1:
            tokens = tokens.flatten()
        
        # Extract time-related tokens (simplified analysis)
        time_tokens = self._extract_time_tokens(tokens)
        
        if len(time_tokens) < 3:
            return {"coherence": 0.1, "regularity": 0.1, "complexity": 0.1}
        
        # Analyze rhythm patterns
        rhythm_coherence = self._analyze_rhythm_coherence(time_tokens)
        rhythm_regularity = self._analyze_rhythm_regularity(time_tokens)
        rhythm_complexity = self._analyze_rhythm_complexity(time_tokens)
        
        return {
            "coherence": rhythm_coherence,
            "regularity": rhythm_regularity,
            "complexity": rhythm_complexity,
            "overall": (rhythm_coherence + rhythm_regularity + rhythm_complexity) / 3.0
        }
    
    def _extract_time_tokens(self, tokens: torch.Tensor) -> List[int]:
        """Extract time-related tokens from sequence."""
        
        # Simplified: assume time tokens are in certain ranges
        # In practice, this would use the vocabulary mapping
        time_tokens = []
        
        for token in tokens:
            token_val = token.item()
            # Heuristic: time shift tokens often in specific ranges
            if 100 <= token_val <= 200:  # Example range
                time_tokens.append(token_val)
        
        return time_tokens
    
    def _analyze_rhythm_coherence(self, time_tokens: List[int]) -> float:
        """Analyze how coherent the rhythm patterns are."""
        
        if len(time_tokens) < 3:
            return 0.1
        
        # Calculate intervals between time tokens
        intervals = [time_tokens[i+1] - time_tokens[i] for i in range(len(time_tokens)-1)]
        
        if not intervals:
            return 0.1
        
        # Coherence based on interval consistency
        interval_variance = np.var(intervals)
        mean_interval = np.mean(intervals)
        
        if mean_interval == 0:
            return 0.1
        
        # Normalize variance by mean
        coherence = 1.0 / (1.0 + interval_variance / (mean_interval + 1e-8))
        
        return np.clip(coherence, 0.0, 1.0)
    
    def _analyze_rhythm_regularity(self, time_tokens: List[int]) -> float:
        """Analyze rhythmic regularity."""
        
        if len(time_tokens) < 4:
            return 0.1
        
        # Look for repeating patterns
        pattern_scores = []
        
        for pattern_length in [2, 3, 4]:
            if len(time_tokens) >= pattern_length * 2:
                pattern_score = self._find_repeating_patterns(time_tokens, pattern_length)
                pattern_scores.append(pattern_score)
        
        if not pattern_scores:
            return 0.1
        
        return np.mean(pattern_scores)
    
    def _analyze_rhythm_complexity(self, time_tokens: List[int]) -> float:
        """Analyze rhythmic complexity."""
        
        if len(time_tokens) < 2:
            return 0.1
        
        # Complexity based on unique intervals and patterns
        intervals = [abs(time_tokens[i+1] - time_tokens[i]) for i in range(len(time_tokens)-1)]
        unique_intervals = len(set(intervals))
        
        # Normalize by sequence length
        complexity = unique_intervals / max(len(intervals), 1)
        
        # Optimal complexity is neither too simple nor too complex
        optimal_complexity = 0.5
        complexity_score = 1.0 - abs(complexity - optimal_complexity) * 2.0
        
        return np.clip(complexity_score, 0.0, 1.0)
    
    def _find_repeating_patterns(self, tokens: List[int], pattern_length: int) -> float:
        """Find repeating rhythmic patterns."""
        
        if len(tokens) < pattern_length * 2:
            return 0.0
        
        patterns = {}
        
        for i in range(len(tokens) - pattern_length + 1):
            pattern = tuple(tokens[i:i + pattern_length])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        if not patterns:
            return 0.0
        
        # Score based on pattern repetition
        total_patterns = len(tokens) - pattern_length + 1
        repeated_patterns = sum(count for count in patterns.values() if count > 1)
        
        regularity = repeated_patterns / max(total_patterns, 1)
        
        return np.clip(regularity, 0.0, 1.0)


class HarmonicAnalyzer:
    """Analyzes harmonic consistency and quality."""
    
    def __init__(self):
        self.chord_progressions = self._load_common_progressions()
        self.key_signatures = self._load_key_signatures()
    
    def analyze_harmonic_consistency(self,
                                   tokens: torch.Tensor,
                                   token_to_event_fn: Callable = None) -> Dict[str, float]:
        """Analyze harmonic consistency of musical sequence."""
        
        if tokens.dim() > 1:
            tokens = tokens.flatten()
        
        # Extract note-related tokens
        note_tokens = self._extract_note_tokens(tokens)
        
        if len(note_tokens) < 3:
            return {"consistency": 0.1, "progression": 0.1, "dissonance": 0.5}
        
        # Analyze different harmonic aspects
        harmonic_consistency = self._analyze_harmonic_consistency(note_tokens)
        chord_progression_quality = self._analyze_chord_progressions(note_tokens)
        dissonance_control = self._analyze_dissonance(note_tokens)
        
        return {
            "consistency": harmonic_consistency,
            "progression": chord_progression_quality,
            "dissonance": dissonance_control,
            "overall": (harmonic_consistency + chord_progression_quality + dissonance_control) / 3.0
        }
    
    def _extract_note_tokens(self, tokens: torch.Tensor) -> List[int]:
        """Extract note-related tokens from sequence."""
        
        note_tokens = []
        
        for token in tokens:
            token_val = token.item()
            # Heuristic: note tokens often in MIDI range
            if 21 <= token_val <= 108:  # Piano key range
                note_tokens.append(token_val)
        
        return note_tokens
    
    def _analyze_harmonic_consistency(self, note_tokens: List[int]) -> float:
        """Analyze overall harmonic consistency."""
        
        if len(note_tokens) < 3:
            return 0.1
        
        # Group notes into potential chords (simplified)
        note_groups = self._group_simultaneous_notes(note_tokens)
        
        if not note_groups:
            return 0.1
        
        # Analyze each chord
        chord_scores = []
        
        for notes in note_groups:
            if len(notes) >= 2:
                chord_quality = self._analyze_chord_quality(notes)
                chord_scores.append(chord_quality)
        
        if not chord_scores:
            return 0.1
        
        return np.mean(chord_scores)
    
    def _analyze_chord_progressions(self, note_tokens: List[int]) -> float:
        """Analyze quality of chord progressions."""
        
        # Simplified progression analysis
        note_groups = self._group_simultaneous_notes(note_tokens)
        
        if len(note_groups) < 2:
            return 0.1
        
        # Analyze transitions between chords
        progression_scores = []
        
        for i in range(len(note_groups) - 1):
            current_chord = note_groups[i]
            next_chord = note_groups[i + 1]
            
            if len(current_chord) >= 2 and len(next_chord) >= 2:
                transition_quality = self._analyze_chord_transition(current_chord, next_chord)
                progression_scores.append(transition_quality)
        
        if not progression_scores:
            return 0.1
        
        return np.mean(progression_scores)
    
    def _analyze_dissonance(self, note_tokens: List[int]) -> float:
        """Analyze dissonance control (higher score = better control)."""
        
        note_groups = self._group_simultaneous_notes(note_tokens)
        
        if not note_groups:
            return 0.5
        
        dissonance_scores = []
        
        for notes in note_groups:
            if len(notes) >= 2:
                dissonance_level = self._calculate_dissonance_level(notes)
                # Convert to quality score (controlled dissonance is good)
                dissonance_quality = self._dissonance_to_quality_score(dissonance_level)
                dissonance_scores.append(dissonance_quality)
        
        if not dissonance_scores:
            return 0.5
        
        return np.mean(dissonance_scores)
    
    def _group_simultaneous_notes(self, note_tokens: List[int]) -> List[List[int]]:
        """Group notes that might be played simultaneously."""
        
        # Simplified: group consecutive notes as potential chords
        groups = []
        current_group = []
        
        for i, note in enumerate(note_tokens):
            if not current_group:
                current_group = [note]
            else:
                # If notes are close together, consider them simultaneous
                if abs(note - current_group[-1]) <= 12:  # Within an octave
                    current_group.append(note)
                else:
                    if len(current_group) >= 2:
                        groups.append(current_group)
                    current_group = [note]
        
        if len(current_group) >= 2:
            groups.append(current_group)
        
        return groups
    
    def _analyze_chord_quality(self, notes: List[int]) -> float:
        """Analyze quality of a chord."""
        
        if len(notes) < 2:
            return 0.1
        
        # Sort notes
        sorted_notes = sorted(set(notes))
        
        if len(sorted_notes) < 2:
            return 0.1
        
        # Calculate intervals
        intervals = [sorted_notes[i+1] - sorted_notes[i] for i in range(len(sorted_notes)-1)]
        
        # Score based on common chord intervals
        common_intervals = [3, 4, 7]  # Minor 3rd, Major 3rd, Perfect 5th
        
        score = 0.0
        for interval in intervals:
            interval_mod = interval % 12
            if interval_mod in common_intervals:
                score += 1.0
            elif interval_mod in [2, 5, 9, 10]:  # Other consonant intervals
                score += 0.5
        
        # Normalize by number of intervals
        return score / max(len(intervals), 1)
    
    def _analyze_chord_transition(self, chord1: List[int], chord2: List[int]) -> float:
        """Analyze quality of transition between two chords."""
        
        # Simplified: good transitions have some common notes or smooth voice leading
        set1 = set(note % 12 for note in chord1)  # Reduce to pitch classes
        set2 = set(note % 12 for note in chord2)
        
        # Common notes between chords
        common_notes = len(set1.intersection(set2))
        total_notes = len(set1.union(set2))
        
        if total_notes == 0:
            return 0.1
        
        # Score based on common notes (some common notes are good, all different can also be good)
        common_ratio = common_notes / total_notes
        
        # Optimal is some common notes but not all
        if 0.2 <= common_ratio <= 0.7:
            return 0.8
        elif common_ratio > 0.7:
            return 0.6  # Too similar
        else:
            return 0.4  # Too different
    
    def _calculate_dissonance_level(self, notes: List[int]) -> float:
        """Calculate dissonance level of a chord."""
        
        if len(notes) < 2:
            return 0.0
        
        sorted_notes = sorted(set(notes))
        dissonance = 0.0
        
        for i in range(len(sorted_notes)):
            for j in range(i + 1, len(sorted_notes)):
                interval = (sorted_notes[j] - sorted_notes[i]) % 12
                
                # Dissonance values for different intervals
                dissonance_values = {
                    0: 0.0,   # Unison
                    1: 0.9,   # Minor 2nd
                    2: 0.7,   # Major 2nd
                    3: 0.2,   # Minor 3rd
                    4: 0.1,   # Major 3rd
                    5: 0.3,   # Perfect 4th
                    6: 0.8,   # Tritone
                    7: 0.1,   # Perfect 5th
                    8: 0.2,   # Minor 6th
                    9: 0.2,   # Major 6th
                    10: 0.4,  # Minor 7th
                    11: 0.5   # Major 7th
                }
                
                dissonance += dissonance_values.get(interval, 0.5)
        
        # Normalize by number of intervals
        num_intervals = len(sorted_notes) * (len(sorted_notes) - 1) // 2
        return dissonance / max(num_intervals, 1)
    
    def _dissonance_to_quality_score(self, dissonance_level: float) -> float:
        """Convert dissonance level to quality score."""
        
        # Some dissonance is good for musical interest
        # Optimal dissonance is around 0.2-0.4
        if 0.1 <= dissonance_level <= 0.4:
            return 0.9
        elif dissonance_level <= 0.6:
            return 0.7
        elif dissonance_level <= 0.8:
            return 0.4
        else:
            return 0.2
    
    def _load_common_progressions(self) -> List[List[int]]:
        """Load common chord progressions."""
        
        # Simplified common progressions in C major
        return [
            [0, 5, 7],   # I-V-vi
            [0, 3, 5, 0], # I-IV-V-I
            [0, 9, 5, 7], # I-vi-V-vi
        ]
    
    def _load_key_signatures(self) -> Dict[str, List[int]]:
        """Load key signatures."""
        
        return {
            "C_major": [0, 2, 4, 5, 7, 9, 11],
            "G_major": [0, 2, 4, 6, 7, 9, 11],
            "F_major": [0, 2, 4, 5, 7, 9, 10],
        }


class MelodicAnalyzer:
    """Analyzes melodic flow and quality."""
    
    def __init__(self):
        self.optimal_leap_ratio = 0.3
        self.max_leap_size = 12
    
    def analyze_melodic_flow(self,
                           tokens: torch.Tensor,
                           token_to_event_fn: Callable = None) -> Dict[str, float]:
        """Analyze melodic flow quality."""
        
        if tokens.dim() > 1:
            tokens = tokens.flatten()
        
        # Extract melodic line (simplified: take note tokens in sequence)
        melody_notes = self._extract_melodic_line(tokens)
        
        if len(melody_notes) < 3:
            return {"contour": 0.1, "intervals": 0.1, "direction": 0.1}
        
        # Analyze different aspects of melody
        contour_quality = self._analyze_melodic_contour(melody_notes)
        interval_quality = self._analyze_melodic_intervals(melody_notes)
        direction_quality = self._analyze_melodic_direction(melody_notes)
        
        return {
            "contour": contour_quality,
            "intervals": interval_quality,
            "direction": direction_quality,
            "overall": (contour_quality + interval_quality + direction_quality) / 3.0
        }
    
    def _extract_melodic_line(self, tokens: torch.Tensor) -> List[int]:
        """Extract main melodic line from tokens."""
        
        melody_notes = []
        
        for token in tokens:
            token_val = token.item()
            # Simplified: take notes in typical melody range
            if 60 <= token_val <= 84:  # C4 to C6
                melody_notes.append(token_val)
        
        return melody_notes
    
    def _analyze_melodic_contour(self, melody_notes: List[int]) -> float:
        """Analyze quality of melodic contour."""
        
        if len(melody_notes) < 3:
            return 0.1
        
        # Calculate contour (up/down/same)
        contour = []
        for i in range(len(melody_notes) - 1):
            diff = melody_notes[i + 1] - melody_notes[i]
            if diff > 0:
                contour.append(1)    # Up
            elif diff < 0:
                contour.append(-1)   # Down
            else:
                contour.append(0)    # Same
        
        if not contour:
            return 0.1
        
        # Good contour has balance and interesting shape
        direction_changes = sum(1 for i in range(len(contour) - 1) 
                              if contour[i] != contour[i + 1] and contour[i] != 0 and contour[i + 1] != 0)
        
        # Optimal contour has some direction changes but not too many
        contour_ratio = direction_changes / max(len(contour) - 1, 1)
        
        if 0.2 <= contour_ratio <= 0.6:
            return 0.9
        elif contour_ratio <= 0.8:
            return 0.6
        else:
            return 0.3
    
    def _analyze_melodic_intervals(self, melody_notes: List[int]) -> float:
        """Analyze quality of melodic intervals."""
        
        if len(melody_notes) < 2:
            return 0.1
        
        intervals = [abs(melody_notes[i + 1] - melody_notes[i]) for i in range(len(melody_notes) - 1)]
        
        if not intervals:
            return 0.1
        
        # Analyze interval distribution
        steps = sum(1 for interval in intervals if interval <= 2)  # Steps
        leaps = sum(1 for interval in intervals if 3 <= interval <= 7)  # Leaps
        large_leaps = sum(1 for interval in intervals if interval > 7)  # Large leaps
        
        total_intervals = len(intervals)
        
        # Good melody has mostly steps with some leaps
        step_ratio = steps / total_intervals
        leap_ratio = leaps / total_intervals
        large_leap_ratio = large_leaps / total_intervals
        
        # Optimal ratios
        if step_ratio >= 0.6 and leap_ratio <= 0.3 and large_leap_ratio <= 0.1:
            return 0.9
        elif step_ratio >= 0.4 and large_leap_ratio <= 0.2:
            return 0.7
        else:
            return 0.4
    
    def _analyze_melodic_direction(self, melody_notes: List[int]) -> float:
        """Analyze melodic direction and balance."""
        
        if len(melody_notes) < 3:
            return 0.1
        
        # Calculate overall direction tendencies
        ups = 0
        downs = 0
        
        for i in range(len(melody_notes) - 1):
            diff = melody_notes[i + 1] - melody_notes[i]
            if diff > 0:
                ups += 1
            elif diff < 0:
                downs += 1
        
        total_moves = ups + downs
        
        if total_moves == 0:
            return 0.1  # No movement
        
        # Good melody has balanced direction or slight bias
        balance = min(ups, downs) / total_moves
        
        if balance >= 0.3:  # Good balance
            return 0.9
        elif balance >= 0.2:
            return 0.7
        else:
            return 0.4


class RealTimeQualityEvaluator:
    """
    Real-time musical quality evaluator for training feedback.
    
    Features:
    - Multi-dimensional quality assessment
    - Performance-optimized evaluation
    - Asynchronous processing for minimal training impact
    - Adaptive evaluation based on training phase
    - Quality-based training recommendations
    """
    
    def __init__(self, config: QualityConfig):
        self.config = config
        
        # Initialize analyzers
        self.rhythmic_analyzer = RhythmicAnalyzer()
        self.harmonic_analyzer = HarmonicAnalyzer()
        self.melodic_analyzer = MelodicAnalyzer()
        
        # Performance tracking
        self.evaluation_count = 0
        self.total_evaluation_time = 0.0
        self.quality_history = deque(maxlen=config.quality_history_window)
        
        # Asynchronous processing
        if config.async_evaluation:
            self.executor = ThreadPoolExecutor(max_workers=config.max_workers)
            self.evaluation_queue = queue.Queue(maxsize=100)
        else:
            self.executor = None
            self.evaluation_queue = None
        
        # Caching
        if config.cache_evaluations:
            self.evaluation_cache = {}
        else:
            self.evaluation_cache = None
        
        logger.info(f"Initialized real-time quality evaluator with mode: {config.mode.value}")
    
    def evaluate_batch(self,
                      generated_tokens: torch.Tensor,
                      batch_idx: int = 0,
                      epoch: int = 0) -> Dict[str, Any]:
        """Evaluate quality of a batch of generated samples."""
        
        if not self._should_evaluate(batch_idx):
            return {"status": "skipped", "reason": "frequency"}
        
        start_time = time.time()
        
        # Select samples to evaluate
        samples_to_evaluate = self._select_samples_for_evaluation(generated_tokens)
        
        if self.config.async_evaluation:
            # Asynchronous evaluation
            return self._evaluate_batch_async(samples_to_evaluate, batch_idx, epoch)
        else:
            # Synchronous evaluation
            return self._evaluate_batch_sync(samples_to_evaluate, batch_idx, epoch, start_time)
    
    def _should_evaluate(self, batch_idx: int) -> bool:
        """Determine if evaluation should be performed for this batch."""
        
        return batch_idx % self.config.evaluation_frequency == 0
    
    def _select_samples_for_evaluation(self, generated_tokens: torch.Tensor) -> torch.Tensor:
        """Select samples from batch for evaluation."""
        
        batch_size = generated_tokens.size(0)
        num_samples = min(self.config.samples_per_evaluation, batch_size)
        
        # Select samples (could be random, best, worst, etc.)
        if batch_size <= num_samples:
            return generated_tokens
        else:
            # Select first N samples for simplicity
            return generated_tokens[:num_samples]
    
    def _evaluate_batch_sync(self,
                           samples: torch.Tensor,
                           batch_idx: int,
                           epoch: int,
                           start_time: float) -> Dict[str, Any]:
        """Synchronous batch evaluation."""
        
        batch_results = []
        
        for i, sample in enumerate(samples):
            sample_id = f"epoch_{epoch}_batch_{batch_idx}_sample_{i}"
            
            # Check cache
            if self.evaluation_cache is not None:
                sample_hash = self._compute_sample_hash(sample)
                if sample_hash in self.evaluation_cache:
                    batch_results.append(self.evaluation_cache[sample_hash])
                    continue
            
            # Evaluate sample
            result = self._evaluate_single_sample(sample, sample_id)
            batch_results.append(result)
            
            # Cache result
            if self.evaluation_cache is not None:
                self.evaluation_cache[sample_hash] = result
                
                # Limit cache size
                if len(self.evaluation_cache) > self.config.cache_size:
                    # Remove oldest entries (simplified)
                    old_keys = list(self.evaluation_cache.keys())[:len(self.evaluation_cache)//2]
                    for key in old_keys:
                        del self.evaluation_cache[key]
        
        # Aggregate results
        evaluation_time = time.time() - start_time
        return self._aggregate_batch_results(batch_results, evaluation_time, batch_idx, epoch)
    
    def _evaluate_batch_async(self,
                            samples: torch.Tensor,
                            batch_idx: int,
                            epoch: int) -> Dict[str, Any]:
        """Asynchronous batch evaluation."""
        
        # Submit evaluation job
        future = self.executor.submit(
            self._evaluate_batch_sync, samples, batch_idx, epoch, time.time()
        )
        
        # Return immediately with future
        return {
            "status": "async",
            "future": future,
            "batch_idx": batch_idx,
            "epoch": epoch
        }
    
    def _evaluate_single_sample(self, tokens: torch.Tensor, sample_id: str) -> QualityResult:
        """Evaluate quality of a single sample."""
        
        start_time = time.time()
        
        result = QualityResult(
            timestamp=start_time,
            sample_id=sample_id,
            sequence_length=len(tokens),
            unique_tokens=len(torch.unique(tokens))
        )
        
        # Truncate if too long
        if len(tokens) > self.config.max_sequence_length:
            tokens = tokens[:self.config.max_sequence_length]
        
        # Evaluate enabled dimensions
        for dimension in self.config.enabled_dimensions:
            if dimension == QualityDimension.RHYTHMIC_COHERENCE:
                rhythmic_result = self.rhythmic_analyzer.analyze_rhythmic_coherence(tokens)
                result.dimension_scores["rhythmic_coherence"] = rhythmic_result["overall"]
                result.rhythmic_features = rhythmic_result
            
            elif dimension == QualityDimension.HARMONIC_CONSISTENCY:
                harmonic_result = self.harmonic_analyzer.analyze_harmonic_consistency(tokens)
                result.dimension_scores["harmonic_consistency"] = harmonic_result["overall"]
                result.harmonic_features = harmonic_result
            
            elif dimension == QualityDimension.MELODIC_FLOW:
                melodic_result = self.melodic_analyzer.analyze_melodic_flow(tokens)
                result.dimension_scores["melodic_flow"] = melodic_result["overall"]
                result.melodic_features = melodic_result
            
            elif dimension == QualityDimension.STRUCTURAL_INTEGRITY:
                structure_score = self._analyze_structural_integrity(tokens)
                result.dimension_scores["structural_integrity"] = structure_score
            
            elif dimension == QualityDimension.OVERALL_QUALITY:
                # Will be calculated after all dimensions
                pass
        
        # Calculate overall quality
        result.overall_score = self._calculate_overall_quality(result.dimension_scores)
        result.dimension_scores["overall_quality"] = result.overall_score
        
        # Determine quality category
        result.quality_category = self._categorize_quality(result.overall_score)
        
        # Generate improvement suggestions
        result.improvement_suggestions = self._generate_improvement_suggestions(result)
        
        # Record evaluation time
        result.evaluation_time = time.time() - start_time
        result.computation_cost = self._categorize_computation_cost(result.evaluation_time)
        
        return result
    
    def _analyze_structural_integrity(self, tokens: torch.Tensor) -> float:
        """Analyze structural integrity (simplified)."""
        
        # Simplified analysis based on sequence properties
        if len(tokens) < 10:
            return 0.1
        
        # Check for reasonable token distribution
        unique_ratio = len(torch.unique(tokens)) / len(tokens)
        
        # Check for patterns and structure
        pattern_score = self._find_structural_patterns(tokens)
        
        # Combine scores
        structure_score = (unique_ratio + pattern_score) / 2.0
        
        return np.clip(structure_score, 0.0, 1.0)
    
    def _find_structural_patterns(self, tokens: torch.Tensor) -> float:
        """Find structural patterns in token sequence."""
        
        tokens_list = tokens.tolist()
        
        if len(tokens_list) < 8:
            return 0.1
        
        # Look for repeating subsequences
        pattern_scores = []
        
        for pattern_length in [4, 6, 8]:
            if len(tokens_list) >= pattern_length * 2:
                pattern_score = self._find_repeating_subsequences(tokens_list, pattern_length)
                pattern_scores.append(pattern_score)
        
        if not pattern_scores:
            return 0.1
        
        return np.mean(pattern_scores)
    
    def _find_repeating_subsequences(self, tokens: List[int], pattern_length: int) -> float:
        """Find repeating subsequences of given length."""
        
        patterns = {}
        
        for i in range(len(tokens) - pattern_length + 1):
            pattern = tuple(tokens[i:i + pattern_length])
            patterns[pattern] = patterns.get(pattern, 0) + 1
        
        if not patterns:
            return 0.0
        
        # Score based on pattern repetition
        repeated_patterns = sum(1 for count in patterns.values() if count > 1)
        total_patterns = len(patterns)
        
        return repeated_patterns / max(total_patterns, 1)
    
    def _calculate_overall_quality(self, dimension_scores: Dict[str, float]) -> float:
        """Calculate overall quality score from dimension scores."""
        
        if not dimension_scores:
            return 0.0
        
        weighted_sum = 0.0
        total_weight = 0.0
        
        for dimension, score in dimension_scores.items():
            weight = self.config.dimension_weights.get(dimension, 1.0)
            weighted_sum += weight * score
            total_weight += weight
        
        if total_weight == 0:
            return np.mean(list(dimension_scores.values()))
        
        return weighted_sum / total_weight
    
    def _categorize_quality(self, overall_score: float) -> str:
        """Categorize quality based on score."""
        
        thresholds = self.config.quality_thresholds
        
        if overall_score >= thresholds.get("excellent", 0.8):
            return "excellent"
        elif overall_score >= thresholds.get("good", 0.6):
            return "good"
        elif overall_score >= thresholds.get("acceptable", 0.4):
            return "acceptable"
        else:
            return "poor"
    
    def _generate_improvement_suggestions(self, result: QualityResult) -> List[str]:
        """Generate suggestions for improving quality."""
        
        suggestions = []
        
        # Check individual dimensions
        for dimension, score in result.dimension_scores.items():
            if score < 0.5:
                if dimension == "rhythmic_coherence":
                    suggestions.append("Improve rhythmic patterns and timing consistency")
                elif dimension == "harmonic_consistency":
                    suggestions.append("Focus on better chord progressions and harmonic flow")
                elif dimension == "melodic_flow":
                    suggestions.append("Enhance melodic contour and interval choices")
                elif dimension == "structural_integrity":
                    suggestions.append("Develop stronger musical structure and form")
        
        # Overall suggestions
        if result.overall_score < 0.3:
            suggestions.append("Consider simplifying musical complexity")
        elif result.overall_score < 0.6:
            suggestions.append("Focus on musical coherence and consistency")
        
        return suggestions[:3]  # Limit to top 3 suggestions
    
    def _categorize_computation_cost(self, evaluation_time: float) -> str:
        """Categorize computational cost of evaluation."""
        
        if evaluation_time < 0.01:
            return "very_low"
        elif evaluation_time < 0.05:
            return "low"
        elif evaluation_time < 0.1:
            return "medium"
        else:
            return "high"
    
    def _aggregate_batch_results(self,
                               batch_results: List[QualityResult],
                               evaluation_time: float,
                               batch_idx: int,
                               epoch: int) -> Dict[str, Any]:
        """Aggregate results from batch evaluation."""
        
        if not batch_results:
            return {"status": "no_results"}
        
        # Calculate statistics
        overall_scores = [result.overall_score for result in batch_results]
        dimension_stats = defaultdict(list)
        
        for result in batch_results:
            for dimension, score in result.dimension_scores.items():
                dimension_stats[dimension].append(score)
        
        # Aggregate statistics
        aggregated_stats = {
            "overall_quality": {
                "mean": np.mean(overall_scores),
                "std": np.std(overall_scores),
                "min": np.min(overall_scores),
                "max": np.max(overall_scores)
            }
        }
        
        for dimension, scores in dimension_stats.items():
            aggregated_stats[dimension] = {
                "mean": np.mean(scores),
                "std": np.std(scores),
                "min": np.min(scores),
                "max": np.max(scores)
            }
        
        # Quality distribution
        quality_categories = [result.quality_category for result in batch_results]
        category_counts = {cat: quality_categories.count(cat) for cat in set(quality_categories)}
        
        # Update global statistics
        self.evaluation_count += len(batch_results)
        self.total_evaluation_time += evaluation_time
        
        # Add to quality history
        mean_quality = aggregated_stats["overall_quality"]["mean"]
        self.quality_history.append(mean_quality)
        
        # Performance metrics
        avg_evaluation_time = evaluation_time / len(batch_results)
        throughput = len(batch_results) / evaluation_time if evaluation_time > 0 else 0
        
        return {
            "status": "completed",
            "batch_idx": batch_idx,
            "epoch": epoch,
            "num_samples": len(batch_results),
            "evaluation_time": evaluation_time,
            "avg_evaluation_time": avg_evaluation_time,
            "throughput": throughput,
            "aggregated_stats": aggregated_stats,
            "quality_distribution": category_counts,
            "individual_results": batch_results,
            "quality_trend": list(self.quality_history)[-10:],  # Last 10 evaluations
            "recommendations": self._generate_training_recommendations(aggregated_stats)
        }
    
    def _generate_training_recommendations(self, stats: Dict[str, Any]) -> List[str]:
        """Generate training recommendations based on quality statistics."""
        
        recommendations = []
        
        overall_mean = stats["overall_quality"]["mean"]
        overall_std = stats["overall_quality"]["std"]
        
        # Overall quality recommendations
        if overall_mean < 0.3:
            recommendations.append("Consider reducing learning rate for stability")
        elif overall_mean < 0.5:
            recommendations.append("Focus on basic musical patterns in curriculum")
        elif overall_mean > 0.8 and overall_std < 0.1:
            recommendations.append("Model converging well - consider increasing complexity")
        
        # Dimension-specific recommendations
        for dimension in ["rhythmic_coherence", "harmonic_consistency", "melodic_flow"]:
            if dimension in stats:
                mean_score = stats[dimension]["mean"]
                if mean_score < 0.4:
                    if dimension == "rhythmic_coherence":
                        recommendations.append("Increase focus on rhythmic loss components")
                    elif dimension == "harmonic_consistency":
                        recommendations.append("Enhance harmonic loss weighting")
                    elif dimension == "melodic_flow":
                        recommendations.append("Improve melodic training data quality")
        
        # Variance recommendations
        if overall_std > 0.3:
            recommendations.append("High quality variance - consider more consistent training data")
        
        return recommendations[:3]  # Limit recommendations
    
    def _compute_sample_hash(self, tokens: torch.Tensor) -> str:
        """Compute hash of sample for caching."""
        
        import hashlib
        
        tokens_bytes = tokens.detach().cpu().numpy().tobytes()
        return hashlib.md5(tokens_bytes).hexdigest()
    
    def get_evaluation_summary(self) -> Dict[str, Any]:
        """Get summary of evaluation performance and results."""
        
        avg_eval_time = self.total_evaluation_time / max(self.evaluation_count, 1)
        
        return {
            "total_evaluations": self.evaluation_count,
            "total_evaluation_time": self.total_evaluation_time,
            "avg_evaluation_time": avg_eval_time,
            "quality_history_length": len(self.quality_history),
            "current_quality_trend": list(self.quality_history)[-5:] if self.quality_history else [],
            "cache_size": len(self.evaluation_cache) if self.evaluation_cache else 0,
            "async_mode": self.config.async_evaluation,
            "evaluation_frequency": self.config.evaluation_frequency
        }


def create_production_quality_config() -> QualityConfig:
    """Create quality configuration for production training."""
    
    return QualityConfig(
        mode=EvaluationMode.BALANCED,
        enabled_dimensions=[
            QualityDimension.RHYTHMIC_COHERENCE,
            QualityDimension.HARMONIC_CONSISTENCY,
            QualityDimension.MELODIC_FLOW,
            QualityDimension.OVERALL_QUALITY
        ],
        dimension_weights={
            "rhythmic_coherence": 0.3,
            "harmonic_consistency": 0.3,
            "melodic_flow": 0.3,
            "overall_quality": 0.1
        },
        batch_evaluation=True,
        max_batch_size=8,
        async_evaluation=True,
        max_workers=2,
        evaluation_frequency=20,
        samples_per_evaluation=2,
        max_sequence_length=512,
        cache_evaluations=True,
        cache_size=500,
        adapt_to_training_phase=True
    )


def create_research_quality_config() -> QualityConfig:
    """Create quality configuration for research experiments."""
    
    return QualityConfig(
        mode=EvaluationMode.COMPREHENSIVE,
        enabled_dimensions=[
            QualityDimension.RHYTHMIC_COHERENCE,
            QualityDimension.HARMONIC_CONSISTENCY,
            QualityDimension.MELODIC_FLOW,
            QualityDimension.STRUCTURAL_INTEGRITY,
            QualityDimension.OVERALL_QUALITY
        ],
        batch_evaluation=True,
        async_evaluation=False,  # Synchronous for research
        evaluation_frequency=10,
        samples_per_evaluation=4,
        max_sequence_length=1024,
        cache_evaluations=True,
        adapt_to_training_phase=False
    )