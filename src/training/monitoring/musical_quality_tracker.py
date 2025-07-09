"""
Musical Quality Metrics Tracking for Aurl.ai Music Generation.

This module provides real-time musical quality assessment during training:
- Rhythm consistency and regularity
- Harmonic coherence evaluation
- Melodic contour analysis
- Dynamic range assessment
- Structural pattern detection
- Style consistency metrics
"""

import torch
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass, field
from pathlib import Path
import json
from datetime import datetime
from collections import deque

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MusicalQualityMetrics:
    """Container for musical quality metrics."""
    
    rhythm_consistency: float = 0.0  # 0-1 score for rhythmic regularity
    rhythm_diversity: float = 0.0    # 0-1 score for rhythmic variety
    harmonic_coherence: float = 0.0  # 0-1 score for harmonic progression quality
    melodic_contour: float = 0.0     # 0-1 score for melodic shape quality
    dynamic_range: float = 0.0       # 0-1 score for velocity variation
    note_density: float = 0.0        # Notes per beat
    pitch_range: int = 0             # Number of unique pitches used
    repetition_score: float = 0.0    # 0-1 score (0=too repetitive, 1=good variety)
    structural_coherence: float = 0.0 # 0-1 score for phrase structure
    overall_quality: float = 0.0     # Weighted average of all metrics
    
    def to_dict(self) -> Dict[str, float]:
        """Convert to dictionary for logging."""
        return {
            "rhythm_consistency": self.rhythm_consistency,
            "rhythm_diversity": self.rhythm_diversity,
            "harmonic_coherence": self.harmonic_coherence,
            "melodic_contour": self.melodic_contour,
            "dynamic_range": self.dynamic_range,
            "note_density": self.note_density,
            "pitch_range": float(self.pitch_range),
            "repetition_score": self.repetition_score,
            "structural_coherence": self.structural_coherence,
            "overall_quality": self.overall_quality
        }


class MusicalQualityTracker:
    """
    Tracks musical quality metrics during training.
    
    Features:
    - Real-time quality assessment
    - Historical quality tracking
    - Quality trend analysis
    - Automatic alerts for quality degradation
    """
    
    def __init__(self,
                 vocab_size: int = 774,
                 window_size: int = 100,
                 quality_threshold: float = 0.5,
                 alert_on_degradation: bool = True):
        """
        Initialize musical quality tracker.
        
        Args:
            vocab_size: Size of the token vocabulary
            window_size: Number of samples to keep in history
            quality_threshold: Minimum acceptable quality score
            alert_on_degradation: Whether to alert on quality drops
        """
        
        self.vocab_size = vocab_size
        self.window_size = window_size
        self.quality_threshold = quality_threshold
        self.alert_on_degradation = alert_on_degradation
        
        # Historical tracking
        self.quality_history = deque(maxlen=window_size)
        self.epoch_quality = {}  # Epoch -> average quality
        
        # Token interpretation (based on Aurl.ai vocabulary)
        self._setup_token_mappings()
        
        logger.info(f"Musical quality tracker initialized - threshold: {quality_threshold}")
    
    def _setup_token_mappings(self):
        """Setup token to musical element mappings."""
        # Token ranges based on Aurl.ai vocabulary structure
        self.note_on_range = (0, 128)      # Note-on events
        self.note_off_range = (128, 256)   # Note-off events
        self.velocity_range = (256, 384)   # Velocity events
        self.time_shift_range = (384, 512) # Time shift events
        self.special_tokens = {
            512: "bar",
            513: "position",
            514: "instrument",
            # Add more as needed
        }
    
    def evaluate_sample(self, 
                       tokens: torch.Tensor,
                       epoch: Optional[int] = None) -> MusicalQualityMetrics:
        """
        Evaluate musical quality of a token sequence.
        
        Args:
            tokens: Token sequence tensor [batch_size, seq_len]
            epoch: Current training epoch
            
        Returns:
            MusicalQualityMetrics object
        """
        
        # Convert to numpy for analysis
        if isinstance(tokens, torch.Tensor):
            tokens_np = tokens.cpu().numpy()
        else:
            tokens_np = np.array(tokens)
        
        # Calculate individual metrics
        metrics = MusicalQualityMetrics()
        
        # Rhythm analysis
        rhythm_scores = self._analyze_rhythm(tokens_np)
        metrics.rhythm_consistency = rhythm_scores["consistency"]
        metrics.rhythm_diversity = rhythm_scores["diversity"]
        
        # Harmonic analysis
        metrics.harmonic_coherence = self._analyze_harmony(tokens_np)
        
        # Melodic analysis
        melodic_scores = self._analyze_melody(tokens_np)
        metrics.melodic_contour = melodic_scores["contour_quality"]
        metrics.pitch_range = melodic_scores["pitch_range"]
        
        # Dynamic analysis
        metrics.dynamic_range = self._analyze_dynamics(tokens_np)
        
        # Density and repetition
        metrics.note_density = self._calculate_note_density(tokens_np)
        metrics.repetition_score = self._calculate_repetition_score(tokens_np)
        
        # Structural analysis
        metrics.structural_coherence = self._analyze_structure(tokens_np)
        
        # Calculate overall quality (weighted average)
        weights = {
            "rhythm": 0.20,
            "harmony": 0.20,
            "melody": 0.20,
            "dynamics": 0.10,
            "repetition": 0.15,
            "structure": 0.15
        }
        
        metrics.overall_quality = (
            weights["rhythm"] * (metrics.rhythm_consistency + metrics.rhythm_diversity) / 2 +
            weights["harmony"] * metrics.harmonic_coherence +
            weights["melody"] * metrics.melodic_contour +
            weights["dynamics"] * metrics.dynamic_range +
            weights["repetition"] * metrics.repetition_score +
            weights["structure"] * metrics.structural_coherence
        )
        
        # Track history
        self.quality_history.append(metrics)
        
        if epoch is not None:
            if epoch not in self.epoch_quality:
                self.epoch_quality[epoch] = []
            self.epoch_quality[epoch].append(metrics.overall_quality)
        
        # Check for quality degradation
        if self.alert_on_degradation:
            self._check_quality_degradation(metrics)
        
        return metrics
    
    def _analyze_rhythm(self, tokens: np.ndarray) -> Dict[str, float]:
        """Analyze rhythmic qualities."""
        
        # Extract time shift tokens
        time_shifts = []
        for batch in tokens:
            shifts = batch[(batch >= self.time_shift_range[0]) & 
                          (batch < self.time_shift_range[1])]
            time_shifts.extend(shifts - self.time_shift_range[0])
        
        if len(time_shifts) < 2:
            return {"consistency": 0.5, "diversity": 0.5}
        
        # Calculate inter-onset intervals
        time_shifts = np.array(time_shifts)
        intervals = np.diff(time_shifts)
        
        # Consistency: Low variance in intervals = high consistency
        if len(intervals) > 0:
            consistency = 1.0 / (1.0 + np.std(intervals) / (np.mean(intervals) + 1e-6))
        else:
            consistency = 0.5
        
        # Diversity: Number of unique interval patterns
        unique_intervals = len(np.unique(intervals))
        diversity = min(1.0, unique_intervals / (len(intervals) + 1))
        
        return {
            "consistency": float(np.clip(consistency, 0, 1)),
            "diversity": float(np.clip(diversity, 0, 1))
        }
    
    def _analyze_harmony(self, tokens: np.ndarray) -> float:
        """Analyze harmonic coherence."""
        
        # Extract note-on events
        notes = []
        for batch in tokens:
            note_ons = batch[(batch >= self.note_on_range[0]) & 
                            (batch < self.note_on_range[1])]
            notes.extend(note_ons)
        
        if len(notes) < 3:
            return 0.5
        
        # Simple harmonic analysis based on intervals
        notes = np.array(notes) % 12  # Convert to pitch classes
        
        # Count consonant intervals (unison, 3rd, 5th, octave)
        consonant_intervals = {0, 3, 4, 5, 7, 8, 9}
        interval_counts = 0
        consonant_counts = 0
        
        for i in range(len(notes) - 1):
            interval = abs(notes[i+1] - notes[i]) % 12
            interval_counts += 1
            if interval in consonant_intervals:
                consonant_counts += 1
        
        coherence = consonant_counts / (interval_counts + 1e-6)
        
        return float(np.clip(coherence, 0, 1))
    
    def _analyze_melody(self, tokens: np.ndarray) -> Dict[str, Any]:
        """Analyze melodic qualities."""
        
        # Extract pitch sequence
        pitches = []
        for batch in tokens:
            note_ons = batch[(batch >= self.note_on_range[0]) & 
                            (batch < self.note_on_range[1])]
            pitches.extend(note_ons)
        
        if len(pitches) < 2:
            return {"contour_quality": 0.5, "pitch_range": 0}
        
        pitches = np.array(pitches)
        
        # Analyze melodic contour
        intervals = np.diff(pitches)
        
        # Good melodies have a mix of steps and leaps
        steps = np.sum(np.abs(intervals) <= 2)  # Whole tone or less
        leaps = np.sum(np.abs(intervals) > 4)   # More than major third
        total = len(intervals)
        
        # Ideal ratio: mostly steps with some leaps
        step_ratio = steps / (total + 1e-6)
        leap_ratio = leaps / (total + 1e-6)
        
        # Contour quality based on balance
        if 0.6 <= step_ratio <= 0.8 and 0.1 <= leap_ratio <= 0.3:
            contour_quality = 1.0
        else:
            contour_quality = 1.0 - abs(step_ratio - 0.7) - abs(leap_ratio - 0.2)
        
        # Pitch range
        pitch_range = len(np.unique(pitches))
        
        return {
            "contour_quality": float(np.clip(contour_quality, 0, 1)),
            "pitch_range": int(pitch_range)
        }
    
    def _analyze_dynamics(self, tokens: np.ndarray) -> float:
        """Analyze dynamic range and expression."""
        
        # Extract velocity values
        velocities = []
        for batch in tokens:
            vel_tokens = batch[(batch >= self.velocity_range[0]) & 
                              (batch < self.velocity_range[1])]
            velocities.extend(vel_tokens - self.velocity_range[0])
        
        if len(velocities) < 2:
            return 0.5
        
        velocities = np.array(velocities)
        
        # Good dynamics have variety but not randomness
        vel_range = np.ptp(velocities)  # Peak to peak
        vel_std = np.std(velocities)
        
        # Normalize to 0-1
        range_score = min(1.0, vel_range / 127.0)
        consistency_score = 1.0 / (1.0 + vel_std / 20.0)  # Penalize too much randomness
        
        dynamic_score = 0.7 * range_score + 0.3 * consistency_score
        
        return float(np.clip(dynamic_score, 0, 1))
    
    def _calculate_note_density(self, tokens: np.ndarray) -> float:
        """Calculate note density (notes per time unit)."""
        
        note_count = 0
        time_count = 0
        
        for batch in tokens:
            # Count note-on events
            note_count += np.sum((batch >= self.note_on_range[0]) & 
                               (batch < self.note_on_range[1]))
            
            # Count time progression
            time_count += np.sum((batch >= self.time_shift_range[0]) & 
                               (batch < self.time_shift_range[1]))
        
        if time_count == 0:
            return 0.0
        
        # Average notes per time unit
        density = note_count / (time_count + 1)
        
        return float(density)
    
    def _calculate_repetition_score(self, tokens: np.ndarray) -> float:
        """Calculate repetition score (0=too repetitive, 1=good variety)."""
        
        # Look for repeated patterns
        all_tokens = tokens.flatten()
        
        # Check for n-gram repetitions (4-gram, 8-gram)
        repetition_scores = []
        
        for n in [4, 8, 16]:
            if len(all_tokens) < n * 2:
                continue
                
            ngrams = [tuple(all_tokens[i:i+n]) for i in range(len(all_tokens)-n+1)]
            unique_ngrams = len(set(ngrams))
            total_ngrams = len(ngrams)
            
            diversity = unique_ngrams / (total_ngrams + 1e-6)
            repetition_scores.append(diversity)
        
        if not repetition_scores:
            return 0.5
        
        # Average across different n-gram sizes
        return float(np.clip(np.mean(repetition_scores), 0, 1))
    
    def _analyze_structure(self, tokens: np.ndarray) -> float:
        """Analyze structural coherence (phrases, sections)."""
        
        # Look for bar markers and structural patterns
        structure_score = 0.5  # Default
        
        for batch in tokens:
            # Check for bar markers
            bar_positions = np.where(batch == 512)[0]  # Bar token
            
            if len(bar_positions) > 1:
                # Check for regular bar spacing
                bar_intervals = np.diff(bar_positions)
                
                # Good structure has consistent bar lengths
                if len(bar_intervals) > 0:
                    consistency = 1.0 / (1.0 + np.std(bar_intervals) / (np.mean(bar_intervals) + 1e-6))
                    structure_score = consistency
        
        return float(np.clip(structure_score, 0, 1))
    
    def _check_quality_degradation(self, current_metrics: MusicalQualityMetrics):
        """Check for quality degradation and alert if needed."""
        
        if len(self.quality_history) < 10:
            return  # Not enough history
        
        # Get recent history
        recent_qualities = [m.overall_quality for m in list(self.quality_history)[-10:]]
        avg_recent = np.mean(recent_qualities)
        
        # Check if current is significantly worse
        if current_metrics.overall_quality < avg_recent * 0.8:
            logger.warning(
                f"Musical quality degradation detected! "
                f"Current: {current_metrics.overall_quality:.3f}, "
                f"Recent avg: {avg_recent:.3f}"
            )
        
        # Check if below threshold
        if current_metrics.overall_quality < self.quality_threshold:
            logger.warning(
                f"Musical quality below threshold! "
                f"Current: {current_metrics.overall_quality:.3f}, "
                f"Threshold: {self.quality_threshold:.3f}"
            )
    
    def get_epoch_summary(self, epoch: int) -> Dict[str, float]:
        """Get quality summary for a specific epoch."""
        
        if epoch not in self.epoch_quality:
            return {}
        
        qualities = self.epoch_quality[epoch]
        
        return {
            "epoch": epoch,
            "average_quality": float(np.mean(qualities)),
            "min_quality": float(np.min(qualities)),
            "max_quality": float(np.max(qualities)),
            "std_quality": float(np.std(qualities)),
            "num_samples": len(qualities)
        }
    
    def get_quality_trend(self) -> Dict[str, Any]:
        """Get quality trend analysis."""
        
        if not self.quality_history:
            return {"trend": "unknown", "improving": False}
        
        # Get quality values over time
        qualities = [m.overall_quality for m in self.quality_history]
        
        if len(qualities) < 5:
            return {"trend": "insufficient_data", "improving": False}
        
        # Simple linear regression for trend
        x = np.arange(len(qualities))
        y = np.array(qualities)
        
        # Calculate slope
        slope = np.polyfit(x, y, 1)[0]
        
        # Determine trend
        if slope > 0.001:
            trend = "improving"
            improving = True
        elif slope < -0.001:
            trend = "degrading"
            improving = False
        else:
            trend = "stable"
            improving = False
        
        return {
            "trend": trend,
            "improving": improving,
            "slope": float(slope),
            "current_quality": float(qualities[-1]),
            "average_quality": float(np.mean(qualities))
        }
    
    def save_quality_report(self, save_path: Path):
        """Save detailed quality report."""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "total_samples_evaluated": len(self.quality_history),
            "quality_threshold": self.quality_threshold,
            "trend_analysis": self.get_quality_trend(),
            "epoch_summaries": {}
        }
        
        # Add epoch summaries
        for epoch in sorted(self.epoch_quality.keys()):
            report["epoch_summaries"][str(epoch)] = self.get_epoch_summary(epoch)
        
        # Add recent samples
        if self.quality_history:
            recent_metrics = list(self.quality_history)[-10:]
            report["recent_samples"] = [m.to_dict() for m in recent_metrics]
        
        # Save report
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Quality report saved to {save_path}")