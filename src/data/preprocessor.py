"""
Aurl.ai Streaming Preprocessor - Phase 2.3

Advanced preprocessing pipeline for musical data with streaming capabilities,
quantization options, and placeholder for musical intelligence features.
"""

import logging
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union, Iterator
from dataclasses import dataclass
from enum import Enum
import numpy as np
import torch
from concurrent.futures import ThreadPoolExecutor
import time

from .midi_parser import MidiData, load_midi_file
from .representation import MusicRepresentationConverter, VocabularyConfig, MusicalRepresentation
from .musical_quantizer import MusicalQuantizer, QuantizationConfig, GridResolution, create_adaptive_quantizer
from .velocity_normalizer import VelocityNormalizer, VelocityNormalizationConfig
from .polyphony_reducer import PolyphonyReducer, PolyphonyReductionConfig
from ..utils.cache import AdvancedCache
from ..utils.config import MidiFlyConfig


class QuantizationMode(Enum):
    """Different quantization approaches for musical timing."""
    NONE = "none"                    # No quantization
    STRICT = "strict"               # Snap to exact grid
    GROOVE_PRESERVING = "groove"    # Preserve swing/groove while quantizing
    ADAPTIVE = "adaptive"           # Smart quantization based on musical context


class VelocityNormalizationMode(Enum):
    """Velocity normalization strategies."""
    NONE = "none"                   # No normalization
    GLOBAL = "global"               # Normalize to global min/max
    PIECE_RELATIVE = "piece"        # Normalize within each piece
    STYLE_PRESERVING = "style"      # Preserve dynamic relationships


@dataclass
class PreprocessingOptions:
    """Configuration options for preprocessing."""
    quantization_mode: QuantizationMode = QuantizationMode.GROOVE_PRESERVING
    quantization_resolution: int = 125  # milliseconds
    quantization_strength: float = 0.8  # 0-1, how strongly to quantize
    velocity_normalization: VelocityNormalizationMode = VelocityNormalizationMode.STYLE_PRESERVING
    preserve_rubato: bool = True
    enable_chord_detection: bool = False  # Placeholder for Phase 6
    max_polyphony: int = 8  # Maximum simultaneous notes
    reduce_polyphony: bool = True
    max_sequence_length: int = 2048
    overlap_ratio: float = 0.1
    cache_processed: bool = True


@dataclass
class PreprocessingResult:
    """Result of preprocessing a single file."""
    representation: MusicalRepresentation
    metadata: Dict
    musical_features: Optional[Dict] = None  # Placeholder for chord detection, etc.
    processing_time: float = 0.0
    cache_hit: bool = False


class ChordDetectionPlaceholder:
    """
    Placeholder for chord detection functionality (Phase 6).
    Sets up the structure without implementing complex music theory.
    """
    
    def __init__(self, config: MidiFlyConfig):
        self.config = config
        self.logger = logging.getLogger(__name__)
        
    def detect_chords(self, midi_data: MidiData) -> Dict:
        """
        Placeholder for chord detection. Currently returns empty structure.
        
        Future implementation will include:
        - Chord progression analysis
        - Key detection
        - Harmonic rhythm analysis
        - Chord quality classification
        """
        return {
            "chords": [],
            "key_signature": None,
            "harmonic_rhythm": None,
            "chord_changes": [],
            "tonal_center": None,
            "note": "Chord detection placeholder - implement in Phase 6"
        }
    
    def analyze_harmony(self, representation: MusicalRepresentation) -> Dict:
        """Placeholder for harmonic analysis."""
        return {
            "harmony_analysis": "Not implemented yet",
            "phase": "Phase 6 - Musical Intelligence"
        }


class StreamingPreprocessor:
    """
    Advanced streaming preprocessor for musical data.
    
    Features:
    - Streaming processing for large datasets
    - Musical quantization with groove preservation
    - Velocity normalization with style preservation
    - Placeholder for chord detection (Phase 6)
    - Advanced caching with intelligent invalidation
    """
    
    def __init__(self, config: MidiFlyConfig, options: PreprocessingOptions = None):
        self.config = config
        self.options = options or PreprocessingOptions()
        self.logger = logging.getLogger(__name__)
        
        # Initialize components
        self.vocab_config = VocabularyConfig()
        self.converter = MusicRepresentationConverter(self.vocab_config)
        self.cache = AdvancedCache(
            cache_dir=Path(config.system.cache_dir) / "preprocessed",
            max_size_gb=config.data.cache_size_gb
        )
        
        # Initialize preprocessing components
        self.quantizer = None  # Created per-file or with specific config
        self.velocity_normalizer = VelocityNormalizer(VelocityNormalizationConfig())
        self.polyphony_reducer = PolyphonyReducer(PolyphonyReductionConfig(
            max_polyphony=self.options.max_polyphony
        ))
        
        # Placeholder for musical intelligence
        self.chord_detector = ChordDetectionPlaceholder(config)
        
        # Statistics tracking
        self.stats = {
            "files_processed": 0,
            "cache_hits": 0,
            "processing_time": 0.0,
            "quantization_applied": 0,
            "velocity_normalized": 0
        }
        
        self.logger.info(f"StreamingPreprocessor initialized with options: {self.options}")
    
    def process_file(self, file_path: Path) -> PreprocessingResult:
        """
        Process a single MIDI file with all preprocessing options.
        
        Args:
            file_path: Path to MIDI file
            
        Returns:
            PreprocessingResult with processed representation and metadata
        """
        start_time = time.time()
        
        # Check cache first
        cache_key = self._generate_cache_key(file_path)
        if self.options.cache_processed:
            cached_result = self.cache.get(cache_key)
            if cached_result is not None:
                self.stats["cache_hits"] += 1
                cached_result.cache_hit = True
                return cached_result
        
        try:
            # Load MIDI file
            midi_data = load_midi_file(file_path)
            
            # Apply preprocessing steps
            processed_midi = self._apply_preprocessing(midi_data)
            
            # Convert to representation
            representation = self.converter.midi_to_representation(processed_midi)
            
            # Extract musical features (placeholder for Phase 6)
            musical_features = None
            if self.options.enable_chord_detection:
                musical_features = self.chord_detector.detect_chords(processed_midi)
            
            # Create result
            result = PreprocessingResult(
                representation=representation,
                metadata=self._extract_metadata(file_path, midi_data),
                musical_features=musical_features,
                processing_time=time.time() - start_time,
                cache_hit=False
            )
            
            # Cache the result
            if self.options.cache_processed:
                self.cache.set(cache_key, result)
            
            # Update statistics
            self.stats["files_processed"] += 1
            self.stats["processing_time"] += result.processing_time
            
            return result
            
        except Exception as e:
            self.logger.error(f"Error processing {file_path}: {e}")
            raise
    
    def stream_process_directory(self, data_dir: Path, max_workers: int = 4) -> Iterator[PreprocessingResult]:
        """
        Stream process all MIDI files in a directory.
        
        Args:
            data_dir: Directory containing MIDI files
            max_workers: Number of parallel workers
            
        Yields:
            PreprocessingResult for each successfully processed file
        """
        midi_files = list(data_dir.glob("*.mid")) + list(data_dir.glob("*.midi"))
        
        self.logger.info(f"Found {len(midi_files)} MIDI files to process")
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {executor.submit(self.process_file, file_path): file_path 
                      for file_path in midi_files}
            
            for future in futures:
                try:
                    result = future.result()
                    yield result
                except Exception as e:
                    file_path = futures[future]
                    self.logger.error(f"Failed to process {file_path}: {e}")
                    continue
    
    def _apply_preprocessing(self, midi_data: MidiData) -> MidiData:
        """
        Apply all preprocessing steps to MIDI data.
        
        Args:
            midi_data: Original MIDI data
            
        Returns:
            Preprocessed MIDI data
        """
        processed_data = midi_data
        
        # Apply polyphony reduction if needed
        if self.options.reduce_polyphony:
            # First analyze polyphony
            poly_analysis = self.polyphony_reducer.analyze_polyphony(processed_data)
            if poly_analysis.get("reduction_needed", False):
                processed_data = self.polyphony_reducer.reduce_polyphony(processed_data)
                self.logger.debug(f"Reduced polyphony from {poly_analysis['max_polyphony']} to {self.options.max_polyphony}")
        
        # Apply quantization
        if self.options.quantization_mode != QuantizationMode.NONE:
            processed_data = self._apply_quantization(processed_data)
            self.stats["quantization_applied"] += 1
        
        # Apply velocity normalization
        if self.options.velocity_normalization != VelocityNormalizationMode.NONE:
            processed_data = self._apply_velocity_normalization(processed_data)
            self.stats["velocity_normalized"] += 1
        
        return processed_data
    
    def _apply_quantization(self, midi_data: MidiData) -> MidiData:
        """
        Apply musical quantization based on the selected mode.
        """
        if self.options.quantization_mode == QuantizationMode.NONE:
            return midi_data
        
        # Create quantizer config based on mode
        if self.options.quantization_mode == QuantizationMode.STRICT:
            config = QuantizationConfig(
                resolution=GridResolution.SIXTEENTH,
                strength=1.0,  # Full snap
                preserve_micro_timing=False
            )
        elif self.options.quantization_mode == QuantizationMode.GROOVE_PRESERVING:
            config = QuantizationConfig(
                resolution=GridResolution.SIXTEENTH,
                strength=self.options.quantization_strength,
                preserve_micro_timing=True,
                swing_ratio=0.0  # Could be made configurable
            )
        elif self.options.quantization_mode == QuantizationMode.ADAPTIVE:
            # Use adaptive quantizer that analyzes the piece
            self.quantizer = create_adaptive_quantizer(midi_data)
            return self.quantizer.quantize_midi_data(midi_data)
        else:
            return midi_data
        
        # Create and apply quantizer
        self.quantizer = MusicalQuantizer(config)
        return self.quantizer.quantize_midi_data(midi_data)
    
    def _apply_velocity_normalization(self, midi_data: MidiData) -> MidiData:
        """Apply velocity normalization based on the selected mode."""
        if self.options.velocity_normalization == VelocityNormalizationMode.NONE:
            return midi_data
        
        # Map our modes to the normalizer's modes
        mode_map = {
            VelocityNormalizationMode.GLOBAL: "global",
            VelocityNormalizationMode.PIECE_RELATIVE: "piece_relative",
            VelocityNormalizationMode.STYLE_PRESERVING: "style_preserving"
        }
        
        mode = mode_map.get(self.options.velocity_normalization, "style_preserving")
        return self.velocity_normalizer.normalize_midi_data(midi_data, mode)
    
    def _extract_metadata(self, file_path: Path, midi_data: MidiData) -> Dict:
        """Extract metadata from file and MIDI data."""
        return {
            "file_path": str(file_path),
            "file_size": file_path.stat().st_size,
            "duration": midi_data.end_time,
            "num_instruments": len(midi_data.instruments),
            "total_notes": sum(len(inst.notes) for inst in midi_data.instruments),
            "preprocessing_options": {
                "quantization_mode": self.options.quantization_mode.value,
                "velocity_normalization": self.options.velocity_normalization.value,
                "preserve_rubato": self.options.preserve_rubato,
                "chord_detection_enabled": self.options.enable_chord_detection
            }
        }
    
    def _generate_cache_key(self, file_path: Path) -> str:
        """Generate cache key based on file path and preprocessing options."""
        import hashlib
        
        # Include file path, modification time, and preprocessing options
        key_data = f"{file_path}:{file_path.stat().st_mtime}:{self.options}"
        return hashlib.md5(key_data.encode()).hexdigest()
    
    def get_statistics(self) -> Dict:
        """Get preprocessing statistics."""
        return {
            **self.stats,
            "cache_hit_rate": self.stats["cache_hits"] / max(1, self.stats["files_processed"]),
            "avg_processing_time": self.stats["processing_time"] / max(1, self.stats["files_processed"])
        }
    
    def clear_cache(self):
        """Clear the preprocessing cache."""
        self.cache.clear()
        self.logger.info("Preprocessing cache cleared")


def create_preprocessor(config: MidiFlyConfig, **kwargs) -> StreamingPreprocessor:
    """
    Factory function to create a StreamingPreprocessor with custom options.
    
    Args:
        config: Aurl.ai configuration
        **kwargs: Additional options for PreprocessingOptions
        
    Returns:
        Configured StreamingPreprocessor instance
    """
    options = PreprocessingOptions(**kwargs)
    return StreamingPreprocessor(config, options)