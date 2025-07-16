"""
Music Generation Module

This module provides comprehensive music generation capabilities including:

Core Components:
- MusicSampler: Advanced sampling strategies (temperature, top-k, top-p, beam search)
- BatchMusicSampler: Optimized batch generation for high throughput
- MusicalConstraintEngine: Music theory-based constraints
- ConditionalMusicGenerator: Full conditional generation with style/attribute control
- InteractiveGenerator: Real-time generation with dynamic conditioning
- MidiExporter: Professional-grade MIDI file export with Finale compatibility

Key Features:
- Multiple sampling strategies optimized for music
- Musical constraint enforcement
- Style and attribute conditioning
- Real-time interactive generation
- Batch processing for production scenarios
- Professional MIDI export with multi-track support
- Finale and DAW compatibility optimization
- Comprehensive statistics and monitoring

Usage Examples:

Basic Generation:
    from src.generation import MusicSampler, GenerationConfig
    
    sampler = MusicSampler(model, device)
    config = GenerationConfig(strategy="top_p", top_p=0.9, temperature=0.8)
    music = sampler.generate(config=config)

Conditional Generation:
    from src.generation import ConditionalMusicGenerator, StyleCondition
    
    generator = ConditionalMusicGenerator(model, device)
    config = ConditionalGenerationConfig(
        style=StyleCondition(genre="jazz", complexity=0.7)
    )
    music = generator.generate(config)

Interactive Generation:
    from src.generation import InteractiveGenerator
    
    interactive = InteractiveGenerator(model, device)
    interactive.start_generation(config)
    
    # Generate in chunks
    for _ in range(10):
        chunk = interactive.generate_next(num_tokens=16)
        # Process chunk in real-time
"""

# Core generation components
from .sampler import (
    SamplingStrategy,
    GenerationConfig,
    MusicSampler,
    BatchMusicSampler
)

# Musical constraints
from .constraints import (
    ConstraintType,
    ConstraintConfig,
    MusicalConstraintEngine,
    StyleSpecificConstraints
)

# Conditional generation
from .conditional import (
    ConditioningType,
    StyleCondition,
    MusicalAttributes,
    StructuralCondition,
    ConditionalGenerationConfig,
    ConditionalMusicGenerator,
    InteractiveGenerator
)

# MIDI Export
from .midi_export import (
    MidiExportConfig,
    MidiExporter,
    ExportStatistics,
    export_tokens_to_midi_file,
    create_standard_config,
    create_performance_config
)

__all__ = [
    # Sampling
    "SamplingStrategy",
    "GenerationConfig", 
    "MusicSampler",
    "BatchMusicSampler",
    
    # Constraints
    "ConstraintType",
    "ConstraintConfig",
    "MusicalConstraintEngine",
    "StyleSpecificConstraints",
    
    # Conditional Generation
    "ConditioningType",
    "StyleCondition",
    "MusicalAttributes", 
    "StructuralCondition",
    "ConditionalGenerationConfig",
    "ConditionalMusicGenerator",
    "InteractiveGenerator",
    
    # MIDI Export
    "MidiExportConfig",
    "MidiExporter",
    "ExportStatistics",
    "export_tokens_to_midi_file",
    "create_standard_config",
    "create_performance_config",
]

# Version and metadata
__version__ = "1.0.0"
__author__ = "Aurl.ai Team"
__description__ = "State-of-the-art music generation with advanced conditioning and constraints"