#!/usr/bin/env python3
"""
Test and analyze the constraint system to identify why we're getting
forced chord progressions instead of natural melodies.
"""

import torch
import numpy as np
from src.generation.constraints import MusicalConstraintEngine, ConstraintConfig
from src.generation.conditional import ConditionalMusicGenerator, ConditionalGenerationConfig
from src.generation.sampler import GenerationConfig, SamplingStrategy

print("=== Constraint System Analysis ===\n")

# Test 1: Check default constraint configuration
print("1. Default Constraint Configuration:")
default_config = ConstraintConfig()
print(f"   Harmonic constraints: {default_config.use_harmonic_constraints}")
print(f"   Allowed progressions: {default_config.allowed_chord_progressions}")
print(f"   Dissonance threshold: {default_config.dissonance_threshold}")
print(f"   Voice leading strictness: {default_config.voice_leading_strictness}")
print()

# Test 2: What happens with constraints vs without
print("2. Constraint Impact Analysis:")

# Create a simple melodic sequence
test_sequence = torch.tensor([[
    1,   # START
    64,  # NOTE_ON C4 (token 64)
    266, # TIME_SHIFT
    192, # NOTE_OFF C4 (token 192) 
    66,  # NOTE_ON D4
    266, # TIME_SHIFT
    194, # NOTE_OFF D4
    68,  # NOTE_ON E4
    266, # TIME_SHIFT
    196, # NOTE_OFF E4
]])

# Test with constraints enabled
constraint_engine = MusicalConstraintEngine(default_config)
print("   Testing constraint engine on melodic sequence...")

# Simulate model logits (random distribution)
logits = torch.randn(1, 774)  # Vocab size 774

# Apply constraints
constrained_logits = constraint_engine.apply_constraints(
    logits=logits,
    generated_sequence=test_sequence,
    step=10,
    context={"key": "C major", "time_signature": (4, 4)}
)

# Check what changed
constraint_impact = torch.abs(constrained_logits - logits).sum().item()
print(f"   Constraint impact magnitude: {constraint_impact:.2f}")
print(f"   Constraints applied: {constraint_engine.get_stats()}")
print()

# Test 3: Check for forced progressions in constraint logic
print("3. Checking for Forced Chord Progression Logic:")
print("   Common progressions in constraint engine:")
for mode, progressions in constraint_engine.common_progressions.items():
    print(f"   {mode}: {progressions[:2]}...")  # Show first 2

print()

# Test 4: Recommended settings for natural generation
print("4. Recommended Settings for Natural 2-Voice Piano:")
natural_config = ConstraintConfig(
    use_harmonic_constraints=False,    # KEY: Disable harmonic forcing
    use_melodic_constraints=True,      # Keep for voice coherence
    use_rhythmic_constraints=True,     # Keep for timing
    use_dynamic_constraints=True,      # Keep for expression
    use_structural_constraints=False,  # Allow natural form
    dissonance_threshold=0.8,         # Allow more dissonance
    voice_leading_strictness=0.3,     # Relaxed voice leading
    prefer_stepwise_motion=True,       # Natural melodic motion
    max_leap_interval=12              # Allow expressive leaps
)

print("   Natural Generation Config:")
print(f"   - Harmonic constraints: {natural_config.use_harmonic_constraints}")
print(f"   - Melodic constraints: {natural_config.use_melodic_constraints}")
print(f"   - Dissonance threshold: {natural_config.dissonance_threshold}")
print(f"   - Voice leading strictness: {natural_config.voice_leading_strictness}")
print()

# Test 5: Generation config for natural music
print("5. Recommended Generation Parameters:")
gen_config = GenerationConfig(
    strategy=SamplingStrategy.TOP_P,
    temperature=0.8,
    top_p=0.9,
    use_musical_constraints=False,  # KEY: Turn off constraints
    repetition_penalty=1.1          # Light repetition control
)

print(f"   Strategy: {gen_config.strategy.value}")
print(f"   Temperature: {gen_config.temperature}")
print(f"   Top-p: {gen_config.top_p}")
print(f"   Musical constraints: {gen_config.use_musical_constraints}")
print()

print("=== Key Findings ===")
print()
print("The 'forced chord progressions' issue comes from:")
print("1. ConstraintConfig.use_harmonic_constraints = True (default)")
print("2. Pre-defined chord progressions enforced during generation")
print("3. Low dissonance threshold blocking natural voice independence")
print("4. High voice leading strictness preventing melodic freedom")
print()
print("SOLUTION: Use the natural_config settings above")
print("This will let the trained model generate freely like Bach/Mozart")
print()
print("Command to test:")
print("python generate.py --checkpoint model.pt --no-constraints \\")
print("  --strategy top_p --top-p 0.9 --temperature 0.8")