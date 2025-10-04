#!/usr/bin/env python3
"""
Demo Chord Analysis - Simplified version for quick results
"""

import sys
from pathlib import Path
import json
import random

# Add project to path
sys.path.insert(0, str(Path(__file__).parent))

from studies.chord_analysis.chord_extractor import AdvancedChordExtractor
from studies.chord_analysis.transition_matrix import TransitionMatrixBuilder
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


def demo_analysis():
    """Run simplified demo analysis on one file."""

    # Find a single MIDI file to analyze
    midi_files = list(Path("data/raw").glob("beethoven_opus10_1.mid"))

    if not midi_files:
        logger.error("No MIDI files found")
        return

    midi_file = str(midi_files[0])
    logger.info(f"\n{'='*60}")
    logger.info("CHORD ANALYSIS DEMONSTRATION")
    logger.info(f"{'='*60}\n")
    logger.info(f"Analyzing: {midi_file}\n")

    # Extract chords
    extractor = AdvancedChordExtractor()
    logger.info("Extracting chords...")
    chords = extractor.extract_chords_from_midi(midi_file)

    if not chords:
        logger.error("No chords extracted")
        return

    logger.info(f"✓ Extracted {len(chords)} chord occurrences\n")

    # Show sample chord analysis
    logger.info("SAMPLE CHORD ANALYSIS:")
    logger.info("-" * 40)

    for i, chord in enumerate(chords[:10]):  # First 10 chords
        logger.info(f"Measure {chord.measure}, Beat {chord.beat:.1f}:")
        logger.info(f"  Chord: {chord.chord_symbol}")
        logger.info(f"  Roman: {chord.roman_numeral}")
        logger.info(f"  Function: {chord.harmonic_function}")
        logger.info(f"  Tension: {chord.tension_score:.2f}")
        logger.info(f"  Consonance: {chord.consonance_score:.2f}")
        if chord.cadence_type:
            logger.info(f"  → CADENCE: {chord.cadence_type}")
        logger.info("")

    # Build transition matrix
    logger.info("\nBUILDING TRANSITION MATRIX...")
    logger.info("-" * 40)

    matrix_builder = TransitionMatrixBuilder()
    matrix = matrix_builder.build_transition_matrix(chords, "Beethoven Op.10 No.1")

    logger.info(f"✓ Matrix built: {len(matrix.chord_vocabulary)} unique chords")
    logger.info(f"  Entropy: {matrix.entropy_score:.3f}")
    logger.info(f"  Sparsity: {matrix.sparsity:.3f}")
    logger.info(f"  Total transitions: {matrix.total_transitions}")

    # Show top transitions
    logger.info("\nTOP CHORD PROGRESSIONS:")
    logger.info("-" * 40)

    for from_chord, to_chord, prob in matrix.top_transitions[:15]:
        logger.info(f"  {from_chord:6} → {to_chord:6} : {prob:.3f}")

    # Show cycles
    if matrix.cycles:
        logger.info("\nHARMONIC CYCLES DETECTED:")
        logger.info("-" * 40)
        for cycle in matrix.cycles[:5]:
            logger.info(f"  {' → '.join(cycle)} → {cycle[0]}")

    # Show attractors
    if matrix.attractors:
        logger.info("\nCHORD ATTRACTORS (Most targeted chords):")
        logger.info("-" * 40)
        for attractor in matrix.attractors:
            logger.info(f"  {attractor}")

    # Generate a progression
    logger.info("\nGENERATED PROGRESSION (Based on learned patterns):")
    logger.info("-" * 40)

    generated = matrix_builder.generate_progression(
        matrix, start_chord='I', length=8, temperature=0.8
    )
    logger.info(f"  {' - '.join(generated)}")

    # Predict next chords
    logger.info("\nNEXT CHORD PREDICTIONS:")
    logger.info("-" * 40)

    test_chords = ['I', 'V', 'ii', 'IV']
    for test in test_chords:
        predictions = matrix_builder.predict_next_chord(matrix, test, 3)
        if predictions:
            logger.info(f"  After {test}:")
            for next_chord, prob in predictions:
                logger.info(f"    → {next_chord}: {prob:.2%}")

    # Summary statistics
    logger.info("\n" + "="*60)
    logger.info("HARMONIC CHARACTERISTICS SUMMARY")
    logger.info("="*60)

    # Count chord qualities
    qualities = {}
    functions = {}
    for chord in chords:
        qualities[chord.quality] = qualities.get(chord.quality, 0) + 1
        if chord.harmonic_function:
            functions[chord.harmonic_function] = functions.get(chord.harmonic_function, 0) + 1

    logger.info("\nChord Quality Distribution:")
    total = sum(qualities.values())
    for quality, count in sorted(qualities.items(), key=lambda x: x[1], reverse=True)[:5]:
        logger.info(f"  {quality}: {count/total:.1%}")

    logger.info("\nHarmonic Function Distribution:")
    total = sum(functions.values())
    for function, count in sorted(functions.items(), key=lambda x: x[1], reverse=True):
        logger.info(f"  {function}: {count/total:.1%}")

    # Tension analysis
    tensions = [c.tension_score for c in chords]
    avg_tension = sum(tensions) / len(tensions)
    logger.info(f"\nAverage Harmonic Tension: {avg_tension:.3f}")

    # Voice leading
    smooth_count = sum(1 for c in chords if c.voice_leading_smooth)
    logger.info(f"Smooth Voice Leading: {smooth_count/len(chords):.1%}")

    logger.info("\n" + "="*60)
    logger.info("Analysis complete!")
    logger.info("="*60)


if __name__ == "__main__":
    demo_analysis()