#!/usr/bin/env python3
"""
Test Enhanced Musical Grammar System

This script tests the new musical grammar loss functions and validators
to ensure they correctly identify and penalize the issues discovered
in the MIDI export diagnosis (velocity=1, duration=0.0s).

Phase 5.1 Testing - Addresses root cause of unusable MIDI output
"""

import sys
import os
import numpy as np
import torch
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from src.training.musical_grammar import (
    MusicalGrammarLoss, MusicalGrammarValidator, MusicalGrammarConfig,
    validate_generated_sequence, get_parameter_quality_report
)
from src.data.representation import VocabularyConfig, EventType

def create_test_vocabulary():
    """Create a test vocabulary configuration."""
    vocab_config = VocabularyConfig()
    
    # Mock the token_to_event_info method for testing
    def mock_token_to_event_info(token_id):
        # Simple mapping for test tokens
        if token_id < 128:  # NOTE_ON tokens
            return EventType.NOTE_ON, token_id
        elif token_id < 256:  # NOTE_OFF tokens
            return EventType.NOTE_OFF, token_id - 128
        elif token_id < 300:  # VELOCITY tokens (44 tokens: 256-299)
            return EventType.VELOCITY_CHANGE, token_id - 256
        elif token_id < 400:  # TIME_SHIFT tokens (100 tokens: 300-399)
            return EventType.TIME_SHIFT, token_id - 300
        else:
            return EventType.UNK_TOKEN, 0
    
    vocab_config.token_to_event_info = mock_token_to_event_info
    vocab_config.velocity_bins = 32  # 32 velocity bins
    vocab_config.time_shift_ms = 15.625  # 15.625ms per time shift
    
    return vocab_config

def test_velocity_parameter_loss():
    """Test velocity parameter loss function."""
    print("Testing Velocity Parameter Loss...")
    
    vocab_config = create_test_vocabulary()
    config = MusicalGrammarConfig()
    grammar_loss = MusicalGrammarLoss(config, vocab_config)
    
    # Create test sequences
    batch_size = 2
    seq_len = 10
    
    # Sequence 1: Contains bad velocity tokens (map to velocity=1)
    bad_velocity_tokens = torch.tensor([
        [60, 256, 188, 62, 130, 257, 188, 64, 132, 300],  # 256,257 -> low velocity
        [60, 256, 188, 62, 130, 258, 188, 64, 132, 301]   # 256,258 -> low velocity
    ])
    
    # Sequence 2: Contains good velocity tokens (map to velocity=64+)
    good_velocity_tokens = torch.tensor([
        [60, 272, 188, 62, 130, 273, 188, 64, 132, 310],  # 272,273 -> good velocity
        [60, 274, 188, 62, 130, 275, 188, 64, 132, 311]   # 274,275 -> good velocity
    ])
    
    # Test bad velocity loss
    bad_loss = grammar_loss._velocity_parameter_loss(bad_velocity_tokens)
    good_loss = grammar_loss._velocity_parameter_loss(good_velocity_tokens)
    
    print(f"Bad velocity loss: {bad_loss:.3f}")
    print(f"Good velocity loss: {good_loss:.3f}")
    
    assert bad_loss > good_loss, "Bad velocity tokens should have higher loss"
    assert bad_loss > 1.0, "Bad velocity tokens should have significant penalty"
    print("✅ Velocity parameter loss test passed!")

def test_timing_parameter_loss():
    """Test timing parameter loss function."""
    print("\nTesting Timing Parameter Loss...")
    
    vocab_config = create_test_vocabulary()
    config = MusicalGrammarConfig()
    grammar_loss = MusicalGrammarLoss(config, vocab_config)
    
    # Create test sequences
    batch_size = 2
    seq_len = 10
    
    # Sequence 1: Contains bad timing tokens (map to 0.0s duration)
    bad_timing_tokens = torch.tensor([
        [60, 270, 300, 62, 130, 270, 300, 64, 132, 301],  # 300,301 -> ~0s duration
        [60, 270, 300, 62, 130, 270, 300, 64, 132, 301]
    ])
    
    # Sequence 2: Contains good timing tokens (map to 0.2s+ duration)
    good_timing_tokens = torch.tensor([
        [60, 270, 320, 62, 130, 270, 325, 64, 132, 330],  # 320+ -> good duration
        [60, 270, 320, 62, 130, 270, 325, 64, 132, 330]
    ])
    
    # Test timing loss
    bad_loss = grammar_loss._timing_parameter_loss(bad_timing_tokens)
    good_loss = grammar_loss._timing_parameter_loss(good_timing_tokens)
    
    print(f"Bad timing loss: {bad_loss:.3f}")
    print(f"Good timing loss: {good_loss:.3f}")
    
    assert bad_loss > good_loss, "Bad timing tokens should have higher loss"
    assert bad_loss > 0.5, "Bad timing tokens should have significant penalty"
    print("✅ Timing parameter loss test passed!")

def test_enhanced_note_pairing():
    """Test enhanced note pairing loss."""
    print("\nTesting Enhanced Note Pairing Loss...")
    
    vocab_config = create_test_vocabulary()
    config = MusicalGrammarConfig()
    grammar_loss = MusicalGrammarLoss(config, vocab_config)
    
    # Good sequence: proper note on/off pairing
    good_sequence = torch.tensor([
        [60, 270, 320, 188, 270, 320, 62, 270, 320, 190],  # NOTE_ON 60, NOTE_OFF 60, NOTE_ON 62, NOTE_OFF 62
    ])
    
    # Bad sequence: mismatched note pairing
    bad_sequence = torch.tensor([
        [60, 270, 320, 190, 270, 320, 62, 270, 320, 188],  # NOTE_ON 60, NOTE_OFF 62, NOTE_ON 62, NOTE_OFF 60
    ])
    
    good_loss = grammar_loss._note_pairing_loss(good_sequence, good_sequence)
    bad_loss = grammar_loss._note_pairing_loss(bad_sequence, bad_sequence)
    
    print(f"Good pairing loss: {good_loss:.3f}")
    print(f"Bad pairing loss: {bad_loss:.3f}")
    
    assert bad_loss > good_loss, "Mismatched note pairing should have higher loss"
    print("✅ Enhanced note pairing loss test passed!")

def test_parameter_quality_validator():
    """Test the enhanced validator with parameter quality checks."""
    print("\nTesting Parameter Quality Validator...")
    
    vocab_config = create_test_vocabulary()
    validator = MusicalGrammarValidator()
    validator.vocab_config = vocab_config
    
    # Test sequence with parameter quality issues
    problematic_sequence = np.array([
        60, 256, 300, 188, 270, 301,  # Low velocity (256) and zero duration (300)
        62, 257, 300, 190, 270, 301   # Low velocity (257) and zero duration (300)
    ])
    
    # Test sequence with good parameters
    good_sequence = np.array([
        60, 270, 320, 188, 270, 325,  # Good velocity (270) and duration (320)
        62, 271, 320, 190, 270, 325   # Good velocity (271) and duration (320)
    ])
    
    problem_results = validator.validate_sequence(problematic_sequence)
    good_results = validator.validate_sequence(good_sequence)
    
    print(f"Problematic sequence grammar score: {problem_results['grammar_score']:.3f}")
    print(f"Good sequence grammar score: {good_results['grammar_score']:.3f}")
    
    assert problem_results['grammar_score'] < good_results['grammar_score'], \
        "Problematic sequence should have lower grammar score"
    
    assert problem_results['velocity_quality_score'] < 0.5, \
        "Should detect poor velocity quality"
    
    assert problem_results['timing_quality_score'] < 0.5, \
        "Should detect poor timing quality"
    
    print("✅ Parameter quality validator test passed!")

def test_quality_report():
    """Test the parameter quality report generation."""
    print("\nTesting Quality Report Generation...")
    
    vocab_config = create_test_vocabulary()
    
    # Test sequence with mixed quality
    test_sequence = np.array([
        60, 256, 300, 188, 270, 320,  # Bad velocity, bad timing, good velocity, good timing
        62, 270, 325, 190, 256, 300   # Good velocity, good timing, bad velocity, bad timing
    ])
    
    report = get_parameter_quality_report(test_sequence, vocab_config)
    print("Generated Quality Report:")
    print(report)
    
    assert "WARNING" in report, "Should contain warnings for parameter issues"
    assert "silent velocity" in report.lower(), "Should warn about silent velocity tokens"
    assert "zero duration" in report.lower(), "Should warn about zero duration tokens"
    
    print("✅ Quality report test passed!")

def run_all_tests():
    """Run all enhanced musical grammar tests."""
    print("🎵 Testing Enhanced Musical Grammar System")
    print("=" * 50)
    
    try:
        test_velocity_parameter_loss()
        test_timing_parameter_loss()
        test_enhanced_note_pairing()
        test_parameter_quality_validator()
        test_quality_report()
        
        print("\n🎉 All Enhanced Musical Grammar Tests Passed!")
        print("\n📊 Summary:")
        print("- Velocity parameter loss correctly penalizes silent tokens")
        print("- Timing parameter loss correctly penalizes zero-duration tokens") 
        print("- Enhanced note pairing loss detects mismatched pitch pairs")
        print("- Parameter quality validator provides comprehensive scoring")
        print("- Quality report generation works with warnings")
        print("\n✅ Phase 5.1 musical grammar enhancements are working correctly!")
        
    except Exception as e:
        print(f"\n❌ Test failed with error: {e}")
        import traceback
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = run_all_tests()
    sys.exit(0 if success else 1)