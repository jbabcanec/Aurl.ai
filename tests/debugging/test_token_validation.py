"""
Test token validation and correction functionality.
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import numpy as np
from src.generation.token_validator import TokenValidator, validate_and_fix_tokens
from src.data.representation import MusicRepresentationConverter, VocabularyConfig


def test_token_validation():
    """Test various token sequences and their validation/correction."""
    print("=== Token Validation Tests ===\n")
    
    vocab_config = VocabularyConfig()
    validator = TokenValidator(vocab_config)
    converter = MusicRepresentationConverter(vocab_config)
    
    # Test cases
    test_cases = [
        {
            "name": "Empty sequence",
            "tokens": np.array([], dtype=np.int32)
        },
        {
            "name": "Only special tokens",
            "tokens": np.array([0, 1, 1, 1, 2, 2, 3], dtype=np.int32)
        },
        {
            "name": "Random tokens",
            "tokens": np.random.randint(0, vocab_config.vocab_size, size=50)
        },
        {
            "name": "Note without velocity",
            "tokens": np.array([0, 43, 212, 131, 1], dtype=np.int32)  # START, NOTE_ON C4, TIME_SHIFT, NOTE_OFF C4, END
        },
        {
            "name": "Unclosed notes",
            "tokens": np.array([0, 708, 43, 45, 47, 1], dtype=np.int32)  # START, VEL, NOTE_ON x3, END
        },
        {
            "name": "Valid sequence",
            "tokens": validator.create_valid_sequence(30)
        }
    ]
    
    for test_case in test_cases:
        print(f"Test: {test_case['name']}")
        tokens = test_case['tokens']
        print(f"  Original length: {len(tokens)}")
        
        # Validate
        result = validator.validate_sequence(tokens, fix_errors=True)
        print(f"  Valid: {result.is_valid}")
        print(f"  Notes found: {result.num_notes}")
        print(f"  Errors: {result.num_errors}")
        
        if result.error_messages:
            print("  Error messages:")
            for err in result.error_messages[:3]:  # Show first 3 errors
                print(f"    - {err}")
            if len(result.error_messages) > 3:
                print(f"    ... and {len(result.error_messages) - 3} more")
        
        # Try to convert to MIDI
        try:
            midi_data = converter.tokens_to_midi(tokens)
            total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
            print(f"  Original MIDI notes: {total_notes}")
        except Exception as e:
            print(f"  Original MIDI conversion failed: {e}")
        
        # If corrected, try the corrected version
        if result.corrected_tokens is not None:
            print(f"  Corrected length: {len(result.corrected_tokens)}")
            try:
                midi_data = converter.tokens_to_midi(result.corrected_tokens)
                total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
                print(f"  Corrected MIDI notes: {total_notes}")
            except Exception as e:
                print(f"  Corrected MIDI conversion failed: {e}")
        
        print()


def test_fix_generated_sequence():
    """Test fixing a sequence that might come from an untrained model."""
    print("=== Fixing Generated Sequences ===\n")
    
    vocab_config = VocabularyConfig()
    
    # Simulate output from untrained model (mostly random with some structure)
    tokens = np.array([
        0,    # START
        43,   # NOTE_ON C4 (missing velocity!)
        212,  # TIME_SHIFT
        45,   # NOTE_ON D4 (missing velocity!)
        131,  # NOTE_OFF C4
        # Missing NOTE_OFF for D4
        1,    # END
        1,    # Extra END tokens
        1,
        55,   # Random token after END
        102,
    ], dtype=np.int32)
    
    print("Original token sequence:")
    for i, token in enumerate(tokens):
        event_type, value = vocab_config.token_to_event_info(int(token))
        print(f"  {i:2d}: Token {token:3d} = {event_type.name} (value={value})")
    
    # Validate and fix
    is_valid, corrected = validate_and_fix_tokens(tokens, vocab_config)
    
    print(f"\nOriginal valid: {is_valid}")
    
    if corrected is not None:
        print("\nCorrected token sequence:")
        for i, token in enumerate(corrected):
            event_type, value = vocab_config.token_to_event_info(int(token))
            print(f"  {i:2d}: Token {token:3d} = {event_type.name} (value={value})")
        
        # Convert both to MIDI and compare
        converter = MusicRepresentationConverter(vocab_config)
        
        original_midi = converter.tokens_to_midi(tokens)
        corrected_midi = converter.tokens_to_midi(corrected)
        
        original_notes = sum(len(inst.notes) for inst in original_midi.instruments)
        corrected_notes = sum(len(inst.notes) for inst in corrected_midi.instruments)
        
        print(f"\nOriginal MIDI notes: {original_notes}")
        print(f"Corrected MIDI notes: {corrected_notes}")
        
        if corrected_notes > 0:
            print("\nCorrected MIDI contains:")
            for inst in corrected_midi.instruments:
                for note in inst.notes:
                    print(f"  Note: pitch={note.pitch}, start={note.start:.3f}, end={note.end:.3f}")


if __name__ == "__main__":
    test_token_validation()
    print("\n" + "="*60 + "\n")
    test_fix_generated_sequence()