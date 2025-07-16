"""
Test the full generation pipeline to debug empty MIDI files.
This script will:
1. Create a mock model that generates meaningful tokens
2. Test the generation sampler
3. Convert generated tokens to MIDI
4. Analyze what's happening at each step
"""

import sys
from pathlib import Path
sys.path.append(str(Path(__file__).parent.parent.parent))

import torch
import numpy as np
from src.generation.sampler import MusicSampler, GenerationConfig, SamplingStrategy
from src.data.representation import (
    MusicRepresentationConverter, VocabularyConfig, EventType
)

class MockMusicModel(torch.nn.Module):
    """Mock model that generates meaningful token sequences."""
    
    def __init__(self, vocab_size=774):
        super().__init__()
        self.vocab_size = vocab_size
        self.vocab_config = VocabularyConfig()
        
        # Pre-compute useful tokens
        self.start_token = 0
        self.end_token = 1
        self.velocity_tokens = []
        self.note_on_tokens = {}
        self.note_off_tokens = {}
        self.time_shift_tokens = []
        
        # Find tokens for common operations
        for token in range(vocab_size):
            event_type, value = self.vocab_config.token_to_event_info(token)
            
            if event_type == EventType.VELOCITY_CHANGE:
                self.velocity_tokens.append(token)
            elif event_type == EventType.NOTE_ON:
                self.note_on_tokens[value] = token
            elif event_type == EventType.NOTE_OFF:
                self.note_off_tokens[value] = token
            elif event_type == EventType.TIME_SHIFT:
                self.time_shift_tokens.append(token)
    
    def forward(self, input_ids):
        """Generate logits that produce a simple melody."""
        batch_size, seq_len = input_ids.shape
        device = input_ids.device
        
        # Create logits tensor
        logits = torch.zeros(batch_size, seq_len, self.vocab_size, device=device)
        
        # For each position in the sequence, determine what token should come next
        for b in range(batch_size):
            for s in range(seq_len):
                # Get last few tokens to determine context
                if s == 0:
                    context = [self.start_token]
                else:
                    context = input_ids[b, max(0, s-5):s+1].tolist()
                
                # Generate appropriate next token probabilities
                next_logits = self._get_next_token_probabilities(context)
                logits[b, s] = next_logits
        
        return logits
    
    def _get_next_token_probabilities(self, context):
        """Determine what token should come next based on context."""
        logits = torch.full((self.vocab_size,), -10.0)  # Low probability for all
        
        last_token = context[-1]
        
        # If we just started, set velocity
        if last_token == self.start_token:
            # Set medium velocity
            if len(self.velocity_tokens) > 16:
                logits[self.velocity_tokens[16]] = 10.0
        
        # If we just set velocity, play a note
        elif last_token in self.velocity_tokens:
            # Play C4
            if 60 in self.note_on_tokens:
                logits[self.note_on_tokens[60]] = 10.0
        
        # If we just played a note, wait a bit
        elif any(last_token == self.note_on_tokens.get(pitch, -1) for pitch in range(128)):
            # Time shift of ~500ms
            if len(self.time_shift_tokens) > 32:
                logits[self.time_shift_tokens[32]] = 10.0
        
        # If we waited, turn off the note
        elif last_token in self.time_shift_tokens:
            # Check what note is playing
            for i in range(len(context)-2, -1, -1):
                token = context[i]
                for pitch, note_on_token in self.note_on_tokens.items():
                    if token == note_on_token:
                        # Found the last note on, turn it off
                        if pitch in self.note_off_tokens:
                            logits[self.note_off_tokens[pitch]] = 10.0
                            return logits
            
            # If no note to turn off, play next note or end
            if len(context) > 20:
                logits[self.end_token] = 10.0
            else:
                # Play next note in scale
                last_pitch = 60
                for i in range(len(context)-1, -1, -1):
                    for pitch, token in self.note_on_tokens.items():
                        if context[i] == token:
                            last_pitch = pitch
                            break
                
                next_pitch = last_pitch + 2  # Whole tone up
                if next_pitch <= 72 and next_pitch in self.note_on_tokens:
                    logits[self.note_on_tokens[next_pitch]] = 10.0
                else:
                    logits[self.end_token] = 10.0
        
        else:
            # Default: end the sequence
            logits[self.end_token] = 10.0
        
        return logits


def test_generation_pipeline():
    """Test the full generation pipeline."""
    print("=== Testing Generation Pipeline ===")
    
    # Create mock model and sampler
    device = torch.device("cpu")
    model = MockMusicModel()
    sampler = MusicSampler(model, device)
    
    # Test different generation strategies
    configs = [
        GenerationConfig(strategy=SamplingStrategy.GREEDY, max_length=50),
        GenerationConfig(strategy=SamplingStrategy.TEMPERATURE, temperature=0.8, max_length=50),
        GenerationConfig(strategy=SamplingStrategy.TOP_K, top_k=5, temperature=0.9, max_length=50),
    ]
    
    for i, config in enumerate(configs):
        print(f"\n--- Test {i+1}: {config.strategy.value} ---")
        
        # Generate tokens
        generated = sampler.generate(config=config)
        tokens = generated[0].cpu().numpy()
        
        print(f"Generated {len(tokens)} tokens")
        
        # Show token sequence
        vocab_config = VocabularyConfig()
        print("\nToken sequence:")
        for j, token in enumerate(tokens[:30]):
            event_type, value = vocab_config.token_to_event_info(int(token))
            print(f"  {j:3d}: Token {token:3d} = {event_type.name:18s} (value={value})")
        
        # Convert to MIDI
        converter = MusicRepresentationConverter(vocab_config)
        midi_data = converter.tokens_to_midi(tokens)
        
        print(f"\nMIDI conversion result:")
        print(f"  Duration: {midi_data.end_time:.2f}s")
        print(f"  Instruments: {len(midi_data.instruments)}")
        total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
        print(f"  Total notes: {total_notes}")
        
        if total_notes > 0:
            print("\n  First few notes:")
            for inst_idx, instrument in enumerate(midi_data.instruments):
                for note_idx, note in enumerate(instrument.notes[:5]):
                    print(f"    Inst {inst_idx}, Note {note_idx}: "
                          f"Pitch {note.pitch}, start={note.start:.3f}, "
                          f"end={note.end:.3f}, velocity={note.velocity}")


def test_random_vs_meaningful_tokens():
    """Compare random tokens vs meaningful tokens."""
    print("\n=== Random vs Meaningful Tokens ===")
    
    vocab_config = VocabularyConfig()
    converter = MusicRepresentationConverter(vocab_config)
    
    # Test 1: Random tokens
    print("\n1. Random tokens (likely to produce empty MIDI):")
    random_tokens = np.random.randint(0, vocab_config.vocab_size, size=50)
    print(f"Random tokens: {random_tokens[:20]}...")
    
    midi_data = converter.tokens_to_midi(random_tokens)
    total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    print(f"Result: {total_notes} notes")
    
    # Test 2: Meaningful sequence
    print("\n2. Meaningful token sequence:")
    tokens = []
    
    # Start
    tokens.append(0)  # START_TOKEN
    
    # Set velocity
    for token in range(vocab_config.vocab_size):
        event_type, value = vocab_config.token_to_event_info(token)
        if event_type == EventType.VELOCITY_CHANGE and value == 16:
            tokens.append(token)
            break
    
    # Play a few notes
    for pitch in [60, 62, 64, 65, 67]:  # C D E F G
        # Note on
        for token in range(vocab_config.vocab_size):
            event_type, value = vocab_config.token_to_event_info(token)
            if event_type == EventType.NOTE_ON and value == pitch:
                tokens.append(token)
                break
        
        # Time shift
        for token in range(vocab_config.vocab_size):
            event_type, value = vocab_config.token_to_event_info(token)
            if event_type == EventType.TIME_SHIFT and value == 32:
                tokens.append(token)
                break
        
        # Note off
        for token in range(vocab_config.vocab_size):
            event_type, value = vocab_config.token_to_event_info(token)
            if event_type == EventType.NOTE_OFF and value == pitch:
                tokens.append(token)
                break
        
        # Gap between notes
        for token in range(vocab_config.vocab_size):
            event_type, value = vocab_config.token_to_event_info(token)
            if event_type == EventType.TIME_SHIFT and value == 16:
                tokens.append(token)
                break
    
    # End
    tokens.append(1)  # END_TOKEN
    
    token_array = np.array(tokens, dtype=np.int32)
    print(f"Meaningful tokens: {token_array[:20]}...")
    
    midi_data = converter.tokens_to_midi(token_array)
    total_notes = sum(len(inst.notes) for inst in midi_data.instruments)
    print(f"Result: {total_notes} notes")
    
    if total_notes > 0:
        print("\nGenerated notes:")
        for inst in midi_data.instruments:
            for note in inst.notes:
                print(f"  Pitch {note.pitch}, start={note.start:.3f}, "
                      f"end={note.end:.3f}, velocity={note.velocity}")


if __name__ == "__main__":
    test_generation_pipeline()
    print("\n" + "="*60 + "\n")
    test_random_vs_meaningful_tokens()