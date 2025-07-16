#!/usr/bin/env python3
"""
Test Generation with Grammar-Enforced Model

Generate MIDI using the grammar-enforced trained model.
"""

import sys
import torch
from pathlib import Path
import json

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.generation.sampler import MusicSampler, SamplingStrategy, GenerationConfig
from src.data.representation import VocabularyConfig
from src.generation.midi_export import export_tokens_to_midi_file, create_standard_config
from src.training.musical_grammar import validate_generated_sequence
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)

def test_grammar_model():
    """Test the grammar-enforced model generation."""
    logger.info("🎵 Testing Grammar-Enforced Model Generation")
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Load checkpoint
    checkpoint_path = "debug/grammar_training_20250713_145739/good_grammar_epoch_0.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab_config = VocabularyConfig()
    
    # Create model with correct architecture
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=128,
        n_layers=2,
        n_heads=2,
        max_sequence_length=256,
        mode="transformer"
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    sampler = MusicSampler(model, device)
    
    # Generate multiple sequences
    generation_configs = [
        GenerationConfig(max_length=64, strategy=SamplingStrategy.TEMPERATURE, temperature=1.5),
        GenerationConfig(max_length=64, strategy=SamplingStrategy.TEMPERATURE, temperature=2.0),
        GenerationConfig(max_length=128, strategy=SamplingStrategy.TEMPERATURE, temperature=1.8),
    ]
    
    output_dir = Path("debug/test_generation")
    output_dir.mkdir(exist_ok=True)
    
    results = []
    
    for i, config in enumerate(generation_configs):
        logger.info(f"Generating sequence {i+1}: {config.strategy.name} temp={config.temperature} len={config.max_length}")
        
        with torch.no_grad():
            generated_tokens = sampler.generate(config=config)
        
        tokens = generated_tokens[0].cpu().numpy()
        
        # Validate grammar
        validation = validate_generated_sequence(tokens, vocab_config)
        grammar_score = validation['grammar_score']
        
        logger.info(f"  Grammar score: {grammar_score:.3f}")
        logger.info(f"  Note pairing: {validation['note_pairing_score']:.3f}")
        logger.info(f"  Timing: {validation['timing_score']:.3f}")
        logger.info(f"  Repetition: {validation['repetition_score']:.3f}")
        
        # Convert to MIDI
        try:
            midi_path = output_dir / f"generation_{i+1}_temp{config.temperature}_grammar{grammar_score:.3f}.mid"
            export_config = create_standard_config()
            
            export_stats = export_tokens_to_midi_file(
                tokens=tokens,
                output_path=str(midi_path),
                config=export_config
            )
            
            success = export_stats is not None
            
            if success:
                logger.info(f"  ✅ MIDI saved: {midi_path}")
                
                # Load and count notes
                import pretty_midi
                midi_data = pretty_midi.PrettyMIDI(str(midi_path))
                total_notes = sum(len(instrument.notes) for instrument in midi_data.instruments)
                logger.info(f"  Notes in MIDI: {total_notes}")
            else:
                logger.error(f"  ❌ MIDI export failed")
                midi_path = None
                total_notes = 0
            
        except Exception as e:
            logger.error(f"  ❌ MIDI conversion failed: {e}")
            midi_path = None
            total_notes = 0
        
        # Save tokens
        tokens_path = output_dir / f"tokens_{i+1}_temp{config.temperature}.json"
        with open(tokens_path, 'w') as f:
            json.dump({
                'tokens': tokens.tolist(),
                'config': {
                    'temperature': config.temperature,
                    'max_length': config.max_length,
                    'strategy': config.strategy.name
                },
                'validation': validation,
                'midi_notes': total_notes
            }, f, indent=2)
        
        results.append({
            'config': config,
            'grammar_score': grammar_score,
            'validation': validation,
            'midi_path': str(midi_path) if midi_path else None,
            'midi_notes': total_notes
        })
    
    # Summary
    logger.info("\n🎯 GENERATION SUMMARY:")
    for i, result in enumerate(results):
        logger.info(f"Sequence {i+1}: Grammar={result['grammar_score']:.3f}, "
                   f"Notes={result['midi_notes']}, "
                   f"MIDI={'✅' if result['midi_path'] else '❌'}")
    
    # Best result
    best_result = max(results, key=lambda x: x['grammar_score'])
    logger.info(f"\n🏆 BEST RESULT: Grammar={best_result['grammar_score']:.3f}, "
               f"Notes={best_result['midi_notes']}")
    
    if best_result['grammar_score'] > 0.6:
        logger.info("✅ SUCCESS: Model generates good musical grammar!")
    else:
        logger.warning("⚠️  Model still needs improvement")
    
    return results

if __name__ == "__main__":
    test_grammar_model()