#!/usr/bin/env python3
"""
Test the constrained generation approach that solves all three token issues.
"""

import sys
import torch
from pathlib import Path

# Add src to path
sys.path.insert(0, str(Path(__file__).parent))

from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.data.representation import VocabularyConfig
from src.generation.midi_export import MidiExporter
from src.training.musical_grammar import validate_generated_sequence, get_parameter_quality_report
from src.utils.base_logger import setup_logger

# Import the constrained sampler from our fix
from fix_token_generation_issues import ConstrainedSampler

logger = setup_logger(__name__)

def test_constrained_generation():
    """Test the constrained generation that solves all token issues."""
    
    # Load the fixed model
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    vocab_config = VocabularyConfig()
    
    # Load model
    checkpoint_path = "outputs/training/fixed/fixed_training_20250713_193713/fixed_model_epoch_2.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=checkpoint.get('d_model', 128),
        n_layers=checkpoint.get('n_layers', 2),
        n_heads=checkpoint.get('n_heads', 8),
        max_sequence_length=checkpoint.get('max_sequence_length', 256),
        mode="transformer"
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    logger.info(f"Loaded fixed model with metrics: {checkpoint.get('metrics', {})}")
    
    # Initialize constrained sampler
    constrained_sampler = ConstrainedSampler(model, device, vocab_config)
    
    # Generate with different temperatures
    temperatures = [0.6, 0.8, 1.0]
    lengths = [40, 60, 80]
    
    results = []
    
    for temp in temperatures:
        for length in lengths:
            logger.info(f"\n🎵 Testing: length={length}, temperature={temp}")
            
            # Generate with constraints
            generated_tokens = constrained_sampler.generate_constrained(
                max_length=length, 
                temperature=temp
            )
            
            sample_tokens = generated_tokens[0].cpu().numpy()
            
            # Validate
            validation = validate_generated_sequence(sample_tokens, vocab_config)
            
            logger.info(f"  Generated {len(sample_tokens)} tokens")
            logger.info(f"  Notes: {validation['note_count']}")
            logger.info(f"  Grammar Score: {validation['grammar_score']:.3f}")
            logger.info(f"  Note Pairing Score: {validation.get('note_pairing_score', 0):.3f}")
            
            # Test MIDI export
            exporter = MidiExporter()
            output_path = f"outputs/generated/constrained_test_len{length}_temp{temp:.1f}.mid"
            
            try:
                stats = exporter.export_tokens_to_midi(
                    sample_tokens, 
                    output_path, 
                    title=f"Constrained Generation L{length} T{temp}"
                )
                logger.info(f"  ✅ MIDI Export: {stats.total_notes} notes, {stats.total_duration:.2f}s")
                
                # Generate quality report
                report = get_parameter_quality_report(sample_tokens, vocab_config)
                report_path = f"outputs/generated/quality_report_len{length}_temp{temp:.1f}.txt"
                with open(report_path, 'w') as f:
                    f.write(report)
                
                results.append({
                    'length': length,
                    'temperature': temp,
                    'notes': validation['note_count'],
                    'midi_notes': stats.total_notes,
                    'grammar': validation['grammar_score'],
                    'pairing': validation.get('note_pairing_score', 0),
                    'duration': stats.total_duration,
                    'tokens': len(sample_tokens)
                })
                
            except Exception as e:
                logger.error(f"  ❌ MIDI Export failed: {e}")
    
    # Summary
    logger.info(f"\n🎉 Test Summary:")
    logger.info(f"Successful generations: {len(results)}")
    
    if results:
        best = max(results, key=lambda x: x['midi_notes'])
        logger.info(f"🏆 Best result: {best['midi_notes']} MIDI notes")
        logger.info(f"   Parameters: length={best['length']}, temp={best['temperature']}")
        logger.info(f"   Grammar: {best['grammar']:.3f}, Pairing: {best['pairing']:.3f}")
        logger.info(f"   Duration: {best['duration']:.2f}s")
        
        # Check if we solved the three issues
        issue1_solved = best['midi_notes'] > 0  # No immediate END
        issue2_solved = best['pairing'] > 0.1   # Good note pairing
        issue3_solved = best['grammar'] > 0.6   # Good musical structure
        
        logger.info(f"\n🔍 Issue Resolution Status:")
        logger.info(f"  Issue 1 (Immediate END tokens): {'✅ SOLVED' if issue1_solved else '❌ NOT SOLVED'}")
        logger.info(f"  Issue 2 (Unmatched note pairs): {'✅ SOLVED' if issue2_solved else '❌ NOT SOLVED'}")
        logger.info(f"  Issue 3 (Poor structure): {'✅ SOLVED' if issue3_solved else '❌ NOT SOLVED'}")
        
        if all([issue1_solved, issue2_solved, issue3_solved]):
            logger.info(f"🎯 ALL THREE ISSUES SUCCESSFULLY RESOLVED! 🎯")
        else:
            logger.info(f"⚠️  Some issues remain - need further training")
    
    return results

if __name__ == "__main__":
    results = test_constrained_generation()