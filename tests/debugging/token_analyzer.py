#!/usr/bin/env python3
"""
Token Analysis Debug Tool

Analyze what tokens the model is generating and why they don't create valid music.
"""

import sys
import torch
import numpy as np
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN
from src.generation.sampler import MusicSampler, SamplingStrategy, GenerationConfig
from src.data.representation import VocabularyConfig
from src.training.musical_grammar import validate_generated_sequence
from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)

def analyze_token_sequence(tokens, vocab_config):
    """Analyze a token sequence to understand musical grammar issues."""
    from src.data.representation import EventType
    
    tokens_np = tokens.cpu().numpy() if torch.is_tensor(tokens) else tokens
    
    # Decode tokens to understand what they represent
    token_types = []
    note_on_tokens = []
    note_off_tokens = []
    timing_tokens = []
    velocity_tokens = []
    
    for i, token in enumerate(tokens_np):
        event_type, value = vocab_config.token_to_event_info(int(token))
        
        if event_type == EventType.NOTE_ON:
            token_types.append(f"NOTE_ON_{value}")
            note_on_tokens.append((i, value))
        elif event_type == EventType.NOTE_OFF:
            token_types.append(f"NOTE_OFF_{value}")
            note_off_tokens.append((i, value))
        elif event_type == EventType.TIME_SHIFT:
            token_types.append(f"TIME_{value}")
            timing_tokens.append((i, value))
        elif event_type == EventType.VELOCITY_CHANGE:
            token_types.append(f"VELOCITY_{value}")
            velocity_tokens.append((i, value))
        elif event_type in [EventType.START_TOKEN, EventType.END_TOKEN, EventType.PAD_TOKEN]:
            token_types.append(f"{event_type.name}")
        else:
            token_types.append(f"{event_type.name}_{value}")
    
    return {
        'token_types': token_types,
        'note_on_tokens': note_on_tokens,
        'note_off_tokens': note_off_tokens,
        'timing_tokens': timing_tokens,
        'velocity_tokens': velocity_tokens,
        'total_tokens': len(tokens_np)
    }

def debug_model_generation():
    """Generate tokens and analyze what's wrong with the musical grammar."""
    logger.info("🔍 Starting Token Analysis Debug")
    
    # Load model
    device = torch.device('cuda' if torch.cuda.is_available() else 
                         'mps' if torch.backends.mps.is_available() else 'cpu')
    
    # Use the new grammar-trained model
    checkpoint_path = "debug/grammar_training_20250713_145739/good_grammar_epoch_0.pt"
    checkpoint = torch.load(checkpoint_path, map_location=device)
    
    vocab_config = VocabularyConfig()
    
    # The grammar-trained model has specific architecture
    model = MusicTransformerVAEGAN(
        vocab_size=vocab_config.vocab_size,
        d_model=128,  # From quick_test config
        n_layers=2,   # From quick_test config  
        n_heads=2,    # From quick_test config
        max_sequence_length=256,  # From quick_test config
        mode="transformer"
    )
    
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    
    sampler = MusicSampler(model, device)
    
    # Generate different types of sequences
    strategies = [
        (SamplingStrategy.TEMPERATURE, {"temperature": 0.5}),
        (SamplingStrategy.TEMPERATURE, {"temperature": 1.0}),
        (SamplingStrategy.TEMPERATURE, {"temperature": 2.0}),
    ]
    
    debug_results = []
    
    for i, (strategy, params) in enumerate(strategies):
        logger.info(f"Testing strategy {i+1}: {strategy.name} with {params}")
        
        config = GenerationConfig(
            max_length=32,
            strategy=strategy,
            **params
        )
        
        with torch.no_grad():
            generated_tokens = sampler.generate(config=config)
        
        # Analyze the first sequence
        tokens = generated_tokens[0]
        analysis = analyze_token_sequence(tokens, vocab_config)
        
        # Get grammar validation
        validation = validate_generated_sequence(tokens.cpu().numpy(), vocab_config)
        
        result = {
            'strategy': strategy.name,
            'params': params,
            'tokens': tokens.cpu().numpy().tolist(),
            'analysis': analysis,
            'validation': validation,
            'grammar_score': validation['grammar_score']
        }
        
        debug_results.append(result)
        
        # Print immediate analysis
        logger.info(f"  Generated {len(tokens)} tokens")
        logger.info(f"  Grammar score: {validation['grammar_score']:.3f}")
        logger.info(f"  Note ON tokens: {len(analysis['note_on_tokens'])}")
        logger.info(f"  Note OFF tokens: {len(analysis['note_off_tokens'])}")
        logger.info(f"  Timing tokens: {len(analysis['timing_tokens'])}")
        
        # Show first 10 tokens
        first_10 = analysis['token_types'][:10]
        logger.info(f"  First 10 tokens: {first_10}")
        
        # Check for repetition
        tokens_list = tokens.cpu().numpy().tolist()
        if len(set(tokens_list)) < len(tokens_list) * 0.5:
            logger.warning(f"  High repetition detected!")
            
    return debug_results

def main():
    debug_results = debug_model_generation()
    
    # Save detailed analysis
    import json
    debug_file = Path("debug/token_analysis_results.json")
    
    # Convert numpy arrays to lists for JSON serialization
    serializable_results = []
    for result in debug_results:
        serializable_result = {
            'strategy': result['strategy'],
            'params': result['params'],
            'tokens': result['tokens'],
            'grammar_score': float(result['grammar_score']),
            'total_tokens': result['analysis']['total_tokens'],
            'note_on_count': len(result['analysis']['note_on_tokens']),
            'note_off_count': len(result['analysis']['note_off_tokens']),
            'timing_count': len(result['analysis']['timing_tokens']),
            'token_types_sample': result['analysis']['token_types'][:20],
            'validation_summary': {
                'grammar_score': float(result['validation']['grammar_score']),
                'note_pairing_score': float(result['validation']['note_pairing_score']),
                'timing_score': float(result['validation']['timing_score']),
                'repetition_score': float(result['validation']['repetition_score'])
            }
        }
        serializable_results.append(serializable_result)
    
    with open(debug_file, 'w') as f:
        json.dump(serializable_results, f, indent=2)
    
    logger.info(f"📊 Debug analysis saved to {debug_file}")
    
    # Print summary
    logger.info("\n🎯 SUMMARY:")
    for result in debug_results:
        logger.info(f"Strategy {result['strategy']}: Grammar={result['grammar_score']:.3f}, "
                   f"Notes={len(result['analysis']['note_on_tokens'])}")

if __name__ == "__main__":
    main()