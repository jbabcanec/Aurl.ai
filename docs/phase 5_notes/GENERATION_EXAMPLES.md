# 🎼 Aurl.ai Music Generation - Usage Examples

## Quick Start

The Aurl.ai music generation system provides state-of-the-art AI music generation with advanced conditional control. This guide shows you how to use the generation system effectively.

## Prerequisites

- Trained model checkpoint (e.g., `outputs/checkpoints/best_model.pt`)
- Python environment with Aurl.ai installed
- CPU or GPU device (CUDA/MPS supported)

## Basic Generation

### Simple Generation
Generate basic music with default settings:

```bash
python generate.py --checkpoint outputs/checkpoints/best_model.pt --length 512
```

### Sampling Strategies
Choose different sampling strategies for varied musical output:

```bash
# Temperature sampling (default, creative)
python generate.py --checkpoint outputs/checkpoints/best_model.pt --strategy temperature --temperature 0.8

# Top-k sampling (focused, coherent)
python generate.py --checkpoint outputs/checkpoints/best_model.pt --strategy top_k --top-k 20

# Top-p/nucleus sampling (balanced)
python generate.py --checkpoint outputs/checkpoints/best_model.pt --strategy top_p --top-p 0.9

# Greedy sampling (deterministic)
python generate.py --checkpoint outputs/checkpoints/best_model.pt --strategy greedy

# Beam search (high quality, slower)
python generate.py --checkpoint outputs/checkpoints/best_model.pt --strategy beam_search --num-beams 4
```

## Conditional Generation

### Style-Based Generation
Generate music in specific styles:

```bash
# Jazz style with medium complexity
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --style jazz --complexity 0.7 --length 256

# Classical style with high complexity
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --style classical --complexity 0.9 --length 512

# Pop style with simple patterns
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --style pop --complexity 0.3 --length 128
```

### Musical Attributes
Control specific musical parameters:

```bash
# Set tempo and key
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --tempo 120 --key "C major" --length 256

# Complex musical setup
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --tempo 140 --key "A minor" --time-signature "4/4" \
    --dynamics mf --length 512
```

### Structural Control
Generate music with specific structural forms:

```bash
# AABA song form with 32 measures
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --structure AABA --measures 32 --length 1024

# Verse-chorus structure
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --structure verse-chorus --measures 48 --length 1536
```

### Combined Conditioning
Combine multiple conditioning parameters:

```bash
# Complete musical specification
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --style jazz --complexity 0.8 \
    --tempo 120 --key "Bb major" --time-signature "4/4" \
    --dynamics mf --structure AABA --measures 32 \
    --length 1024 --strategy top_p --top-p 0.85
```

## Batch Generation

### Multiple Samples
Generate multiple variations efficiently:

```bash
# Generate 5 samples in parallel
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --num-samples 5 --batch-size 5 --length 256

# Generate 10 jazz pieces
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --style jazz --num-samples 10 --batch-size 5 --length 512
```

## Interactive Generation

### Real-time Generation
Start an interactive session for real-time music creation:

```bash
# Interactive mode with dynamic control
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --interactive --length 1024 --chunk-size 16
```

In interactive mode, you can:
- Generate music in real-time chunks
- Dynamically change tempo, style, and dynamics
- Build extended compositions progressively

## Output Options

### File Formats
Control output format and location:

```bash
# Save as tokens only
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --format tokens --output my_output --filename my_piece

# Save both tokens and MIDI (when implemented)
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --format both --output my_output --filename jazz_piece_001
```

### Custom Output Directory
```bash
# Specify custom output location
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --output /path/to/my/music --filename custom_composition
```

## Advanced Usage

### Reproducible Generation
Use seeds for reproducible results:

```bash
# Same seed = same output
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --seed 42 --length 256
```

### Memory Optimization
For memory-constrained environments:

```bash
# Disable caching for lower memory usage
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --no-cache --length 256
```

### Verbose Output
Get detailed generation statistics:

```bash
# Verbose logging with performance metrics
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --verbose --length 256
```

## Generation Statistics

The system provides comprehensive statistics about generation:

- **Generation Speed**: Tokens per second
- **Conditioning Usage**: Which conditions were applied
- **Constraint Statistics**: Musical rules enforced
- **Model Performance**: Memory usage and timing

Example output:
```
Generation complete. Stats: {
    'total_conditional_generations': 1, 
    'conditioning_types_used': {'style': 1, 'tempo': 1}, 
    'constraint_stats': {
        'total_applied': 31, 
        'violations_corrected': 0, 
        'constraint_counts': {
            'harmonic': 31, 'melodic': 31, 'rhythmic': 31
        }
    }
}
```

## Musical Constraints

The system automatically applies musical theory constraints:

- **Harmonic Constraints**: Ensures chord progressions follow harmonic rules
- **Melodic Constraints**: Maintains melodic coherence and voice leading
- **Rhythmic Constraints**: Enforces rhythmic patterns and meter
- **Dynamic Constraints**: Maintains consistent dynamics and expression
- **Structural Constraints**: Follows musical form and phrase structure

These can be disabled with `--use-constraints false` if desired.

## Device Configuration

The generation system supports multiple compute devices:

```bash
# Automatic device selection (recommended)
python generate.py --checkpoint outputs/checkpoints/best_model.pt --device auto

# Force CPU usage
python generate.py --checkpoint outputs/checkpoints/best_model.pt --device cpu

# Use CUDA GPU
python generate.py --checkpoint outputs/checkpoints/best_model.pt --device cuda

# Use Apple Silicon MPS
python generate.py --checkpoint outputs/checkpoints/best_model.pt --device mps
```

## Performance Tips

1. **Use appropriate sequence lengths**: Start with 256-512 tokens for experimentation
2. **Batch generation**: Use `--batch-size` for generating multiple pieces efficiently
3. **Choose optimal sampling**: `top_p` offers best balance of quality and diversity
4. **Leverage conditioning**: Specific conditions produce more coherent results
5. **Monitor memory**: Use `--no-cache` if running out of memory

## Troubleshooting

### Common Issues

**Out of Memory**:
```bash
# Use smaller batch size and disable cache
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --batch-size 1 --no-cache --length 128
```

**Slow Generation**:
```bash
# Use simpler sampling strategy
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --strategy greedy --length 256
```

**Poor Quality**:
```bash
# Use conditional generation with constraints
python generate.py --checkpoint outputs/checkpoints/best_model.pt \
    --style classical --use-constraints --strategy top_p --top-p 0.9
```

## Integration with Python Code

You can also use the generation system programmatically:

```python
import torch
from src.generation import MusicSampler, GenerationConfig, SamplingStrategy
from src.models.music_transformer_vae_gan import MusicTransformerVAEGAN

# Load model
device = torch.device("cpu")
checkpoint = torch.load("outputs/checkpoints/best_model.pt", map_location=device)
model = MusicTransformerVAEGAN(**model_config)
model.load_state_dict(checkpoint['model_state_dict'])

# Create sampler
sampler = MusicSampler(model, device)

# Generate music
config = GenerationConfig(
    max_length=256,
    strategy=SamplingStrategy.TOP_P,
    top_p=0.9,
    temperature=0.8
)

generated_tokens = sampler.generate(config=config)
print(f"Generated {generated_tokens.size(1)} tokens")
```

For more advanced programmatic usage, see the test files in `tests/phase_7_tests/`.

## Next Steps

- **Phase 7.2**: MIDI export functionality for direct audio file creation
- **Phase 7.3**: Model optimization for faster inference and mobile deployment
- **Phase 8**: Web interface and API for easy access

The generation system is production-ready and provides state-of-the-art music generation capabilities with comprehensive conditional control.