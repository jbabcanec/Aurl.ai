# How to Generate Music with Aurl.ai

This comprehensive guide explains how to use Aurl.ai's music generation system to create high-quality MIDI files that can be opened in professional music software like Finale, Sibelius, and DAWs.

## Overview

Aurl.ai's music generation system consists of several key components:

1. **Token-based Music Representation**: Music is encoded as sequences of tokens representing notes, timing, and musical events
2. **Advanced Generation Methods**: Multiple sampling strategies and constraint-based generation
3. **Professional MIDI Export**: High-quality MIDI file creation with full musical nuance preservation
4. **Style Conditioning**: Generate music in different styles (classical, jazz, pop, etc.)

## Quick Start

### Prerequisites

1. A trained Aurl.ai model checkpoint (`.pth` file)
2. Python environment with all dependencies installed
3. MIDI playback software (optional, for testing outputs)

### Basic Generation

```bash
# Generate basic music sequence
python generate.py --checkpoint path/to/model.pth --length 200 --output my_music.mid

# Generate with specific style
python generate.py --checkpoint path/to/model.pth --style classical --tempo 120 --output classical_piece.mid

# Generate multiple variations
python generate.py --checkpoint path/to/model.pth --num-samples 5 --batch-size 5 --output variations
```

## Generation Methods

### 1. Basic Generation

The simplest form of generation using the model's learned patterns:

```bash
python generate.py \
  --checkpoint models/my_model.pth \
  --length 500 \
  --temperature 0.8 \
  --output basic_generation.mid
```

**Best for**: Simple melodies, experimentation, quick results

### 2. Conditional Generation

Generate music with specific attributes and constraints:

```bash
python generate.py \
  --checkpoint models/my_model.pth \
  --style classical \
  --tempo 120 \
  --key "C major" \
  --time-signature "4/4" \
  --complexity 0.7 \
  --dynamics mf \
  --output conditional_piece.mid
```

**Best for**: Targeted compositions, specific musical requirements

### 3. Constraint-Based Generation (Most Powerful)

Uses the musical constraint engine for sophisticated musical rules:

```bash
python generate.py \
  --checkpoint models/my_model.pth \
  --use-constraints \
  --style jazz \
  --structure "AABA" \
  --measures 32 \
  --output constrained_jazz.mid
```

**Best for**: Professional compositions, complex musical structures, harmonic correctness

## Sampling Strategies

### Temperature Sampling
- **Low temperature (0.1-0.5)**: Conservative, repetitive patterns
- **Medium temperature (0.6-0.9)**: Balanced creativity and coherence
- **High temperature (1.0+)**: Highly creative, potentially chaotic

### Top-p Sampling (Recommended)
```bash
python generate.py \
  --checkpoint model.pth \
  --strategy top_p \
  --top-p 0.9 \
  --output creative_piece.mid
```

### Top-k Sampling
```bash
python generate.py \
  --checkpoint model.pth \
  --strategy top_k \
  --top-k 50 \
  --output focused_piece.mid
```

### Beam Search
```bash
python generate.py \
  --checkpoint model.pth \
  --strategy beam_search \
  --num-beams 5 \
  --output structured_piece.mid
```

## Musical Styles and Conditioning

### Available Styles
- **Classical**: Traditional classical music patterns
- **Jazz**: Jazz harmonies and rhythms
- **Pop**: Popular music structures
- **Blues**: Blues progressions and scales
- **Rock**: Rock rhythms and patterns
- **Electronic**: Electronic music elements

### Style-Specific Generation
```bash
# Classical with specific parameters
python generate.py \
  --checkpoint model.pth \
  --style classical \
  --tempo 120 \
  --complexity 0.8 \
  --dynamics p \
  --structure rondo \
  --output classical_rondo.mid

# Jazz improvisation
python generate.py \
  --checkpoint model.pth \
  --style jazz \
  --tempo 140 \
  --use-constraints \
  --structure "AABA" \
  --output jazz_standard.mid
```

## MIDI Export Features

### Standard MIDI Export
All generated music is exported as professional-grade MIDI files with:

- **High Resolution**: 480 ticks per quarter note for precise timing
- **Multi-track Support**: Separate tracks for different instruments
- **Professional Metadata**: Title, copyright, and generation info
- **Standard Compatibility**: Works with all major music software

### Export Formats
```bash
# MIDI only (default)
python generate.py --checkpoint model.pth --format midi --output song.mid

# Token sequence only
python generate.py --checkpoint model.pth --format tokens --output song.tokens

# Both MIDI and tokens
python generate.py --checkpoint model.pth --format both --output song
```

### MIDI Configuration

The system uses professional MIDI export settings:
- **Format Type 1**: Multi-track MIDI for complex arrangements
- **Conductor Track**: Global tempo and time signature information
- **Instrument Assignment**: Automatic instrument selection based on style
- **Velocity Dynamics**: Preserved musical expression

## Advanced Features

### Interactive Generation
```bash
# Interactive mode for real-time adjustments
python generate.py --checkpoint model.pth --interactive
```

### Batch Generation
```bash
# Generate multiple pieces at once
python generate.py \
  --checkpoint model.pth \
  --batch-size 10 \
  --num-samples 10 \
  --output batch_generation
```

### Prompted Generation
```bash
# Start with a specific musical phrase
python generate.py \
  --checkpoint model.pth \
  --prompt "C4 E4 G4 C5" \
  --length 200 \
  --output prompted_piece.mid
```

### Chunked Generation (for long pieces)
```bash
# Generate very long compositions efficiently
python generate.py \
  --checkpoint model.pth \
  --length 2000 \
  --chunk-size 100 \
  --output long_composition.mid
```

## Best Practices

### For Classical Music
```bash
python generate.py \
  --checkpoint model.pth \
  --style classical \
  --strategy top_p \
  --top-p 0.85 \
  --use-constraints \
  --tempo 120 \
  --dynamics mp \
  --structure rondo \
  --complexity 0.7
```

### For Jazz
```bash
python generate.py \
  --checkpoint model.pth \
  --style jazz \
  --strategy top_p \
  --top-p 0.9 \
  --use-constraints \
  --tempo 140 \
  --structure "AABA" \
  --complexity 0.8
```

### For Pop Music
```bash
python generate.py \
  --checkpoint model.pth \
  --style pop \
  --strategy temperature \
  --temperature 0.7 \
  --tempo 120 \
  --structure "verse-chorus" \
  --measures 32
```

## Opening in Music Software

### Finale
1. Open Finale
2. File → Open → Select your `.mid` file
3. The MIDI will be automatically converted to notation
4. Adjust quantization if needed (typically 16th or 8th notes)

### Sibelius
1. Open Sibelius
2. File → Open → Select your `.mid` file
3. Use the MIDI import wizard
4. Choose appropriate quantization and staff assignments

### DAWs (Logic, Pro Tools, Cubase, etc.)
1. Create new project
2. Import MIDI file or drag-and-drop
3. Assign instruments to tracks
4. The timing and notes will be preserved perfectly

### MuseScore (Free)
1. Open MuseScore
2. File → Open → Select your `.mid` file
3. Adjust import settings for optimal notation display

## Troubleshooting

### No Sound in Generated MIDI
- MIDI files contain note data, not audio
- Use a software synthesizer or import into a DAW
- Check that your playback software supports MIDI

### Timing Issues
- Adjust the `--tempo` parameter
- Use `--use-constraints` for better rhythmic structure
- Try different quantization settings in your music software

### Style Not Matching Expectations
- Ensure model was trained on the desired style
- Adjust `--complexity` parameter (0.0-1.0)
- Use style-specific parameters and constraints

### Long Generation Times
- Reduce `--length` parameter
- Use `--chunk-size` for very long pieces
- Consider using a GPU-enabled environment

## Example Workflows

### Creating a Classical Sonata Movement
```bash
# First movement - Allegro
python generate.py \
  --checkpoint classical_model.pth \
  --style classical \
  --use-constraints \
  --tempo 140 \
  --structure rondo \
  --complexity 0.8 \
  --dynamics f \
  --measures 120 \
  --output sonata_movement1.mid
```

### Jazz Standards Generation
```bash
# Generate a jazz standard with typical AABA form
python generate.py \
  --checkpoint jazz_model.pth \
  --style jazz \
  --use-constraints \
  --structure "AABA" \
  --tempo 120 \
  --measures 32 \
  --complexity 0.7 \
  --output jazz_standard.mid
```

### Pop Song Creation
```bash
# Generate verse-chorus structure
python generate.py \
  --checkpoint pop_model.pth \
  --style pop \
  --structure "verse-chorus" \
  --tempo 120 \
  --measures 64 \
  --complexity 0.6 \
  --output pop_song.mid
```

## Advanced Configuration

### Custom MIDI Export Settings

For advanced users, you can modify MIDI export settings by editing the configuration in `src/generation/midi_export.py`:

```python
# High-resolution configuration
config = MidiExportConfig(
    resolution=960,         # Very high resolution
    format_type=1,          # Multi-track
    quantize_timing=False,  # Preserve natural timing
    humanize_timing=True,   # Add slight variations
    humanize_velocity=True  # Natural velocity curves
)
```

### Custom Constraint Rules

Modify constraint rules in `src/generation/constraints.py` for specific musical requirements.

## Performance Tips

1. **Use GPU**: Ensure PyTorch can access GPU for faster generation
2. **Batch Processing**: Generate multiple pieces simultaneously
3. **Chunk Large Pieces**: Use `--chunk-size` for compositions over 1000 tokens
4. **Cache Models**: The system caches loaded models for repeated use

## Output Quality

The MIDI export system is designed for professional use:

- **Professional Resolution**: 480-960 ticks per quarter note
- **Multi-track Support**: Separate tracks for different instruments
- **Preserved Dynamics**: Velocity information maintained
- **Accurate Timing**: Precise temporal relationships
- **Metadata Inclusion**: Title, style, and generation information
- **Standard Compliance**: Compatible with all major music software

## Getting the Best Results

1. **Choose the right generation method**:
   - Basic: Quick experimentation
   - Conditional: Targeted results
   - Constraint-based: Professional quality

2. **Use appropriate sampling**:
   - Top-p (0.9): Most creative while coherent
   - Temperature (0.7-0.8): Balanced approach
   - Beam search: Most structured

3. **Style conditioning**:
   - Always specify style for best results
   - Use appropriate tempo and dynamics for the style
   - Enable constraints for style-appropriate harmonic progressions

4. **Post-processing**:
   - Import MIDI into professional software
   - Adjust quantization as needed
   - Add appropriate instruments and effects

This system provides a complete pipeline from AI generation to professional music notation, suitable for composers, educators, and music professionals.