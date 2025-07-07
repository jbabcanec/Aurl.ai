# 🎼 How to Generate Music with MidiFly

This guide shows you how to generate new music using a trained MidiFly model.

## Prerequisites

1. **Trained Model**: You need a trained model checkpoint (see [HOW_TO_TRAIN.md](HOW_TO_TRAIN.md))
2. **Environment**: Same as training environment
   ```bash
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

## Quick Start

```bash
# Generate a single piece with default settings
python generate_pipeline.py --model outputs/checkpoints/best.pt

# Generate multiple pieces
python generate_pipeline.py --model outputs/checkpoints/best.pt --num_pieces 10

# Generate with specific style
python generate_pipeline.py --model outputs/checkpoints/best.pt --style classical
```

## Generation Modes

### 1. Unconditional Generation
Generate completely new pieces from scratch:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --mode unconditional \
    --length 2048 \
    --output outputs/generated/new_piece.mid
```

### 2. Style-Based Generation
Generate in the style of your training data subsets:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --mode style \
    --style "baroque" \
    --length 2048
```

Available styles depend on your training data organization.

### 3. Continuation
Continue an existing MIDI file:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --mode continuation \
    --input data/raw/mozart_sonata.mid \
    --start_measure 16 \
    --generate_measures 32
```

### 4. Variation
Create variations of existing pieces:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --mode variation \
    --input data/raw/original.mid \
    --variation_strength 0.5  # 0.0 = identical, 1.0 = very different
```

### 5. Interpolation
Blend between two pieces:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --mode interpolation \
    --input_a data/raw/piece1.mid \
    --input_b data/raw/piece2.mid \
    --interpolation_steps 5
```

### 6. Conditional Generation
Generate with specific musical constraints:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --mode conditional \
    --key "C_major" \
    --time_signature "4/4" \
    --tempo 120 \
    --chord_progression "C-Am-F-G"
```

## Sampling Parameters

### Temperature
Controls randomness (0.1 = conservative, 2.0 = very random):

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --temperature 0.8
```

### Top-k Sampling
Only consider top k most likely next tokens:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --top_k 40
```

### Top-p (Nucleus) Sampling
Sample from tokens that sum to probability p:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --top_p 0.9
```

### Beam Search
Generate multiple candidates and pick best:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --beam_size 5 \
    --length_penalty 0.6
```

## Advanced Options

### Batch Generation
Generate multiple pieces efficiently:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --batch_generate \
    --num_pieces 100 \
    --output_dir outputs/generated/batch_001/
```

### Interactive Generation
Real-time generation with MIDI input:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --interactive \
    --midi_input_device "Your MIDI Keyboard"
```

### Constrained Generation
Apply music theory constraints:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --constraints "avoid_parallel_fifths,voice_leading" \
    --constraint_strength 0.8
```

### Multi-Track Generation
Generate full arrangements:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --tracks "piano,strings,bass,drums" \
    --arrangement_style "orchestral"
```

## Output Formats

### MIDI Export
```bash
# Standard MIDI file
python generate_pipeline.py --model outputs/checkpoints/best.pt --format midi

# With specific MIDI settings
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --format midi \
    --midi_program 1 \  # Acoustic Grand Piano
    --midi_resolution 480
```

### Audio Rendering
```bash
# Render to audio using FluidSynth
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --render_audio \
    --soundfont "path/to/soundfont.sf2" \
    --audio_format "wav"
```

### Music XML
```bash
# For notation software
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --format musicxml
```

### JSON (for analysis)
```bash
# Structured data format
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --format json
```

## Generation Strategies

### 1. Seed-Based Generation
Use random seeds for reproducibility:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --seed 42 \
    --num_pieces 5
```

### 2. Progressive Generation
Generate in sections for long pieces:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --progressive \
    --section_length 512 \
    --num_sections 8 \
    --overlap 64
```

### 3. Ensemble Generation
Use multiple models:

```bash
python generate_pipeline.py \
    --models "model1.pt,model2.pt,model3.pt" \
    --ensemble_method "voting" \
    --ensemble_temperature 0.7
```

## Quality Control

### Filtering
Automatically filter generated pieces:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --filter_quality \
    --min_quality_score 0.7 \
    --num_attempts 50 \
    --keep_best 10
```

### Post-Processing
Clean up generated music:

```bash
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --post_process \
    --quantize_to "16th" \
    --remove_short_notes 0.1 \
    --smooth_dynamics
```

## Performance Tips

### GPU Acceleration
```bash
# Use specific GPU
python generate_pipeline.py --model outputs/checkpoints/best.pt --gpu 0

# CPU-only generation
python generate_pipeline.py --model outputs/checkpoints/best.pt --cpu
```

### Memory Management
```bash
# For long sequences
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --use_memory_efficient_attention \
    --max_batch_size 1
```

### Speed Optimization
```bash
# Faster generation with some quality trade-off
python generate_pipeline.py \
    --model outputs/checkpoints/best.pt \
    --fast_generation \
    --cache_size 2048
```

## Examples

### Classical Style Sonata
```bash
python generate_pipeline.py \
    --model outputs/checkpoints/classical_model.pt \
    --style "classical_sonata" \
    --key "D_major" \
    --time_signature "3/4" \
    --length 4096 \
    --temperature 0.7 \
    --output "outputs/generated/sonata_in_d.mid"
```

### Jazz Improvisation
```bash
python generate_pipeline.py \
    --model outputs/checkpoints/jazz_model.pt \
    --mode continuation \
    --input "data/jazz_standards/autumn_leaves.mid" \
    --start_measure 8 \
    --style "bebop" \
    --temperature 1.2 \
    --swing_factor 0.67
```

### Film Score
```bash
python generate_pipeline.py \
    --model outputs/checkpoints/cinematic_model.pt \
    --mode conditional \
    --emotion "epic" \
    --intensity_curve "0.2,0.5,0.8,1.0,0.7,0.3" \
    --length 8192 \
    --tracks "strings,brass,percussion,choir"
```

## Troubleshooting

### Poor Quality Output
- Lower temperature: `--temperature 0.6`
- Use beam search: `--beam_size 10`
- Try different checkpoints: `--model outputs/checkpoints/epoch_80.pt`

### Repetitive Output
- Increase temperature: `--temperature 1.2`
- Use top-p sampling: `--top_p 0.95`
- Add repetition penalty: `--repetition_penalty 1.2`

### Out of Memory
- Reduce length: `--length 1024`
- Generate progressively: `--progressive`
- Use CPU: `--cpu`

### Incorrect Style
- Check model training data
- Use style-specific checkpoint
- Adjust conditioning strength: `--style_strength 0.9`

## Integration

### Python API
```python
from midifly import Generator

# Initialize generator
gen = Generator("outputs/checkpoints/best.pt")

# Generate
midi_data = gen.generate(
    length=2048,
    temperature=0.8,
    style="classical"
)

# Save
midi_data.save("output.mid")
```

### REST API
```bash
# Start API server
python -m midifly.api --model outputs/checkpoints/best.pt --port 8080

# Generate via API
curl -X POST http://localhost:8080/generate \
    -H "Content-Type: application/json" \
    -d '{"length": 2048, "temperature": 0.8}'
```

For more details, see [api_reference.md](api_reference.md) and [architecture.md](architecture.md).