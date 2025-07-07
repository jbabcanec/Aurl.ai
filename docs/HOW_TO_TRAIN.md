# 🎵 How to Train MidiFly

This guide walks you through training your own music generation model with MidiFly.

## Prerequisites

1. **Environment Setup**
   ```bash
   # Create virtual environment
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   
   # Install dependencies
   pip install -r requirements.txt
   
   # Verify GPU availability (optional but recommended)
   python -c "import torch; print(torch.cuda.is_available())"
   ```

2. **Data Preparation**
   - Place your MIDI files in `data/raw/`
   - Supported formats: `.mid`, `.midi`
   - Recommended: At least 100 MIDI files for meaningful training
   - Files should be organized by style/genre in subdirectories (optional)

## Quick Start

```bash
# Basic training with default settings
python train_pipeline.py

# Training with custom configuration
python train_pipeline.py --config configs/custom_training.yaml

# Resume from checkpoint
python train_pipeline.py --resume outputs/checkpoints/latest.pt
```

## Detailed Training Process

### Step 1: Data Analysis
First, analyze your dataset to understand its characteristics:

```bash
python train_pipeline.py --mode analyze --data_dir data/raw/
```

This generates:
- `data/metadata/dataset_stats.json` - Statistical analysis
- `data/metadata/data_report.pdf` - Visual analysis report

### Step 2: Configure Training

Edit `configs/training_configs/default.yaml` or create your own:

```yaml
# Model Configuration
model:
  type: "hierarchical_vae_gan"
  latent_dim: 64
  encoder_layers: 6
  decoder_layers: 6
  attention_heads: 8
  
# Training Configuration  
training:
  batch_size: 32  # Reduce if GPU memory limited
  learning_rate: 0.001
  epochs: 100
  early_stopping_patience: 10
  
# Data Configuration
data:
  sequence_length: 1024  # Tokens per sequence
  augmentation:
    transpose: true
    time_stretch: true
    velocity_scale: true
  cache_size_gb: 5  # Disk cache size
  
# Logging
logging:
  log_every_n_steps: 50
  save_samples_every_n_epochs: 5
  tensorboard: true
```

### Step 3: Start Training

```bash
# Full training pipeline with monitoring
python train_pipeline.py \
    --config configs/training_configs/my_config.yaml \
    --experiment_name "music_gen_v1" \
    --gpu 0
```

### Step 4: Monitor Progress

1. **TensorBoard** (recommended):
   ```bash
   tensorboard --logdir logs/training/
   ```
   Navigate to http://localhost:6006

2. **Training Logs**:
   ```bash
   # Real-time log monitoring
   tail -f logs/training/music_gen_v1/train.log
   
   # Check latest metrics
   cat logs/training/music_gen_v1/metrics.json | jq '.'
   ```

3. **Generated Samples**:
   - Check `outputs/generated/` for sample outputs every N epochs
   - Listen to progression of quality over training

## Advanced Options

### Multi-GPU Training
```bash
# Use all available GPUs
python train_pipeline.py --gpu all

# Use specific GPUs
python train_pipeline.py --gpu 0,1,2
```

### Curriculum Learning
Train progressively from simple to complex:

```bash
python train_pipeline.py \
    --curriculum \
    --curriculum_stages "simple:20,medium:30,complex:50"
```

### Custom Data Augmentation
```bash
python train_pipeline.py \
    --augment_transpose_range -6,6 \
    --augment_time_stretch 0.8,1.2 \
    --augment_on_the_fly
```

### Experiment Tracking
```bash
# With Weights & Biases
python train_pipeline.py --wandb --wandb_project "midifly"

# With MLflow
python train_pipeline.py --mlflow --mlflow_uri "http://localhost:5000"
```

## Training Stages

1. **Warm-up (Epochs 1-10)**
   - Model learns basic note patterns
   - Loss should decrease rapidly
   - Generated samples will be mostly noise

2. **Structure Learning (Epochs 10-40)**
   - Model begins to understand rhythm and timing
   - Samples start to have musical structure
   - Chord progressions emerge

3. **Refinement (Epochs 40-80)**
   - Fine details improve
   - Longer coherent sequences
   - Style consistency improves

4. **Fine-tuning (Epochs 80+)**
   - Subtle improvements
   - Watch for overfitting
   - Early stopping may trigger

## Troubleshooting

### Out of Memory
```bash
# Reduce batch size
python train_pipeline.py --batch_size 16

# Use gradient accumulation
python train_pipeline.py --batch_size 8 --accumulate_grad_batches 4

# Enable memory efficient attention
python train_pipeline.py --efficient_attention
```

### Slow Training
```bash
# Profile to find bottlenecks
python train_pipeline.py --profile --profile_steps 100

# Use mixed precision training
python train_pipeline.py --mixed_precision

# Increase number of data workers
python train_pipeline.py --num_workers 8
```

### Poor Quality Results
- Increase model size: `--latent_dim 128 --encoder_layers 8`
- Train longer: Remove early stopping with `--no_early_stopping`
- More data augmentation: `--augment_all`
- Check data quality: `--validate_data`

## Best Practices

1. **Start Small**: Begin with a subset of data to verify pipeline
2. **Monitor Actively**: Check samples every few epochs
3. **Save Checkpoints**: Use `--checkpoint_every_n_epochs 5`
4. **Log Everything**: Enable verbose logging with `--verbose`
5. **Version Control**: Tag experiments with meaningful names

## Post-Training

Once training completes:

1. **Evaluate Model**:
   ```bash
   python scripts/evaluate_model.py --checkpoint outputs/checkpoints/best.pt
   ```

2. **Generate Samples**:
   ```bash
   python generate_pipeline.py --model outputs/checkpoints/best.pt
   ```

3. **Export for Production**:
   ```bash
   python scripts/export_model.py \
       --checkpoint outputs/checkpoints/best.pt \
       --format onnx \
       --optimize
   ```

## Configuration Templates

### Quick Training (Testing)
```bash
python train_pipeline.py --preset quick
```

### High Quality (Production)
```bash
python train_pipeline.py --preset quality
```

### Memory Constrained
```bash
python train_pipeline.py --preset low_memory
```

For more details, see [architecture.md](architecture.md) and [api_reference.md](api_reference.md).