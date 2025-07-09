# 🎵 How to Train Aurl.ai Music Generation Model

This guide walks you through training your own music generation model with Aurl.ai's state-of-the-art VAE-GAN architecture.

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
   - **Pre-loaded**: 190+ classical MIDI files already included
   - Additional files can be added to `data/raw/` subdirectories

## Quick Start

### 🚀 **Recommended First Run (5-minute test):**
```bash
# Quick validation and test training
python train_pipeline.py --config configs/training_configs/quick_test.yaml
```

### 🏃 **Validate Setup (No Training):**
```bash
# Dry run - validates everything without training
python train_pipeline.py --config configs/training_configs/quick_test.yaml --dry-run
```

### 🎯 **Production Training:**
```bash
# Full training with default settings (100 epochs)
python train_pipeline.py

# Training with custom configuration
python train_pipeline.py --config configs/training_configs/high_quality.yaml

# Resume from checkpoint
python train_pipeline.py --resume outputs/checkpoints/latest.pt
```

## Advanced Training Options

### 🔍 **Hyperparameter Optimization:**
```bash
# Auto-optimize hyperparameters, then train
python train_pipeline.py --optimize-hyperparameters
```

### 🖥️ **Hardware Configuration:**
```bash
# Use specific GPU
python train_pipeline.py --gpu 0

# Use multiple GPUs
python train_pipeline.py --gpu 0,1,2

# Use CPU only
python train_pipeline.py --gpu cpu
```

### 🎛️ **Training Customization:**
```bash
# Custom training parameters
python train_pipeline.py --epochs 50 --batch-size 64 --learning-rate 0.0001

# Custom experiment name
python train_pipeline.py --experiment-name "my_music_model_v1"

# Enable debugging
python train_pipeline.py --debug
```

## Training Configurations

### 📁 **Available Configurations:**
- `configs/default.yaml` - Default production settings (100 epochs)
- `configs/training_configs/quick_test.yaml` - Fast 5-epoch test
- `configs/training_configs/high_quality.yaml` - High-quality training

### 🔧 **Key Configuration Options:**

```yaml
# Model Configuration
model:
  mode: "vae_gan"  # transformer | vae_only | vae_gan
  hidden_dim: 512
  num_layers: 8
  latent_dim: 128
  
# Training Configuration  
training:
  batch_size: 32  # Reduce if GPU memory limited
  learning_rate: 1.0e-4
  num_epochs: 100
  early_stopping: true
  mixed_precision: true
  
# Data Configuration
data:
  sequence_length: 2048  # Tokens per sequence
  augmentation:
    transpose: true
    time_stretch: true
    velocity_scale: true
  cache_size_gb: 5  # Disk cache size
```

## Monitoring Training Progress

### 📊 **Real-time Monitoring:**
1. **TensorBoard Dashboard** (recommended):
   ```bash
   tensorboard --logdir logs/training/
   ```
   Navigate to http://localhost:6006 for live metrics

2. **Console Output:**
   - Real-time loss values and metrics
   - Training speed (samples/second)
   - Memory usage
   - Checkpoint saves

3. **Generated Samples:**
   - Check `outputs/generated/` for sample outputs every few epochs
   - Listen to progression of quality over training

### 📝 **Log Files:**
```bash
# Real-time log monitoring
tail -f logs/training/[experiment_name]/train.log

# Check training progress
ls outputs/checkpoints/  # Model checkpoints
ls outputs/generated/    # Generated samples
```

## Expected Training Progression

### 🎵 **Training Stages:**
1. **Epochs 1-10:** Model learns basic note patterns, loss drops rapidly
2. **Epochs 10-30:** Rhythm and timing understanding emerges
3. **Epochs 30-60:** Musical structure and chord progressions develop
4. **Epochs 60+:** Fine-tuning and style consistency improvement

### 📈 **What to Expect:**
- **Quick Test (5 epochs):** Basic validation, noisy output
- **Short Training (20 epochs):** Recognizable musical patterns
- **Full Training (100 epochs):** High-quality, coherent music generation

## Experiment Tracking

### 🔬 **Experiment Results:**
- **Hyperparameter optimization:** Results saved to `experiments/hyperparameter_optimization/`
- **Training checkpoints:** Saved to `outputs/checkpoints/`
- **Generated samples:** Saved to `outputs/generated/`
- **TensorBoard logs:** Saved to `logs/training/`

### 📊 **Additional Tracking:**
```bash
# Enable Weights & Biases (if available)
python train_pipeline.py --wandb
```

## Troubleshooting

### 💾 **Out of Memory Issues:**
```bash
# Reduce batch size
python train_pipeline.py --batch-size 16

# Use CPU training
python train_pipeline.py --gpu cpu

# Enable gradient accumulation (effective larger batch size)
python train_pipeline.py --batch-size 8  # Will use accumulation automatically
```

### 🐌 **Slow Training:**
```bash
# Use mixed precision (default enabled)
python train_pipeline.py  # Mixed precision enabled by default

# Enable performance profiling
python train_pipeline.py --profile

# Check GPU utilization
python train_pipeline.py --debug  # Shows detailed performance info
```

### 📉 **Training Issues:**
- **Unstable loss:** Try lower learning rate: `--learning-rate 5e-5`
- **Poor quality:** Train longer or use high-quality config
- **Overfitting:** Early stopping is enabled by default
- **Slow convergence:** Try hyperparameter optimization: `--optimize-hyperparameters`

### 🔧 **Common Solutions:**
```bash
# Conservative training (stable but slower)
python train_pipeline.py --learning-rate 5e-5 --batch-size 16

# Aggressive training (faster but may be unstable)
python train_pipeline.py --learning-rate 2e-4 --batch-size 64
```

## Best Practices

### 🎯 **Recommended Workflow:**
1. **Start with dry run:** `python train_pipeline.py --dry-run`
2. **Test quickly:** `python train_pipeline.py --config configs/training_configs/quick_test.yaml`
3. **Monitor actively:** Use TensorBoard and check generated samples
4. **Scale up gradually:** Move to full training after validation

### 📋 **Training Checklist:**
- ✅ Run dry run first to validate setup
- ✅ Start with quick test configuration
- ✅ Monitor TensorBoard dashboard
- ✅ Check generated samples every few epochs
- ✅ Use meaningful experiment names
- ✅ Save checkpoints regularly (automatic)

## Post-Training

### 🎼 **After Training Completes:**
1. **Best model saved automatically** to `outputs/checkpoints/final_model.pt`
2. **Generated samples** available in `outputs/generated/`
3. **TensorBoard logs** for analysis in `logs/training/`
4. **Ready for Phase 7:** Generation pipeline (coming soon)

### 📊 **Evaluate Training Results:**
```bash
# Check final model
ls outputs/checkpoints/final_model.pt

# Review generated samples
ls outputs/generated/

# Analyze training logs
tensorboard --logdir logs/training/
```

## Next Steps

After successful training, you'll be ready for:
- **Phase 5:** Evaluation & Metrics (enhanced evaluation)
- **Phase 6:** Musical Intelligence Studies (advanced features)
- **Phase 7:** Generation & Deployment (production music generation)

---

**Ready to start training?** Begin with: `python train_pipeline.py --config configs/training_configs/quick_test.yaml`