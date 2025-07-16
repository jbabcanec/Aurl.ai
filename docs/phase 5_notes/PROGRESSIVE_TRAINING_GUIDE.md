# 🎼 Progressive Training Guide for Aurl.ai

## Overview

The Aurl.ai training system has been completely overhauled to use a **progressive training approach** that builds complexity gradually. This solves the previous issues with trying to train Transformer + VAE + GAN simultaneously.

## 🚀 Quick Start

```bash
# Run progressive training (all 3 stages automatically)
python train_progressive.py --config configs/training_configs/quick_test.yaml

# Run individual stages
python train_progressive.py --config configs/training_configs/stage1_transformer.yaml
python train_progressive.py --config configs/training_configs/stage2_vae.yaml
python train_progressive.py --config configs/training_configs/stage3_vae_gan.yaml

# Run tests to validate fixes
python tests/test_progressive_training.py
```

## 📊 Training Stages

### Stage 1: Transformer-Only (Baseline)
- **Purpose**: Establish basic sequence modeling capability
- **Duration**: ~20 epochs
- **Config**: `configs/training_configs/stage1_transformer.yaml`
- **Key Features**:
  - Simple autoregressive modeling
  - Standard attention mechanism
  - Cross-entropy loss only
  - Light data augmentation (20%)

### Stage 2: Transformer + VAE
- **Purpose**: Add latent variable modeling
- **Duration**: ~30 epochs  
- **Config**: `configs/training_configs/stage2_vae.yaml`
- **Key Features**:
  - Hierarchical VAE with 3 levels
  - KL annealing over 10 epochs
  - Free bits to prevent posterior collapse
  - Moderate augmentation (30%)

### Stage 3: Full VAE-GAN
- **Purpose**: Add adversarial training for realism
- **Duration**: ~50 epochs
- **Config**: `configs/training_configs/stage3_vae_gan.yaml`
- **Key Features**:
  - Multi-scale discriminator
  - Progressive adversarial weight increase
  - Spectral normalization for stability
  - Full augmentation (50%)

## 🔧 Key Fixes Implemented

### 1. In-Place Operation Fixes
- Fixed gradient computation errors in `attention.py`
- Fixed PlanarFlow operations in `vae_components.py`
- Added proper tensor cloning where needed

### 2. Mixed Precision Compatibility
- Autocast now properly handles CPU/MPS devices
- Mixed precision only enabled on CUDA devices

### 3. Progressive Complexity
- Start simple (transformer) and validate
- Add complexity only after previous stage succeeds
- Each stage loads compatible weights from previous

### 4. Robust Error Handling
- Comprehensive error diagnostics
- Automatic recovery suggestions
- Gradient monitoring and anomaly detection

### 5. Data Augmentation Integration
- Augmentation now properly integrated into training
- Progressive augmentation intensity per stage
- Supports all 5 augmentation types

## 📈 Training Best Practices

### Start with Quick Test
```bash
# Test the pipeline with minimal epochs
python train_progressive.py --config configs/training_configs/quick_test.yaml
```

### Monitor Training Progress
- Check TensorBoard: `tensorboard --logdir logs/tensorboard`
- Review stage checkpoints in `outputs/stage*/checkpoints/`
- Watch for validation gates between stages

### Handle Errors
The new error handling system provides:
- Detailed diagnostics for gradient errors
- Memory usage tracking and suggestions
- Automatic checkpoint saving on errors

### Customize Stages
Each stage config can be customized:
```yaml
# Adjust batch size for your GPU
training:
  batch_size: 8  # Increase if you have more memory

# Modify learning rate
training:
  learning_rate: 3.0e-4  # Decrease if unstable

# Control augmentation
data:
  augmentation:
    probability: 0.3  # 0.0 to disable, 1.0 for maximum
```

## 🎯 Expected Results

### Stage 1 Completion
- Reconstruction loss < 3.0
- Stable gradient norms
- Basic sequence generation working

### Stage 2 Completion  
- KL loss stabilized (not collapsed)
- ELBO improving steadily
- Latent space showing structure

### Stage 3 Completion
- Discriminator loss balanced (~0.69)
- Generator producing realistic sequences
- Musical quality metrics improving

## 🐛 Troubleshooting

### "Gradient computation error"
- Check error diagnostics in logs
- Ensure all in-place operations are fixed
- Enable anomaly detection: `torch.autograd.set_detect_anomaly(True)`

### "Out of memory"
- Reduce batch size in config
- Enable gradient accumulation
- Use gradient checkpointing

### "Training unstable"
- Reduce learning rate
- Increase gradient clipping
- Check for NaN/Inf in losses

### "Stage validation failed"
- Review validation criteria in logs
- Check generated samples quality
- Ensure previous stage completed successfully

## 📚 Architecture Overview

```
Progressive Training Pipeline:
┌─────────────────┐
│ Stage 1:        │
│ Transformer     │ ──── Basic sequence modeling
└────────┬────────┘      Cross-entropy loss only
         │
         ▼
┌─────────────────┐
│ Stage 2:        │
│ Transformer+VAE │ ──── Add latent variables
└────────┬────────┘      KL + Reconstruction loss
         │
         ▼
┌─────────────────┐
│ Stage 3:        │
│ VAE-GAN         │ ──── Add adversarial training
└─────────────────┘      Full loss framework
```

## 🔍 Monitoring Training

### Real-time Metrics
- Loss curves per stage
- Gradient norms and stability
- Memory usage tracking
- Sample generation quality

### Checkpoints
Each stage saves:
- `stage{N}_final.pt` - Final model weights
- Error states if training fails
- Training configuration used
- Validation results

### Logs
- Structured logging in `logs/stage*/`
- Error diagnostics in `outputs/stage*/checkpoints/error_*/`
- TensorBoard logs for visualization

## 🎉 Success Criteria

Training is considered successful when:
1. All 3 stages complete without errors
2. Validation gates pass between stages
3. Final model generates coherent music
4. No gradient computation errors
5. Memory usage remains stable

## 🚧 Future Improvements

1. **Automated Hyperparameter Tuning**: Integration with Phase 4.5 hyperopt
2. **Distributed Training**: Multi-GPU progressive training
3. **Online Monitoring Dashboard**: Real-time web interface
4. **Automated Quality Assessment**: Musical Turing tests

## 📞 Need Help?

If you encounter issues:
1. Run the test suite: `python tests/test_progressive_training.py`
2. Check error logs in `outputs/*/checkpoints/error_*/`
3. Review TensorBoard for training curves
4. Enable debug mode: `--debug` flag

Remember: **Start simple, validate often, add complexity gradually!**