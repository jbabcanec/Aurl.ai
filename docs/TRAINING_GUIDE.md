# 🎼 Aurl.ai Master Training Guide

## 🚀 **NEW TRAINING SYSTEM**

We now have a **single comprehensive training script** with automatic progression and state persistence!

---

## 📋 **Quick Start**

```bash
# Start complete training pipeline
python train.py

# Check training status anytime
python train.py --status

# Clear everything and start fresh
python train.py --clear
```

---

## 🎯 **3-Stage Training Schedule**

### **Stage 1: Base Training** (20 epochs)
- **Purpose**: Fast training on clean data to establish baseline
- **Data**: No augmentation - uses cached representations
- **Model**: Transformer mode (166K-500K parameters)
- **Speed**: ~10 samples/sec
- **Output**: Solid musical foundation

### **Stage 2: Augmented Training** (30 epochs) 
- **Purpose**: Rich training with 5x data variety
- **Data**: Full augmentation pipeline
  - Pitch transpose (±6 semitones)
  - Time stretch (85%-115% tempo)
  - Velocity scaling (70%-130%)
  - Instrument substitution
  - Rhythmic variation
- **Model**: Enhanced transformer (1M+ parameters)
- **Output**: Musical diversity and robustness

### **Stage 3: Advanced Training** (50 epochs)
- **Purpose**: High-quality generation with VAE-GAN
- **Data**: Enhanced augmentation (±8 semitones, wider ranges)
- **Model**: Full VAE-GAN architecture
- **Output**: Production-quality music generation

---

## 💾 **State Persistence**

**Automatic Resume**: Training can be interrupted and resumed at any time
- **State File**: `training_state.json` tracks progress
- **Checkpoints**: Best model saved to `outputs/checkpoints/best_model.pt`
- **Stage Tracking**: Knows which stage and epoch you're on

**Resume Examples**:
```bash
python train.py           # Automatically resumes where you left off
python train.py --resume  # Explicit resume command
python train.py --stage 2 # Jump to specific stage
```

---

## 📊 **What's Being Learned**

### **Master Model**: `outputs/checkpoints/best_model.pt`
- **Architecture**: `MusicTransformerVAEGAN` (supports all 3 modes)
- **Vocabulary**: 774 musical tokens
- **Sequences**: Up to 2,048 tokens (handles complex pieces)
- **Intelligence**: Rhythm, harmony, velocity, timing relationships

### **Training Progression**:
```
Stage 1: Learn basic musical patterns (clean data)
    ↓
Stage 2: Learn musical variety (5x augmented data)  
    ↓
Stage 3: Learn high-quality generation (VAE-GAN)
    ↓
Production-ready model
```

---

## ⚙️ **Advanced Usage**

### **Individual Stages**
```bash
python train.py --stage 1  # Base training only
python train.py --stage 2  # Augmented training only  
python train.py --stage 3  # Advanced training only
```

### **Status Monitoring**
```bash
python train.py --status   # Detailed progress report
```

### **Clean Start**
```bash
python train.py --clear    # Remove all training state
```

---

## 📁 **File Organization**

### **New Structure**:
- **`train.py`** - 🌟 **MASTER TRAINING SCRIPT** (single entry point)
- **`train_simple.py`** - Fast testing script (kept for quick tests)
- **`scripts/training/`** - Legacy scripts (organized but not primary)

### **Configs**:
- **`configs/training_configs/stage1_base.yaml`** - Base training
- **`configs/training_configs/stage2_augmented.yaml`** - Augmented training
- **`configs/training_configs/stage3_advanced.yaml`** - Advanced training

### **State Files**:
- **`training_state.json`** - Current progress and state
- **`outputs/checkpoints/best_model.pt`** - Best model weights
- **`logs/tensorboard/stage*/`** - TensorBoard logs per stage

---

## 🎵 **Training Schedule Answer**

**Yes!** We now have the exact training schedule you wanted:

1. **✅ Unaugmented Data First**: Stage 1 trains on clean cached data
2. **✅ Then Augmented Data**: Stage 2 adds 5x variety through augmentation  
3. **✅ State Tracking**: `training_state.json` tracks exactly where we are
4. **✅ Resume Capability**: Can stop/resume at any point
5. **✅ Single Script**: `train.py` handles everything

---

## 🚀 **Ready to Start!**

```bash
# Start the complete training pipeline
python train.py
```

The system will:
1. Start with Stage 1 (clean data, fast)
2. Automatically progress to Stage 2 (augmented data)
3. Finish with Stage 3 (VAE-GAN quality)
4. Save state at every step
5. Allow resumption if interrupted

**Perfect training schedule with state persistence!** 🎉