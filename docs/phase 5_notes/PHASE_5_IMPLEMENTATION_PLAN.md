# Phase 5: Critical Model Retraining - Implementation Plan

## Overview

**Status**: Current model suffers from musical grammar collapse - generates tokens that don't form proper musical sequences.

**Goal**: Implement musical grammar training to ensure generated tokens form valid musical structures.

## Root Cause Analysis

### Problem Identified (2025-07-13)
- Model generates NOTE_ON token 43 (pitch 60) paired with NOTE_OFF token 171 (pitch 100)
- **Different pitches!** Results in no proper note on/off pairs
- MIDI export produces 0 notes because notes never get properly closed
- Model has learned token frequency but not musical grammar

### Infrastructure Status
- ✅ **Generation Pipeline**: Fully functional
- ✅ **MIDI Export**: Working perfectly when given proper tokens
- ✅ **Constraint System**: Complete and can be disabled
- ❌ **Model Training**: Needs musical grammar integration

## Implementation Plan

### Phase 5.1: Musical Grammar Training Infrastructure

#### 1. Musical Grammar Loss Functions ✅ **STARTED**
**File**: `src/training/musical_grammar.py` (created)

**Components**:
- `MusicalGrammarLoss`: Loss functions for proper note pairing
- `MusicalGrammarValidator`: Real-time validation during training
- `MusicalGrammarConfig`: Configuration for grammar validation

**Key Features**:
- Note on/off pairing validation
- Sequence coherence scoring
- Repetition penalty
- Real-time grammar checking

#### 2. Integration with Training Loop
**Files to modify**:
- `src/training/trainer.py`: Add grammar loss to training loop
- `src/training/losses.py`: Integrate musical grammar losses
- `train.py`: Update main training script

**Tasks**:
- [ ] Add MusicalGrammarLoss to training loop
- [ ] Implement real-time validation every N batches
- [ ] Add grammar score to early stopping criteria
- [ ] Create automatic model rollback on collapse detection

#### 3. Enhanced Training Data Pipeline
**Files to modify**:
- `src/data/dataset.py`: Add grammar validation to data loading
- `src/data/preprocessor.py`: Validate training sequences

**Tasks**:
- [ ] Validate all training sequences for musical grammar
- [ ] Filter out sequences with poor grammar
- [ ] Create musical quality scoring for training data
- [ ] Implement real-time data quality monitoring

### Phase 5.2: Training Data Preparation

#### 1. High-Quality Data Curation
**Target Sources**:
- Bach Two-Part Inventions (perfect 2-voice structure)
- Mozart Piano Sonatas (simple movements)
- Chopin Études (selected simple ones)
- Debussy Arabesque No. 1 (impressionist style)

**Quality Criteria**:
- Proper note on/off pairing in token sequences
- Clear 2-voice texture
- Reasonable note durations
- Musical coherence

#### 2. Data Validation Pipeline
**Create**:
- `src/data/musical_validation.py`: Data quality validation
- `scripts/validate_training_data.py`: Batch validation script

**Tasks**:
- [ ] Implement MIDI-to-token-to-MIDI round-trip validation
- [ ] Create musical quality scoring system
- [ ] Build training data analysis tools
- [ ] Generate data quality reports

### Phase 5.3: Enhanced Training Pipeline

#### 1. Grammar-Aware Training Loop
**Modifications to `src/training/trainer.py`**:
```python
# Add to training loop
grammar_loss = MusicalGrammarLoss(config.grammar_config)
total_loss, loss_components = grammar_loss(
    predicted_tokens=model_output,
    target_tokens=batch_tokens,
    base_loss=base_loss
)

# Real-time validation
if step % config.validate_every_n_batches == 0:
    validation_result = grammar_validator.validate_sequence(generated_tokens)
    if validation_result["grammar_score"] < threshold:
        logger.warning("Grammar score degraded - consider rollback")
```

#### 2. Training Configuration Updates
**File**: `configs/training_configs/musical_grammar.yaml`
```yaml
model:
  mode: "transformer"
  vocab_size: 774
  d_model: 256
  n_layers: 4

training:
  musical_grammar:
    enabled: true
    note_pairing_weight: 10.0
    sequence_coherence_weight: 5.0
    repetition_penalty_weight: 3.0
    validate_every_n_batches: 100
    grammar_score_threshold: 0.8
    early_stop_patience: 5

data:
  validate_grammar: true
  filter_poor_grammar: true
  min_grammar_score: 0.9
```

## Implementation Timeline

### Week 1: Infrastructure
- **Day 1-2**: Complete musical grammar loss implementation
- **Day 3-4**: Integrate with training loop
- **Day 5-7**: Create data validation pipeline

### Week 2: Training & Validation
- **Day 8-10**: Curate and validate training data
- **Day 11-12**: Retrain model with grammar losses
- **Day 13-14**: Validate musical output quality

## Success Criteria

### Technical Metrics
- [ ] Generated tokens form proper note on/off pairs (100% validation)
- [ ] MIDI files contain actual playable notes (>0 notes)
- [ ] No token repetition loops (max 3 consecutive identical tokens)
- [ ] Grammar validation score >0.8 consistently

### Musical Quality Metrics
- [ ] 2-voice piano texture is natural and musical
- [ ] Proper phrase structure and timing
- [ ] Reasonable note durations and spacing
- [ ] Musical coherence across sequences

### Integration Metrics
- [ ] Training pipeline stable with grammar losses
- [ ] Real-time validation working during training
- [ ] Automatic quality monitoring and alerts
- [ ] Model rollback triggers when needed

## Risk Mitigation

### Potential Issues
1. **Training Instability**: Grammar losses might interfere with convergence
   - **Mitigation**: Gradual weight ramping, careful hyperparameter tuning

2. **Data Quality**: Limited high-quality training data
   - **Mitigation**: Careful curation, quality validation pipeline

3. **Overfitting to Grammar**: Model might become too rigid
   - **Mitigation**: Balanced loss weights, diversity penalties

### Monitoring Plan
- Real-time grammar score tracking
- Generation quality sampling every 100 batches
- Automatic alerts on quality degradation
- Model checkpoint comparison and rollback capability

## Next Steps

1. **Immediate**: Complete `MusicalGrammarLoss` implementation
2. **This Week**: Integrate with training loop and test
3. **Next Week**: Curate training data and retrain model
4. **Validation**: Confirm musical output quality in Finale

This plan addresses the root cause of the musical grammar collapse and provides a path to natural 2-voice piano generation.