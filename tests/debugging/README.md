# Debugging Tests

This directory contains debugging and diagnostic tests used during development to identify and fix training issues.

## Files

### Gradient Issues Investigation
- `debug_inplace.py` - Debug in-place operation errors
- `debug_simple.py` - Simple debugging utilities
- `deep_inplace_debug.py` - Deep analysis of in-place operations
- `hunt_inplace_bug.py` - Hunt for specific in-place bugs
- `minimal_bug_repro.py` - Minimal reproduction of bugs
- `test_inplace_fix.py` - Test fixes for in-place operations
- `test_inplace_simple.py` - Simple in-place operation tests
- `trace_training_step.py` - Trace training steps for debugging

### Training Issues
- `final_bug_hunt.py` - Final debugging session
- `synthetic_training_test.py` - Test with synthetic data
- `test_no_grad.py` - Test gradient disabling

## Status

These files were used to debug and fix the major gradient computation errors that were blocking training. The issues have been resolved:

- ✅ In-place operation errors fixed in `src/models/attention.py`
- ✅ Mixed precision compatibility issues resolved
- ✅ VAE/GAN API mismatches fixed
- ✅ Training pipeline stability achieved

## Historical Note

These debugging files represent the investigation process that led to identifying and fixing critical training issues. They are preserved for reference but should not be needed for normal operation.