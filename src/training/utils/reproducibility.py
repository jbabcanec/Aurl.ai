"""
Training Reproducibility and Seed Management for Aurl.ai Music Generation.

This module implements comprehensive reproducibility guarantees:
- Deterministic seed management across all random sources
- State synchronization for distributed training
- Checkpoint-based reproducibility validation
- Cross-platform reproducibility handling
- Musical generation reproducibility testing
- Random state debugging and validation

Designed to ensure fully reproducible music generation training and inference.
"""

import torch
import torch.nn as nn
import torch.backends.cudnn as cudnn
import numpy as np
import random
import os
import hashlib
import json
from typing import Dict, List, Optional, Tuple, Any, Union
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import time
from collections import defaultdict
import warnings

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class ReproducibilityLevel(Enum):
    """Levels of reproducibility enforcement."""
    BASIC = "basic"              # Basic seed setting
    DETERMINISTIC = "deterministic"  # Full deterministic mode
    VALIDATED = "validated"      # With validation checks
    CROSS_PLATFORM = "cross_platform"  # Cross-platform reproducible


class RandomSource(Enum):
    """Different sources of randomness to manage."""
    PYTHON_RANDOM = "python_random"
    NUMPY = "numpy"
    TORCH = "torch"
    TORCH_CUDA = "torch_cuda"
    DATALOADER = "dataloader"
    AUGMENTATION = "augmentation"
    DROPOUT = "dropout"
    MODEL_INIT = "model_init"


@dataclass
class ReproducibilityConfig:
    """Configuration for reproducibility management."""
    
    # Reproducibility level
    level: ReproducibilityLevel = ReproducibilityLevel.DETERMINISTIC
    
    # Seeds
    master_seed: int = 42
    component_seeds: Dict[str, int] = field(default_factory=dict)
    auto_generate_seeds: bool = True
    
    # Deterministic settings
    deterministic_algorithms: bool = True
    benchmark_mode: bool = False  # Set to False for reproducibility
    warn_only: bool = False  # For deterministic algorithms
    
    # Validation settings
    validate_reproducibility: bool = True
    validation_frequency: int = 100  # epochs
    validation_tolerance: float = 1e-8
    
    # Cross-platform settings
    force_cpu_fallback: bool = False
    platform_specific_seeds: bool = True
    
    # State management
    save_random_states: bool = True
    random_state_file: str = "random_states.json"
    validate_state_loading: bool = True
    
    # Debugging
    debug_mode: bool = False
    log_random_calls: bool = False
    track_non_deterministic_ops: bool = True


@dataclass
class RandomState:
    """Container for all random states."""
    
    timestamp: float
    epoch: int
    step: int
    
    # Random states
    python_random_state: Any
    numpy_random_state: Dict[str, Any]
    torch_random_state: torch.Tensor
    torch_cuda_states: Dict[int, torch.Tensor] = field(default_factory=dict)
    
    # Component-specific states
    dataloader_state: Dict[str, Any] = field(default_factory=dict)
    augmentation_state: Dict[str, Any] = field(default_factory=dict)
    
    # Validation data
    state_hash: str = ""
    validation_tensor: Optional[torch.Tensor] = None


class SeedManager:
    """Manages seed generation and distribution across components."""
    
    def __init__(self, master_seed: int = 42):
        self.master_seed = master_seed
        self.component_seeds = {}
        self.seed_history = []
        
        # Generate deterministic seeds for each component
        self._generate_component_seeds()
    
    def _generate_component_seeds(self):
        """Generate deterministic seeds for all components."""
        
        # Use master seed to generate component seeds
        rng = np.random.RandomState(self.master_seed)
        
        components = [
            "python_random", "numpy", "torch", "torch_cuda",
            "dataloader", "augmentation", "model_init", "dropout",
            "curriculum", "distillation", "optimizer", "scheduler"
        ]
        
        for component in components:
            # Generate seed in range [0, 2^32-1]
            seed = rng.randint(0, 2**32 - 1)
            self.component_seeds[component] = seed
        
        logger.info(f"Generated {len(self.component_seeds)} component seeds from master seed {self.master_seed}")
    
    def get_seed(self, component: str) -> int:
        """Get seed for specific component."""
        
        if component not in self.component_seeds:
            # Generate new seed deterministically
            base_hash = hashlib.md5(f"{self.master_seed}_{component}".encode()).hexdigest()
            seed = int(base_hash[:8], 16) % (2**32)
            self.component_seeds[component] = seed
            logger.debug(f"Generated new seed for {component}: {seed}")
        
        return self.component_seeds[component]
    
    def get_epoch_seed(self, component: str, epoch: int) -> int:
        """Get epoch-specific seed for component."""
        
        base_seed = self.get_seed(component)
        # Combine base seed with epoch for temporal variation
        epoch_hash = hashlib.md5(f"{base_seed}_{epoch}".encode()).hexdigest()
        epoch_seed = int(epoch_hash[:8], 16) % (2**32)
        
        return epoch_seed
    
    def record_seed_usage(self, component: str, seed: int, context: str = ""):
        """Record seed usage for debugging."""
        
        usage_record = {
            "timestamp": time.time(),
            "component": component,
            "seed": seed,
            "context": context
        }
        self.seed_history.append(usage_record)
        
        if len(self.seed_history) > 10000:  # Prevent memory bloat
            self.seed_history = self.seed_history[-5000:]  # Keep recent history


class ReproducibilityManager:
    """
    Comprehensive reproducibility manager for training and inference.
    
    Features:
    - Deterministic seed management across all components
    - Random state capture and restoration
    - Reproducibility validation and testing
    - Cross-platform compatibility handling
    - Non-deterministic operation detection
    """
    
    def __init__(self, config: ReproducibilityConfig):
        self.config = config
        self.seed_manager = SeedManager(config.master_seed)
        self.random_states = []
        self.validation_results = []
        self.non_deterministic_warnings = []
        
        # Initialize reproducibility
        self._setup_reproducibility()
        
        logger.info(f"Initialized reproducibility manager with level: {config.level.value}")
        logger.info(f"Master seed: {config.master_seed}")
    
    def _setup_reproducibility(self):
        """Setup reproducibility based on configuration level."""
        
        if self.config.level in [ReproducibilityLevel.BASIC, 
                                ReproducibilityLevel.DETERMINISTIC,
                                ReproducibilityLevel.VALIDATED,
                                ReproducibilityLevel.CROSS_PLATFORM]:
            self._setup_deterministic_mode()
        
        if self.config.level in [ReproducibilityLevel.VALIDATED,
                                ReproducibilityLevel.CROSS_PLATFORM]:
            self._setup_validation_mode()
        
        if self.config.level == ReproducibilityLevel.CROSS_PLATFORM:
            self._setup_cross_platform_mode()
    
    def _setup_deterministic_mode(self):
        """Setup full deterministic mode."""
        
        # Set all basic seeds
        self.set_global_seeds()
        
        # PyTorch deterministic settings
        if self.config.deterministic_algorithms:
            torch.use_deterministic_algorithms(True, warn_only=self.config.warn_only)
            
            # Set environment variable for deterministic operations
            os.environ['CUBLAS_WORKSPACE_CONFIG'] = ':4096:8'
        
        # Disable CUDNN auto-tuner for reproducibility
        if torch.cuda.is_available():
            cudnn.deterministic = True
            cudnn.benchmark = self.config.benchmark_mode
            
            if self.config.benchmark_mode:
                logger.warning("CUDNN benchmark mode enabled - may reduce reproducibility")
        
        # Setup warning filters for non-deterministic operations
        if self.config.track_non_deterministic_ops:
            self._setup_deterministic_warnings()
        
        logger.info("Deterministic mode configured")
    
    def _setup_validation_mode(self):
        """Setup reproducibility validation."""
        
        # Enable state saving
        self.config.save_random_states = True
        
        # Setup validation tensor for testing
        self._create_validation_tensor()
        
        logger.info("Validation mode configured")
    
    def _setup_cross_platform_mode(self):
        """Setup cross-platform reproducibility."""
        
        # Force CPU for certain operations if needed
        if self.config.force_cpu_fallback:
            os.environ['CUDA_VISIBLE_DEVICES'] = ''
            logger.warning("Forced CPU mode for cross-platform reproducibility")
        
        # Additional platform-specific settings
        if self.config.platform_specific_seeds:
            platform_seed = self._get_platform_specific_seed()
            self.seed_manager.master_seed = platform_seed
            self.seed_manager._generate_component_seeds()
        
        logger.info("Cross-platform mode configured")
    
    def set_global_seeds(self, epoch: Optional[int] = None):
        """Set all global random seeds."""
        
        # Get base seeds
        if epoch is not None:
            python_seed = self.seed_manager.get_epoch_seed("python_random", epoch)
            numpy_seed = self.seed_manager.get_epoch_seed("numpy", epoch)
            torch_seed = self.seed_manager.get_epoch_seed("torch", epoch)
        else:
            python_seed = self.seed_manager.get_seed("python_random")
            numpy_seed = self.seed_manager.get_seed("numpy")
            torch_seed = self.seed_manager.get_seed("torch")
        
        # Set Python random seed
        random.seed(python_seed)
        self.seed_manager.record_seed_usage("python_random", python_seed, f"epoch_{epoch}")
        
        # Set NumPy seed
        np.random.seed(numpy_seed)
        self.seed_manager.record_seed_usage("numpy", numpy_seed, f"epoch_{epoch}")
        
        # Set PyTorch seed
        torch.manual_seed(torch_seed)
        self.seed_manager.record_seed_usage("torch", torch_seed, f"epoch_{epoch}")
        
        # Set CUDA seeds for all devices
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(torch_seed)
            for device_id in range(torch.cuda.device_count()):
                self.seed_manager.record_seed_usage("torch_cuda", torch_seed, f"device_{device_id}_epoch_{epoch}")
        
        if epoch is not None:
            logger.debug(f"Set epoch {epoch} seeds: Python={python_seed}, NumPy={numpy_seed}, Torch={torch_seed}")
        else:
            logger.info(f"Set global seeds: Python={python_seed}, NumPy={numpy_seed}, Torch={torch_seed}")
    
    def set_dataloader_seed(self, epoch: int, worker_id: Optional[int] = None) -> int:
        """Set deterministic seed for dataloader worker."""
        
        if worker_id is not None:
            # Per-worker seed
            base_seed = self.seed_manager.get_epoch_seed("dataloader", epoch)
            worker_seed = (base_seed + worker_id) % (2**32)
        else:
            worker_seed = self.seed_manager.get_epoch_seed("dataloader", epoch)
        
        # Set worker seeds
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)
        
        self.seed_manager.record_seed_usage("dataloader", worker_seed, f"epoch_{epoch}_worker_{worker_id}")
        
        return worker_seed
    
    def get_augmentation_seed(self, epoch: int, batch_idx: int) -> int:
        """Get deterministic seed for augmentation."""
        
        base_seed = self.seed_manager.get_epoch_seed("augmentation", epoch)
        aug_seed = (base_seed + batch_idx) % (2**32)
        
        self.seed_manager.record_seed_usage("augmentation", aug_seed, f"epoch_{epoch}_batch_{batch_idx}")
        
        return aug_seed
    
    def capture_random_state(self, epoch: int, step: int) -> RandomState:
        """Capture current random state."""
        
        state = RandomState(
            timestamp=time.time(),
            epoch=epoch,
            step=step,
            python_random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
            torch_random_state=torch.get_rng_state()
        )
        
        # Capture CUDA states
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                state.torch_cuda_states[device_id] = torch.cuda.get_rng_state(device_id)
        
        # Create state hash for validation
        state.state_hash = self._compute_state_hash(state)
        
        # Create validation tensor
        if self.config.validate_reproducibility:
            state.validation_tensor = torch.randn(10, device='cpu')
        
        # Store state
        self.random_states.append(state)
        
        # Limit stored states to prevent memory issues
        if len(self.random_states) > 1000:
            self.random_states = self.random_states[-500:]
        
        logger.debug(f"Captured random state at epoch {epoch}, step {step} (hash: {state.state_hash[:8]})")
        
        return state
    
    def restore_random_state(self, state: RandomState):
        """Restore random state."""
        
        # Restore Python random
        random.setstate(state.python_random_state)
        
        # Restore NumPy random
        np.random.set_state(state.numpy_random_state)
        
        # Restore PyTorch random
        torch.set_rng_state(state.torch_random_state)
        
        # Restore CUDA states
        if torch.cuda.is_available():
            for device_id, cuda_state in state.torch_cuda_states.items():
                if device_id < torch.cuda.device_count():
                    torch.cuda.set_rng_state(cuda_state, device_id)
        
        # Validate restoration if configured
        if self.config.validate_state_loading:
            current_hash = self._compute_current_state_hash()
            if current_hash != state.state_hash:
                logger.warning(f"State hash mismatch after restoration: {current_hash[:8]} != {state.state_hash[:8]}")
            else:
                logger.debug(f"Successfully restored random state (hash: {state.state_hash[:8]})")
    
    def validate_reproducibility(self, epoch: int) -> Dict[str, Any]:
        """Validate reproducibility by re-running operations."""
        
        if not self.config.validate_reproducibility:
            return {"status": "disabled"}
        
        try:
            # Capture current state
            current_state = self.capture_random_state(epoch, 0)
            
            # Generate test tensor 1
            test_tensor_1 = torch.randn(100, 50)
            test_hash_1 = hashlib.md5(test_tensor_1.numpy().tobytes()).hexdigest()
            
            # Restore state and regenerate
            self.restore_random_state(current_state)
            test_tensor_2 = torch.randn(100, 50)
            test_hash_2 = hashlib.md5(test_tensor_2.numpy().tobytes()).hexdigest()
            
            # Compare results
            reproducible = test_hash_1 == test_hash_2
            max_diff = torch.max(torch.abs(test_tensor_1 - test_tensor_2)).item()
            
            validation_result = {
                "epoch": epoch,
                "reproducible": reproducible,
                "tensor_hash_match": test_hash_1 == test_hash_2,
                "max_difference": max_diff,
                "tolerance": self.config.validation_tolerance,
                "within_tolerance": max_diff <= self.config.validation_tolerance,
                "timestamp": time.time()
            }
            
            self.validation_results.append(validation_result)
            
            if not reproducible:
                logger.warning(f"Reproducibility validation failed at epoch {epoch}: max_diff={max_diff}")
            else:
                logger.debug(f"Reproducibility validation passed at epoch {epoch}")
            
            return validation_result
            
        except Exception as e:
            logger.error(f"Reproducibility validation error at epoch {epoch}: {e}")
            return {"status": "error", "error": str(e)}
    
    def _compute_state_hash(self, state: RandomState) -> str:
        """Compute hash of random state for validation."""
        
        # Combine all state components
        state_data = {
            "python": str(state.python_random_state),
            "numpy": str(state.numpy_random_state),
            "torch": state.torch_random_state.numpy().tobytes().hex(),
            "cuda": {str(k): v.cpu().numpy().tobytes().hex() for k, v in state.torch_cuda_states.items()}
        }
        
        state_str = json.dumps(state_data, sort_keys=True)
        return hashlib.md5(state_str.encode()).hexdigest()
    
    def _compute_current_state_hash(self) -> str:
        """Compute hash of current random state."""
        
        temp_state = RandomState(
            timestamp=time.time(),
            epoch=0,
            step=0,
            python_random_state=random.getstate(),
            numpy_random_state=np.random.get_state(),
            torch_random_state=torch.get_rng_state()
        )
        
        if torch.cuda.is_available():
            for device_id in range(torch.cuda.device_count()):
                temp_state.torch_cuda_states[device_id] = torch.cuda.get_rng_state(device_id)
        
        return self._compute_state_hash(temp_state)
    
    def _create_validation_tensor(self):
        """Create validation tensor for reproducibility testing."""
        
        # Set known seed for validation tensor creation
        torch.manual_seed(self.config.master_seed)
        self.validation_tensor = torch.randn(50, 30)
        logger.debug("Created validation tensor for reproducibility testing")
    
    def _get_platform_specific_seed(self) -> int:
        """Get platform-specific seed for cross-platform reproducibility."""
        
        import platform
        import sys
        
        # Create platform signature
        platform_info = {
            "system": platform.system(),
            "python_version": sys.version_info[:2],
            "pytorch_version": torch.__version__.split('+')[0],  # Remove CUDA version suffix
        }
        
        platform_str = json.dumps(platform_info, sort_keys=True)
        platform_hash = hashlib.md5(platform_str.encode()).hexdigest()
        
        # Combine with master seed
        combined_seed = (self.config.master_seed + int(platform_hash[:8], 16)) % (2**32)
        
        logger.info(f"Platform-specific seed: {combined_seed} (base: {self.config.master_seed})")
        return combined_seed
    
    def _setup_deterministic_warnings(self):
        """Setup warnings for non-deterministic operations."""
        
        # Custom warning filter for non-deterministic operations
        def deterministic_warning_filter(message, category, filename, lineno, file=None, line=None):
            warning_text = str(message)
            
            # Track specific non-deterministic operations
            non_deterministic_indicators = [
                "non-deterministic",
                "not deterministic",
                "may not be deterministic",
                "atomicAdd",
                "scatter_add",
                "index_add"
            ]
            
            if any(indicator in warning_text.lower() for indicator in non_deterministic_indicators):
                warning_info = {
                    "timestamp": time.time(),
                    "message": warning_text,
                    "filename": filename,
                    "lineno": lineno,
                    "category": category.__name__ if category else "Unknown"
                }
                self.non_deterministic_warnings.append(warning_info)
                
                if self.config.debug_mode:
                    logger.warning(f"Non-deterministic operation detected: {warning_text}")
            
            # Show the warning
            original_showwarning(message, category, filename, lineno, file, line)
        
        # Replace warning system if not already done
        if not hasattr(warnings, '_original_showwarning'):
            warnings._original_showwarning = warnings.showwarning
            original_showwarning = warnings._original_showwarning
            warnings.showwarning = deterministic_warning_filter
    
    def save_state(self, filepath: Path):
        """Save reproducibility state to file."""
        
        if not self.config.save_random_states:
            return
        
        state_data = {
            "config": {
                "master_seed": self.config.master_seed,
                "level": self.config.level.value,
                "deterministic_algorithms": self.config.deterministic_algorithms
            },
            "seed_manager": {
                "master_seed": self.seed_manager.master_seed,
                "component_seeds": self.seed_manager.component_seeds,
                "seed_history": self.seed_manager.seed_history[-100:]  # Last 100 entries
            },
            "random_states": [
                {
                    "timestamp": state.timestamp,
                    "epoch": state.epoch,
                    "step": state.step,
                    "state_hash": state.state_hash
                }
                for state in self.random_states[-10:]  # Last 10 states
            ],
            "validation_results": self.validation_results[-20:],  # Last 20 validations
            "non_deterministic_warnings": self.non_deterministic_warnings[-50:]  # Last 50 warnings
        }
        
        with open(filepath, 'w') as f:
            json.dump(state_data, f, indent=2, default=str)
        
        logger.info(f"Saved reproducibility state to {filepath}")
    
    def load_state(self, filepath: Path):
        """Load reproducibility state from file."""
        
        with open(filepath, 'r') as f:
            state_data = json.load(f)
        
        # Restore seed manager
        if "seed_manager" in state_data:
            seed_data = state_data["seed_manager"]
            self.seed_manager.master_seed = seed_data.get("master_seed", self.config.master_seed)
            self.seed_manager.component_seeds = seed_data.get("component_seeds", {})
            self.seed_manager.seed_history = seed_data.get("seed_history", [])
        
        # Restore validation results
        self.validation_results = state_data.get("validation_results", [])
        self.non_deterministic_warnings = state_data.get("non_deterministic_warnings", [])
        
        logger.info(f"Loaded reproducibility state from {filepath}")
    
    def get_reproducibility_report(self) -> Dict[str, Any]:
        """Get comprehensive reproducibility report."""
        
        return {
            "config": {
                "level": self.config.level.value,
                "master_seed": self.config.master_seed,
                "deterministic_algorithms": self.config.deterministic_algorithms,
                "validation_enabled": self.config.validate_reproducibility
            },
            "seeds": {
                "master_seed": self.seed_manager.master_seed,
                "component_count": len(self.seed_manager.component_seeds),
                "seed_usage_count": len(self.seed_manager.seed_history)
            },
            "validation": {
                "total_validations": len(self.validation_results),
                "successful_validations": sum(1 for r in self.validation_results if r.get("reproducible", False)),
                "recent_validation": self.validation_results[-1] if self.validation_results else None
            },
            "warnings": {
                "non_deterministic_warnings": len(self.non_deterministic_warnings),
                "recent_warnings": self.non_deterministic_warnings[-5:] if self.non_deterministic_warnings else []
            },
            "state_management": {
                "captured_states": len(self.random_states),
                "save_states_enabled": self.config.save_random_states
            }
        }


def create_production_reproducibility_config() -> ReproducibilityConfig:
    """Create reproducibility configuration for production training."""
    
    return ReproducibilityConfig(
        level=ReproducibilityLevel.DETERMINISTIC,
        master_seed=42,
        auto_generate_seeds=True,
        deterministic_algorithms=True,
        benchmark_mode=False,
        warn_only=False,
        validate_reproducibility=True,
        validation_frequency=50,
        validation_tolerance=1e-10,
        save_random_states=True,
        validate_state_loading=True,
        debug_mode=False,
        track_non_deterministic_ops=True
    )


def create_research_reproducibility_config() -> ReproducibilityConfig:
    """Create reproducibility configuration for research experiments."""
    
    return ReproducibilityConfig(
        level=ReproducibilityLevel.VALIDATED,
        master_seed=42,
        auto_generate_seeds=True,
        deterministic_algorithms=True,
        benchmark_mode=False,
        warn_only=True,
        validate_reproducibility=True,
        validation_frequency=25,
        validation_tolerance=1e-8,
        save_random_states=True,
        validate_state_loading=True,
        debug_mode=True,
        log_random_calls=True,
        track_non_deterministic_ops=True
    )


def setup_reproducible_training(seed: int = 42, 
                               level: ReproducibilityLevel = ReproducibilityLevel.DETERMINISTIC) -> ReproducibilityManager:
    """
    Setup reproducible training with specified seed and level.
    
    Args:
        seed: Master seed for reproducibility
        level: Level of reproducibility enforcement
        
    Returns:
        Configured reproducibility manager
    """
    
    config = ReproducibilityConfig(
        level=level,
        master_seed=seed,
        deterministic_algorithms=True,
        validate_reproducibility=True
    )
    
    manager = ReproducibilityManager(config)
    
    logger.info(f"Setup reproducible training with seed {seed} and level {level.value}")
    
    return manager