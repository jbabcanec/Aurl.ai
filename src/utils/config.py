"""
Configuration management system for MidiFly.

This module provides a flexible, hierarchical configuration system with
YAML support, validation, inheritance, and environment-specific overrides.
"""

import os
import yaml
from pathlib import Path
from typing import Dict, Any, Optional, Union, List
from dataclasses import dataclass, field
from omegaconf import OmegaConf, DictConfig
import logging

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


class ConfigError(Exception):
    """Exception raised for configuration-related errors."""
    pass


@dataclass
class ModelConfig:
    """Model architecture configuration."""
    mode: str = "vae_gan"  # transformer, vae, vae_gan
    hidden_dim: int = 512
    num_layers: int = 8
    num_heads: int = 8
    dropout: float = 0.1
    max_sequence_length: int = 2048
    vocab_size: int = 512
    
    # VAE specific
    latent_dim: int = 128
    encoder_layers: int = 6
    decoder_layers: int = 6
    beta: float = 1.0  # Beta-VAE parameter
    
    # GAN specific
    discriminator_layers: int = 5
    discriminator_hidden_dim: int = 256
    spectral_norm: bool = True
    
    # Attention specific
    attention_type: str = "scaled_dot_product"  # scaled_dot_product, relative, sparse
    relative_position_embedding: bool = True
    flash_attention: bool = True


@dataclass
class TrainingConfig:
    """Training configuration."""
    batch_size: int = 32
    learning_rate: float = 1e-4
    num_epochs: int = 100
    warmup_steps: int = 1000
    weight_decay: float = 1e-5
    gradient_clip_norm: float = 1.0
    accumulate_grad_batches: int = 1
    
    # Scheduling
    scheduler: str = "cosine"  # cosine, linear, exponential
    min_learning_rate: float = 1e-6
    
    # Early stopping
    early_stopping: bool = True
    patience: int = 10
    min_delta: float = 1e-4
    
    # Checkpointing
    save_every_n_epochs: int = 5
    keep_best_n_checkpoints: int = 3
    
    # Mixed precision
    mixed_precision: bool = True
    
    # Loss weights
    reconstruction_weight: float = 1.0
    kl_weight: float = 1.0
    adversarial_weight: float = 0.1


@dataclass
class DataConfig:
    """Data processing configuration."""
    sequence_length: int = 2048
    overlap: int = 256
    min_sequence_length: int = 64
    
    # Preprocessing
    normalize_velocity: bool = True
    quantize_timing: bool = True
    quantization_level: float = 1.0  # 16th note = 1.0
    
    # Augmentation
    augmentation: Dict[str, Any] = field(default_factory=lambda: {
        "transpose": True,
        "transpose_range": (-6, 6),
        "time_stretch": True,
        "time_stretch_range": (0.9, 1.1),
        "velocity_scale": True,
        "velocity_scale_range": (0.8, 1.2),
        "probability": 0.5
    })
    
    # Caching
    cache_processed_data: bool = True
    cache_size_gb: int = 5
    num_workers: int = 4


@dataclass
class ExperimentConfig:
    """Experiment tracking configuration."""
    name: str = "default_experiment"
    group: str = "midifly"
    tags: List[str] = field(default_factory=list)
    notes: str = ""
    
    # Tracking backends
    use_wandb: bool = False
    wandb_project: str = "midifly"
    wandb_entity: Optional[str] = None
    
    use_tensorboard: bool = True
    tensorboard_log_dir: str = "logs/tensorboard"
    
    use_mlflow: bool = False
    mlflow_tracking_uri: str = "http://localhost:5000"
    
    # Logging frequency
    log_every_n_steps: int = 50
    save_samples_every_n_epochs: int = 5


@dataclass
class SystemConfig:
    """System and hardware configuration."""
    device: str = "auto"  # auto, cpu, cuda, mps
    gpu_ids: List[int] = field(default_factory=lambda: [0])
    num_workers: int = 4
    pin_memory: bool = True
    
    # Paths
    data_dir: str = "data/raw"
    cache_dir: str = "data/cache"
    output_dir: str = "outputs"
    log_dir: str = "logs"
    
    # Random seed
    seed: Optional[int] = None
    deterministic: bool = False


@dataclass
class MidiFlyConfig:
    """Complete MidiFly configuration."""
    model: ModelConfig = field(default_factory=ModelConfig)
    training: TrainingConfig = field(default_factory=TrainingConfig)
    data: DataConfig = field(default_factory=DataConfig)
    experiment: ExperimentConfig = field(default_factory=ExperimentConfig)
    system: SystemConfig = field(default_factory=SystemConfig)


class ConfigManager:
    """Manages configuration loading, validation, and inheritance."""
    
    def __init__(self, config_dir: str = "configs"):
        self.config_dir = Path(config_dir)
        self.config_dir.mkdir(parents=True, exist_ok=True)
    
    def load_config(self, config_path: Union[str, Path]) -> DictConfig:
        """Load configuration from YAML file."""
        config_path = Path(config_path)
        
        if not config_path.exists():
            raise ConfigError(f"Config file not found: {config_path}")
        
        try:
            with open(config_path, 'r') as f:
                config_dict = yaml.safe_load(f)
            
            # Convert to OmegaConf for advanced features
            config = OmegaConf.create(config_dict)
            logger.info(f"Loaded config from {config_path}")
            
            return config
        
        except yaml.YAMLError as e:
            raise ConfigError(f"Error parsing YAML config: {e}")
        except Exception as e:
            raise ConfigError(f"Error loading config: {e}")
    
    def save_config(self, config: DictConfig, config_path: Union[str, Path]) -> None:
        """Save configuration to YAML file."""
        config_path = Path(config_path)
        config_path.parent.mkdir(parents=True, exist_ok=True)
        
        try:
            with open(config_path, 'w') as f:
                yaml.dump(OmegaConf.to_yaml(config), f, default_flow_style=False)
            
            logger.info(f"Saved config to {config_path}")
        
        except Exception as e:
            raise ConfigError(f"Error saving config: {e}")
    
    def merge_configs(self, base_config: DictConfig, override_config: DictConfig) -> DictConfig:
        """Merge two configurations with override taking precedence."""
        return OmegaConf.merge(base_config, override_config)
    
    def create_config_from_dataclass(self, config_obj: MidiFlyConfig) -> DictConfig:
        """Create OmegaConf from dataclass."""
        return OmegaConf.structured(config_obj)
    
    def validate_config(self, config: DictConfig) -> bool:
        """Validate configuration against schema."""
        try:
            # Convert to structured config for validation
            structured_config = OmegaConf.to_object(config)
            
            # Basic validation checks
            self._validate_model_config(structured_config.get("model", {}))
            self._validate_training_config(structured_config.get("training", {}))
            self._validate_data_config(structured_config.get("data", {}))
            self._validate_system_config(structured_config.get("system", {}))
            
            logger.info("Configuration validation passed")
            return True
        
        except Exception as e:
            logger.error(f"Configuration validation failed: {e}")
            return False
    
    def _validate_model_config(self, model_config: Dict[str, Any]) -> None:
        """Validate model configuration."""
        if "mode" in model_config:
            valid_modes = ["transformer", "vae", "vae_gan"]
            if model_config["mode"] not in valid_modes:
                raise ConfigError(f"Invalid model mode: {model_config['mode']}")
        
        if "hidden_dim" in model_config:
            if model_config["hidden_dim"] <= 0:
                raise ConfigError("hidden_dim must be positive")
        
        if "num_heads" in model_config and "hidden_dim" in model_config:
            if model_config["hidden_dim"] % model_config["num_heads"] != 0:
                raise ConfigError("hidden_dim must be divisible by num_heads")
    
    def _validate_training_config(self, training_config: Dict[str, Any]) -> None:
        """Validate training configuration."""
        if "batch_size" in training_config:
            if training_config["batch_size"] <= 0:
                raise ConfigError("batch_size must be positive")
        
        if "learning_rate" in training_config:
            if training_config["learning_rate"] <= 0:
                raise ConfigError("learning_rate must be positive")
        
        if "num_epochs" in training_config:
            if training_config["num_epochs"] <= 0:
                raise ConfigError("num_epochs must be positive")
    
    def _validate_data_config(self, data_config: Dict[str, Any]) -> None:
        """Validate data configuration."""
        if "sequence_length" in data_config:
            if data_config["sequence_length"] <= 0:
                raise ConfigError("sequence_length must be positive")
        
        if "num_workers" in data_config:
            if data_config["num_workers"] < 0:
                raise ConfigError("num_workers must be non-negative")
    
    def _validate_system_config(self, system_config: Dict[str, Any]) -> None:
        """Validate system configuration."""
        if "device" in system_config:
            valid_devices = ["auto", "cpu", "cuda", "mps"]
            if system_config["device"] not in valid_devices:
                raise ConfigError(f"Invalid device: {system_config['device']}")


def load_config(config_path: Union[str, Path]) -> DictConfig:
    """Convenience function to load configuration."""
    manager = ConfigManager()
    return manager.load_config(config_path)


def merge_configs(base_config: DictConfig, *override_configs: DictConfig) -> DictConfig:
    """Convenience function to merge multiple configurations."""
    result = base_config
    for override_config in override_configs:
        result = OmegaConf.merge(result, override_config)
    return result


def get_default_config() -> MidiFlyConfig:
    """Get default configuration."""
    return MidiFlyConfig()


def create_config_from_args(args: Dict[str, Any]) -> DictConfig:
    """Create configuration from command line arguments."""
    # Start with default config
    default_config = get_default_config()
    config = OmegaConf.structured(default_config)
    
    # Apply command line overrides
    for key, value in args.items():
        if value is not None:
            # Handle nested keys like "model.hidden_dim"
            if "." in key:
                # Use OmegaConf.update to set nested values
                nested_dict = {}
                keys = key.split(".")
                current = nested_dict
                for k in keys[:-1]:
                    current[k] = {}
                    current = current[k]
                current[keys[-1]] = value
                config = OmegaConf.merge(config, OmegaConf.create(nested_dict))
            else:
                # Try to set at top level
                if hasattr(config, key):
                    setattr(config, key, value)
    
    return config


def resolve_config_path(config_name: str, config_dir: str = "configs") -> Path:
    """Resolve configuration file path."""
    config_path = Path(config_dir) / f"{config_name}.yaml"
    
    if not config_path.exists():
        # Try with .yml extension
        config_path = Path(config_dir) / f"{config_name}.yml"
    
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_name}")
    
    return config_path


def get_environment() -> str:
    """Get current environment (dev/test/prod)."""
    return os.getenv("MIDIFLY_ENV", "dev")


def load_environment_config(environment: str = None) -> DictConfig:
    """Load environment-specific configuration."""
    if environment is None:
        environment = get_environment()
    
    config_path = resolve_config_path(f"env_{environment}")
    return load_config(config_path)