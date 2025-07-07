"""
Unit tests for configuration system.
"""

import pytest
import tempfile
import os
from pathlib import Path
from omegaconf import OmegaConf

from src.utils.config import (
    ConfigManager, MidiFlyConfig, ModelConfig, TrainingConfig,
    DataConfig, ExperimentConfig, SystemConfig, ConfigError,
    load_config, merge_configs, get_default_config, create_config_from_args,
    resolve_config_path, get_environment, load_environment_config
)


class TestConfigDataclasses:
    """Test configuration dataclasses."""
    
    def test_model_config_defaults(self):
        """Test ModelConfig default values."""
        config = ModelConfig()
        assert config.mode == "vae_gan"
        assert config.hidden_dim == 512
        assert config.num_layers == 8
        assert config.latent_dim == 128
    
    def test_training_config_defaults(self):
        """Test TrainingConfig default values."""
        config = TrainingConfig()
        assert config.batch_size == 32
        assert config.learning_rate == 1e-4
        assert config.early_stopping is True
        assert config.mixed_precision is True
    
    def test_data_config_defaults(self):
        """Test DataConfig default values."""
        config = DataConfig()
        assert config.sequence_length == 2048
        assert config.normalize_velocity is True
        assert "transpose" in config.augmentation
    
    def test_complete_config_creation(self):
        """Test creating complete MidiFlyConfig."""
        config = MidiFlyConfig()
        assert isinstance(config.model, ModelConfig)
        assert isinstance(config.training, TrainingConfig)
        assert isinstance(config.data, DataConfig)
        assert isinstance(config.experiment, ExperimentConfig)
        assert isinstance(config.system, SystemConfig)


class TestConfigManager:
    """Test ConfigManager class."""
    
    def test_config_manager_creation(self, temp_dir):
        """Test ConfigManager creation."""
        manager = ConfigManager(temp_dir)
        assert manager.config_dir == Path(temp_dir)
        assert manager.config_dir.exists()
    
    def test_save_and_load_config(self, temp_dir):
        """Test saving and loading configuration."""
        manager = ConfigManager(temp_dir)
        
        # Create test config
        config = OmegaConf.create({
            "model": {"hidden_dim": 256},
            "training": {"batch_size": 16}
        })
        
        # Save config
        config_path = Path(temp_dir) / "test_config.yaml"
        manager.save_config(config, config_path)
        
        # Load config
        loaded_config = manager.load_config(config_path)
        
        assert loaded_config.model.hidden_dim == 256
        assert loaded_config.training.batch_size == 16
    
    def test_merge_configs(self):
        """Test configuration merging."""
        manager = ConfigManager()
        
        base_config = OmegaConf.create({
            "model": {"hidden_dim": 512, "num_layers": 8},
            "training": {"batch_size": 32}
        })
        
        override_config = OmegaConf.create({
            "model": {"hidden_dim": 256},  # Override
            "training": {"learning_rate": 1e-3}  # Add new
        })
        
        merged = manager.merge_configs(base_config, override_config)
        
        assert merged.model.hidden_dim == 256  # Overridden
        assert merged.model.num_layers == 8    # Preserved
        assert merged.training.batch_size == 32  # Preserved
        assert merged.training.learning_rate == 1e-3  # Added
    
    def test_config_validation(self):
        """Test configuration validation."""
        manager = ConfigManager()
        
        # Valid config
        valid_config = OmegaConf.create({
            "model": {"mode": "transformer", "hidden_dim": 512, "num_heads": 8},
            "training": {"batch_size": 32, "learning_rate": 1e-4},
            "data": {"sequence_length": 1024},
            "system": {"device": "cpu"}
        })
        
        assert manager.validate_config(valid_config) is True
        
        # Invalid config - bad mode
        invalid_config = OmegaConf.create({
            "model": {"mode": "invalid_mode"}
        })
        
        assert manager.validate_config(invalid_config) is False
    
    def test_config_validation_errors(self):
        """Test specific validation errors."""
        manager = ConfigManager()
        
        # Test invalid model mode
        config = OmegaConf.create({"model": {"mode": "invalid"}})
        assert not manager.validate_config(config)
        
        # Test invalid hidden_dim/num_heads combination
        config = OmegaConf.create({
            "model": {"hidden_dim": 513, "num_heads": 8}
        })
        assert not manager.validate_config(config)
        
        # Test negative batch_size
        config = OmegaConf.create({"training": {"batch_size": -1}})
        assert not manager.validate_config(config)
    
    def test_load_nonexistent_config(self, temp_dir):
        """Test loading non-existent configuration."""
        manager = ConfigManager(temp_dir)
        
        with pytest.raises(ConfigError):
            manager.load_config("nonexistent.yaml")


class TestConfigUtilities:
    """Test configuration utility functions."""
    
    def test_get_default_config(self):
        """Test getting default configuration."""
        config = get_default_config()
        assert isinstance(config, MidiFlyConfig)
        assert config.model.mode == "vae_gan"
        assert config.training.batch_size == 32
    
    def test_create_config_from_args(self):
        """Test creating config from command line arguments."""
        args = {
            "model.hidden_dim": 256,
            "training.batch_size": 16,
            "experiment.name": "test_experiment"
        }
        
        config = create_config_from_args(args)
        
        assert config.model.hidden_dim == 256
        assert config.training.batch_size == 16
        assert config.experiment.name == "test_experiment"
    
    def test_resolve_config_path(self, temp_dir):
        """Test configuration path resolution."""
        # Create test config file
        config_file = Path(temp_dir) / "test_config.yaml"
        config_file.write_text("model:\n  hidden_dim: 256")
        
        # Test resolution
        resolved_path = resolve_config_path("test_config", temp_dir)
        assert resolved_path == config_file
        
        # Test non-existent config
        with pytest.raises(ConfigError):
            resolve_config_path("nonexistent", temp_dir)
    
    @pytest.mark.parametrize("env_var,expected", [
        ("dev", "dev"),
        ("test", "test"),
        ("prod", "prod"),
        (None, "dev")  # Default
    ])
    def test_get_environment(self, env_var, expected, monkeypatch):
        """Test environment detection."""
        if env_var is None:
            monkeypatch.delenv("MIDIFLY_ENV", raising=False)
        else:
            monkeypatch.setenv("MIDIFLY_ENV", env_var)
        
        assert get_environment() == expected


class TestConfigInheritance:
    """Test configuration inheritance and overrides."""
    
    def test_simple_inheritance(self):
        """Test simple configuration inheritance."""
        base = OmegaConf.create({
            "model": {"hidden_dim": 512, "num_layers": 8},
            "training": {"batch_size": 32}
        })
        
        override = OmegaConf.create({
            "model": {"hidden_dim": 256}
        })
        
        merged = merge_configs(base, override)
        
        assert merged.model.hidden_dim == 256
        assert merged.model.num_layers == 8
        assert merged.training.batch_size == 32
    
    def test_nested_inheritance(self):
        """Test nested configuration inheritance."""
        base = OmegaConf.create({
            "data": {
                "augmentation": {
                    "transpose": True,
                    "transpose_range": [-6, 6],
                    "time_stretch": False
                }
            }
        })
        
        override = OmegaConf.create({
            "data": {
                "augmentation": {
                    "transpose_range": [-12, 12],
                    "time_stretch": True
                }
            }
        }
        )
        
        merged = merge_configs(base, override)
        
        assert merged.data.augmentation.transpose is True  # Preserved
        assert merged.data.augmentation.transpose_range == [-12, 12]  # Overridden
        assert merged.data.augmentation.time_stretch is True  # Overridden
    
    def test_multiple_inheritance(self):
        """Test inheriting from multiple configurations."""
        base = OmegaConf.create({"model": {"hidden_dim": 512}})
        override1 = OmegaConf.create({"model": {"num_layers": 8}})
        override2 = OmegaConf.create({"training": {"batch_size": 32}})
        
        merged = merge_configs(base, override1, override2)
        
        assert merged.model.hidden_dim == 512
        assert merged.model.num_layers == 8
        assert merged.training.batch_size == 32


class TestEnvironmentConfigs:
    """Test environment-specific configurations."""
    
    def test_load_actual_config_files(self):
        """Test loading actual config files."""
        # This tests the actual config files we created
        try:
            config = load_config("configs/default.yaml")
            assert config.model.mode == "vae_gan"
            assert config.training.batch_size == 32
        except FileNotFoundError:
            pytest.skip("Config files not found - may not be in test environment")
    
    def test_environment_config_differences(self):
        """Test that environment configs have expected differences."""
        try:
            dev_config = load_config("configs/env_dev.yaml")
            prod_config = load_config("configs/env_prod.yaml")
            
            # Dev should have smaller model
            assert dev_config.model.hidden_dim < prod_config.model.hidden_dim
            
            # Dev should have fewer epochs
            assert dev_config.training.num_epochs < prod_config.training.num_epochs
        except FileNotFoundError:
            pytest.skip("Environment config files not found")


if __name__ == '__main__':
    pytest.main([__file__])