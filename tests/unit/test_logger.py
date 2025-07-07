"""
Unit tests for logging functionality.
"""

import pytest
import logging
import tempfile
import os
from pathlib import Path
from unittest.mock import patch, MagicMock

from src.utils.logger import (
    setup_logger, get_experiment_logger, log_system_info,
    log_training_progress, log_model_info, ColoredFormatter
)


class TestSetupLogger:
    """Test logger setup functionality."""
    
    def test_basic_logger_setup(self):
        """Test basic logger creation."""
        logger = setup_logger("test_logger", level="INFO")
        assert logger.name == "test_logger"
        assert logger.level == logging.INFO
        assert len(logger.handlers) == 1  # Console handler only
    
    def test_file_logger_setup(self):
        """Test logger with file output."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = setup_logger("test_logger", log_file=log_file)
            
            assert len(logger.handlers) == 2  # File + console handlers
            logger.info("Test message")
            
            # Check file was created and contains message
            assert os.path.exists(log_file)
            with open(log_file, 'r') as f:
                content = f.read()
                assert "Test message" in content
    
    def test_logger_levels(self):
        """Test different log levels."""
        for level in ["DEBUG", "INFO", "WARNING", "ERROR", "CRITICAL"]:
            logger = setup_logger(f"test_{level}", level=level)
            expected_level = getattr(logging, level)
            assert logger.level == expected_level
    
    def test_console_output_disabled(self):
        """Test logger with console output disabled."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = setup_logger("test_logger", log_file=log_file, console_output=False)
            
            assert len(logger.handlers) == 1  # File handler only
    
    def test_handler_cleanup(self):
        """Test that existing handlers are cleared."""
        logger_name = "test_cleanup"
        
        # Create logger with handlers
        logger1 = setup_logger(logger_name)
        initial_handlers = len(logger1.handlers)
        
        # Setup same logger again
        logger2 = setup_logger(logger_name)
        
        # Should not duplicate handlers
        assert len(logger2.handlers) == initial_handlers


class TestColoredFormatter:
    """Test colored log formatter."""
    
    def test_colored_formatter_creation(self):
        """Test ColoredFormatter can be created."""
        formatter = ColoredFormatter("[%(levelname)s] %(message)s")
        assert formatter is not None
    
    def test_color_application(self):
        """Test that colors are applied to log levels."""
        formatter = ColoredFormatter("[%(levelname)s] %(message)s")
        
        # Create a log record
        record = logging.LogRecord(
            name="test", level=logging.INFO, pathname="", lineno=0,
            msg="Test message", args=(), exc_info=None
        )
        
        formatted = formatter.format(record)
        # Should contain ANSI color codes
        assert '\033[' in formatted  # ANSI escape sequence
        assert 'INFO' in formatted


class TestExperimentLogger:
    """Test experiment logger functionality."""
    
    def test_experiment_logger_creation(self):
        """Test experiment logger creation."""
        with tempfile.TemporaryDirectory() as temp_dir:
            logger = get_experiment_logger("test_experiment", base_dir=temp_dir)
            
            assert "midifly.test_experiment" in logger.name
            assert logger.level == logging.INFO
            
            # Check log file exists in correct location
            logs_dir = Path(temp_dir) / "experiments"
            log_files = list(logs_dir.glob("test_experiment_*.log"))
            assert len(log_files) == 1


class TestLoggingUtilities:
    """Test logging utility functions."""
    
    @patch('src.utils.logger.torch')
    @patch('src.utils.logger.psutil')
    @patch('src.utils.logger.platform')
    def test_log_system_info(self, mock_platform, mock_psutil, mock_torch):
        """Test system information logging."""
        # Mock dependencies
        mock_platform.platform.return_value = "Test Platform"
        mock_platform.python_version.return_value = "3.10.0"
        mock_torch.__version__ = "2.0.0"
        mock_torch.cuda.is_available.return_value = True
        mock_torch.cuda.device_count.return_value = 1
        mock_torch.cuda.current_device.return_value = 0
        mock_torch.cuda.get_device_name.return_value = "Test GPU"
        mock_torch.cuda.get_device_capability.return_value = (8, 0)
        
        # Mock memory info
        mock_memory = MagicMock()
        mock_memory.total = 16 * 1024**3  # 16GB
        mock_memory.available = 8 * 1024**3  # 8GB
        mock_psutil.virtual_memory.return_value = mock_memory
        
        # Test logging
        logger = setup_logger("test_system")
        with patch.object(logger, 'info') as mock_info:
            log_system_info(logger)
            
            # Check that system info was logged
            assert mock_info.call_count > 5  # Multiple info calls
            
            # Check specific calls
            calls = [call[0][0] for call in mock_info.call_args_list]
            assert any("Platform:" in call for call in calls)
            assert any("Python version:" in call for call in calls)
            assert any("PyTorch version:" in call for call in calls)
    
    def test_log_training_progress(self):
        """Test training progress logging."""
        logger = setup_logger("test_training")
        
        with patch.object(logger, 'info') as mock_info:
            log_training_progress(
                logger=logger,
                epoch=10,
                total_epochs=100,
                batch=50,
                total_batches=200,
                losses={"total": 0.5, "reconstruction": 0.3, "kl": 0.2},
                metrics={"accuracy": 0.85, "perplexity": 2.1},
                learning_rate=1e-4,
                files_processed=1000,
                augmented_count=5000,
                memory_usage={"gpu": 4.2, "ram": 8.1},
                sample_path="outputs/sample.mid"
            )
            
            # Check that progress was logged
            assert mock_info.call_count >= 5
            
            # Check content of calls
            calls = [call[0][0] for call in mock_info.call_args_list]
            assert any("Epoch: 10/100" in call for call in calls)
            assert any("Files processed: 1000" in call for call in calls)
            assert any("total: 0.5000" in call for call in calls)
            assert any("accuracy: 0.8500" in call for call in calls)
    
    def test_log_model_info(self):
        """Test model information logging."""
        # Mock model
        mock_model = MagicMock()
        mock_param1 = MagicMock()
        mock_param1.numel.return_value = 1000
        mock_param1.requires_grad = True
        mock_param2 = MagicMock()
        mock_param2.numel.return_value = 500
        mock_param2.requires_grad = False
        
        mock_model.parameters.return_value = [mock_param1, mock_param2]
        
        config = {
            "hidden_dim": 512,
            "num_layers": 8,
            "nested": {"learning_rate": 1e-4, "batch_size": 32}
        }
        
        logger = setup_logger("test_model")
        
        with patch.object(logger, 'info') as mock_info:
            log_model_info(logger, mock_model, config)
            
            # Check that model info was logged
            assert mock_info.call_count > 5
            
            # Check content
            calls = [call[0][0] for call in mock_info.call_args_list]
            assert any("Total parameters: 1,500" in call for call in calls)
            assert any("Trainable parameters: 1,000" in call for call in calls)
            assert any("hidden_dim: 512" in call for call in calls)


class TestLoggerIntegration:
    """Test logger integration scenarios."""
    
    def test_multiple_loggers(self):
        """Test creating multiple loggers."""
        logger1 = setup_logger("logger1")
        logger2 = setup_logger("logger2")
        
        assert logger1.name != logger2.name
        assert logger1 is not logger2
    
    def test_logger_inheritance(self):
        """Test logger hierarchy."""
        parent_logger = setup_logger("parent")
        child_logger = setup_logger("parent.child")
        
        # Child should inherit from parent
        assert child_logger.parent == parent_logger or child_logger.parent.name.startswith("parent")
    
    def test_file_rotation_config(self):
        """Test file rotation configuration."""
        with tempfile.TemporaryDirectory() as temp_dir:
            log_file = os.path.join(temp_dir, "test.log")
            logger = setup_logger(
                "test_rotation",
                log_file=log_file,
                max_bytes=1024,  # Small size for testing
                backup_count=3
            )
            
            # Find the rotating file handler
            file_handler = None
            for handler in logger.handlers:
                if hasattr(handler, 'maxBytes'):
                    file_handler = handler
                    break
            
            assert file_handler is not None
            assert file_handler.maxBytes == 1024
            assert file_handler.backupCount == 3


if __name__ == '__main__':
    pytest.main([__file__])