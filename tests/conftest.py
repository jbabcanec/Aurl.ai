"""
Pytest configuration and shared fixtures for MidiFly tests.
"""

import pytest
import tempfile
import os
from pathlib import Path
from typing import Generator
import numpy as np
import torch

# Configure pytest
pytest_plugins = []


@pytest.fixture
def temp_dir() -> Generator[str, None, None]:
    """Create a temporary directory for test files."""
    with tempfile.TemporaryDirectory() as tmp_dir:
        yield tmp_dir


@pytest.fixture
def sample_midi_data():
    """Generate sample MIDI-like data for testing."""
    return {
        "notes": np.array([60, 64, 67, 72]),  # C major chord
        "velocities": np.array([80, 75, 85, 90]),
        "start_times": np.array([0.0, 0.0, 0.0, 1.0]),
        "durations": np.array([1.0, 1.0, 1.0, 0.5]),
        "channels": np.array([0, 0, 0, 0]),
    }


@pytest.fixture
def sample_config():
    """Sample configuration for testing."""
    return {
        "model": {
            "hidden_dim": 256,
            "num_layers": 4,
            "num_heads": 4,
            "dropout": 0.1,
            "vocab_size": 512,
        },
        "training": {
            "batch_size": 16,
            "learning_rate": 1e-4,
            "num_epochs": 10,
            "warmup_steps": 100,
        },
        "data": {
            "sequence_length": 512,
            "augmentation": {
                "transpose": True,
                "time_stretch": True,
            },
        },
    }


@pytest.fixture
def device():
    """Get the appropriate device for testing."""
    if torch.cuda.is_available():
        return torch.device("cuda")
    elif hasattr(torch.backends, 'mps') and torch.backends.mps.is_available():
        return torch.device("mps")
    else:
        return torch.device("cpu")


@pytest.fixture
def sample_tensor(device):
    """Create a sample tensor for testing."""
    return torch.randn(4, 128, device=device)


@pytest.fixture(scope="session")
def test_data_dir():
    """Create a session-wide test data directory."""
    test_dir = Path(__file__).parent / "test_data"
    test_dir.mkdir(exist_ok=True)
    return test_dir


# Test markers
def pytest_configure(config):
    """Configure custom pytest markers."""
    config.addinivalue_line(
        "markers", "slow: marks tests as slow (deselect with '-m \"not slow\"')"
    )
    config.addinivalue_line(
        "markers", "integration: marks tests as integration tests"
    )
    config.addinivalue_line(
        "markers", "unit: marks tests as unit tests"
    )
    config.addinivalue_line(
        "markers", "gpu: marks tests that require GPU"
    )


# Skip GPU tests if no GPU available
def pytest_runtest_setup(item):
    """Setup function for each test."""
    if "gpu" in item.keywords:
        if not torch.cuda.is_available() and not (
            hasattr(torch.backends, 'mps') and torch.backends.mps.is_available()
        ):
            pytest.skip("GPU not available")


# Custom test collection
def pytest_collection_modifyitems(config, items):
    """Modify test collection to add markers automatically."""
    for item in items:
        # Add 'unit' marker to tests in unit/ directory
        if "unit" in str(item.fspath):
            item.add_marker(pytest.mark.unit)
        # Add 'integration' marker to tests in integration/ directory
        elif "integration" in str(item.fspath):
            item.add_marker(pytest.mark.integration)
        # Add 'slow' marker to tests with 'slow' in name
        if "slow" in item.name.lower():
            item.add_marker(pytest.mark.slow)