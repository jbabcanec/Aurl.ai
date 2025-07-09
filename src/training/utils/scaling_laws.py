"""
Model Scaling Laws Analysis for Aurl.ai Music Generation.

This module implements comprehensive scaling laws analysis for optimal model sizing:
- Parameter count vs performance analysis
- Compute vs performance scaling relationships
- Data requirements for different model sizes
- Optimal model architecture search based on scaling laws
- Training efficiency vs model size analysis
- Musical domain-specific scaling considerations
- Predictive scaling for resource planning

Designed to determine optimal model architectures for given computational budgets.
"""

import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import math
from collections import defaultdict
from scipy.optimize import curve_fit
from scipy.stats import pearsonr
import warnings

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class ScalingDimension(Enum):
    """Dimensions along which to analyze scaling."""
    PARAMETERS = "parameters"  # Model parameter count
    COMPUTE = "compute"  # Training FLOPs
    DATA = "data"  # Training data size
    SEQUENCE_LENGTH = "sequence_length"  # Maximum sequence length
    BATCH_SIZE = "batch_size"  # Training batch size
    LAYERS = "layers"  # Number of model layers
    HIDDEN_SIZE = "hidden_size"  # Hidden dimension size
    ATTENTION_HEADS = "attention_heads"  # Number of attention heads


class PerformanceMetric(Enum):
    """Performance metrics for scaling analysis."""
    VALIDATION_LOSS = "validation_loss"
    PERPLEXITY = "perplexity"
    MUSICAL_QUALITY = "musical_quality"
    TRAINING_TIME = "training_time"
    INFERENCE_TIME = "inference_time"
    MEMORY_USAGE = "memory_usage"
    CONVERGENCE_SPEED = "convergence_speed"


class ScalingLawType(Enum):
    """Types of scaling laws to fit."""
    POWER_LAW = "power_law"  # y = a * x^b
    EXPONENTIAL = "exponential"  # y = a * exp(b * x)
    LOGARITHMIC = "logarithmic"  # y = a * log(x) + b
    CHINCHILLA = "chinchilla"  # Chinchilla scaling law
    KAPLAN = "kaplan"  # Kaplan scaling law


@dataclass
class ScalingExperiment:
    """Configuration for a scaling experiment."""
    
    experiment_id: str
    
    # Model configurations to test
    model_configs: List[Dict[str, Any]]
    
    # Training configuration
    training_config: Dict[str, Any]
    
    # Scaling dimensions to vary
    scaling_dimensions: List[ScalingDimension]
    
    # Performance metrics to measure
    metrics: List[PerformanceMetric]
    
    # Experiment settings
    max_epochs: int = 50
    early_stopping: bool = True
    patience: int = 10
    
    # Resource constraints
    max_compute_budget: float = 1000.0  # GPU hours
    max_memory_gb: float = 80.0
    
    # Musical domain settings
    musical_genres: List[str] = field(default_factory=lambda: ["classical", "pop", "jazz"])
    sequence_lengths: List[int] = field(default_factory=lambda: [256, 512, 1024])
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if not self.experiment_id:
            raise ValueError("experiment_id cannot be empty")
        if not self.model_configs:
            raise ValueError("model_configs cannot be empty")
        if not self.scaling_dimensions:
            raise ValueError("scaling_dimensions cannot be empty")
        if not self.metrics:
            raise ValueError("metrics cannot be empty")
        if self.max_epochs <= 0:
            raise ValueError("max_epochs must be positive")
        if self.patience <= 0:
            raise ValueError("patience must be positive")
        if self.max_compute_budget <= 0:
            raise ValueError("max_compute_budget must be positive")
        if self.max_memory_gb <= 0:
            raise ValueError("max_memory_gb must be positive")
    
    # Analysis settings
    confidence_level: float = 0.95
    bootstrap_samples: int = 1000


@dataclass
class ScalingResult:
    """Result of a single scaling experiment."""
    
    experiment_id: str
    model_config: Dict[str, Any]
    
    # Model characteristics
    parameter_count: int
    compute_flops: float
    memory_usage: float
    
    # Performance metrics
    final_loss: float
    best_loss: float
    training_time: float
    convergence_epoch: int
    
    # Musical metrics
    musical_quality: float
    genre_performance: Dict[str, float]
    
    # Efficiency metrics
    loss_per_parameter: float
    loss_per_flop: float
    loss_per_hour: float
    
    # Training dynamics
    loss_history: List[float]
    validation_history: List[float]
    
    # Status
    converged: bool
    stable: bool
    error_message: Optional[str] = None


@dataclass
class ScalingLaw:
    """Fitted scaling law relationship."""
    
    dimension: ScalingDimension
    metric: PerformanceMetric
    law_type: ScalingLawType
    
    # Fitted parameters
    coefficients: List[float]
    r_squared: float
    confidence_interval: Tuple[float, float]
    
    # Data points used for fitting
    x_values: List[float]
    y_values: List[float]
    
    # Predictions
    prediction_range: Tuple[float, float]
    
    # Model equation
    equation: str
    
    # Statistical analysis
    p_value: float
    standard_error: float
    
    def predict(self, x: float) -> float:
        """Predict performance for given scaling dimension value."""
        
        if self.law_type == ScalingLawType.POWER_LAW:
            # y = a * x^b
            a, b = self.coefficients
            return a * (x ** b)
        
        elif self.law_type == ScalingLawType.EXPONENTIAL:
            # y = a * exp(b * x)
            a, b = self.coefficients
            return a * np.exp(b * x)
        
        elif self.law_type == ScalingLawType.LOGARITHMIC:
            # y = a * log(x) + b
            a, b = self.coefficients
            return a * np.log(x) + b
        
        elif self.law_type == ScalingLawType.CHINCHILLA:
            # Chinchilla optimal scaling
            # L = E + A/N^α + B/D^β
            E, A, alpha, B, beta = self.coefficients
            N, D = x  # x is tuple of (parameters, data)
            return E + A / (N ** alpha) + B / (D ** beta)
        
        elif self.law_type == ScalingLawType.KAPLAN:
            # Kaplan scaling law
            # L = (N_c/N)^α_N
            N_c, alpha_N = self.coefficients
            return (N_c / x) ** alpha_N
        
        else:
            raise ValueError(f"Unknown scaling law type: {self.law_type}")


class ModelScalingAnalyzer:
    """Analyzer for model scaling laws and optimal sizing."""
    
    def __init__(self, config: Dict[str, Any] = None):
        self.config = config or {}
        self.experiments = []
        self.results = []
        self.scaling_laws = {}
        
        # Set random seed for reproducibility
        torch.manual_seed(42)
        np.random.seed(42)
        
        logger.info("Model scaling analyzer initialized")
    
    def run_scaling_experiments(self, experiment: ScalingExperiment) -> List[ScalingResult]:
        """Run a series of scaling experiments."""
        
        logger.info(f"Starting scaling experiment: {experiment.experiment_id}")
        logger.info(f"Testing {len(experiment.model_configs)} model configurations")
        
        results = []
        
        for i, model_config in enumerate(experiment.model_configs):
            logger.info(f"Running experiment {i+1}/{len(experiment.model_configs)}: {model_config}")
            
            try:
                result = self._run_single_experiment(experiment, model_config)
                results.append(result)
                
                logger.info(f"Experiment {i+1} completed: loss={result.final_loss:.4f}, "
                           f"params={result.parameter_count:,}")
                
            except Exception as e:
                logger.error(f"Experiment {i+1} failed: {e}")
                
                # Create failed result
                failed_result = ScalingResult(
                    experiment_id=experiment.experiment_id,
                    model_config=model_config,
                    parameter_count=0,
                    compute_flops=0.0,
                    memory_usage=0.0,
                    final_loss=float('inf'),
                    best_loss=float('inf'),
                    training_time=0.0,
                    convergence_epoch=-1,
                    musical_quality=0.0,
                    genre_performance={},
                    loss_per_parameter=float('inf'),
                    loss_per_flop=float('inf'),
                    loss_per_hour=float('inf'),
                    loss_history=[],
                    validation_history=[],
                    converged=False,
                    stable=False,
                    error_message=str(e)
                )
                results.append(failed_result)
        
        # Store results
        self.experiments.append(experiment)
        self.results.extend(results)
        
        logger.info(f"Scaling experiment completed: {len(results)} results")
        
        return results
    
    def _run_single_experiment(self, experiment: ScalingExperiment, model_config: Dict[str, Any]) -> ScalingResult:
        """Run a single scaling experiment."""
        
        # Calculate model characteristics
        parameter_count = self._estimate_parameter_count(model_config)
        compute_flops = self._estimate_compute_flops(model_config, experiment.training_config)
        memory_usage = self._estimate_memory_usage(model_config, experiment.training_config)
        
        # Simulate training (in production, this would be actual training)
        training_result = self._simulate_training(model_config, experiment)
        
        # Calculate efficiency metrics
        loss_per_parameter = training_result["final_loss"] / max(parameter_count, 1)
        loss_per_flop = training_result["final_loss"] / max(compute_flops, 1)
        loss_per_hour = training_result["final_loss"] / max(training_result["training_time"], 1)
        
        return ScalingResult(
            experiment_id=experiment.experiment_id,
            model_config=model_config,
            parameter_count=parameter_count,
            compute_flops=compute_flops,
            memory_usage=memory_usage,
            final_loss=training_result["final_loss"],
            best_loss=training_result["best_loss"],
            training_time=training_result["training_time"],
            convergence_epoch=training_result["convergence_epoch"],
            musical_quality=training_result["musical_quality"],
            genre_performance=training_result["genre_performance"],
            loss_per_parameter=loss_per_parameter,
            loss_per_flop=loss_per_flop,
            loss_per_hour=loss_per_hour,
            loss_history=training_result["loss_history"],
            validation_history=training_result["validation_history"],
            converged=training_result["converged"],
            stable=training_result["stable"]
        )
    
    def _estimate_parameter_count(self, model_config: Dict[str, Any]) -> int:
        """Estimate parameter count from model configuration."""
        
        # Extract key parameters
        d_model = model_config.get("d_model", 512)
        n_layers = model_config.get("n_layers", 6)
        n_heads = model_config.get("n_heads", 8)
        vocab_size = model_config.get("vocab_size", 10000)
        max_seq_length = model_config.get("max_seq_length", 1024)
        
        # Estimate parameters for transformer-like architecture
        # Embedding layer
        embedding_params = vocab_size * d_model
        
        # Position embeddings
        position_params = max_seq_length * d_model
        
        # Transformer layers
        # Self-attention: Q, K, V projections + output projection
        attention_params = 4 * d_model * d_model
        
        # Feed-forward network (typically 4x expansion)
        ff_params = 2 * d_model * (4 * d_model)
        
        # Layer normalization (2 per layer)
        ln_params = 2 * d_model * 2
        
        # Total per layer
        layer_params = attention_params + ff_params + ln_params
        
        # Total parameters
        total_params = embedding_params + position_params + (n_layers * layer_params)
        
        # Add output layer
        total_params += d_model * vocab_size
        
        return total_params
    
    def _estimate_compute_flops(self, model_config: Dict[str, Any], training_config: Dict[str, Any]) -> float:
        """Estimate compute FLOPs for training."""
        
        # Model parameters
        d_model = model_config.get("d_model", 512)
        n_layers = model_config.get("n_layers", 6)
        seq_length = model_config.get("max_seq_length", 1024)
        
        # Training parameters
        batch_size = training_config.get("batch_size", 32)
        epochs = training_config.get("epochs", 50)
        steps_per_epoch = training_config.get("steps_per_epoch", 1000)
        
        # FLOPs per forward pass (approximation)
        # Self-attention: O(seq_length^2 * d_model)
        attention_flops = seq_length * seq_length * d_model
        
        # Feed-forward: O(seq_length * d_model^2)
        ff_flops = seq_length * d_model * (4 * d_model)
        
        # Per layer
        layer_flops = attention_flops + ff_flops
        
        # Total forward pass
        forward_flops = n_layers * layer_flops
        
        # Backward pass is approximately 2x forward
        backward_flops = 2 * forward_flops
        
        # Total per batch
        batch_flops = (forward_flops + backward_flops) * batch_size
        
        # Total training FLOPs
        total_flops = batch_flops * steps_per_epoch * epochs
        
        return total_flops
    
    def _estimate_memory_usage(self, model_config: Dict[str, Any], training_config: Dict[str, Any]) -> float:
        """Estimate memory usage in GB."""
        
        parameter_count = self._estimate_parameter_count(model_config)
        batch_size = training_config.get("batch_size", 32)
        seq_length = model_config.get("max_seq_length", 1024)
        d_model = model_config.get("d_model", 512)
        
        # Model parameters (float32 = 4 bytes)
        model_memory = parameter_count * 4
        
        # Optimizer states (Adam: 2x model params)
        optimizer_memory = parameter_count * 8
        
        # Gradients
        gradient_memory = parameter_count * 4
        
        # Activations (approximate)
        activation_memory = batch_size * seq_length * d_model * 4 * 10  # Rough estimate
        
        # Total in bytes
        total_memory = model_memory + optimizer_memory + gradient_memory + activation_memory
        
        # Convert to GB
        return total_memory / (1024**3)
    
    def _simulate_training(self, model_config: Dict[str, Any], experiment: ScalingExperiment) -> Dict[str, Any]:
        """Simulate training for scaling analysis."""
        
        # Extract model parameters
        parameter_count = self._estimate_parameter_count(model_config)
        d_model = model_config.get("d_model", 512)
        n_layers = model_config.get("n_layers", 6)
        
        # Simulate scaling relationships
        # Larger models generally achieve lower loss but require more compute
        
        # Base loss decreases with model size (power law)
        base_loss = 4.0 * (parameter_count / 1e6) ** (-0.076)  # Chinchilla-like scaling
        
        # Add noise and musical domain factors
        musical_factor = 1.0 + 0.1 * np.random.randn()  # Musical complexity
        noise = 0.05 * np.random.randn()
        
        final_loss = base_loss * musical_factor + noise
        best_loss = final_loss * 0.95  # Best is slightly better than final
        
        # Training time scales with model size and complexity
        base_training_time = 0.1 * (parameter_count / 1e6) ** 0.8  # Hours
        training_time = base_training_time * (1 + 0.2 * np.random.randn())
        
        # Convergence epoch depends on model size
        convergence_epoch = max(10, int(30 - 5 * math.log10(parameter_count / 1e6)))
        
        # Musical quality correlates with lower loss
        musical_quality = max(0.1, 0.9 - 0.5 * final_loss)
        
        # Genre performance (simulated)
        genre_performance = {}
        for genre in experiment.musical_genres:
            genre_performance[genre] = musical_quality * (0.9 + 0.2 * np.random.randn())
        
        # Create training history
        loss_history = []
        validation_history = []
        
        for epoch in range(experiment.max_epochs):
            # Learning curve (exponential decay)
            epoch_loss = final_loss + (4.0 - final_loss) * np.exp(-epoch / 10.0)
            val_loss = epoch_loss * (1.0 + 0.1 * np.random.randn())
            
            loss_history.append(epoch_loss)
            validation_history.append(val_loss)
            
            # Early stopping
            if experiment.early_stopping and epoch > experiment.patience:
                recent_improvement = validation_history[-experiment.patience] - val_loss
                if recent_improvement < 0.01:
                    break
        
        return {
            "final_loss": final_loss,
            "best_loss": best_loss,
            "training_time": training_time,
            "convergence_epoch": convergence_epoch,
            "musical_quality": musical_quality,
            "genre_performance": genre_performance,
            "loss_history": loss_history,
            "validation_history": validation_history,
            "converged": len(loss_history) < experiment.max_epochs,
            "stable": np.std(loss_history[-10:]) < 0.05 if len(loss_history) >= 10 else False
        }
    
    def fit_scaling_laws(self, results: List[ScalingResult] = None) -> Dict[str, ScalingLaw]:
        """Fit scaling laws to experimental results."""
        
        if results is None:
            results = self.results
        
        if not results:
            logger.warning("No results available for scaling law fitting")
            return {}
        
        logger.info(f"Fitting scaling laws to {len(results)} experimental results")
        
        scaling_laws = {}
        
        # Fit different scaling relationships
        scaling_relationships = [
            (ScalingDimension.PARAMETERS, PerformanceMetric.VALIDATION_LOSS),
            (ScalingDimension.COMPUTE, PerformanceMetric.VALIDATION_LOSS),
            (ScalingDimension.PARAMETERS, PerformanceMetric.TRAINING_TIME),
            (ScalingDimension.PARAMETERS, PerformanceMetric.MUSICAL_QUALITY),
        ]
        
        for dimension, metric in scaling_relationships:
            try:
                law = self._fit_single_scaling_law(results, dimension, metric)
                if law:
                    key = f"{dimension.value}_{metric.value}"
                    scaling_laws[key] = law
                    logger.info(f"Fitted scaling law: {key} with R²={law.r_squared:.3f}")
            
            except Exception as e:
                logger.error(f"Failed to fit scaling law {dimension.value}_{metric.value}: {e}")
        
        # Store scaling laws
        self.scaling_laws.update(scaling_laws)
        
        return scaling_laws
    
    def _fit_single_scaling_law(self, results: List[ScalingResult], 
                               dimension: ScalingDimension, 
                               metric: PerformanceMetric) -> Optional[ScalingLaw]:
        """Fit a single scaling law."""
        
        # Extract data points
        x_values = []
        y_values = []
        
        for result in results:
            if result.error_message:
                continue  # Skip failed experiments
            
            # Get x value (scaling dimension)
            if dimension == ScalingDimension.PARAMETERS:
                x = result.parameter_count
            elif dimension == ScalingDimension.COMPUTE:
                x = result.compute_flops
            else:
                continue  # Dimension not supported
            
            # Get y value (performance metric)
            if metric == PerformanceMetric.VALIDATION_LOSS:
                y = result.final_loss
            elif metric == PerformanceMetric.TRAINING_TIME:
                y = result.training_time
            elif metric == PerformanceMetric.MUSICAL_QUALITY:
                y = result.musical_quality
            else:
                continue  # Metric not supported
            
            if x > 0 and np.isfinite(y):
                x_values.append(x)
                y_values.append(y)
        
        if len(x_values) < 3:
            logger.warning(f"Insufficient data points for {dimension.value}_{metric.value}")
            return None
        
        # Convert to numpy arrays
        x_array = np.array(x_values)
        y_array = np.array(y_values)
        
        # Try different law types and pick the best fit
        best_law = None
        best_r_squared = -1
        
        law_types = [ScalingLawType.POWER_LAW, ScalingLawType.LOGARITHMIC]
        
        for law_type in law_types:
            try:
                law = self._fit_specific_law(x_array, y_array, dimension, metric, law_type)
                if law and law.r_squared > best_r_squared:
                    best_law = law
                    best_r_squared = law.r_squared
            except Exception as e:
                logger.debug(f"Failed to fit {law_type.value}: {e}")
        
        return best_law
    
    def _fit_specific_law(self, x_array: np.ndarray, y_array: np.ndarray,
                         dimension: ScalingDimension, metric: PerformanceMetric,
                         law_type: ScalingLawType) -> Optional[ScalingLaw]:
        """Fit a specific type of scaling law."""
        
        if law_type == ScalingLawType.POWER_LAW:
            # Fit y = a * x^b
            def power_law(x, a, b):
                return a * np.power(x, b)
            
            # Use log-log fit for initial guess
            log_x = np.log(x_array)
            log_y = np.log(y_array)
            
            # Remove any infinite values
            valid_mask = np.isfinite(log_x) & np.isfinite(log_y)
            if np.sum(valid_mask) < 3:
                return None
            
            log_x = log_x[valid_mask]
            log_y = log_y[valid_mask]
            
            # Linear fit in log space
            coeffs = np.polyfit(log_x, log_y, 1)
            b_init = coeffs[0]
            a_init = np.exp(coeffs[1])
            
            # Fit the actual power law
            popt, pcov = curve_fit(power_law, x_array, y_array, p0=[a_init, b_init])
            
            # Calculate R²
            y_pred = power_law(x_array, *popt)
            r_squared = 1 - np.sum((y_array - y_pred)**2) / np.sum((y_array - np.mean(y_array))**2)
            
            # Calculate confidence interval (simplified)
            confidence_interval = (0.95, 0.99)  # Placeholder
            
            # P-value and standard error (simplified)
            correlation, p_value = pearsonr(x_array, y_pred)
            standard_error = np.sqrt(np.diag(pcov)).mean()
            
            equation = f"y = {popt[0]:.3f} * x^{popt[1]:.3f}"
            
            return ScalingLaw(
                dimension=dimension,
                metric=metric,
                law_type=law_type,
                coefficients=popt.tolist(),
                r_squared=r_squared,
                confidence_interval=confidence_interval,
                x_values=x_array.tolist(),
                y_values=y_array.tolist(),
                prediction_range=(x_array.min(), x_array.max() * 2),
                equation=equation,
                p_value=p_value,
                standard_error=standard_error
            )
        
        elif law_type == ScalingLawType.LOGARITHMIC:
            # Fit y = a * log(x) + b
            def log_law(x, a, b):
                return a * np.log(x) + b
            
            popt, pcov = curve_fit(log_law, x_array, y_array)
            
            # Calculate R²
            y_pred = log_law(x_array, *popt)
            r_squared = 1 - np.sum((y_array - y_pred)**2) / np.sum((y_array - np.mean(y_array))**2)
            
            # Calculate confidence interval (simplified)
            confidence_interval = (0.95, 0.99)  # Placeholder
            
            # P-value and standard error (simplified)
            correlation, p_value = pearsonr(x_array, y_pred)
            standard_error = np.sqrt(np.diag(pcov)).mean()
            
            equation = f"y = {popt[0]:.3f} * log(x) + {popt[1]:.3f}"
            
            return ScalingLaw(
                dimension=dimension,
                metric=metric,
                law_type=law_type,
                coefficients=popt.tolist(),
                r_squared=r_squared,
                confidence_interval=confidence_interval,
                x_values=x_array.tolist(),
                y_values=y_array.tolist(),
                prediction_range=(x_array.min(), x_array.max() * 2),
                equation=equation,
                p_value=p_value,
                standard_error=standard_error
            )
        
        return None
    
    def predict_optimal_model_size(self, compute_budget: float, 
                                  performance_target: float = None) -> Dict[str, Any]:
        """Predict optimal model size for given compute budget."""
        
        if not self.scaling_laws:
            raise ValueError("No scaling laws fitted. Run fit_scaling_laws() first.")
        
        # Get parameter vs compute scaling law
        param_compute_law = self.scaling_laws.get("parameters_training_time")
        param_loss_law = self.scaling_laws.get("parameters_validation_loss")
        
        if not param_compute_law or not param_loss_law:
            raise ValueError("Required scaling laws not available")
        
        # Search for optimal parameter count
        param_candidates = np.logspace(5, 8, 100)  # 100K to 100M parameters
        
        best_config = None
        best_score = float('inf')
        
        for param_count in param_candidates:
            # Predict compute time
            compute_time = param_compute_law.predict(param_count)
            
            # Check if within budget
            if compute_time > compute_budget:
                continue
            
            # Predict performance
            predicted_loss = param_loss_law.predict(param_count)
            
            # Calculate score (lower is better)
            if performance_target:
                if predicted_loss > performance_target:
                    continue
                score = compute_time  # Minimize compute for target performance
            else:
                score = predicted_loss  # Minimize loss within budget
            
            if score < best_score:
                best_score = score
                best_config = {
                    "parameter_count": int(param_count),
                    "predicted_loss": predicted_loss,
                    "predicted_compute_time": compute_time,
                    "score": score
                }
        
        if best_config:
            # Estimate model architecture
            param_count = best_config["parameter_count"]
            
            # Rough estimates for transformer architecture
            if param_count < 1e6:  # < 1M parameters
                d_model = 256
                n_layers = 4
            elif param_count < 10e6:  # < 10M parameters
                d_model = 512
                n_layers = 6
            elif param_count < 100e6:  # < 100M parameters
                d_model = 768
                n_layers = 8
            else:  # >= 100M parameters
                d_model = 1024
                n_layers = 12
            
            best_config["suggested_architecture"] = {
                "d_model": d_model,
                "n_layers": n_layers,
                "n_heads": min(16, d_model // 64),
                "vocab_size": 10000,
                "max_seq_length": 1024
            }
        
        return best_config or {"error": "No valid configuration found within budget"}
    
    def generate_scaling_report(self) -> Dict[str, Any]:
        """Generate comprehensive scaling analysis report."""
        
        if not self.results:
            return {"error": "No experimental results available"}
        
        # Basic statistics
        successful_results = [r for r in self.results if not r.error_message]
        
        if not successful_results:
            return {"error": "No successful experiments"}
        
        # Parameter statistics
        param_counts = [r.parameter_count for r in successful_results]
        losses = [r.final_loss for r in successful_results]
        training_times = [r.training_time for r in successful_results]
        
        # Efficiency statistics
        efficiency_metrics = {
            "parameter_efficiency": {
                "mean": np.mean([r.loss_per_parameter for r in successful_results]),
                "std": np.std([r.loss_per_parameter for r in successful_results]),
                "min": np.min([r.loss_per_parameter for r in successful_results]),
                "max": np.max([r.loss_per_parameter for r in successful_results])
            },
            "compute_efficiency": {
                "mean": np.mean([r.loss_per_flop for r in successful_results]),
                "std": np.std([r.loss_per_flop for r in successful_results]),
                "min": np.min([r.loss_per_flop for r in successful_results]),
                "max": np.max([r.loss_per_flop for r in successful_results])
            }
        }
        
        # Best performing models
        best_loss = min(successful_results, key=lambda x: x.final_loss)
        best_efficiency = min(successful_results, key=lambda x: x.loss_per_parameter)
        
        # Scaling law summary
        law_summary = {}
        for key, law in self.scaling_laws.items():
            law_summary[key] = {
                "equation": law.equation,
                "r_squared": law.r_squared,
                "p_value": law.p_value
            }
        
        return {
            "experiment_summary": {
                "total_experiments": len(self.results),
                "successful_experiments": len(successful_results),
                "success_rate": len(successful_results) / len(self.results)
            },
            "parameter_range": {
                "min": min(param_counts),
                "max": max(param_counts),
                "mean": np.mean(param_counts)
            },
            "performance_range": {
                "min_loss": min(losses),
                "max_loss": max(losses),
                "mean_loss": np.mean(losses)
            },
            "efficiency_metrics": efficiency_metrics,
            "best_models": {
                "best_loss": {
                    "parameter_count": best_loss.parameter_count,
                    "final_loss": best_loss.final_loss,
                    "config": best_loss.model_config
                },
                "best_efficiency": {
                    "parameter_count": best_efficiency.parameter_count,
                    "loss_per_parameter": best_efficiency.loss_per_parameter,
                    "config": best_efficiency.model_config
                }
            },
            "scaling_laws": law_summary,
            "recommendations": self._generate_scaling_recommendations()
        }
    
    def _generate_scaling_recommendations(self) -> List[str]:
        """Generate recommendations based on scaling analysis."""
        
        recommendations = []
        
        if not self.results:
            return ["Run scaling experiments to get recommendations"]
        
        successful_results = [r for r in self.results if not r.error_message]
        
        if not successful_results:
            return ["No successful experiments - check model configurations"]
        
        # Parameter efficiency analysis
        param_efficiencies = [r.loss_per_parameter for r in successful_results]
        best_efficiency_idx = np.argmin(param_efficiencies)
        best_model = successful_results[best_efficiency_idx]
        
        recommendations.append(f"Most parameter-efficient model has {best_model.parameter_count:,} parameters")
        
        # Scaling law recommendations
        if "parameters_validation_loss" in self.scaling_laws:
            law = self.scaling_laws["parameters_validation_loss"]
            if law.r_squared > 0.8:
                recommendations.append(f"Strong scaling relationship found (R²={law.r_squared:.3f})")
                
                # Check if we're in the optimal scaling regime
                if len(law.coefficients) >= 2:
                    exponent = law.coefficients[1]
                    if exponent < -0.05:
                        recommendations.append("Model size increases still provide significant improvements")
                    else:
                        recommendations.append("Diminishing returns from larger models - consider other optimizations")
        
        # Compute efficiency recommendations
        training_times = [r.training_time for r in successful_results]
        if max(training_times) > 10 * min(training_times):
            recommendations.append("Large variation in training time - consider compute-optimal sizing")
        
        return recommendations[:5]  # Return top 5 recommendations
    
    def save_results(self, filepath: Path):
        """Save scaling analysis results to file."""
        
        results_data = {
            "experiments": [
                {
                    "experiment_id": exp.experiment_id,
                    "model_configs": exp.model_configs,
                    "scaling_dimensions": [d.value for d in exp.scaling_dimensions],
                    "metrics": [m.value for m in exp.metrics]
                }
                for exp in self.experiments
            ],
            "results": [
                {
                    "experiment_id": r.experiment_id,
                    "parameter_count": r.parameter_count,
                    "final_loss": r.final_loss,
                    "training_time": r.training_time,
                    "musical_quality": r.musical_quality,
                    "converged": r.converged,
                    "model_config": r.model_config
                }
                for r in self.results
            ],
            "scaling_laws": {
                key: {
                    "dimension": law.dimension.value,
                    "metric": law.metric.value,
                    "law_type": law.law_type.value,
                    "coefficients": law.coefficients,
                    "r_squared": law.r_squared,
                    "equation": law.equation
                }
                for key, law in self.scaling_laws.items()
            }
        }
        
        with open(filepath, 'w') as f:
            json.dump(results_data, f, indent=2)
        
        logger.info(f"Scaling analysis results saved to {filepath}")


def create_scaling_experiment(experiment_id: str, 
                            parameter_ranges: Dict[str, List[Any]],
                            compute_budget: float = 1000.0) -> ScalingExperiment:
    """Create a scaling experiment with parameter ranges."""
    
    # Generate model configurations
    model_configs = []
    
    # Create all combinations of parameters
    import itertools
    
    param_names = list(parameter_ranges.keys())
    param_values = list(parameter_ranges.values())
    
    for combination in itertools.product(*param_values):
        config = dict(zip(param_names, combination))
        model_configs.append(config)
    
    return ScalingExperiment(
        experiment_id=experiment_id,
        model_configs=model_configs,
        training_config={
            "batch_size": 32,
            "epochs": 50,
            "steps_per_epoch": 1000
        },
        scaling_dimensions=[
            ScalingDimension.PARAMETERS,
            ScalingDimension.COMPUTE
        ],
        metrics=[
            PerformanceMetric.VALIDATION_LOSS,
            PerformanceMetric.TRAINING_TIME,
            PerformanceMetric.MUSICAL_QUALITY
        ],
        max_compute_budget=compute_budget
    )


def analyze_model_scaling(parameter_ranges: Dict[str, List[Any]], 
                         compute_budget: float = 1000.0) -> Dict[str, Any]:
    """
    Analyze model scaling laws for music generation.
    
    Args:
        parameter_ranges: Dictionary of parameter ranges to test
        compute_budget: Maximum compute budget in GPU hours
        
    Returns:
        Scaling analysis results and recommendations
    """
    
    # Create experiment
    experiment = create_scaling_experiment("music_generation_scaling", parameter_ranges, compute_budget)
    
    # Run analysis
    analyzer = ModelScalingAnalyzer()
    results = analyzer.run_scaling_experiments(experiment)
    
    # Fit scaling laws
    scaling_laws = analyzer.fit_scaling_laws(results)
    
    # Generate report
    report = analyzer.generate_scaling_report()
    
    # Find optimal model size
    try:
        optimal_config = analyzer.predict_optimal_model_size(compute_budget)
        report["optimal_configuration"] = optimal_config
    except Exception as e:
        report["optimal_configuration"] = {"error": str(e)}
    
    logger.info(f"Model scaling analysis completed:")
    logger.info(f"  Experiments: {len(results)}")
    logger.info(f"  Scaling laws fitted: {len(scaling_laws)}")
    logger.info(f"  Success rate: {report['experiment_summary']['success_rate']:.1%}")
    
    return report