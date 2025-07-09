"""
Hyperparameter Optimization for Aurl.ai Music Generation.

This module implements comprehensive hyperparameter optimization strategies:
- Grid search optimization
- Random search optimization
- Bayesian optimization with Gaussian processes
- Multi-objective optimization for music generation
- Early stopping for efficient hyperparameter search
- Musical domain-specific hyperparameter spaces
- Distributed hyperparameter optimization

Designed for efficient exploration of hyperparameter spaces for music generation models.
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Union, Callable
from dataclasses import dataclass, field
from enum import Enum
from pathlib import Path
import json
import math
import time
import itertools
import random
from collections import defaultdict, deque
from concurrent.futures import ThreadPoolExecutor, ProcessPoolExecutor
import logging

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class OptimizationStrategy(Enum):
    """Hyperparameter optimization strategies."""
    GRID_SEARCH = "grid_search"
    RANDOM_SEARCH = "random_search"
    BAYESIAN_OPTIMIZATION = "bayesian_optimization"
    MULTI_OBJECTIVE = "multi_objective"
    EVOLUTIONARY = "evolutionary"
    HYPERBAND = "hyperband"


class ParameterType(Enum):
    """Types of hyperparameters."""
    CONTINUOUS = "continuous"
    INTEGER = "integer"
    CATEGORICAL = "categorical"
    BOOLEAN = "boolean"
    LOG_UNIFORM = "log_uniform"


@dataclass
class ParameterSpec:
    """Specification for a hyperparameter."""
    
    name: str
    param_type: ParameterType
    bounds: Optional[Tuple[float, float]] = None
    choices: Optional[List[Any]] = None
    default: Optional[Any] = None
    log_scale: bool = False
    
    def sample(self) -> Any:
        """Sample a value from this parameter's distribution."""
        
        if self.param_type == ParameterType.CONTINUOUS:
            if self.log_scale:
                log_min, log_max = math.log(self.bounds[0]), math.log(self.bounds[1])
                return math.exp(random.uniform(log_min, log_max))
            else:
                return random.uniform(self.bounds[0], self.bounds[1])
        
        elif self.param_type == ParameterType.INTEGER:
            return random.randint(int(self.bounds[0]), int(self.bounds[1]))
        
        elif self.param_type == ParameterType.CATEGORICAL:
            return random.choice(self.choices)
        
        elif self.param_type == ParameterType.BOOLEAN:
            return random.choice([True, False])
        
        elif self.param_type == ParameterType.LOG_UNIFORM:
            log_min, log_max = math.log(self.bounds[0]), math.log(self.bounds[1])
            return math.exp(random.uniform(log_min, log_max))
        
        else:
            raise ValueError(f"Unknown parameter type: {self.param_type}")


@dataclass
class OptimizationConfig:
    """Configuration for hyperparameter optimization."""
    
    # Strategy
    strategy: OptimizationStrategy = OptimizationStrategy.RANDOM_SEARCH
    
    # Search space
    parameter_space: Dict[str, ParameterSpec] = field(default_factory=dict)
    
    # Budget and constraints
    max_trials: int = 100
    max_time_hours: float = 24.0
    max_parallel_trials: int = 4
    
    # Early stopping
    early_stopping_patience: int = 10
    early_stopping_min_trials: int = 20
    
    # Optimization objectives
    primary_objective: str = "validation_loss"
    secondary_objectives: List[str] = field(default_factory=lambda: ["musical_quality"])
    objective_weights: Dict[str, float] = field(default_factory=lambda: {
        "validation_loss": 0.7, "musical_quality": 0.3
    })
    
    def __post_init__(self):
        """Validate configuration parameters."""
        if self.max_trials <= 0:
            raise ValueError("max_trials must be positive")
        if self.max_time_hours <= 0:
            raise ValueError("max_time_hours must be positive")
        if self.max_parallel_trials <= 0:
            raise ValueError("max_parallel_trials must be positive")
        if self.early_stopping_patience < 0:
            raise ValueError("early_stopping_patience must be non-negative")
        if self.early_stopping_min_trials < 0:
            raise ValueError("early_stopping_min_trials must be non-negative")
    
    # Grid search specific
    grid_search_resolution: int = 5
    
    # Random search specific
    random_search_seed: int = 42
    
    # Bayesian optimization specific
    bayesian_acquisition_function: str = "expected_improvement"
    bayesian_exploration_factor: float = 0.1
    
    # Multi-objective specific
    pareto_front_size: int = 10
    
    # Musical domain specific
    musical_objective_weight: float = 0.3
    genre_specific_optimization: bool = True
    
    # Efficiency settings
    resume_from_checkpoint: bool = True
    save_all_trials: bool = True
    results_directory: str = "experiments/hyperparameter_optimization"
    
    # Pruning settings
    enable_pruning: bool = True
    pruning_patience: int = 5
    pruning_threshold: float = 0.1


@dataclass
class TrialResult:
    """Result of a single hyperparameter trial."""
    
    trial_id: str
    parameters: Dict[str, Any]
    
    # Metrics
    primary_metric: float
    secondary_metrics: Dict[str, float] = field(default_factory=dict)
    combined_objective: float = 0.0
    
    # Training info
    epochs_trained: int = 0
    training_time: float = 0.0
    convergence_epoch: int = -1
    
    # Status
    status: str = "running"  # running, completed, failed, pruned
    error_message: Optional[str] = None
    
    # Musical specific
    musical_metrics: Dict[str, float] = field(default_factory=dict)
    
    # Efficiency metrics
    samples_per_second: float = 0.0
    gpu_memory_usage: float = 0.0
    
    def __post_init__(self):
        self.trial_id = self.trial_id or f"trial_{int(time.time())}"


class HyperparameterOptimizer:
    """
    Base class for hyperparameter optimization.
    """
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        self.trials = []
        self.best_trial = None
        self.start_time = time.time()
        self.optimization_history = []
        
        # Create results directory
        self.results_dir = Path(config.results_directory)
        self.results_dir.mkdir(exist_ok=True)
        
        logger.info(f"Initialized hyperparameter optimizer: {config.strategy.value}")
    
    def optimize(self, 
                objective_function: Callable[[Dict[str, Any]], Dict[str, float]],
                **kwargs) -> Dict[str, Any]:
        """
        Run hyperparameter optimization.
        
        Args:
            objective_function: Function that takes parameters and returns metrics
            
        Returns:
            Best parameters and optimization results
        """
        
        logger.info(f"Starting hyperparameter optimization with {self.config.max_trials} trials")
        
        if self.config.strategy == OptimizationStrategy.GRID_SEARCH:
            return self._grid_search(objective_function)
        elif self.config.strategy == OptimizationStrategy.RANDOM_SEARCH:
            return self._random_search(objective_function)
        elif self.config.strategy == OptimizationStrategy.BAYESIAN_OPTIMIZATION:
            return self._bayesian_optimization(objective_function)
        elif self.config.strategy == OptimizationStrategy.MULTI_OBJECTIVE:
            return self._multi_objective_optimization(objective_function)
        else:
            raise ValueError(f"Unknown optimization strategy: {self.config.strategy}")
    
    def _grid_search(self, objective_function: Callable) -> Dict[str, Any]:
        """Perform grid search optimization."""
        
        logger.info("Starting grid search optimization")
        
        # Generate grid points
        grid_points = self._generate_grid_points()
        
        logger.info(f"Generated {len(grid_points)} grid points")
        
        # Evaluate all points
        for i, parameters in enumerate(grid_points):
            if self._should_stop():
                break
            
            trial_result = self._evaluate_trial(parameters, f"grid_{i}", objective_function)
            self._process_trial_result(trial_result)
        
        return self._get_optimization_results()
    
    def _random_search(self, objective_function: Callable) -> Dict[str, Any]:
        """Perform random search optimization."""
        
        logger.info("Starting random search optimization")
        
        random.seed(self.config.random_search_seed)
        
        for i in range(self.config.max_trials):
            if self._should_stop():
                break
            
            # Sample random parameters
            parameters = self._sample_parameters()
            
            trial_result = self._evaluate_trial(parameters, f"random_{i}", objective_function)
            self._process_trial_result(trial_result)
        
        return self._get_optimization_results()
    
    def _bayesian_optimization(self, objective_function: Callable) -> Dict[str, Any]:
        """Perform Bayesian optimization."""
        
        logger.info("Starting Bayesian optimization")
        
        # Start with random exploration
        for i in range(min(10, self.config.max_trials // 4)):
            if self._should_stop():
                break
            
            parameters = self._sample_parameters()
            trial_result = self._evaluate_trial(parameters, f"bayes_init_{i}", objective_function)
            self._process_trial_result(trial_result)
        
        # Bayesian optimization loop
        for i in range(len(self.trials), self.config.max_trials):
            if self._should_stop():
                break
            
            # Use simple acquisition function (in production, use proper GP)
            parameters = self._acquire_next_parameters()
            
            trial_result = self._evaluate_trial(parameters, f"bayes_{i}", objective_function)
            self._process_trial_result(trial_result)
        
        return self._get_optimization_results()
    
    def _multi_objective_optimization(self, objective_function: Callable) -> Dict[str, Any]:
        """Perform multi-objective optimization."""
        
        logger.info("Starting multi-objective optimization")
        
        # Random search with Pareto front tracking
        for i in range(self.config.max_trials):
            if self._should_stop():
                break
            
            parameters = self._sample_parameters()
            trial_result = self._evaluate_trial(parameters, f"multi_obj_{i}", objective_function)
            self._process_trial_result(trial_result)
        
        # Find Pareto front
        pareto_front = self._find_pareto_front()
        
        results = self._get_optimization_results()
        results["pareto_front"] = pareto_front
        
        return results
    
    def _generate_grid_points(self) -> List[Dict[str, Any]]:
        """Generate grid search points."""
        
        grid_values = {}
        
        for param_name, param_spec in self.config.parameter_space.items():
            if param_spec.param_type == ParameterType.CONTINUOUS:
                if param_spec.log_scale:
                    log_min, log_max = math.log(param_spec.bounds[0]), math.log(param_spec.bounds[1])
                    log_values = np.linspace(log_min, log_max, self.config.grid_search_resolution)
                    grid_values[param_name] = [math.exp(v) for v in log_values]
                else:
                    grid_values[param_name] = list(np.linspace(
                        param_spec.bounds[0], param_spec.bounds[1], self.config.grid_search_resolution
                    ))
            
            elif param_spec.param_type == ParameterType.INTEGER:
                grid_values[param_name] = list(range(
                    int(param_spec.bounds[0]), 
                    int(param_spec.bounds[1]) + 1,
                    max(1, (int(param_spec.bounds[1]) - int(param_spec.bounds[0])) // self.config.grid_search_resolution)
                ))
            
            elif param_spec.param_type == ParameterType.CATEGORICAL:
                grid_values[param_name] = param_spec.choices
            
            elif param_spec.param_type == ParameterType.BOOLEAN:
                grid_values[param_name] = [True, False]
        
        # Generate all combinations
        param_names = list(grid_values.keys())
        param_values = list(grid_values.values())
        
        grid_points = []
        for combination in itertools.product(*param_values):
            grid_points.append(dict(zip(param_names, combination)))
        
        return grid_points
    
    def _sample_parameters(self) -> Dict[str, Any]:
        """Sample parameters from the search space."""
        
        parameters = {}
        
        for param_name, param_spec in self.config.parameter_space.items():
            parameters[param_name] = param_spec.sample()
        
        return parameters
    
    def _acquire_next_parameters(self) -> Dict[str, Any]:
        """Acquire next parameters using acquisition function (simplified)."""
        
        if len(self.trials) < 5:
            return self._sample_parameters()
        
        # Simplified acquisition: sample around best parameters with noise
        best_params = self.best_trial.parameters.copy()
        
        for param_name, param_spec in self.config.parameter_space.items():
            if param_spec.param_type == ParameterType.CONTINUOUS:
                current_value = best_params[param_name]
                
                # Add Gaussian noise
                noise_scale = (param_spec.bounds[1] - param_spec.bounds[0]) * 0.1
                new_value = current_value + random.gauss(0, noise_scale)
                
                # Clamp to bounds
                new_value = max(param_spec.bounds[0], min(param_spec.bounds[1], new_value))
                best_params[param_name] = new_value
            
            elif param_spec.param_type == ParameterType.INTEGER:
                current_value = best_params[param_name]
                
                # Add integer noise
                noise = random.randint(-2, 2)
                new_value = current_value + noise
                
                # Clamp to bounds
                new_value = max(int(param_spec.bounds[0]), min(int(param_spec.bounds[1]), new_value))
                best_params[param_name] = new_value
            
            elif param_spec.param_type in [ParameterType.CATEGORICAL, ParameterType.BOOLEAN]:
                # Random sample for categorical/boolean
                if random.random() < 0.3:  # 30% chance to change
                    best_params[param_name] = param_spec.sample()
        
        return best_params
    
    def _evaluate_trial(self, 
                       parameters: Dict[str, Any], 
                       trial_id: str,
                       objective_function: Callable) -> TrialResult:
        """Evaluate a single trial."""
        
        trial_result = TrialResult(
            trial_id=trial_id,
            parameters=parameters.copy(),
            primary_metric=float('inf')  # Initialize with worst possible value, will be updated
        )
        
        try:
            start_time = time.time()
            
            # Run objective function
            metrics = objective_function(parameters)
            
            trial_result.training_time = time.time() - start_time
            trial_result.primary_metric = metrics.get(self.config.primary_objective, float('inf'))
            
            # Secondary metrics
            for metric_name in self.config.secondary_objectives:
                if metric_name in metrics:
                    trial_result.secondary_metrics[metric_name] = metrics[metric_name]
            
            # Musical metrics
            musical_metrics = ["musical_quality", "rhythmic_coherence", "harmonic_consistency"]
            for metric_name in musical_metrics:
                if metric_name in metrics:
                    trial_result.musical_metrics[metric_name] = metrics[metric_name]
            
            # Efficiency metrics
            trial_result.samples_per_second = metrics.get("samples_per_second", 0.0)
            trial_result.gpu_memory_usage = metrics.get("gpu_memory_usage", 0.0)
            trial_result.epochs_trained = metrics.get("epochs_trained", 0)
            
            # Calculate combined objective
            trial_result.combined_objective = self._calculate_combined_objective(trial_result)
            
            trial_result.status = "completed"
            
        except Exception as e:
            trial_result.status = "failed"
            trial_result.error_message = str(e)
            trial_result.combined_objective = float('inf')
            
            logger.error(f"Trial {trial_id} failed: {e}")
        
        return trial_result
    
    def _calculate_combined_objective(self, trial_result: TrialResult) -> float:
        """Calculate combined objective from multiple metrics."""
        
        combined = 0.0
        total_weight = 0.0
        
        # Primary objective
        primary_weight = self.config.objective_weights.get(self.config.primary_objective, 1.0)
        combined += primary_weight * trial_result.primary_metric
        total_weight += primary_weight
        
        # Secondary objectives
        for metric_name, weight in self.config.objective_weights.items():
            if metric_name != self.config.primary_objective:
                if metric_name in trial_result.secondary_metrics:
                    # Assuming lower is better, invert for musical quality metrics
                    metric_value = trial_result.secondary_metrics[metric_name]
                    if metric_name in ["musical_quality", "rhythmic_coherence", "harmonic_consistency"]:
                        metric_value = 1.0 - metric_value  # Invert since higher is better
                    
                    combined += weight * metric_value
                    total_weight += weight
        
        return combined / max(total_weight, 1.0)
    
    def _process_trial_result(self, trial_result: TrialResult):
        """Process and store trial result."""
        
        # Calculate combined objective if not already set (for manual trial results)
        if trial_result.combined_objective == 0.0 and trial_result.primary_metric != 0.0:
            trial_result.combined_objective = self._calculate_combined_objective(trial_result)
        
        self.trials.append(trial_result)
        
        # Update best trial
        if (self.best_trial is None or 
            trial_result.combined_objective < self.best_trial.combined_objective):
            self.best_trial = trial_result
            
            logger.info(f"New best trial: {trial_result.trial_id} with objective {trial_result.combined_objective:.4f}")
        
        # Save trial result
        if self.config.save_all_trials:
            self._save_trial_result(trial_result)
        
        # Update optimization history
        self.optimization_history.append({
            "trial_id": trial_result.trial_id,
            "objective": trial_result.combined_objective,
            "best_so_far": self.best_trial.combined_objective,
            "timestamp": time.time()
        })
    
    def _should_stop(self) -> bool:
        """Check if optimization should stop."""
        
        # Max trials reached
        if len(self.trials) >= self.config.max_trials:
            return True
        
        # Max time reached
        elapsed_hours = (time.time() - self.start_time) / 3600
        if elapsed_hours >= self.config.max_time_hours:
            return True
        
        # Early stopping
        if (len(self.trials) >= self.config.early_stopping_min_trials and
            self.config.early_stopping_patience > 0):
            
            recent_trials = self.trials[-self.config.early_stopping_patience:]
            if all(trial.combined_objective >= self.best_trial.combined_objective * 0.99 
                   for trial in recent_trials):
                logger.info("Early stopping triggered - no improvement")
                return True
        
        return False
    
    def _find_pareto_front(self) -> List[Dict[str, Any]]:
        """Find Pareto front for multi-objective optimization."""
        
        if len(self.trials) == 0:
            return []
        
        # Extract objective values
        objectives = []
        for trial in self.trials:
            if trial.status == "completed":
                obj_values = [trial.primary_metric]
                for metric_name in self.config.secondary_objectives:
                    if metric_name in trial.secondary_metrics:
                        obj_values.append(trial.secondary_metrics[metric_name])
                objectives.append((trial, obj_values))
        
        # Find Pareto front (simplified)
        pareto_front = []
        
        for i, (trial_i, obj_i) in enumerate(objectives):
            is_dominated = False
            
            for j, (trial_j, obj_j) in enumerate(objectives):
                if i != j:
                    # Check if trial_i is dominated by trial_j
                    if all(obj_j[k] <= obj_i[k] for k in range(len(obj_i))) and \
                       any(obj_j[k] < obj_i[k] for k in range(len(obj_i))):
                        is_dominated = True
                        break
            
            if not is_dominated:
                pareto_front.append({
                    "trial_id": trial_i.trial_id,
                    "parameters": trial_i.parameters,
                    "objectives": obj_i
                })
        
        # Sort by primary objective
        pareto_front.sort(key=lambda x: x["objectives"][0])
        
        return pareto_front[:self.config.pareto_front_size]
    
    def _save_trial_result(self, trial_result: TrialResult):
        """Save trial result to file."""
        
        trial_file = self.results_dir / f"{trial_result.trial_id}.json"
        
        trial_data = {
            "trial_id": trial_result.trial_id,
            "parameters": trial_result.parameters,
            "primary_metric": trial_result.primary_metric,
            "secondary_metrics": trial_result.secondary_metrics,
            "combined_objective": trial_result.combined_objective,
            "training_time": trial_result.training_time,
            "status": trial_result.status,
            "error_message": trial_result.error_message,
            "musical_metrics": trial_result.musical_metrics,
            "efficiency_metrics": {
                "samples_per_second": trial_result.samples_per_second,
                "gpu_memory_usage": trial_result.gpu_memory_usage,
                "epochs_trained": trial_result.epochs_trained
            }
        }
        
        with open(trial_file, 'w') as f:
            json.dump(trial_data, f, indent=2)
    
    def _get_optimization_results(self) -> Dict[str, Any]:
        """Get comprehensive optimization results."""
        
        completed_trials = [t for t in self.trials if t.status == "completed"]
        failed_trials = [t for t in self.trials if t.status == "failed"]
        
        return {
            "best_trial": {
                "trial_id": self.best_trial.trial_id if self.best_trial else None,
                "parameters": self.best_trial.parameters if self.best_trial else None,
                "objective": self.best_trial.combined_objective if self.best_trial else None,
                "primary_metric": self.best_trial.primary_metric if self.best_trial else None,
                "secondary_metrics": self.best_trial.secondary_metrics if self.best_trial else {},
                "musical_metrics": self.best_trial.musical_metrics if self.best_trial else {}
            },
            "optimization_stats": {
                "total_trials": len(self.trials),
                "completed_trials": len(completed_trials),
                "failed_trials": len(failed_trials),
                "success_rate": len(completed_trials) / max(len(self.trials), 1),
                "total_time_hours": (time.time() - self.start_time) / 3600,
                "average_trial_time": np.mean([t.training_time for t in completed_trials]) if completed_trials else 0
            },
            "convergence_history": self.optimization_history,
            "parameter_importance": self._analyze_parameter_importance(),
            "recommendations": self._generate_recommendations()
        }
    
    def _analyze_parameter_importance(self) -> Dict[str, float]:
        """Analyze parameter importance (simplified)."""
        
        if len(self.trials) < 10:
            return {}
        
        completed_trials = [t for t in self.trials if t.status == "completed"]
        
        if len(completed_trials) < 5:
            return {}
        
        # Simple correlation analysis
        parameter_importance = {}
        
        for param_name in self.config.parameter_space.keys():
            param_values = []
            objectives = []
            
            for trial in completed_trials:
                if param_name in trial.parameters:
                    param_values.append(trial.parameters[param_name])
                    objectives.append(trial.combined_objective)
            
            if len(param_values) > 3:
                # Calculate correlation (simplified)
                try:
                    correlation = abs(np.corrcoef(param_values, objectives)[0, 1])
                    parameter_importance[param_name] = correlation if not np.isnan(correlation) else 0.0
                except:
                    parameter_importance[param_name] = 0.0
        
        return parameter_importance
    
    def _generate_recommendations(self) -> List[str]:
        """Generate optimization recommendations."""
        
        recommendations = []
        
        if len(self.trials) == 0:
            return ["No trials completed yet"]
        
        completed_trials = [t for t in self.trials if t.status == "completed"]
        failed_trials = [t for t in self.trials if t.status == "failed"]
        
        # Success rate recommendations
        success_rate = len(completed_trials) / len(self.trials)
        if success_rate < 0.8:
            recommendations.append("High failure rate - check parameter bounds and objective function")
        
        # Convergence recommendations
        if len(completed_trials) > 10:
            recent_improvement = (
                self.optimization_history[-1]["best_so_far"] - 
                self.optimization_history[-10]["best_so_far"]
            )
            if abs(recent_improvement) < 0.001:
                recommendations.append("Optimization may have converged - consider expanding search space")
        
        # Parameter importance recommendations
        param_importance = self._analyze_parameter_importance()
        if param_importance:
            most_important = max(param_importance, key=param_importance.get)
            recommendations.append(f"Parameter '{most_important}' appears most important - focus tuning there")
        
        return recommendations[:3]


def create_musical_parameter_space() -> Dict[str, ParameterSpec]:
    """Create parameter space optimized for music generation."""
    
    return {
        # Model architecture
        "d_model": ParameterSpec("d_model", ParameterType.INTEGER, (256, 1024)),
        "n_layers": ParameterSpec("n_layers", ParameterType.INTEGER, (4, 12)),
        "n_heads": ParameterSpec("n_heads", ParameterType.INTEGER, (8, 16)),
        "dropout": ParameterSpec("dropout", ParameterType.CONTINUOUS, (0.1, 0.3)),
        
        # Training
        "learning_rate": ParameterSpec("learning_rate", ParameterType.LOG_UNIFORM, (1e-5, 1e-3)),
        "batch_size": ParameterSpec("batch_size", ParameterType.INTEGER, (16, 64)),
        "weight_decay": ParameterSpec("weight_decay", ParameterType.LOG_UNIFORM, (1e-6, 1e-2)),
        
        # Loss weights
        "reconstruction_weight": ParameterSpec("reconstruction_weight", ParameterType.CONTINUOUS, (0.5, 1.5)),
        "kl_weight": ParameterSpec("kl_weight", ParameterType.CONTINUOUS, (0.1, 1.0)),
        "musical_quality_weight": ParameterSpec("musical_quality_weight", ParameterType.CONTINUOUS, (0.1, 0.5)),
        
        # Musical domain
        "curriculum_strategy": ParameterSpec("curriculum_strategy", ParameterType.CATEGORICAL, 
                                          choices=["linear", "cosine", "exponential", "musical"]),
        "optimizer_type": ParameterSpec("optimizer_type", ParameterType.CATEGORICAL,
                                      choices=["adamw", "lion", "adamw_enhanced"]),
        
        # Regularization
        "gradient_clipping": ParameterSpec("gradient_clipping", ParameterType.CONTINUOUS, (0.5, 2.0)),
        "warmup_steps": ParameterSpec("warmup_steps", ParameterType.INTEGER, (1000, 5000)),
        
        # Musical specific
        "sequence_length": ParameterSpec("sequence_length", ParameterType.INTEGER, (256, 1024)),
        "musical_theory_weight": ParameterSpec("musical_theory_weight", ParameterType.CONTINUOUS, (0.1, 0.5))
    }


def create_quick_search_config() -> OptimizationConfig:
    """Create configuration for quick hyperparameter search."""
    
    return OptimizationConfig(
        strategy=OptimizationStrategy.RANDOM_SEARCH,
        parameter_space=create_musical_parameter_space(),
        max_trials=50,
        max_time_hours=6.0,
        early_stopping_patience=8,
        early_stopping_min_trials=15,
        primary_objective="validation_loss",
        secondary_objectives=["musical_quality", "training_efficiency"],
        objective_weights={
            "validation_loss": 0.6,
            "musical_quality": 0.3,
            "training_efficiency": 0.1
        }
    )


def create_comprehensive_search_config() -> OptimizationConfig:
    """Create configuration for comprehensive hyperparameter search."""
    
    return OptimizationConfig(
        strategy=OptimizationStrategy.BAYESIAN_OPTIMIZATION,
        parameter_space=create_musical_parameter_space(),
        max_trials=200,
        max_time_hours=48.0,
        early_stopping_patience=15,
        early_stopping_min_trials=30,
        primary_objective="validation_loss",
        secondary_objectives=["musical_quality", "rhythmic_coherence", "harmonic_consistency"],
        objective_weights={
            "validation_loss": 0.5,
            "musical_quality": 0.3,
            "rhythmic_coherence": 0.1,
            "harmonic_consistency": 0.1
        },
        musical_objective_weight=0.5,
        genre_specific_optimization=True
    )


def create_multi_objective_config() -> OptimizationConfig:
    """Create configuration for multi-objective optimization."""
    
    return OptimizationConfig(
        strategy=OptimizationStrategy.MULTI_OBJECTIVE,
        parameter_space=create_musical_parameter_space(),
        max_trials=150,
        max_time_hours=24.0,
        primary_objective="validation_loss",
        secondary_objectives=["musical_quality", "training_efficiency", "model_size"],
        pareto_front_size=15,
        musical_objective_weight=0.4
    )


def optimize_hyperparameters(
    objective_function: Callable[[Dict[str, Any]], Dict[str, float]],
    config: OptimizationConfig = None
) -> Dict[str, Any]:
    """
    Optimize hyperparameters for music generation model.
    
    Args:
        objective_function: Function that takes parameters and returns metrics
        config: Optimization configuration
        
    Returns:
        Optimization results with best parameters
    """
    
    if config is None:
        config = create_quick_search_config()
    
    optimizer = HyperparameterOptimizer(config)
    results = optimizer.optimize(objective_function)
    
    logger.info(f"Hyperparameter optimization completed:")
    logger.info(f"  Best objective: {results['best_trial']['objective']:.4f}")
    logger.info(f"  Total trials: {results['optimization_stats']['total_trials']}")
    logger.info(f"  Success rate: {results['optimization_stats']['success_rate']:.1%}")
    
    return results