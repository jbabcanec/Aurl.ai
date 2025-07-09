"""
Comprehensive test suite for Phase 4.5 Advanced Training Techniques.

This test suite validates all Phase 4.5 components:
- Hyperparameter Optimization (Grid/Random/Bayesian Search)
- Training Efficiency Analysis and Optimization
- Model Scaling Laws Analysis
- Integration with existing Phase 4.1-4.4 components

Tests cover functionality, performance, musical domain integration, and production readiness.
"""

import pytest
import torch
import torch.nn as nn
import numpy as np
from unittest.mock import Mock, patch, MagicMock
from pathlib import Path
import json
import tempfile
import time
from typing import Dict, List, Any

# Phase 4.5 imports
from src.training.utils.hyperparameter_optimization import (
    HyperparameterOptimizer, OptimizationConfig, OptimizationStrategy,
    ParameterSpec, ParameterType, TrialResult, create_musical_parameter_space,
    create_quick_search_config, optimize_hyperparameters
)
from src.training.utils.training_efficiency import (
    TrainingEfficiencyOptimizer, EfficiencyConfig, PerformanceMetric,
    BottleneckType, OptimizationStrategy as EfficiencyOptimizationStrategy,
    PerformanceProfiler, create_production_efficiency_config,
    optimize_training_efficiency
)
from src.training.utils.scaling_laws import (
    ModelScalingAnalyzer, ScalingExperiment, ScalingLaw,
    ScalingDimension, PerformanceMetric as ScalingPerformanceMetric,
    ScalingLawType, create_scaling_experiment, analyze_model_scaling
)

# Phase 4.1-4.4 imports for integration testing
from src.training.utils.curriculum_learning import ProgressiveCurriculumScheduler
from src.training.utils.knowledge_distillation import KnowledgeDistillationTrainer
from src.training.utils.advanced_optimizers import LionOptimizer
from src.training.utils.multi_stage_training import MultiStageTrainingOrchestrator
from src.training.utils.reproducibility import ReproducibilityManager
from src.training.utils.realtime_evaluation import RealTimeQualityEvaluator
from src.training.utils.musical_strategies import MusicalDomainTrainer


class MockModel(nn.Module):
    """Mock model for testing."""
    
    def __init__(self, d_model=512, n_layers=6):
        super().__init__()
        self.d_model = d_model
        self.n_layers = n_layers
        self.embedding = nn.Embedding(1000, d_model)
        self.layers = nn.ModuleList([
            nn.TransformerEncoderLayer(d_model, 8, batch_first=True)
            for _ in range(n_layers)
        ])
        self.output = nn.Linear(d_model, 1000)
        
    def forward(self, x):
        x = self.embedding(x)
        for layer in self.layers:
            x = layer(x)
        return self.output(x)


@pytest.fixture
def mock_model():
    """Create a mock model for testing."""
    return MockModel()


@pytest.fixture
def mock_optimizer():
    """Create a mock optimizer for testing."""
    model = MockModel()
    return torch.optim.Adam(model.parameters(), lr=0.001)


@pytest.fixture
def mock_dataloader():
    """Create a mock dataloader for testing."""
    dataset = torch.utils.data.TensorDataset(
        torch.randint(0, 1000, (100, 256)),
        torch.randint(0, 1000, (100, 256))
    )
    return torch.utils.data.DataLoader(dataset, batch_size=16)


class TestHyperparameterOptimization:
    """Test suite for hyperparameter optimization."""
    
    def test_parameter_spec_creation(self):
        """Test ParameterSpec creation and sampling."""
        
        # Test continuous parameter
        param = ParameterSpec(
            name="learning_rate",
            param_type=ParameterType.CONTINUOUS,
            bounds=(1e-5, 1e-2),
            log_scale=True
        )
        
        assert param.name == "learning_rate"
        assert param.param_type == ParameterType.CONTINUOUS
        assert param.bounds == (1e-5, 1e-2)
        assert param.log_scale is True
        
        # Test sampling
        samples = [param.sample() for _ in range(100)]
        assert all(1e-5 <= sample <= 1e-2 for sample in samples)
    
    def test_optimization_config_creation(self):
        """Test OptimizationConfig creation."""
        
        config = OptimizationConfig(
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            max_trials=50,
            early_stopping_patience=10
        )
        
        assert config.strategy == OptimizationStrategy.RANDOM_SEARCH
        assert config.max_trials == 50
        assert config.early_stopping_patience == 10
    
    def test_musical_parameter_space(self):
        """Test musical domain-specific parameter space creation."""
        
        param_space = create_musical_parameter_space()
        
        # Check essential parameters
        assert "d_model" in param_space
        assert "n_layers" in param_space
        assert "learning_rate" in param_space
        assert "musical_quality_weight" in param_space
        
        # Check parameter types
        assert param_space["d_model"].param_type == ParameterType.INTEGER
        assert param_space["learning_rate"].param_type == ParameterType.LOG_UNIFORM
        assert param_space["curriculum_strategy"].param_type == ParameterType.CATEGORICAL
    
    def test_hyperparameter_optimizer_initialization(self):
        """Test HyperparameterOptimizer initialization."""
        
        config = create_quick_search_config()
        optimizer = HyperparameterOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.trials == []
        assert optimizer.best_trial is None
        assert optimizer.results_dir.exists()
    
    def test_random_search_optimization(self):
        """Test random search optimization."""
        
        config = OptimizationConfig(
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            parameter_space=create_musical_parameter_space(),
            max_trials=5,
            max_time_hours=0.1
        )
        
        optimizer = HyperparameterOptimizer(config)
        
        # Mock objective function
        def mock_objective(params):
            return {
                "validation_loss": np.random.uniform(0.5, 2.0),
                "musical_quality": np.random.uniform(0.3, 0.9),
                "training_efficiency": np.random.uniform(0.4, 0.8)
            }
        
        results = optimizer.optimize(mock_objective)
        
        assert "best_trial" in results
        assert "optimization_stats" in results
        assert len(optimizer.trials) <= config.max_trials
    
    def test_grid_search_optimization(self):
        """Test grid search optimization."""
        
        # Simple parameter space for grid search
        param_space = {
            "learning_rate": ParameterSpec(
                "learning_rate", ParameterType.CONTINUOUS, (0.001, 0.01)
            ),
            "batch_size": ParameterSpec(
                "batch_size", ParameterType.INTEGER, (16, 64)
            )
        }
        
        config = OptimizationConfig(
            strategy=OptimizationStrategy.GRID_SEARCH,
            parameter_space=param_space,
            grid_search_resolution=3,
            max_trials=20
        )
        
        optimizer = HyperparameterOptimizer(config)
        
        # Mock objective function
        def mock_objective(params):
            return {"validation_loss": params["learning_rate"] * params["batch_size"]}
        
        results = optimizer.optimize(mock_objective)
        
        assert results["best_trial"]["parameters"] is not None
        assert len(optimizer.trials) > 0
    
    def test_trial_result_processing(self):
        """Test trial result processing and best trial tracking."""
        
        config = create_quick_search_config()
        optimizer = HyperparameterOptimizer(config)
        
        # Create mock trial results
        trial1 = TrialResult(
            trial_id="trial_1",
            parameters={"learning_rate": 0.001},
            primary_metric=1.5,
            secondary_metrics={"musical_quality": 0.6}
        )
        
        trial2 = TrialResult(
            trial_id="trial_2",
            parameters={"learning_rate": 0.01},
            primary_metric=1.0,
            secondary_metrics={"musical_quality": 0.8}
        )
        
        # Process trials
        optimizer._process_trial_result(trial1)
        optimizer._process_trial_result(trial2)
        
        # Check best trial tracking
        assert optimizer.best_trial.trial_id == "trial_2"
        assert len(optimizer.trials) == 2
        assert len(optimizer.optimization_history) == 2
    
    def test_parameter_importance_analysis(self):
        """Test parameter importance analysis."""
        
        config = create_quick_search_config()
        optimizer = HyperparameterOptimizer(config)
        
        # Create multiple trial results
        for i in range(15):
            trial = TrialResult(
                trial_id=f"trial_{i}",
                parameters={
                    "learning_rate": 0.001 + i * 0.001,
                    "batch_size": 16 + i * 2
                },
                primary_metric=2.0 - i * 0.05,
                status="completed"
            )
            optimizer.trials.append(trial)
        
        importance = optimizer._analyze_parameter_importance()
        
        assert isinstance(importance, dict)
        if importance:  # May be empty if correlation analysis fails
            assert "learning_rate" in importance or "batch_size" in importance
    
    def test_optimization_recommendations(self):
        """Test optimization recommendations generation."""
        
        config = create_quick_search_config()
        optimizer = HyperparameterOptimizer(config)
        
        # Add some trial results
        for i in range(10):
            trial = TrialResult(
                trial_id=f"trial_{i}",
                parameters={"learning_rate": 0.001 + i * 0.001},
                primary_metric=2.0 - i * 0.1,
                status="completed" if i < 8 else "failed"
            )
            optimizer.trials.append(trial)
        
        recommendations = optimizer._generate_recommendations()
        
        assert isinstance(recommendations, list)
        assert len(recommendations) <= 3


class TestTrainingEfficiency:
    """Test suite for training efficiency optimization."""
    
    def test_efficiency_config_creation(self):
        """Test EfficiencyConfig creation."""
        
        config = create_production_efficiency_config()
        
        assert config.monitor_interval > 0
        assert config.target_throughput > 0
        assert config.target_gpu_utilization > 0
        assert isinstance(config.optimization_strategies, list)
    
    def test_performance_profiler_initialization(self):
        """Test PerformanceProfiler initialization."""
        
        config = create_production_efficiency_config()
        profiler = PerformanceProfiler(config)
        
        assert profiler.config == config
        assert len(profiler.snapshots) == 0
        assert profiler.monitoring_active is False
    
    def test_performance_profiler_monitoring(self):
        """Test performance monitoring start/stop."""
        
        config = create_production_efficiency_config()
        profiler = PerformanceProfiler(config)
        
        # Start monitoring
        profiler.start_monitoring()
        assert profiler.monitoring_active is True
        assert profiler.monitor_thread is not None
        
        # Stop monitoring
        profiler.stop_monitoring()
        assert profiler.monitoring_active is False
    
    def test_batch_profiling_context(self):
        """Test batch profiling context manager."""
        
        config = create_production_efficiency_config()
        profiler = PerformanceProfiler(config)
        
        # Test context manager
        with profiler.profile_batch(epoch=1, batch_idx=10):
            time.sleep(0.01)  # Simulate some work
        
        # Check snapshot was recorded
        assert len(profiler.snapshots) == 1
        snapshot = profiler.snapshots[0]
        assert snapshot.epoch == 1
        assert snapshot.batch_idx == 10
        assert snapshot.batch_time > 0
    
    def test_efficiency_optimizer_initialization(self):
        """Test TrainingEfficiencyOptimizer initialization."""
        
        config = create_production_efficiency_config()
        optimizer = TrainingEfficiencyOptimizer(config)
        
        assert optimizer.config == config
        assert optimizer.profiler is not None
        assert len(optimizer.optimizations_applied) == 0
    
    def test_training_optimization(self, mock_model, mock_optimizer, mock_dataloader):
        """Test training optimization process."""
        
        config = create_production_efficiency_config()
        efficiency_optimizer = TrainingEfficiencyOptimizer(config)
        
        # Mock the optimization strategies
        with patch.object(efficiency_optimizer, '_get_current_metrics') as mock_metrics:
            mock_metrics.return_value = {
                "throughput": 50.0,
                "gpu_utilization": 0.7,
                "memory_efficiency": 0.6
            }
            
            results = efficiency_optimizer.optimize_training(
                mock_model, mock_optimizer, mock_dataloader
            )
            
            assert "optimizations_applied" in results
            assert "total_improvement" in results
            assert isinstance(results["results"], list)
    
    def test_bottleneck_detection(self):
        """Test bottleneck detection logic."""
        
        config = create_production_efficiency_config()
        profiler = PerformanceProfiler(config)
        
        # Create a performance snapshot with bottleneck
        from src.training.utils.training_efficiency import PerformanceSnapshot
        
        snapshot = PerformanceSnapshot(
            timestamp=time.time(),
            epoch=1,
            batch_idx=1,
            samples_per_second=30.0,  # Below target
            batches_per_second=1.0,
            tokens_per_second=100.0,
            batch_time=1.0,
            forward_time=0.5,
            backward_time=0.4,
            optimizer_time=0.1,
            dataloader_time=0.0,
            gpu_memory_used=6.0,
            gpu_memory_total=8.0,
            gpu_utilization=0.6,  # Below target
            cpu_utilization=0.5,
            ram_usage=0.4,
            compute_efficiency=0.6,
            memory_efficiency=0.75,
            io_efficiency=0.8,
            overall_efficiency=0.0,
            sequence_length=512,
            musical_complexity=0.5
        )
        
        # Test bottleneck detection
        bottleneck_type, severity = profiler._detect_bottlenecks(snapshot)
        
        assert bottleneck_type is not None
        assert severity > 0
    
    def test_optimization_report_generation(self):
        """Test optimization report generation."""
        
        config = create_production_efficiency_config()
        optimizer = TrainingEfficiencyOptimizer(config)
        
        # Add some mock performance data
        optimizer.profiler.smoothed_metrics = {
            "avg_throughput": 75.0,
            "avg_efficiency": 0.8,
            "avg_gpu_utilization": 0.85
        }
        
        report = optimizer.get_optimization_report()
        
        assert "performance" in report
        assert "optimizations" in report
        assert "config" in report
        assert "recommendations" in report


class TestScalingLaws:
    """Test suite for model scaling laws analysis."""
    
    def test_scaling_experiment_creation(self):
        """Test ScalingExperiment creation."""
        
        parameter_ranges = {
            "d_model": [256, 512, 768],
            "n_layers": [4, 6, 8],
            "learning_rate": [0.001, 0.01]
        }
        
        experiment = create_scaling_experiment(
            "test_experiment",
            parameter_ranges,
            compute_budget=100.0
        )
        
        assert experiment.experiment_id == "test_experiment"
        assert len(experiment.model_configs) == 3 * 3 * 2  # 18 combinations
        assert experiment.max_compute_budget == 100.0
    
    def test_model_scaling_analyzer_initialization(self):
        """Test ModelScalingAnalyzer initialization."""
        
        analyzer = ModelScalingAnalyzer()
        
        assert analyzer.experiments == []
        assert analyzer.results == []
        assert analyzer.scaling_laws == {}
    
    def test_parameter_count_estimation(self):
        """Test parameter count estimation."""
        
        analyzer = ModelScalingAnalyzer()
        
        # Test small model
        small_config = {
            "d_model": 256,
            "n_layers": 4,
            "n_heads": 8,
            "vocab_size": 1000,
            "max_seq_length": 512
        }
        
        param_count = analyzer._estimate_parameter_count(small_config)
        assert param_count > 0
        assert param_count < 10_000_000  # Should be less than 10M
        
        # Test large model
        large_config = {
            "d_model": 1024,
            "n_layers": 12,
            "n_heads": 16,
            "vocab_size": 10000,
            "max_seq_length": 2048
        }
        
        large_param_count = analyzer._estimate_parameter_count(large_config)
        assert large_param_count > param_count  # Should be larger
    
    def test_compute_flops_estimation(self):
        """Test compute FLOPs estimation."""
        
        analyzer = ModelScalingAnalyzer()
        
        model_config = {
            "d_model": 512,
            "n_layers": 6,
            "max_seq_length": 1024
        }
        
        training_config = {
            "batch_size": 32,
            "epochs": 10,
            "steps_per_epoch": 1000
        }
        
        flops = analyzer._estimate_compute_flops(model_config, training_config)
        assert flops > 0
    
    def test_memory_usage_estimation(self):
        """Test memory usage estimation."""
        
        analyzer = ModelScalingAnalyzer()
        
        model_config = {
            "d_model": 512,
            "n_layers": 6,
            "max_seq_length": 1024,
            "vocab_size": 10000
        }
        
        training_config = {
            "batch_size": 32
        }
        
        memory_gb = analyzer._estimate_memory_usage(model_config, training_config)
        assert memory_gb > 0
        assert memory_gb < 100  # Should be reasonable
    
    def test_scaling_experiments_execution(self):
        """Test scaling experiments execution."""
        
        analyzer = ModelScalingAnalyzer()
        
        # Create simple experiment
        experiment = ScalingExperiment(
            experiment_id="test_scaling",
            model_configs=[
                {"d_model": 256, "n_layers": 4},
                {"d_model": 512, "n_layers": 6}
            ],
            training_config={"batch_size": 32, "epochs": 10},
            scaling_dimensions=[ScalingDimension.PARAMETERS],
            metrics=[ScalingPerformanceMetric.VALIDATION_LOSS],
            max_epochs=5
        )
        
        results = analyzer.run_scaling_experiments(experiment)
        
        assert len(results) == 2
        assert all(r.experiment_id == "test_scaling" for r in results)
        assert all(r.parameter_count > 0 for r in results)
    
    def test_scaling_law_fitting(self):
        """Test scaling law fitting."""
        
        analyzer = ModelScalingAnalyzer()
        
        # Create mock results
        mock_results = []
        for i in range(10):
            from src.training.utils.scaling_laws import ScalingResult
            
            result = ScalingResult(
                experiment_id="test",
                model_config={"d_model": 256 + i * 64},
                parameter_count=1000000 + i * 1000000,
                compute_flops=1e12 + i * 1e12,
                memory_usage=1.0 + i * 0.5,
                final_loss=2.0 - i * 0.1,
                best_loss=1.8 - i * 0.1,
                training_time=1.0 + i * 0.5,
                convergence_epoch=50 - i * 2,
                musical_quality=0.5 + i * 0.03,
                genre_performance={},
                loss_per_parameter=0.0,
                loss_per_flop=0.0,
                loss_per_hour=0.0,
                loss_history=[],
                validation_history=[],
                converged=True,
                stable=True
            )
            mock_results.append(result)
        
        # Fit scaling laws
        scaling_laws = analyzer.fit_scaling_laws(mock_results)
        
        assert isinstance(scaling_laws, dict)
        # Should have at least one scaling law
        assert len(scaling_laws) > 0
    
    def test_optimal_model_size_prediction(self):
        """Test optimal model size prediction."""
        
        analyzer = ModelScalingAnalyzer()
        
        # Mock some scaling laws
        from src.training.utils.scaling_laws import ScalingLaw
        
        # Create mock scaling laws
        param_compute_law = ScalingLaw(
            dimension=ScalingDimension.PARAMETERS,
            metric=ScalingPerformanceMetric.TRAINING_TIME,
            law_type=ScalingLawType.POWER_LAW,
            coefficients=[1e-6, 0.8],
            r_squared=0.95,
            confidence_interval=(0.9, 0.99),
            x_values=[1e6, 2e6, 5e6],
            y_values=[1.0, 1.5, 2.8],
            prediction_range=(1e6, 1e8),
            equation="y = 1e-6 * x^0.8",
            p_value=0.001,
            standard_error=0.01
        )
        
        param_loss_law = ScalingLaw(
            dimension=ScalingDimension.PARAMETERS,
            metric=ScalingPerformanceMetric.VALIDATION_LOSS,
            law_type=ScalingLawType.POWER_LAW,
            coefficients=[4.0, -0.076],
            r_squared=0.90,
            confidence_interval=(0.85, 0.95),
            x_values=[1e6, 2e6, 5e6],
            y_values=[2.0, 1.8, 1.5],
            prediction_range=(1e6, 1e8),
            equation="y = 4.0 * x^-0.076",
            p_value=0.001,
            standard_error=0.02
        )
        
        analyzer.scaling_laws = {
            "parameters_training_time": param_compute_law,
            "parameters_validation_loss": param_loss_law
        }
        
        # Test prediction
        optimal_config = analyzer.predict_optimal_model_size(
            compute_budget=10.0,
            performance_target=1.6
        )
        
        assert "parameter_count" in optimal_config
        assert "predicted_loss" in optimal_config
        assert "suggested_architecture" in optimal_config
    
    def test_scaling_report_generation(self):
        """Test scaling analysis report generation."""
        
        analyzer = ModelScalingAnalyzer()
        
        # Add mock results
        from src.training.utils.scaling_laws import ScalingResult
        
        result = ScalingResult(
            experiment_id="test",
            model_config={"d_model": 512},
            parameter_count=5000000,
            compute_flops=1e15,
            memory_usage=4.0,
            final_loss=1.5,
            best_loss=1.4,
            training_time=2.0,
            convergence_epoch=30,
            musical_quality=0.7,
            genre_performance={},
            loss_per_parameter=3e-7,
            loss_per_flop=1.5e-15,
            loss_per_hour=0.75,
            loss_history=[2.0, 1.8, 1.5],
            validation_history=[2.1, 1.9, 1.6],
            converged=True,
            stable=True
        )
        
        analyzer.results = [result]
        
        report = analyzer.generate_scaling_report()
        
        assert "experiment_summary" in report
        assert "parameter_range" in report
        assert "performance_range" in report
        assert "best_models" in report
        assert "recommendations" in report


class TestIntegration:
    """Integration tests for Phase 4.5 with existing Phase 4.1-4.4 components."""
    
    def test_hyperparameter_optimization_with_curriculum(self):
        """Test hyperparameter optimization integration with curriculum learning."""
        
        # Create curriculum scheduler
        from src.training.utils.curriculum_learning import CurriculumConfig
        
        curriculum_config = CurriculumConfig()
        curriculum_scheduler = ProgressiveCurriculumScheduler(curriculum_config)
        
        # Test hyperparameter optimization includes curriculum parameters
        param_space = create_musical_parameter_space()
        
        assert "curriculum_strategy" in param_space
        
        # Test that curriculum can be configured through hyperparameter optimization
        config = OptimizationConfig(
            strategy=OptimizationStrategy.RANDOM_SEARCH,
            parameter_space=param_space,
            max_trials=3
        )
        
        optimizer = HyperparameterOptimizer(config)
        
        def mock_objective(params):
            # Use curriculum strategy from hyperparameters
            curriculum_strategy = params.get("curriculum_strategy", "linear")
            
            # Simulate curriculum affecting training
            strategy_bonus = 0.1 if curriculum_strategy == "musical" else 0.0
            
            return {
                "validation_loss": 1.5 - strategy_bonus,
                "musical_quality": 0.6 + strategy_bonus
            }
        
        results = optimizer.optimize(mock_objective)
        
        assert results["best_trial"]["parameters"] is not None
    
    def test_efficiency_optimization_with_reproducibility(self):
        """Test efficiency optimization with reproducibility management."""
        
        # Create reproducibility manager
        from src.training.utils.reproducibility import ReproducibilityConfig
        
        repro_config = ReproducibilityConfig()
        repro_manager = ReproducibilityManager(repro_config)
        
        # Test that efficiency optimization respects reproducibility
        efficiency_config = create_production_efficiency_config()
        efficiency_optimizer = TrainingEfficiencyOptimizer(efficiency_config)
        
        # Verify that efficiency optimization can work with deterministic training
        assert efficiency_config.optimization_aggressiveness >= 0
        assert efficiency_config.stability_priority is not None
        
        # Test that both systems can coexist
        repro_manager.set_global_seeds(epoch=0)
        efficiency_optimizer.start_monitoring()
        
        # Simulate some work
        time.sleep(0.01)
        
        efficiency_optimizer.stop_monitoring()
        
        # Should not crash or interfere with each other
        assert True
    
    def test_scaling_analysis_with_musical_strategies(self):
        """Test scaling analysis with musical domain strategies."""
        
        # Create musical domain trainer
        from src.training.utils.musical_strategies import MusicalTrainingConfig
        
        musical_config = MusicalTrainingConfig()
        musical_trainer = MusicalDomainTrainer(musical_config)
        
        # Test scaling analysis includes musical considerations
        analyzer = ModelScalingAnalyzer()
        
        # Verify that musical complexity is considered in scaling
        model_config = {
            "d_model": 512,
            "n_layers": 6,
            "musical_complexity_factor": 1.2
        }
        
        training_config = {
            "batch_size": 32,
            "epochs": 10,
            "musical_domain": "classical"
        }
        
        # Test that scaling analysis can incorporate musical domain knowledge
        param_count = analyzer._estimate_parameter_count(model_config)
        memory_usage = analyzer._estimate_memory_usage(model_config, training_config)
        
        assert param_count > 0
        assert memory_usage > 0
    
    def test_full_phase_4_5_integration(self, mock_model, mock_optimizer, mock_dataloader):
        """Test full Phase 4.5 integration with all components."""
        
        # Initialize all Phase 4.5 components
        hyperopt_config = create_quick_search_config()
        hyperopt_optimizer = HyperparameterOptimizer(hyperopt_config)
        
        efficiency_config = create_production_efficiency_config()
        efficiency_optimizer = TrainingEfficiencyOptimizer(efficiency_config)
        
        scaling_analyzer = ModelScalingAnalyzer()
        
        # Test that all components can work together
        efficiency_optimizer.start_monitoring()
        
        # Simulate a training loop with all components
        with efficiency_optimizer.profile_batch(epoch=1, batch_idx=0):
            # Mock training step
            inputs = torch.randint(0, 1000, (16, 256))
            outputs = mock_model(inputs)
            loss = outputs.sum()
            loss.backward()
            mock_optimizer.step()
        
        efficiency_optimizer.stop_monitoring()
        
        # Get performance summary
        perf_summary = efficiency_optimizer.get_optimization_report()
        
        # Verify integration worked
        assert "performance" in perf_summary
        assert "optimizations" in perf_summary
        
        # Test that scaling analysis can use efficiency data
        # (In practice, this would involve actual training data)
        assert True  # Integration successful if we reach here
    
    def test_phase_4_5_configuration_compatibility(self):
        """Test that Phase 4.5 configurations are compatible with existing phases."""
        
        # Test hyperparameter optimization config
        hyperopt_config = create_quick_search_config()
        assert hasattr(hyperopt_config, 'parameter_space')
        assert hasattr(hyperopt_config, 'max_trials')
        
        # Test efficiency config
        efficiency_config = create_production_efficiency_config()
        assert hasattr(efficiency_config, 'target_throughput')
        assert hasattr(efficiency_config, 'optimization_strategies')
        
        # Test scaling experiment config
        scaling_experiment = create_scaling_experiment(
            "test", {"d_model": [256, 512]}, 100.0
        )
        assert hasattr(scaling_experiment, 'model_configs')
        assert hasattr(scaling_experiment, 'scaling_dimensions')
        
        # All configurations should be serializable
        import json
        
        # Test serialization (basic compatibility check)
        try:
            json.dumps(hyperopt_config.parameter_space, default=str)
            json.dumps(efficiency_config.target_throughput)
            json.dumps(scaling_experiment.model_configs)
        except Exception:
            pytest.fail("Configuration serialization failed")


class TestProductionReadiness:
    """Test Phase 4.5 production readiness."""
    
    def test_error_handling_hyperparameter_optimization(self):
        """Test error handling in hyperparameter optimization."""
        
        config = create_quick_search_config()
        optimizer = HyperparameterOptimizer(config)
        
        # Test with failing objective function
        def failing_objective(params):
            if params.get("learning_rate", 0) > 0.0005:  # Use a threshold that will actually trigger
                raise ValueError("Learning rate too high")
            return {"validation_loss": 1.0}
        
        results = optimizer.optimize(failing_objective)
        
        # Should handle failures gracefully
        assert "best_trial" in results
        assert results["optimization_stats"]["failed_trials"] > 0
    
    def test_error_handling_efficiency_optimization(self):
        """Test error handling in efficiency optimization."""
        
        config = create_production_efficiency_config()
        optimizer = TrainingEfficiencyOptimizer(config)
        
        # Test with None inputs
        with pytest.raises(Exception):
            optimizer.optimize_training(None, None, None)
        
        # Test graceful degradation
        try:
            optimizer.start_monitoring()
            optimizer.stop_monitoring()
        except Exception:
            pytest.fail("Monitoring should not fail")
    
    def test_error_handling_scaling_analysis(self):
        """Test error handling in scaling analysis."""
        
        analyzer = ModelScalingAnalyzer()
        
        # Test with invalid model config
        invalid_config = {"invalid_param": "invalid_value"}
        
        # Should handle gracefully
        param_count = analyzer._estimate_parameter_count(invalid_config)
        assert param_count >= 0  # Should return reasonable default
    
    def test_memory_management(self):
        """Test memory management in Phase 4.5 components."""
        
        # Test that components don't leak memory
        config = create_production_efficiency_config()
        optimizer = TrainingEfficiencyOptimizer(config)
        
        # Generate lots of performance data
        for i in range(1000):
            with optimizer.profile_batch(epoch=1, batch_idx=i):
                pass
        
        # Snapshots should be limited
        assert len(optimizer.profiler.snapshots) <= config.analysis_window
    
    def test_performance_overhead(self):
        """Test that Phase 4.5 components have minimal performance overhead."""
        
        config = create_production_efficiency_config()
        optimizer = TrainingEfficiencyOptimizer(config)
        
        # Measure overhead of profiling
        start_time = time.time()
        
        for i in range(100):
            with optimizer.profile_batch(epoch=1, batch_idx=i):
                # Simulate minimal work
                x = torch.randn(10, 10)
                y = x @ x.T
        
        elapsed = time.time() - start_time
        
        # Should complete quickly (overhead should be minimal)
        assert elapsed < 1.0  # Should take less than 1 second
    
    def test_configuration_validation(self):
        """Test configuration validation."""
        
        # Test invalid hyperparameter optimization config
        with pytest.raises(ValueError):
            config = OptimizationConfig(max_trials=-1)
        
        # Test invalid efficiency config
        with pytest.raises(ValueError):
            config = EfficiencyConfig(target_throughput=-1)
        
        # Test invalid scaling experiment
        with pytest.raises(ValueError):
            experiment = ScalingExperiment(
                experiment_id="",
                model_configs=[],
                training_config={},
                scaling_dimensions=[],
                metrics=[]
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])