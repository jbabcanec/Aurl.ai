"""
Enhanced Anomaly Detection for Aurl.ai Music Generation Training.

This module provides sophisticated anomaly detection during training:
- Multi-metric anomaly detection
- Statistical outlier detection
- Pattern-based anomaly detection
- Adaptive thresholds
- Root cause analysis
- Automated recovery suggestions
"""

import numpy as np
from typing import Dict, List, Optional, Tuple, Any, Deque
from dataclasses import dataclass, field
from collections import deque, defaultdict
from datetime import datetime, timedelta
import json
from pathlib import Path
from enum import Enum

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class AnomalyType(Enum):
    """Types of training anomalies."""
    GRADIENT_EXPLOSION = "gradient_explosion"
    GRADIENT_VANISHING = "gradient_vanishing"
    LOSS_SPIKE = "loss_spike"
    LOSS_PLATEAU = "loss_plateau"
    LOSS_OSCILLATION = "loss_oscillation"
    MEMORY_OVERFLOW = "memory_overflow"
    THROUGHPUT_DEGRADATION = "throughput_degradation"
    LEARNING_RATE_ISSUE = "learning_rate_issue"
    NAN_VALUES = "nan_values"
    TRAINING_DIVERGENCE = "training_divergence"
    OVERFITTING = "overfitting"
    UNDERFITTING = "underfitting"


class AnomalySeverity(Enum):
    """Severity levels for anomalies."""
    INFO = "info"
    WARNING = "warning"
    CRITICAL = "critical"
    FATAL = "fatal"


@dataclass
class Anomaly:
    """Container for detected anomaly information."""
    
    timestamp: datetime
    anomaly_type: AnomalyType
    severity: AnomalySeverity
    metric_name: str
    metric_value: float
    expected_range: Tuple[float, float]
    details: str
    recovery_suggestions: List[str] = field(default_factory=list)
    epoch: Optional[int] = None
    batch: Optional[int] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to dictionary for logging."""
        return {
            "timestamp": self.timestamp.isoformat(),
            "type": self.anomaly_type.value,
            "severity": self.severity.value,
            "metric": self.metric_name,
            "value": self.metric_value,
            "expected_range": list(self.expected_range),
            "details": self.details,
            "recovery_suggestions": self.recovery_suggestions,
            "epoch": self.epoch,
            "batch": self.batch
        }


class EnhancedAnomalyDetector:
    """
    Enhanced anomaly detection system for training monitoring.
    
    Features:
    - Statistical anomaly detection (z-score, IQR)
    - Pattern-based detection (oscillations, plateaus)
    - Adaptive thresholds that adjust during training
    - Multi-metric correlation analysis
    - Root cause analysis
    - Automated recovery suggestions
    """
    
    def __init__(self,
                 window_size: int = 100,
                 z_score_threshold: float = 3.0,
                 enable_adaptive_thresholds: bool = True,
                 alert_callback: Optional[callable] = None):
        """
        Initialize anomaly detector.
        
        Args:
            window_size: Size of sliding window for statistics
            z_score_threshold: Z-score threshold for outlier detection
            enable_adaptive_thresholds: Whether to adapt thresholds during training
            alert_callback: Function to call when anomaly detected
        """
        
        self.window_size = window_size
        self.z_score_threshold = z_score_threshold
        self.enable_adaptive_thresholds = enable_adaptive_thresholds
        self.alert_callback = alert_callback
        
        # Metric history
        self.metric_history: Dict[str, Deque[float]] = defaultdict(
            lambda: deque(maxlen=window_size)
        )
        
        # Anomaly history
        self.anomaly_history: List[Anomaly] = []
        self.anomaly_counts: Dict[AnomalyType, int] = defaultdict(int)
        
        # Adaptive thresholds
        self.adaptive_thresholds: Dict[str, Dict[str, float]] = {}
        
        # Pattern detection state
        self.oscillation_detector = OscillationDetector(window_size=20)
        self.plateau_detector = PlateauDetector(window_size=50)
        
        # Recovery strategies
        self._setup_recovery_strategies()
        
        logger.info(f"Enhanced anomaly detector initialized - window: {window_size}")
    
    def _setup_recovery_strategies(self):
        """Setup recovery suggestions for each anomaly type."""
        
        self.recovery_strategies = {
            AnomalyType.GRADIENT_EXPLOSION: [
                "Reduce learning rate by 50%",
                "Increase gradient clipping threshold",
                "Check for numerical instabilities in model",
                "Consider using gradient accumulation"
            ],
            AnomalyType.GRADIENT_VANISHING: [
                "Increase learning rate by 20%",
                "Check model initialization",
                "Consider using residual connections",
                "Verify activation functions (avoid sigmoid/tanh in deep layers)"
            ],
            AnomalyType.LOSS_SPIKE: [
                "Reduce learning rate temporarily",
                "Check for corrupted data in batch",
                "Verify loss calculation correctness",
                "Consider reverting to previous checkpoint"
            ],
            AnomalyType.LOSS_PLATEAU: [
                "Adjust learning rate schedule",
                "Increase model capacity",
                "Add data augmentation",
                "Check if model has converged"
            ],
            AnomalyType.LOSS_OSCILLATION: [
                "Reduce learning rate",
                "Increase batch size",
                "Use learning rate warmup",
                "Check for batch normalization issues"
            ],
            AnomalyType.MEMORY_OVERFLOW: [
                "Reduce batch size",
                "Enable gradient checkpointing",
                "Clear GPU cache",
                "Consider model parallelism"
            ],
            AnomalyType.THROUGHPUT_DEGRADATION: [
                "Check for I/O bottlenecks",
                "Verify data loading efficiency",
                "Consider increasing number of workers",
                "Check for memory leaks"
            ],
            AnomalyType.LEARNING_RATE_ISSUE: [
                "Verify learning rate schedule",
                "Check optimizer state",
                "Consider using adaptive learning rate",
                "Review warmup configuration"
            ],
            AnomalyType.NAN_VALUES: [
                "Check for division by zero",
                "Verify loss function stability",
                "Add epsilon to denominators",
                "Check data preprocessing"
            ],
            AnomalyType.TRAINING_DIVERGENCE: [
                "Stop training immediately",
                "Revert to last stable checkpoint",
                "Significantly reduce learning rate",
                "Check for exploding gradients"
            ],
            AnomalyType.OVERFITTING: [
                "Increase dropout rate",
                "Add weight decay/L2 regularization",
                "Reduce model capacity",
                "Increase data augmentation"
            ],
            AnomalyType.UNDERFITTING: [
                "Increase model capacity",
                "Train for more epochs",
                "Reduce regularization",
                "Check if learning rate is too low"
            ]
        }
    
    def check_metrics(self,
                     metrics: Dict[str, float],
                     epoch: Optional[int] = None,
                     batch: Optional[int] = None) -> List[Anomaly]:
        """
        Check metrics for anomalies.
        
        Args:
            metrics: Dictionary of metric name -> value
            epoch: Current epoch number
            batch: Current batch number
            
        Returns:
            List of detected anomalies
        """
        
        detected_anomalies = []
        
        # Update history
        for metric_name, value in metrics.items():
            self.metric_history[metric_name].append(value)
        
        # Check each metric
        for metric_name, value in metrics.items():
            # Skip if not enough history
            if len(self.metric_history[metric_name]) < 10:
                continue
            
            # Statistical anomaly detection
            anomaly = self._check_statistical_anomaly(metric_name, value, epoch, batch)
            if anomaly:
                detected_anomalies.append(anomaly)
            
            # Pattern-based detection
            if metric_name in ["total_loss", "reconstruction_loss"]:
                # Check for oscillations
                if self.oscillation_detector.check(self.metric_history[metric_name]):
                    anomaly = self._create_anomaly(
                        AnomalyType.LOSS_OSCILLATION,
                        AnomalySeverity.WARNING,
                        metric_name,
                        value,
                        "Loss is oscillating significantly",
                        epoch,
                        batch
                    )
                    detected_anomalies.append(anomaly)
                
                # Check for plateau
                if self.plateau_detector.check(self.metric_history[metric_name]):
                    anomaly = self._create_anomaly(
                        AnomalyType.LOSS_PLATEAU,
                        AnomalySeverity.WARNING,
                        metric_name,
                        value,
                        "Loss has plateaued",
                        epoch,
                        batch
                    )
                    detected_anomalies.append(anomaly)
        
        # Specific metric checks
        detected_anomalies.extend(self._check_specific_metrics(metrics, epoch, batch))
        
        # Multi-metric correlation checks
        detected_anomalies.extend(self._check_metric_correlations(metrics, epoch, batch))
        
        # Process detected anomalies
        for anomaly in detected_anomalies:
            self._process_anomaly(anomaly)
        
        return detected_anomalies
    
    def _check_statistical_anomaly(self,
                                 metric_name: str,
                                 value: float,
                                 epoch: Optional[int],
                                 batch: Optional[int]) -> Optional[Anomaly]:
        """Check for statistical anomalies using z-score and IQR methods."""
        
        history = list(self.metric_history[metric_name])
        
        # Check for NaN
        if np.isnan(value) or np.isinf(value):
            return self._create_anomaly(
                AnomalyType.NAN_VALUES,
                AnomalySeverity.CRITICAL,
                metric_name,
                value,
                f"NaN or Inf detected in {metric_name}",
                epoch,
                batch
            )
        
        # Calculate statistics
        mean = np.mean(history[:-1])  # Exclude current value
        std = np.std(history[:-1])
        
        if std > 0:
            z_score = abs((value - mean) / std)
            
            # Adaptive threshold
            threshold = self._get_adaptive_threshold(metric_name, self.z_score_threshold)
            
            if z_score > threshold:
                # Determine anomaly type based on metric
                if "loss" in metric_name.lower():
                    if value > mean + threshold * std:
                        anomaly_type = AnomalyType.LOSS_SPIKE
                    else:
                        anomaly_type = AnomalyType.TRAINING_DIVERGENCE
                elif "gradient" in metric_name.lower():
                    if value > mean + threshold * std:
                        anomaly_type = AnomalyType.GRADIENT_EXPLOSION
                    else:
                        anomaly_type = AnomalyType.GRADIENT_VANISHING
                else:
                    return None  # No specific type for this metric
                
                severity = self._determine_severity(z_score, threshold)
                
                return self._create_anomaly(
                    anomaly_type,
                    severity,
                    metric_name,
                    value,
                    f"Z-score: {z_score:.2f} (threshold: {threshold:.2f})",
                    epoch,
                    batch,
                    expected_range=(mean - threshold * std, mean + threshold * std)
                )
        
        return None
    
    def _check_specific_metrics(self,
                              metrics: Dict[str, float],
                              epoch: Optional[int],
                              batch: Optional[int]) -> List[Anomaly]:
        """Check specific metrics for known issues."""
        
        anomalies = []
        
        # Gradient norm checks
        if "gradient_norm" in metrics:
            grad_norm = metrics["gradient_norm"]
            
            if grad_norm > 100.0:
                anomalies.append(self._create_anomaly(
                    AnomalyType.GRADIENT_EXPLOSION,
                    AnomalySeverity.CRITICAL,
                    "gradient_norm",
                    grad_norm,
                    f"Gradient norm extremely high: {grad_norm:.2f}",
                    epoch,
                    batch,
                    expected_range=(0, 10.0)
                ))
            elif grad_norm < 1e-6:
                anomalies.append(self._create_anomaly(
                    AnomalyType.GRADIENT_VANISHING,
                    AnomalySeverity.WARNING,
                    "gradient_norm",
                    grad_norm,
                    f"Gradient norm near zero: {grad_norm:.6f}",
                    epoch,
                    batch,
                    expected_range=(0.001, 10.0)
                ))
        
        # Memory checks
        if "gpu_allocated" in metrics and "gpu_max_allocated" in metrics:
            usage_ratio = metrics["gpu_allocated"] / metrics["gpu_max_allocated"]
            
            if usage_ratio > 0.95:
                anomalies.append(self._create_anomaly(
                    AnomalyType.MEMORY_OVERFLOW,
                    AnomalySeverity.CRITICAL,
                    "memory_usage",
                    usage_ratio * 100,
                    f"GPU memory usage at {usage_ratio*100:.1f}%",
                    epoch,
                    batch,
                    expected_range=(0, 90)
                ))
        
        # Throughput checks
        if "samples_per_second" in metrics:
            throughput = metrics["samples_per_second"]
            history = list(self.metric_history["samples_per_second"])
            
            if len(history) > 10:
                avg_throughput = np.mean(history[-10:-1])
                if throughput < avg_throughput * 0.5:
                    anomalies.append(self._create_anomaly(
                        AnomalyType.THROUGHPUT_DEGRADATION,
                        AnomalySeverity.WARNING,
                        "samples_per_second",
                        throughput,
                        f"Throughput dropped by {(1-throughput/avg_throughput)*100:.1f}%",
                        epoch,
                        batch,
                        expected_range=(avg_throughput * 0.8, avg_throughput * 1.2)
                    ))
        
        return anomalies
    
    def _check_metric_correlations(self,
                                 metrics: Dict[str, float],
                                 epoch: Optional[int],
                                 batch: Optional[int]) -> List[Anomaly]:
        """Check for anomalies in metric correlations."""
        
        anomalies = []
        
        # Check train vs validation loss for overfitting
        if "train_loss" in metrics and "val_loss" in metrics:
            train_loss = metrics["train_loss"]
            val_loss = metrics["val_loss"]
            
            if val_loss > train_loss * 1.5 and epoch and epoch > 10:
                anomalies.append(self._create_anomaly(
                    AnomalyType.OVERFITTING,
                    AnomalySeverity.WARNING,
                    "loss_ratio",
                    val_loss / train_loss,
                    f"Validation loss significantly higher than training loss",
                    epoch,
                    batch,
                    expected_range=(0.9, 1.2)
                ))
        
        # Check if losses are not decreasing (underfitting)
        if "total_loss" in metrics and len(self.metric_history["total_loss"]) > 20:
            recent_losses = list(self.metric_history["total_loss"])[-20:]
            loss_trend = np.polyfit(range(len(recent_losses)), recent_losses, 1)[0]
            
            if loss_trend > 0 and epoch and epoch > 5:
                anomalies.append(self._create_anomaly(
                    AnomalyType.UNDERFITTING,
                    AnomalySeverity.WARNING,
                    "loss_trend",
                    loss_trend,
                    "Loss is increasing over recent batches",
                    epoch,
                    batch,
                    expected_range=(-0.01, 0)
                ))
        
        return anomalies
    
    def _create_anomaly(self,
                       anomaly_type: AnomalyType,
                       severity: AnomalySeverity,
                       metric_name: str,
                       metric_value: float,
                       details: str,
                       epoch: Optional[int],
                       batch: Optional[int],
                       expected_range: Tuple[float, float] = (0, 0)) -> Anomaly:
        """Create an anomaly object with recovery suggestions."""
        
        anomaly = Anomaly(
            timestamp=datetime.now(),
            anomaly_type=anomaly_type,
            severity=severity,
            metric_name=metric_name,
            metric_value=metric_value,
            expected_range=expected_range,
            details=details,
            recovery_suggestions=self.recovery_strategies.get(anomaly_type, []),
            epoch=epoch,
            batch=batch
        )
        
        return anomaly
    
    def _determine_severity(self, z_score: float, threshold: float) -> AnomalySeverity:
        """Determine anomaly severity based on z-score."""
        
        if z_score > threshold * 3:
            return AnomalySeverity.CRITICAL
        elif z_score > threshold * 2:
            return AnomalySeverity.WARNING
        else:
            return AnomalySeverity.INFO
    
    def _get_adaptive_threshold(self, metric_name: str, base_threshold: float) -> float:
        """Get adaptive threshold for a metric."""
        
        if not self.enable_adaptive_thresholds:
            return base_threshold
        
        # Adjust threshold based on training progress and metric stability
        if metric_name not in self.adaptive_thresholds:
            self.adaptive_thresholds[metric_name] = {"threshold": base_threshold, "adjustments": 0}
        
        # Get metric stability
        history = list(self.metric_history[metric_name])
        if len(history) > 20:
            recent_std = np.std(history[-20:])
            overall_std = np.std(history)
            
            # If metric is becoming more stable, tighten threshold
            if recent_std < overall_std * 0.8:
                self.adaptive_thresholds[metric_name]["threshold"] *= 0.95
                self.adaptive_thresholds[metric_name]["adjustments"] += 1
            # If metric is becoming less stable, loosen threshold
            elif recent_std > overall_std * 1.2:
                self.adaptive_thresholds[metric_name]["threshold"] *= 1.05
                self.adaptive_thresholds[metric_name]["adjustments"] += 1
        
        return self.adaptive_thresholds[metric_name]["threshold"]
    
    def _process_anomaly(self, anomaly: Anomaly):
        """Process detected anomaly."""
        
        # Add to history
        self.anomaly_history.append(anomaly)
        self.anomaly_counts[anomaly.anomaly_type] += 1
        
        # Log anomaly
        logger.warning(
            f"Anomaly detected: {anomaly.anomaly_type.value} - "
            f"{anomaly.metric_name}={anomaly.metric_value:.4f} - "
            f"{anomaly.details}"
        )
        
        # Call alert callback if provided
        if self.alert_callback:
            self.alert_callback(anomaly)
    
    def get_anomaly_summary(self) -> Dict[str, Any]:
        """Get summary of detected anomalies."""
        
        summary = {
            "total_anomalies": len(self.anomaly_history),
            "anomaly_counts": {k.value: v for k, v in self.anomaly_counts.items()},
            "severity_distribution": defaultdict(int),
            "recent_anomalies": []
        }
        
        # Count by severity
        for anomaly in self.anomaly_history:
            summary["severity_distribution"][anomaly.severity.value] += 1
        
        # Get recent anomalies
        if self.anomaly_history:
            recent = self.anomaly_history[-10:]
            summary["recent_anomalies"] = [a.to_dict() for a in recent]
        
        return dict(summary)
    
    def save_anomaly_report(self, save_path: Path):
        """Save detailed anomaly report."""
        
        report = {
            "timestamp": datetime.now().isoformat(),
            "summary": self.get_anomaly_summary(),
            "adaptive_thresholds": self.adaptive_thresholds,
            "anomaly_history": [a.to_dict() for a in self.anomaly_history]
        }
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(report, f, indent=2)
        
        logger.info(f"Anomaly report saved to {save_path}")


class OscillationDetector:
    """Detects oscillation patterns in metrics."""
    
    def __init__(self, window_size: int = 20, threshold: float = 0.3):
        self.window_size = window_size
        self.threshold = threshold
    
    def check(self, values: Deque[float]) -> bool:
        """Check if values are oscillating."""
        
        if len(values) < self.window_size:
            return False
        
        recent = list(values)[-self.window_size:]
        
        # Count direction changes
        direction_changes = 0
        for i in range(1, len(recent) - 1):
            if (recent[i] - recent[i-1]) * (recent[i+1] - recent[i]) < 0:
                direction_changes += 1
        
        # High number of direction changes indicates oscillation
        oscillation_ratio = direction_changes / (len(recent) - 2)
        
        return oscillation_ratio > self.threshold


class PlateauDetector:
    """Detects plateau patterns in metrics."""
    
    def __init__(self, window_size: int = 50, threshold: float = 0.01):
        self.window_size = window_size
        self.threshold = threshold
    
    def check(self, values: Deque[float]) -> bool:
        """Check if values have plateaued."""
        
        if len(values) < self.window_size:
            return False
        
        recent = list(values)[-self.window_size:]
        
        # Calculate coefficient of variation
        mean = np.mean(recent)
        std = np.std(recent)
        
        if mean == 0:
            return False
        
        cv = std / abs(mean)
        
        # Low coefficient of variation indicates plateau
        return cv < self.threshold