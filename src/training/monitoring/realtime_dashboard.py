"""
Real-time Training Dashboard for Aurl.ai Music Generation.

This module provides live monitoring capabilities including:
- Real-time loss tracking and visualization
- Training performance metrics
- Data usage monitoring
- System resource tracking
- Musical quality metrics (when available)
- Training progress estimation

Designed for production training with immediate feedback and anomaly detection.
"""

import time
import threading
import queue
from pathlib import Path
from typing import Dict, List, Optional, Any, Callable
from datetime import datetime, timedelta
from collections import deque, defaultdict
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from matplotlib.widgets import Button
import seaborn as sns
import json
import warnings
warnings.filterwarnings('ignore', category=UserWarning)

from src.utils.base_logger import setup_logger

logger = setup_logger(__name__)


class MetricsBuffer:
    """Thread-safe buffer for real-time metrics."""
    
    def __init__(self, max_size: int = 1000):
        self.max_size = max_size
        self.data = deque(maxlen=max_size)
        self.timestamps = deque(maxlen=max_size)
        self._lock = threading.Lock()
    
    def add(self, metrics: Dict[str, float], timestamp: Optional[datetime] = None):
        """Add metrics to buffer."""
        if timestamp is None:
            timestamp = datetime.now()
        
        with self._lock:
            self.data.append(metrics.copy())
            self.timestamps.append(timestamp)
    
    def get_recent(self, n: int = 100) -> tuple:
        """Get recent n metrics."""
        with self._lock:
            if not self.data:
                return [], []
            
            recent_data = list(self.data)[-n:]
            recent_timestamps = list(self.timestamps)[-n:]
            return recent_data, recent_timestamps
    
    def get_metric_history(self, metric_name: str, n: int = 100) -> tuple:
        """Get history for specific metric."""
        recent_data, recent_timestamps = self.get_recent(n)
        
        values = []
        timestamps = []
        
        for data, timestamp in zip(recent_data, recent_timestamps):
            if metric_name in data:
                values.append(data[metric_name])
                timestamps.append(timestamp)
        
        return timestamps, values
    
    def clear(self):
        """Clear buffer."""
        with self._lock:
            self.data.clear()
            self.timestamps.clear()


class AnomalyDetector:
    """Detect training anomalies in real-time."""
    
    def __init__(self, window_size: int = 50):
        self.window_size = window_size
        self.loss_history = deque(maxlen=window_size)
        self.gradient_history = deque(maxlen=window_size)
        self.anomalies = []
    
    def check_anomalies(self, metrics: Dict[str, float]) -> List[str]:
        """Check for training anomalies."""
        anomalies = []
        
        # Check for exploding gradients
        if 'gradient_norm' in metrics:
            grad_norm = metrics['gradient_norm']
            self.gradient_history.append(grad_norm)
            
            if len(self.gradient_history) > 10:
                recent_avg = np.mean(list(self.gradient_history)[-10:])
                if grad_norm > recent_avg * 5 and grad_norm > 10:
                    anomalies.append(f"Exploding gradients detected: {grad_norm:.2f}")
        
        # Check for NaN/Inf losses
        if 'total_loss' in metrics:
            loss = metrics['total_loss']
            if np.isnan(loss) or np.isinf(loss):
                anomalies.append(f"Invalid loss detected: {loss}")
            else:
                self.loss_history.append(loss)
        
        # Check for loss explosion
        if len(self.loss_history) > 20:
            recent_losses = list(self.loss_history)[-20:]
            if recent_losses[-1] > np.mean(recent_losses[:-1]) * 3:
                anomalies.append(f"Loss explosion detected: {recent_losses[-1]:.4f}")
        
        # Check for training stagnation
        if len(self.loss_history) >= self.window_size:
            recent_losses = list(self.loss_history)[-self.window_size//2:]
            if np.std(recent_losses) < 0.001:
                anomalies.append("Training stagnation detected: loss not decreasing")
        
        return anomalies


class RealTimeDashboard:
    """
    Real-time training dashboard with live visualizations.
    
    Features:
    - Live loss tracking
    - Performance metrics
    - Anomaly detection
    - Resource monitoring
    - Progress estimation
    """
    
    def __init__(self, 
                 save_dir: Path,
                 experiment_id: str,
                 update_interval: float = 2.0,
                 enable_gui: bool = True):
        
        self.save_dir = Path(save_dir)
        self.experiment_id = experiment_id
        self.update_interval = update_interval
        self.enable_gui = enable_gui
        
        # Dashboard directory
        self.dashboard_dir = self.save_dir / "dashboard"
        self.dashboard_dir.mkdir(exist_ok=True)
        
        # Metrics storage
        self.metrics_buffer = MetricsBuffer(max_size=2000)
        self.anomaly_detector = AnomalyDetector()
        
        # Training state
        self.training_start_time = datetime.now()
        self.current_epoch = 0
        self.total_epochs = 100
        self.best_loss = float('inf')
        
        # Dashboard state
        self.dashboard_active = False
        self.dashboard_thread = None
        
        # Status tracking
        self.status_log = []
        self.alerts = deque(maxlen=10)
        
        logger.info(f"Initialized real-time dashboard for experiment {experiment_id}")
    
    def update_metrics(self, metrics: Dict[str, float]):
        """Update metrics and check for anomalies."""
        timestamp = datetime.now()
        
        # Add to buffer
        self.metrics_buffer.add(metrics, timestamp)
        
        # Check for anomalies
        anomalies = self.anomaly_detector.check_anomalies(metrics)
        
        # Log anomalies
        for anomaly in anomalies:
            self.alerts.append({
                'timestamp': timestamp,
                'type': 'anomaly',
                'message': anomaly
            })
            logger.warning(f"Training anomaly: {anomaly}")
        
        # Update best loss
        if 'total_loss' in metrics:
            if metrics['total_loss'] < self.best_loss:
                self.best_loss = metrics['total_loss']
                self.alerts.append({
                    'timestamp': timestamp,
                    'type': 'improvement',
                    'message': f"New best loss: {self.best_loss:.4f}"
                })
    
    def update_training_state(self, epoch: int, total_epochs: int):
        """Update training state information."""
        self.current_epoch = epoch
        self.total_epochs = total_epochs
    
    def start_dashboard(self):
        """Start real-time dashboard."""
        if self.dashboard_active:
            return
        
        self.dashboard_active = True
        
        if self.enable_gui:
            self.dashboard_thread = threading.Thread(
                target=self._run_matplotlib_dashboard,
                daemon=True
            )
            self.dashboard_thread.start()
        
        # Start metrics logging
        self._start_metrics_logging()
        
        logger.info("Started real-time dashboard")
    
    def stop_dashboard(self):
        """Stop real-time dashboard."""
        self.dashboard_active = False
        
        if self.dashboard_thread and self.dashboard_thread.is_alive():
            self.dashboard_thread.join(timeout=5)
        
        logger.info("Stopped real-time dashboard")
    
    def _run_matplotlib_dashboard(self):
        """Run matplotlib-based real-time dashboard."""
        try:
            # Set up the figure
            plt.style.use('dark_background')
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Aurl.ai Training Dashboard - {self.experiment_id}', 
                        fontsize=16, fontweight='bold', color='white')
            
            # Initialize empty plots
            loss_line, = axes[0, 0].plot([], [], 'r-', linewidth=2, label='Total Loss')
            recon_line, = axes[0, 0].plot([], [], 'b-', linewidth=2, label='Reconstruction')
            kl_line, = axes[0, 0].plot([], [], 'g-', linewidth=2, label='KL Divergence')
            
            throughput_line, = axes[0, 1].plot([], [], 'purple', linewidth=2, label='Samples/sec')
            tokens_line, = axes[0, 1].plot([], [], 'orange', linewidth=2, label='Tokens/sec')
            
            memory_line, = axes[0, 2].plot([], [], 'cyan', linewidth=2, label='GPU Memory')
            
            lr_line, = axes[1, 0].plot([], [], 'yellow', linewidth=2, label='Learning Rate')
            grad_line, = axes[1, 1].plot([], [], 'red', linewidth=2, label='Gradient Norm')
            
            # Set up axes
            self._setup_axes(axes)
            
            # Animation function
            def animate(frame):
                if not self.dashboard_active:
                    return
                
                try:
                    self._update_plots(axes, loss_line, recon_line, kl_line, 
                                     throughput_line, tokens_line, memory_line,
                                     lr_line, grad_line)
                except Exception as e:
                    logger.error(f"Dashboard update error: {e}")
            
            # Start animation
            ani = animation.FuncAnimation(
                fig, animate, interval=int(self.update_interval * 1000),
                blit=False, cache_frame_data=False
            )
            
            plt.tight_layout()
            plt.show()
            
        except Exception as e:
            logger.error(f"Dashboard error: {e}")
    
    def _setup_axes(self, axes):
        """Set up dashboard axes."""
        
        # Loss plot
        axes[0, 0].set_title('Training Losses', fontweight='bold', color='white')
        axes[0, 0].set_ylabel('Loss Value', color='white')
        axes[0, 0].legend()
        axes[0, 0].grid(True, alpha=0.3)
        
        # Throughput plot
        axes[0, 1].set_title('Training Throughput', fontweight='bold', color='white')
        axes[0, 1].set_ylabel('Throughput', color='white')
        axes[0, 1].legend()
        axes[0, 1].grid(True, alpha=0.3)
        
        # Memory plot
        axes[0, 2].set_title('GPU Memory Usage', fontweight='bold', color='white')
        axes[0, 2].set_ylabel('Memory (GB)', color='white')
        axes[0, 2].grid(True, alpha=0.3)
        
        # Learning rate plot
        axes[1, 0].set_title('Learning Rate', fontweight='bold', color='white')
        axes[1, 0].set_ylabel('Learning Rate', color='white')
        axes[1, 0].grid(True, alpha=0.3)
        
        # Gradient norm plot
        axes[1, 1].set_title('Gradient Norm', fontweight='bold', color='white')
        axes[1, 1].set_ylabel('Gradient Norm', color='white')
        axes[1, 1].grid(True, alpha=0.3)
        
        # Progress/status text
        axes[1, 2].axis('off')
        axes[1, 2].set_title('Training Status', fontweight='bold', color='white')
    
    def _update_plots(self, axes, loss_line, recon_line, kl_line, 
                     throughput_line, tokens_line, memory_line, lr_line, grad_line):
        """Update all plots with latest data."""
        
        # Get recent data
        recent_data, recent_timestamps = self.metrics_buffer.get_recent(200)
        
        if not recent_data:
            return
        
        # Convert timestamps to minutes from start
        time_minutes = [(ts - self.training_start_time).total_seconds() / 60 
                       for ts in recent_timestamps]
        
        # Update loss plots
        self._update_line_plot(loss_line, time_minutes, recent_data, 'total_loss')
        self._update_line_plot(recon_line, time_minutes, recent_data, 'reconstruction_loss')
        self._update_line_plot(kl_line, time_minutes, recent_data, 'kl_loss')
        
        axes[0, 0].relim()
        axes[0, 0].autoscale_view()
        
        # Update throughput plots
        self._update_line_plot(throughput_line, time_minutes, recent_data, 'samples_per_second')
        
        # Create twin axis for tokens/sec
        ax_twin = axes[0, 1].twinx()
        ax_twin.clear()
        tokens_values = [d.get('tokens_per_second', 0) for d in recent_data]
        if any(v > 0 for v in tokens_values):
            ax_twin.plot(time_minutes, tokens_values, 'orange', linewidth=2, label='Tokens/sec')
            ax_twin.set_ylabel('Tokens/sec', color='orange')
        
        axes[0, 1].relim()
        axes[0, 1].autoscale_view()
        
        # Update memory plot
        self._update_line_plot(memory_line, time_minutes, recent_data, 'gpu_allocated', scale=1e-9)
        axes[0, 2].relim()
        axes[0, 2].autoscale_view()
        
        # Update learning rate plot
        self._update_line_plot(lr_line, time_minutes, recent_data, 'learning_rate')
        axes[1, 0].relim()
        axes[1, 0].autoscale_view()
        
        # Update gradient norm plot
        self._update_line_plot(grad_line, time_minutes, recent_data, 'gradient_norm')
        axes[1, 1].relim()
        axes[1, 1].autoscale_view()
        
        # Update status text
        self._update_status_text(axes[1, 2], recent_data)
    
    def _update_line_plot(self, line, time_data, metrics_data, metric_name, scale=1.0):
        """Update a single line plot."""
        values = []
        times = []
        
        for time_val, data in zip(time_data, metrics_data):
            if metric_name in data and data[metric_name] is not None:
                values.append(data[metric_name] * scale)
                times.append(time_val)
        
        if values:
            line.set_data(times, values)
    
    def _update_status_text(self, ax, recent_data):
        """Update status text panel."""
        ax.clear()
        ax.axis('off')
        
        # Current metrics
        if recent_data:
            latest = recent_data[-1]
            
            status_text = []
            status_text.append(f"Epoch: {self.current_epoch}/{self.total_epochs}")
            
            if 'total_loss' in latest:
                status_text.append(f"Current Loss: {latest['total_loss']:.4f}")
            
            status_text.append(f"Best Loss: {self.best_loss:.4f}")
            
            # Training time
            elapsed = datetime.now() - self.training_start_time
            hours = int(elapsed.total_seconds() // 3600)
            minutes = int((elapsed.total_seconds() % 3600) // 60)
            status_text.append(f"Training Time: {hours:02d}:{minutes:02d}")
            
            # ETA calculation
            if self.current_epoch > 0:
                avg_time_per_epoch = elapsed.total_seconds() / self.current_epoch
                remaining_epochs = self.total_epochs - self.current_epoch
                eta_seconds = remaining_epochs * avg_time_per_epoch
                eta_hours = int(eta_seconds // 3600)
                eta_minutes = int((eta_seconds % 3600) // 60)
                status_text.append(f"ETA: {eta_hours:02d}:{eta_minutes:02d}")
            
            # Recent alerts
            if self.alerts:
                status_text.append("\nRecent Alerts:")
                for alert in list(self.alerts)[-3:]:
                    alert_time = alert['timestamp'].strftime('%H:%M:%S')
                    status_text.append(f"{alert_time}: {alert['message']}")
        
        # Display text
        text_str = '\n'.join(status_text)
        ax.text(0.05, 0.95, text_str, transform=ax.transAxes, 
               fontsize=12, verticalalignment='top', color='white',
               fontfamily='monospace')
    
    def _start_metrics_logging(self):
        """Start periodic metrics logging to file."""
        def log_metrics():
            while self.dashboard_active:
                try:
                    # Save current metrics snapshot
                    recent_data, recent_timestamps = self.metrics_buffer.get_recent(10)
                    
                    if recent_data:
                        snapshot = {
                            'timestamp': datetime.now().isoformat(),
                            'experiment_id': self.experiment_id,
                            'current_epoch': self.current_epoch,
                            'recent_metrics': recent_data[-1] if recent_data else {},
                            'alerts': list(self.alerts)
                        }
                        
                        # Save to file
                        snapshot_file = self.dashboard_dir / f"metrics_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                        with open(snapshot_file, 'w') as f:
                            json.dump(snapshot, f, indent=2, default=str)
                    
                    time.sleep(60)  # Log every minute
                    
                except Exception as e:
                    logger.error(f"Metrics logging error: {e}")
                    time.sleep(60)
        
        # Start logging thread
        logging_thread = threading.Thread(target=log_metrics, daemon=True)
        logging_thread.start()
    
    def save_dashboard_snapshot(self) -> Path:
        """Save current dashboard state as static image."""
        
        try:
            # Create snapshot figure
            fig, axes = plt.subplots(2, 3, figsize=(18, 12))
            fig.suptitle(f'Training Dashboard Snapshot - {self.experiment_id}', 
                        fontsize=16, fontweight='bold')
            
            # Get recent data
            recent_data, recent_timestamps = self.metrics_buffer.get_recent(200)
            
            if recent_data:
                time_minutes = [(ts - self.training_start_time).total_seconds() / 60 
                               for ts in recent_timestamps]
                
                # Plot losses
                loss_values = [d.get('total_loss', 0) for d in recent_data]
                recon_values = [d.get('reconstruction_loss', 0) for d in recent_data]
                kl_values = [d.get('kl_loss', 0) for d in recent_data]
                
                axes[0, 0].plot(time_minutes, loss_values, 'r-', linewidth=2, label='Total Loss')
                axes[0, 0].plot(time_minutes, recon_values, 'b-', linewidth=2, label='Reconstruction')
                axes[0, 0].plot(time_minutes, kl_values, 'g-', linewidth=2, label='KL Divergence')
                axes[0, 0].set_title('Training Losses')
                axes[0, 0].legend()
                axes[0, 0].grid(True, alpha=0.3)
                
                # Plot throughput
                throughput_values = [d.get('samples_per_second', 0) for d in recent_data]
                axes[0, 1].plot(time_minutes, throughput_values, 'purple', linewidth=2)
                axes[0, 1].set_title('Samples per Second')
                axes[0, 1].grid(True, alpha=0.3)
                
                # Add more plots as needed...
            
            plt.tight_layout()
            
            # Save snapshot
            snapshot_path = self.dashboard_dir / f"dashboard_snapshot_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png"
            plt.savefig(snapshot_path, dpi=300, bbox_inches='tight')
            plt.close()
            
            logger.info(f"Saved dashboard snapshot: {snapshot_path}")
            return snapshot_path
            
        except Exception as e:
            logger.error(f"Failed to save dashboard snapshot: {e}")
            return None
    
    def get_training_summary(self) -> Dict[str, Any]:
        """Get current training summary."""
        
        recent_data, _ = self.metrics_buffer.get_recent(10)
        
        summary = {
            'experiment_id': self.experiment_id,
            'current_epoch': self.current_epoch,
            'total_epochs': self.total_epochs,
            'training_time': str(datetime.now() - self.training_start_time),
            'best_loss': self.best_loss,
            'dashboard_active': self.dashboard_active,
            'total_alerts': len(self.alerts),
            'recent_alerts': list(self.alerts)[-5:] if self.alerts else []
        }
        
        if recent_data:
            summary['latest_metrics'] = recent_data[-1]
        
        return summary


def create_console_progress_bar(current: int, total: int, prefix: str = "", suffix: str = "", 
                               decimals: int = 1, length: int = 50, fill: str = '█') -> str:
    """Create a console progress bar."""
    percent = ("{0:." + str(decimals) + "f}").format(100 * (current / float(total)))
    filled_length = int(length * current // total)
    bar = fill * filled_length + '-' * (length - filled_length)
    return f'\r{prefix} |{bar}| {percent}% {suffix}'


class ConsoleLogger:
    """Enhanced console logging for training progress."""
    
    def __init__(self, experiment_id: str):
        self.experiment_id = experiment_id
        self.last_update = time.time()
        self.update_interval = 5.0  # Update every 5 seconds
    
    def log_training_progress(self, 
                            epoch: int, 
                            batch: int, 
                            total_batches: int,
                            losses: Dict[str, float],
                            throughput: Dict[str, float],
                            eta_minutes: int = None):
        """Log detailed training progress to console."""
        
        current_time = time.time()
        if current_time - self.last_update < self.update_interval:
            return
        
        self.last_update = current_time
        
        # Progress bar
        progress_bar = create_console_progress_bar(
            batch, total_batches, 
            prefix=f"Epoch {epoch}",
            suffix=f"Batch {batch}/{total_batches}"
        )
        
        # Metrics summary
        loss_str = f"Loss: {losses.get('total_loss', 0):.4f}"
        throughput_str = f"Throughput: {throughput.get('samples_per_second', 0):.1f} samples/sec"
        eta_str = f"ETA: {eta_minutes}min" if eta_minutes else ""
        
        # Print with colors (if terminal supports it)
        print(f"\033[2K{progress_bar}")  # Clear line and print progress
        print(f"\033[2K  📊 {loss_str} | 🚀 {throughput_str} | ⏱️ {eta_str}")
        print("\033[F\033[F", end="")  # Move cursor up 2 lines
    
    def log_epoch_complete(self, epoch: int, duration: float, losses: Dict[str, float]):
        """Log epoch completion."""
        print(f"\033[2K\n\033[2K")  # Clear and move down
        print(f"✅ Epoch {epoch} completed in {duration:.1f}s")
        print(f"   Final loss: {losses.get('total_loss', 0):.4f}")
        print("-" * 60)