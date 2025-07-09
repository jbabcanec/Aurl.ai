"""
Create Combined Visual Dashboard for Logging System Test Results.

This script creates comprehensive visualizations combining:
- Training progress and loss curves
- Musical quality metrics over time
- Anomaly detection timeline
- Data usage statistics
- Memory and throughput monitoring
"""

import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from pathlib import Path
import pandas as pd
from datetime import datetime
import warnings
warnings.filterwarnings('ignore')

# Set style for professional plots
plt.style.use('seaborn-v0_8')
sns.set_palette("husl")

def load_test_data(base_dir: Path):
    """Load all test data from the integration test."""
    
    # Find the experiment directory
    exp_dirs = list(base_dir.glob("**/full_integration_demo_*"))
    if not exp_dirs:
        raise FileNotFoundError("No integration test results found")
    
    exp_dir = exp_dirs[0]
    
    data = {}
    
    # Load batch metrics
    batch_file = exp_dir / "batch_metrics.jsonl"
    if batch_file.exists():
        batch_data = []
        with open(batch_file, 'r') as f:
            for line in f:
                batch_data.append(json.loads(line))
        data['batches'] = batch_data
    
    # Load epoch summaries
    epoch_file = exp_dir / "epoch_summaries.jsonl"
    if epoch_file.exists():
        epoch_data = []
        with open(epoch_file, 'r') as f:
            for line in f:
                epoch_data.append(json.loads(line))
        data['epochs'] = epoch_data
    
    # Load musical quality report
    quality_file = base_dir / "musical_quality_report.json"
    if quality_file.exists():
        with open(quality_file, 'r') as f:
            data['quality'] = json.load(f)
    
    # Load anomaly report
    anomaly_file = base_dir / "anomaly_report.json"
    if anomaly_file.exists():
        with open(anomaly_file, 'r') as f:
            data['anomalies'] = json.load(f)
    
    return data, exp_dir

def create_combined_dashboard(data, save_dir: Path):
    """Create comprehensive combined dashboard."""
    
    # Create figure with subplots
    fig = plt.figure(figsize=(20, 16))
    
    # Define grid layout
    gs = fig.add_gridspec(4, 4, hspace=0.3, wspace=0.3)
    
    # 1. Training Loss Curves (top left, 2x2)
    ax1 = fig.add_subplot(gs[0:2, 0:2])
    plot_training_losses(data, ax1)
    
    # 2. Musical Quality Timeline (top right, 2x1)
    ax2 = fig.add_subplot(gs[0, 2:4])
    plot_musical_quality(data, ax2)
    
    # 3. Anomaly Detection Timeline (middle right, 1x2)
    ax3 = fig.add_subplot(gs[1, 2:4])
    plot_anomaly_timeline(data, ax3)
    
    # 4. Memory Usage (bottom left, 1x2)
    ax4 = fig.add_subplot(gs[2, 0:2])
    plot_memory_usage(data, ax4)
    
    # 5. Throughput Metrics (bottom middle, 1x2)
    ax5 = fig.add_subplot(gs[2, 2:4])
    plot_throughput(data, ax5)
    
    # 6. Data Usage Statistics (bottom, 1x4)
    ax6 = fig.add_subplot(gs[3, 0:2])
    plot_data_usage_stats(data, ax6)
    
    # 7. Quality Metrics Breakdown (bottom right, 1x2)
    ax7 = fig.add_subplot(gs[3, 2:4])
    plot_quality_breakdown(data, ax7)
    
    # Main title
    fig.suptitle('🎼 Aurl.ai Logging System - Complete Training Dashboard', 
                 fontsize=20, fontweight='bold', y=0.98)
    
    # Add timestamp
    timestamp = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
    fig.text(0.99, 0.01, f'Generated: {timestamp}', ha='right', va='bottom', 
             fontsize=10, style='italic', alpha=0.7)
    
    # Save the dashboard
    dashboard_path = save_dir / "combined_training_dashboard.png"
    plt.savefig(dashboard_path, dpi=300, bbox_inches='tight', 
                facecolor='white', edgecolor='none')
    
    return dashboard_path

def plot_training_losses(data, ax):
    """Plot training loss curves."""
    if 'batches' not in data:
        ax.text(0.5, 0.5, 'No batch data available', ha='center', va='center')
        ax.set_title('Training Loss Curves')
        return
    
    batches = data['batches']
    
    # Extract loss data
    batch_nums = []
    recon_losses = []
    kl_losses = []
    adv_losses = []
    total_losses = []
    
    for i, batch in enumerate(batches):
        batch_nums.append(i)
        losses = batch.get('losses', {})
        recon_losses.append(losses.get('reconstruction', 0))
        kl_losses.append(losses.get('kl_divergence', 0))
        adv_losses.append(losses.get('adversarial', 0))
        total_losses.append(losses.get('total', 0))
    
    # Plot loss curves
    ax.plot(batch_nums, recon_losses, label='Reconstruction', linewidth=2, alpha=0.8)
    ax.plot(batch_nums, kl_losses, label='KL Divergence', linewidth=2, alpha=0.8)
    ax.plot(batch_nums, adv_losses, label='Adversarial', linewidth=2, alpha=0.8)
    ax.plot(batch_nums, total_losses, label='Total Loss', linewidth=3, color='black', alpha=0.9)
    
    ax.set_xlabel('Batch')
    ax.set_ylabel('Loss')
    ax.set_title('Training Loss Progression')
    ax.legend()
    ax.grid(True, alpha=0.3)

def plot_musical_quality(data, ax):
    """Plot musical quality metrics over time."""
    if 'quality' not in data:
        ax.text(0.5, 0.5, 'No quality data available', ha='center', va='center')
        ax.set_title('Musical Quality')
        return
    
    quality_data = data['quality']
    epoch_summaries = quality_data.get('epoch_summaries', {})
    
    epochs = []
    qualities = []
    
    for epoch_str, summary in epoch_summaries.items():
        epochs.append(int(epoch_str))
        qualities.append(summary['average_quality'])
    
    if epochs:
        ax.plot(epochs, qualities, 'o-', linewidth=3, markersize=8, color='green')
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Quality Score')
        ax.set_title('Musical Quality Over Time')
        ax.grid(True, alpha=0.3)
        ax.set_ylim(0, 1)
        
        # Add trend line
        if len(epochs) > 1:
            z = np.polyfit(epochs, qualities, 1)
            p = np.poly1d(z)
            ax.plot(epochs, p(epochs), "--", alpha=0.7, color='red')
            
            trend = "↗️ Improving" if z[0] > 0 else "↘️ Declining"
            ax.text(0.95, 0.95, trend, transform=ax.transAxes, ha='right', va='top',
                   bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgray', alpha=0.7))

def plot_anomaly_timeline(data, ax):
    """Plot anomaly detection timeline."""
    if 'anomalies' not in data:
        ax.text(0.5, 0.5, 'No anomaly data available', ha='center', va='center')
        ax.set_title('Anomaly Detection')
        return
    
    anomaly_data = data['anomalies']
    summary = anomaly_data.get('summary', {})
    
    # Plot anomaly counts by severity
    severity_counts = summary.get('severity_distribution', {})
    
    if severity_counts:
        severities = list(severity_counts.keys())
        counts = list(severity_counts.values())
        colors = {'info': 'blue', 'warning': 'orange', 'critical': 'red', 'fatal': 'darkred'}
        
        bars = ax.bar(severities, counts, color=[colors.get(s, 'gray') for s in severities])
        ax.set_ylabel('Count')
        ax.set_title('Anomalies Detected by Severity')
        
        # Add count labels on bars
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{int(height)}', ha='center', va='bottom')
    
    total_anomalies = summary.get('total_anomalies', 0)
    ax.text(0.95, 0.95, f'Total: {total_anomalies}', transform=ax.transAxes, 
           ha='right', va='top', fontweight='bold')

def plot_memory_usage(data, ax):
    """Plot memory usage over time."""
    if 'batches' not in data:
        ax.text(0.5, 0.5, 'No memory data available', ha='center', va='center')
        ax.set_title('Memory Usage')
        return
    
    batches = data['batches']
    
    batch_nums = []
    gpu_memory = []
    cpu_memory = []
    
    for i, batch in enumerate(batches):
        batch_nums.append(i)
        memory = batch.get('memory_metrics', {})
        gpu_memory.append(memory.get('gpu_allocated', 0))
        cpu_memory.append(memory.get('cpu_rss', 0))
    
    if batch_nums:
        ax.plot(batch_nums, gpu_memory, label='GPU Memory (GB)', linewidth=2)
        ax.plot(batch_nums, cpu_memory, label='CPU Memory (GB)', linewidth=2)
        ax.set_xlabel('Batch')
        ax.set_ylabel('Memory (GB)')
        ax.set_title('Memory Usage Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)

def plot_throughput(data, ax):
    """Plot throughput metrics."""
    if 'batches' not in data:
        ax.text(0.5, 0.5, 'No throughput data available', ha='center', va='center')
        ax.set_title('Throughput')
        return
    
    batches = data['batches']
    
    batch_nums = []
    samples_per_sec = []
    
    for i, batch in enumerate(batches):
        batch_nums.append(i)
        throughput = batch.get('throughput_metrics', {})
        samples_per_sec.append(throughput.get('samples_per_second', 0))
    
    if batch_nums:
        ax.plot(batch_nums, samples_per_sec, linewidth=2, color='purple')
        ax.set_xlabel('Batch')
        ax.set_ylabel('Samples/Second')
        ax.set_title('Training Throughput')
        ax.grid(True, alpha=0.3)
        
        # Add average line
        if samples_per_sec:
            avg_throughput = np.mean(samples_per_sec)
            ax.axhline(y=avg_throughput, color='red', linestyle='--', alpha=0.7)
            ax.text(0.95, 0.95, f'Avg: {avg_throughput:.1f}', transform=ax.transAxes,
                   ha='right', va='top', bbox=dict(boxstyle="round,pad=0.3", 
                   facecolor='lightcoral', alpha=0.7))

def plot_data_usage_stats(data, ax):
    """Plot data usage statistics."""
    if 'epochs' not in data:
        ax.text(0.5, 0.5, 'No epoch data available', ha='center', va='center')
        ax.set_title('Data Usage')
        return
    
    epochs = data['epochs']
    
    epoch_nums = []
    files_processed = []
    files_augmented = []
    
    for epoch in epochs:
        epoch_nums.append(epoch['epoch'])
        files_processed.append(epoch.get('files_processed', 0))
        files_augmented.append(epoch.get('files_augmented', 0))
    
    if epoch_nums:
        width = 0.35
        x = np.arange(len(epoch_nums))
        
        ax.bar(x - width/2, files_processed, width, label='Total Files', alpha=0.8)
        ax.bar(x + width/2, files_augmented, width, label='Augmented Files', alpha=0.8)
        
        ax.set_xlabel('Epoch')
        ax.set_ylabel('Number of Files')
        ax.set_title('Data Processing Statistics')
        ax.set_xticks(x)
        ax.set_xticklabels(epoch_nums)
        ax.legend()

def plot_quality_breakdown(data, ax):
    """Plot breakdown of musical quality metrics."""
    if 'quality' not in data or 'recent_samples' not in data['quality']:
        ax.text(0.5, 0.5, 'No quality breakdown available', ha='center', va='center')
        ax.set_title('Quality Metrics')
        return
    
    recent_samples = data['quality']['recent_samples']
    
    if recent_samples:
        # Get the latest sample
        latest_sample = recent_samples[-1]
        
        # Extract metrics (excluding overall_quality and pitch_range)
        metrics = {
            'Rhythm Consistency': latest_sample.get('rhythm_consistency', 0),
            'Rhythm Diversity': latest_sample.get('rhythm_diversity', 0),
            'Harmonic Coherence': latest_sample.get('harmonic_coherence', 0),
            'Melodic Contour': latest_sample.get('melodic_contour', 0),
            'Dynamic Range': latest_sample.get('dynamic_range', 0),
            'Repetition Score': latest_sample.get('repetition_score', 0),
            'Structural Coherence': latest_sample.get('structural_coherence', 0)
        }
        
        # Create radar chart
        angles = np.linspace(0, 2 * np.pi, len(metrics), endpoint=False).tolist()
        angles += angles[:1]  # Complete the circle
        
        values = list(metrics.values())
        values += values[:1]  # Complete the circle
        
        ax.plot(angles, values, 'o-', linewidth=2, color='green')
        ax.fill(angles, values, alpha=0.25, color='green')
        ax.set_xticks(angles[:-1])
        ax.set_xticklabels(metrics.keys(), fontsize=8)
        ax.set_ylim(0, 1)
        ax.set_title('Latest Musical Quality Breakdown')
        ax.grid(True)

def create_summary_dashboard(data, save_dir: Path):
    """Create a summary dashboard with key metrics."""
    
    fig, ((ax1, ax2), (ax3, ax4)) = plt.subplots(2, 2, figsize=(16, 12))
    
    # 1. Loss progression summary
    if 'batches' in data:
        batches = data['batches']
        total_losses = [b.get('losses', {}).get('total', 0) for b in batches]
        epochs_approx = [i // 20 + 1 for i in range(len(total_losses))]  # 20 batches per epoch
        
        ax1.plot(total_losses, linewidth=3, color='darkblue')
        ax1.set_title('Training Loss Convergence', fontsize=14, fontweight='bold')
        ax1.set_xlabel('Batch')
        ax1.set_ylabel('Total Loss')
        ax1.grid(True, alpha=0.3)
        
        # Add final loss
        if total_losses:
            final_loss = total_losses[-1]
            ax1.text(0.95, 0.95, f'Final: {final_loss:.3f}', transform=ax1.transAxes,
                    ha='right', va='top', fontweight='bold', fontsize=12,
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightblue'))
    
    # 2. Quality progression
    if 'quality' in data:
        epoch_summaries = data['quality'].get('epoch_summaries', {})
        epochs = sorted([int(k) for k in epoch_summaries.keys()])
        qualities = [epoch_summaries[str(e)]['average_quality'] for e in epochs]
        
        ax2.plot(epochs, qualities, 'o-', linewidth=3, markersize=10, color='green')
        ax2.set_title('Musical Quality Evolution', fontsize=14, fontweight='bold')
        ax2.set_xlabel('Epoch')
        ax2.set_ylabel('Quality Score')
        ax2.set_ylim(0, 1)
        ax2.grid(True, alpha=0.3)
        
        if qualities:
            trend_info = data['quality'].get('trend_analysis', {})
            trend_direction = "↗️" if trend_info.get('improving', False) else "↘️"
            ax2.text(0.95, 0.95, f'{trend_direction} {trend_info.get("trend", "unknown")}',
                    transform=ax2.transAxes, ha='right', va='top', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lightgreen'))
    
    # 3. Anomaly summary
    if 'anomalies' in data:
        anomaly_summary = data['anomalies'].get('summary', {})
        severity_dist = anomaly_summary.get('severity_distribution', {})
        
        if severity_dist:
            colors = {'info': 'lightblue', 'warning': 'orange', 'critical': 'red'}
            labels = list(severity_dist.keys())
            sizes = list(severity_dist.values())
            
            wedges, texts, autotexts = ax3.pie(sizes, labels=labels, autopct='%1.0f',
                                              colors=[colors.get(l, 'gray') for l in labels])
            ax3.set_title('Anomaly Detection Summary', fontsize=14, fontweight='bold')
            
            total_anomalies = anomaly_summary.get('total_anomalies', 0)
            ax3.text(0, -1.3, f'Total Anomalies: {total_anomalies}', ha='center', 
                    fontweight='bold', fontsize=12)
    
    # 4. Performance summary
    if 'batches' in data:
        batches = data['batches']
        throughputs = [b.get('throughput_metrics', {}).get('samples_per_second', 0) for b in batches]
        
        if throughputs:
            ax4.hist(throughputs, bins=20, alpha=0.7, color='purple', edgecolor='black')
            ax4.set_title('Throughput Distribution', fontsize=14, fontweight='bold')
            ax4.set_xlabel('Samples per Second')
            ax4.set_ylabel('Frequency')
            
            avg_throughput = np.mean(throughputs)
            ax4.axvline(avg_throughput, color='red', linestyle='--', linewidth=2)
            ax4.text(0.95, 0.95, f'Avg: {avg_throughput:.1f} samples/sec',
                    transform=ax4.transAxes, ha='right', va='top', fontweight='bold',
                    bbox=dict(boxstyle="round,pad=0.3", facecolor='lavender'))
    
    plt.suptitle('🎼 Aurl.ai Logging System - Test Summary Dashboard', 
                 fontsize=18, fontweight='bold', y=0.98)
    plt.tight_layout()
    
    summary_path = save_dir / "summary_dashboard.png"
    plt.savefig(summary_path, dpi=300, bbox_inches='tight',
                facecolor='white', edgecolor='none')
    
    return summary_path

def main():
    """Create combined dashboards from test results."""
    
    # Find test results
    base_dir = Path("outputs/logging_system/full_integration_test/experiments")
    
    if not base_dir.exists():
        print("❌ No integration test results found")
        return
    
    print("📊 Creating Combined Visual Dashboards...")
    
    try:
        # Load test data
        data, exp_dir = load_test_data(base_dir)
        print(f"✅ Loaded test data from {exp_dir.name}")
        
        # Create save directory
        dashboard_dir = base_dir / "combined_dashboards"
        dashboard_dir.mkdir(exist_ok=True)
        
        # Create comprehensive dashboard
        print("🎨 Creating comprehensive dashboard...")
        comprehensive_path = create_combined_dashboard(data, dashboard_dir)
        print(f"✅ Comprehensive dashboard: {comprehensive_path}")
        
        # Create summary dashboard
        print("📋 Creating summary dashboard...")
        summary_path = create_summary_dashboard(data, dashboard_dir)
        print(f"✅ Summary dashboard: {summary_path}")
        
        print("\n🎉 Visual Dashboards Created Successfully!")
        print(f"📁 Location: {dashboard_dir}")
        print(f"📊 Files created:")
        print(f"  - {comprehensive_path.name}")
        print(f"  - {summary_path.name}")
        
        return True
        
    except Exception as e:
        print(f"❌ Failed to create dashboards: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    main()