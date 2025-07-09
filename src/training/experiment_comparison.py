"""
Automated Experiment Comparison for Aurl.ai Music Generation.

This module provides automated comparison of multiple training experiments:
- Performance metric comparison
- Hyperparameter impact analysis
- Best configuration identification
- Visualization generation
- Statistical significance testing
- Automated insights generation
"""

import json
import numpy as np
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

from src.utils.logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class ExperimentResult:
    """Container for experiment results."""
    
    experiment_id: str
    experiment_name: str
    hyperparameters: Dict[str, Any]
    final_metrics: Dict[str, float]
    best_metrics: Dict[str, float]
    training_time: float
    total_epochs: int
    convergence_epoch: Optional[int]
    data_stats: Dict[str, Any]
    
    def get_key_metrics(self) -> Dict[str, float]:
        """Get key metrics for comparison."""
        return {
            "final_loss": self.final_metrics.get("total_loss", float('inf')),
            "best_loss": self.best_metrics.get("total_loss", float('inf')),
            "final_val_loss": self.final_metrics.get("val_loss", float('inf')),
            "best_val_loss": self.best_metrics.get("val_loss", float('inf')),
            "convergence_speed": self.convergence_epoch / self.total_epochs if self.convergence_epoch else 1.0,
            "training_efficiency": self.total_epochs / (self.training_time / 3600) if self.training_time > 0 else 0
        }


class ExperimentComparator:
    """
    Automated experiment comparison and analysis.
    
    Features:
    - Multi-experiment performance comparison
    - Hyperparameter importance analysis
    - Statistical significance testing
    - Automated insight generation
    - Visualization creation
    """
    
    def __init__(self, experiments_dir: Path):
        """
        Initialize experiment comparator.
        
        Args:
            experiments_dir: Directory containing experiment results
        """
        
        self.experiments_dir = Path(experiments_dir)
        self.experiments: Dict[str, ExperimentResult] = {}
        
        # Load all experiments
        self._load_experiments()
        
        logger.info(f"Experiment comparator initialized with {len(self.experiments)} experiments")
    
    def _load_experiments(self):
        """Load all experiment results from directory."""
        
        for exp_file in self.experiments_dir.glob("**/experiment_summary.json"):
            try:
                with open(exp_file, 'r') as f:
                    data = json.load(f)
                
                # Extract experiment info
                exp_id = data.get("experiment_id", exp_file.parent.name)
                
                result = ExperimentResult(
                    experiment_id=exp_id,
                    experiment_name=data.get("experiment_name", exp_id),
                    hyperparameters=data.get("config", {}).get("training", {}),
                    final_metrics=data.get("performance_metrics", {}).get("final_losses", {}),
                    best_metrics=data.get("performance_metrics", {}).get("best_losses", {}),
                    training_time=data.get("duration_seconds", 0),
                    total_epochs=data.get("total_epochs", 0),
                    convergence_epoch=data.get("performance_metrics", {}).get("best_epoch"),
                    data_stats=data.get("data_usage_statistics", {})
                )
                
                self.experiments[exp_id] = result
                
            except Exception as e:
                logger.warning(f"Failed to load experiment from {exp_file}: {e}")
    
    def compare_experiments(self, 
                          experiment_ids: Optional[List[str]] = None,
                          metrics: Optional[List[str]] = None) -> Dict[str, Any]:
        """
        Compare multiple experiments.
        
        Args:
            experiment_ids: List of experiment IDs to compare (None for all)
            metrics: List of metrics to compare (None for default key metrics)
            
        Returns:
            Comparison results with rankings and insights
        """
        
        if experiment_ids is None:
            experiment_ids = list(self.experiments.keys())
        
        if not experiment_ids:
            logger.warning("No experiments to compare")
            return {}
        
        # Filter experiments
        selected_experiments = {
            exp_id: exp for exp_id, exp in self.experiments.items()
            if exp_id in experiment_ids
        }
        
        # Get metrics for comparison
        comparison_data = {}
        for exp_id, exp in selected_experiments.items():
            comparison_data[exp_id] = exp.get_key_metrics()
        
        # Rank experiments
        rankings = self._rank_experiments(comparison_data)
        
        # Analyze hyperparameter impact
        hp_analysis = self._analyze_hyperparameter_impact(selected_experiments)
        
        # Statistical analysis
        stat_analysis = self._statistical_analysis(comparison_data)
        
        # Generate insights
        insights = self._generate_insights(rankings, hp_analysis, stat_analysis)
        
        # Create comparison report
        report = {
            "timestamp": datetime.now().isoformat(),
            "num_experiments": len(selected_experiments),
            "experiment_ids": experiment_ids,
            "rankings": rankings,
            "hyperparameter_analysis": hp_analysis,
            "statistical_analysis": stat_analysis,
            "insights": insights,
            "detailed_metrics": comparison_data
        }
        
        return report
    
    def _rank_experiments(self, comparison_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Rank experiments by different criteria."""
        
        rankings = {
            "by_final_loss": [],
            "by_best_loss": [],
            "by_convergence_speed": [],
            "by_training_efficiency": [],
            "overall": []
        }
        
        # Rank by each metric
        for metric in ["final_loss", "best_loss"]:
            sorted_exps = sorted(
                comparison_data.items(),
                key=lambda x: x[1].get(metric, float('inf'))
            )
            rankings[f"by_{metric}"] = [(exp_id, data[metric]) for exp_id, data in sorted_exps]
        
        # Rank by convergence speed (lower is better)
        sorted_exps = sorted(
            comparison_data.items(),
            key=lambda x: x[1].get("convergence_speed", 1.0)
        )
        rankings["by_convergence_speed"] = [(exp_id, data["convergence_speed"]) for exp_id, data in sorted_exps]
        
        # Rank by training efficiency (higher is better)
        sorted_exps = sorted(
            comparison_data.items(),
            key=lambda x: x[1].get("training_efficiency", 0),
            reverse=True
        )
        rankings["by_training_efficiency"] = [(exp_id, data["training_efficiency"]) for exp_id, data in sorted_exps]
        
        # Calculate overall ranking (weighted combination)
        overall_scores = {}
        for exp_id, data in comparison_data.items():
            # Normalize metrics for fair comparison
            loss_score = 1.0 / (1.0 + data.get("best_loss", 1.0))
            convergence_score = 1.0 - data.get("convergence_speed", 1.0)
            efficiency_score = min(1.0, data.get("training_efficiency", 0) / 100)
            
            # Weighted combination
            overall_scores[exp_id] = (
                0.5 * loss_score +
                0.3 * convergence_score +
                0.2 * efficiency_score
            )
        
        sorted_overall = sorted(overall_scores.items(), key=lambda x: x[1], reverse=True)
        rankings["overall"] = sorted_overall
        
        return rankings
    
    def _analyze_hyperparameter_impact(self, experiments: Dict[str, ExperimentResult]) -> Dict[str, Any]:
        """Analyze impact of hyperparameters on performance."""
        
        if len(experiments) < 3:
            return {"message": "Not enough experiments for hyperparameter analysis"}
        
        # Collect hyperparameter values and corresponding performance
        hp_performance = {}
        
        for exp_id, exp in experiments.items():
            for hp_name, hp_value in exp.hyperparameters.items():
                if hp_name not in hp_performance:
                    hp_performance[hp_name] = {"values": [], "losses": []}
                
                hp_performance[hp_name]["values"].append(hp_value)
                hp_performance[hp_name]["losses"].append(exp.best_metrics.get("total_loss", float('inf')))
        
        # Analyze correlation for numeric hyperparameters
        hp_impact = {}
        
        for hp_name, data in hp_performance.items():
            values = data["values"]
            losses = data["losses"]
            
            # Skip if not numeric or not enough variation
            if not all(isinstance(v, (int, float)) for v in values):
                continue
            
            if len(set(values)) < 2:
                continue
            
            # Calculate correlation
            correlation, p_value = stats.pearsonr(values, losses)
            
            hp_impact[hp_name] = {
                "correlation": float(correlation),
                "p_value": float(p_value),
                "significant": p_value < 0.05,
                "impact": "negative" if correlation > 0 else "positive",
                "strength": self._classify_correlation_strength(abs(correlation))
            }
        
        return hp_impact
    
    def _classify_correlation_strength(self, correlation: float) -> str:
        """Classify correlation strength."""
        if correlation > 0.7:
            return "strong"
        elif correlation > 0.4:
            return "moderate"
        elif correlation > 0.2:
            return "weak"
        else:
            return "negligible"
    
    def _statistical_analysis(self, comparison_data: Dict[str, Dict[str, float]]) -> Dict[str, Any]:
        """Perform statistical analysis on experiment results."""
        
        # Extract loss values
        final_losses = [data["final_loss"] for data in comparison_data.values() 
                       if data["final_loss"] != float('inf')]
        best_losses = [data["best_loss"] for data in comparison_data.values()
                      if data["best_loss"] != float('inf')]
        
        if len(final_losses) < 2:
            return {"message": "Not enough data for statistical analysis"}
        
        analysis = {
            "final_loss_stats": {
                "mean": float(np.mean(final_losses)),
                "std": float(np.std(final_losses)),
                "min": float(np.min(final_losses)),
                "max": float(np.max(final_losses)),
                "median": float(np.median(final_losses))
            },
            "best_loss_stats": {
                "mean": float(np.mean(best_losses)),
                "std": float(np.std(best_losses)),
                "min": float(np.min(best_losses)),
                "max": float(np.max(best_losses)),
                "median": float(np.median(best_losses))
            }
        }
        
        # Perform ANOVA if more than 2 experiments
        if len(comparison_data) > 2:
            # Group experiments by key hyperparameters
            # This is simplified - in practice, you'd group by specific hyperparameters
            f_stat, p_value = stats.f_oneway(*[
                [data["best_loss"]] for data in comparison_data.values()
                if data["best_loss"] != float('inf')
            ])
            
            analysis["anova"] = {
                "f_statistic": float(f_stat) if not np.isnan(f_stat) else None,
                "p_value": float(p_value) if not np.isnan(p_value) else None,
                "significant_difference": p_value < 0.05 if not np.isnan(p_value) else False
            }
        
        return analysis
    
    def _generate_insights(self, 
                         rankings: Dict[str, Any],
                         hp_analysis: Dict[str, Any],
                         stat_analysis: Dict[str, Any]) -> List[str]:
        """Generate automated insights from analysis."""
        
        insights = []
        
        # Best performing experiment
        if rankings["overall"]:
            best_exp_id, best_score = rankings["overall"][0]
            insights.append(f"Best overall experiment: {best_exp_id} (score: {best_score:.3f})")
        
        # Loss insights
        if "by_best_loss" in rankings and rankings["by_best_loss"]:
            best_loss_exp, best_loss = rankings["by_best_loss"][0]
            insights.append(f"Lowest loss achieved: {best_loss:.4f} by {best_loss_exp}")
        
        # Convergence insights
        if "by_convergence_speed" in rankings and rankings["by_convergence_speed"]:
            fastest_exp, convergence_ratio = rankings["by_convergence_speed"][0]
            insights.append(f"Fastest convergence: {fastest_exp} (converged at {convergence_ratio*100:.1f}% of training)")
        
        # Hyperparameter insights
        for hp_name, impact in hp_analysis.items():
            if isinstance(impact, dict) and impact.get("significant"):
                direction = "increases" if impact["impact"] == "negative" else "decreases"
                insights.append(
                    f"Hyperparameter '{hp_name}' has {impact['strength']} impact: "
                    f"higher values {direction} loss (p={impact['p_value']:.3f})"
                )
        
        # Statistical insights
        if "final_loss_stats" in stat_analysis:
            stats_data = stat_analysis["final_loss_stats"]
            insights.append(
                f"Loss statistics: mean={stats_data['mean']:.4f}, "
                f"std={stats_data['std']:.4f}, range=[{stats_data['min']:.4f}, {stats_data['max']:.4f}]"
            )
        
        # Significant difference insight
        if "anova" in stat_analysis and stat_analysis["anova"].get("significant_difference"):
            insights.append("Statistically significant differences found between experiments (ANOVA p<0.05)")
        
        return insights
    
    def visualize_comparison(self,
                           comparison_report: Dict[str, Any],
                           save_dir: Optional[Path] = None) -> Dict[str, Path]:
        """
        Create visualizations for experiment comparison.
        
        Args:
            comparison_report: Report from compare_experiments()
            save_dir: Directory to save plots
            
        Returns:
            Dictionary of visualization type -> file path
        """
        
        if save_dir:
            save_dir = Path(save_dir)
            save_dir.mkdir(parents=True, exist_ok=True)
        
        viz_paths = {}
        
        # Set style
        plt.style.use('seaborn-v0_8')
        
        # 1. Loss comparison bar plot
        fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))
        
        # Extract data
        exp_ids = list(comparison_report["detailed_metrics"].keys())
        final_losses = [comparison_report["detailed_metrics"][exp_id]["final_loss"] 
                       for exp_id in exp_ids]
        best_losses = [comparison_report["detailed_metrics"][exp_id]["best_loss"] 
                      for exp_id in exp_ids]
        
        # Plot
        x = np.arange(len(exp_ids))
        width = 0.35
        
        ax1.bar(x - width/2, final_losses, width, label='Final Loss', alpha=0.8)
        ax1.bar(x + width/2, best_losses, width, label='Best Loss', alpha=0.8)
        ax1.set_xlabel('Experiment')
        ax1.set_ylabel('Loss')
        ax1.set_title('Loss Comparison Across Experiments')
        ax1.set_xticks(x)
        ax1.set_xticklabels([exp_id[:8] for exp_id in exp_ids], rotation=45)
        ax1.legend()
        ax1.grid(True, alpha=0.3)
        
        # 2. Convergence speed plot
        convergence_speeds = [comparison_report["detailed_metrics"][exp_id]["convergence_speed"] 
                            for exp_id in exp_ids]
        
        ax2.bar(x, convergence_speeds, alpha=0.8, color='green')
        ax2.set_xlabel('Experiment')
        ax2.set_ylabel('Convergence Ratio')
        ax2.set_title('Convergence Speed Comparison')
        ax2.set_xticks(x)
        ax2.set_xticklabels([exp_id[:8] for exp_id in exp_ids], rotation=45)
        ax2.grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_dir:
            loss_plot_path = save_dir / "loss_comparison.png"
            plt.savefig(loss_plot_path, dpi=150, bbox_inches='tight')
            viz_paths["loss_comparison"] = loss_plot_path
        
        plt.close()
        
        # 3. Hyperparameter impact heatmap (if available)
        hp_analysis = comparison_report.get("hyperparameter_analysis", {})
        
        if hp_analysis and isinstance(list(hp_analysis.values())[0], dict):
            # Create correlation matrix
            hp_names = []
            correlations = []
            
            for hp_name, impact in hp_analysis.items():
                if isinstance(impact, dict) and "correlation" in impact:
                    hp_names.append(hp_name)
                    correlations.append(impact["correlation"])
            
            if hp_names:
                fig, ax = plt.subplots(figsize=(8, 6))
                
                # Create bar plot for correlations
                y_pos = np.arange(len(hp_names))
                colors = ['red' if c > 0 else 'green' for c in correlations]
                
                ax.barh(y_pos, correlations, color=colors, alpha=0.7)
                ax.set_yticks(y_pos)
                ax.set_yticklabels(hp_names)
                ax.set_xlabel('Correlation with Loss')
                ax.set_title('Hyperparameter Impact on Performance')
                ax.grid(True, alpha=0.3)
                ax.axvline(x=0, color='black', linestyle='-', linewidth=0.5)
                
                # Add significance markers
                for i, (hp_name, impact) in enumerate(hp_analysis.items()):
                    if isinstance(impact, dict) and impact.get("significant"):
                        ax.text(correlations[i], i, '*', ha='center', va='center', 
                               fontsize=16, fontweight='bold')
                
                plt.tight_layout()
                
                if save_dir:
                    hp_plot_path = save_dir / "hyperparameter_impact.png"
                    plt.savefig(hp_plot_path, dpi=150, bbox_inches='tight')
                    viz_paths["hyperparameter_impact"] = hp_plot_path
                
                plt.close()
        
        # 4. Overall ranking plot
        overall_rankings = comparison_report["rankings"]["overall"]
        
        if overall_rankings:
            fig, ax = plt.subplots(figsize=(8, 6))
            
            exp_names = [exp_id for exp_id, _ in overall_rankings]
            scores = [score for _, score in overall_rankings]
            
            y_pos = np.arange(len(exp_names))
            
            ax.barh(y_pos, scores, alpha=0.8, color='skyblue')
            ax.set_yticks(y_pos)
            ax.set_yticklabels(exp_names)
            ax.set_xlabel('Overall Score')
            ax.set_title('Experiment Overall Ranking')
            ax.grid(True, alpha=0.3)
            
            # Add score values
            for i, score in enumerate(scores):
                ax.text(score, i, f'{score:.3f}', ha='left', va='center', 
                       fontweight='bold', fontsize=10)
            
            plt.tight_layout()
            
            if save_dir:
                ranking_plot_path = save_dir / "overall_ranking.png"
                plt.savefig(ranking_plot_path, dpi=150, bbox_inches='tight')
                viz_paths["overall_ranking"] = ranking_plot_path
            
            plt.close()
        
        logger.info(f"Created {len(viz_paths)} comparison visualizations")
        
        return viz_paths
    
    def save_comparison_report(self, 
                             comparison_report: Dict[str, Any],
                             save_path: Path):
        """Save comparison report to file."""
        
        save_path = Path(save_path)
        save_path.parent.mkdir(parents=True, exist_ok=True)
        
        with open(save_path, 'w') as f:
            json.dump(comparison_report, f, indent=2)
        
        logger.info(f"Comparison report saved to {save_path}")
    
    def find_best_configuration(self) -> Optional[ExperimentResult]:
        """Find the best configuration across all experiments."""
        
        if not self.experiments:
            return None
        
        # Compare all experiments
        comparison = self.compare_experiments()
        
        if comparison and "rankings" in comparison and "overall" in comparison["rankings"]:
            best_exp_id, _ = comparison["rankings"]["overall"][0]
            return self.experiments.get(best_exp_id)
        
        return None