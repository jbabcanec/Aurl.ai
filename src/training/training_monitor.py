"""
Training Monitor with Auto-Correction

This module monitors training progress and automatically corrects
issues like note pairing collapse, sparse generation, and other
musical quality problems.
"""

import torch
import torch.nn as nn
from typing import Dict, Optional, List, Tuple, Any
from dataclasses import dataclass, field
import numpy as np
from collections import deque
import json
from pathlib import Path
from datetime import datetime

from ..training.musical_grammar import MusicalGrammarLoss
from ..generation.token_validator import TokenValidator
from ..utils.base_logger import setup_logger

logger = setup_logger(__name__)


@dataclass
class MonitoringConfig:
    """Configuration for training monitor."""

    # Monitoring frequency
    check_frequency: int = 100  # Check every N batches
    detailed_check_frequency: int = 500  # Detailed check every N batches

    # Issue detection thresholds
    note_pairing_threshold: float = 0.7
    grammar_threshold: float = 0.7
    velocity_presence_threshold: float = 0.1  # Min fraction of velocity tokens
    repetition_threshold: float = 0.5  # Max allowed repetition

    # Auto-correction settings
    enable_auto_correction: bool = True
    correction_patience: int = 3  # Issues must persist for N checks
    rollback_on_critical: bool = True

    # Critical issue thresholds (immediate action)
    critical_note_pairing: float = 0.3
    critical_grammar: float = 0.4
    critical_loss_spike: float = 10.0  # Loss increase factor

    # Recovery actions
    learning_rate_reduction: float = 0.5
    guidance_boost: float = 1.5
    constraint_tightening: float = 1.2

    # Logging
    save_monitor_logs: bool = True
    log_dir: str = "outputs/monitoring"


class IssueTracker:
    """Tracks detected issues over time."""

    def __init__(self, patience: int = 3):
        self.patience = patience
        self.issues = {}  # issue_type -> count
        self.history = deque(maxlen=100)

    def report_issue(self, issue_type: str, severity: str = "warning"):
        """Report an issue detection."""
        if issue_type not in self.issues:
            self.issues[issue_type] = 0

        self.issues[issue_type] += 1

        self.history.append({
            'timestamp': datetime.now().isoformat(),
            'issue': issue_type,
            'severity': severity,
            'count': self.issues[issue_type]
        })

    def should_correct(self, issue_type: str) -> bool:
        """Check if issue persists enough to warrant correction."""
        return self.issues.get(issue_type, 0) >= self.patience

    def reset_issue(self, issue_type: str):
        """Reset counter for resolved issue."""
        if issue_type in self.issues:
            self.issues[issue_type] = 0


class TrainingMonitor(nn.Module):
    """
    Monitors training progress and automatically corrects issues.
    """

    def __init__(
        self,
        config: MonitoringConfig,
        model: nn.Module,
        optimizer: torch.optim.Optimizer
    ):
        super().__init__()
        self.config = config
        self.model = model
        self.optimizer = optimizer

        # Initialize components
        self.grammar_loss = MusicalGrammarLoss()
        self.token_validator = TokenValidator()

        # Tracking
        self.issue_tracker = IssueTracker(config.correction_patience)
        self.batch_count = 0
        self.correction_history = []
        self.performance_history = deque(maxlen=50)

        # State for rollback
        self.checkpoint_states = deque(maxlen=5)
        self.last_good_state = None

        # Setup logging
        if config.save_monitor_logs:
            self.log_path = Path(config.log_dir) / f"monitor_{datetime.now():%Y%m%d_%H%M%S}.json"
            self.log_path.parent.mkdir(parents=True, exist_ok=True)

    def check(
        self,
        generated_tokens: torch.Tensor,
        loss: float,
        batch_idx: int
    ) -> Dict[str, Any]:
        """
        Check training state and correct issues if needed.

        Args:
            generated_tokens: Recently generated token sequences
            loss: Current training loss
            batch_idx: Current batch index

        Returns:
            Dictionary of monitoring results and actions taken
        """
        self.batch_count = batch_idx
        results = {}

        # Quick checks every check_frequency batches
        if batch_idx % self.config.check_frequency == 0:
            results.update(self._quick_check(generated_tokens, loss))

        # Detailed checks less frequently
        if batch_idx % self.config.detailed_check_frequency == 0:
            results.update(self._detailed_check(generated_tokens))

        # Apply corrections if needed
        if self.config.enable_auto_correction:
            corrections = self._apply_corrections(results)
            results['corrections'] = corrections

        # Save monitoring log
        if self.config.save_monitor_logs:
            self._save_log(results)

        return results

    def _quick_check(
        self,
        generated_tokens: torch.Tensor,
        loss: float
    ) -> Dict:
        """Perform quick checks on training state."""
        results = {
            'batch': self.batch_count,
            'loss': loss,
            'checks': {}
        }

        # Check for loss spike
        if len(self.performance_history) > 0:
            avg_loss = np.mean([h['loss'] for h in self.performance_history])
            if loss > avg_loss * self.config.critical_loss_spike:
                self.issue_tracker.report_issue('loss_spike', 'critical')
                results['checks']['loss_spike'] = True
                logger.warning(f"Critical loss spike detected: {loss:.4f} (avg: {avg_loss:.4f})")

        # Check note pairing
        pairing_score = self._check_note_pairing(generated_tokens)
        results['checks']['note_pairing_score'] = pairing_score

        if pairing_score < self.config.critical_note_pairing:
            self.issue_tracker.report_issue('critical_note_pairing', 'critical')
            results['checks']['critical_note_pairing'] = True
            logger.error(f"Critical note pairing failure: {pairing_score:.3f}")

        elif pairing_score < self.config.note_pairing_threshold:
            self.issue_tracker.report_issue('note_pairing', 'warning')
            results['checks']['note_pairing_issue'] = True

        # Store in history
        self.performance_history.append({
            'batch': self.batch_count,
            'loss': loss,
            'note_pairing': pairing_score
        })

        return results

    def _detailed_check(self, generated_tokens: torch.Tensor) -> Dict:
        """Perform detailed analysis of generated sequences."""
        results = {'detailed_checks': {}}

        # Comprehensive grammar check
        grammar_results = self.grammar_loss.compute_grammar_score(generated_tokens)
        results['detailed_checks']['grammar'] = grammar_results

        if grammar_results['overall_score'] < self.config.critical_grammar:
            self.issue_tracker.report_issue('critical_grammar', 'critical')
            logger.error(f"Critical grammar failure: {grammar_results['overall_score']:.3f}")

        elif grammar_results['overall_score'] < self.config.grammar_threshold:
            self.issue_tracker.report_issue('grammar', 'warning')

        # Check velocity presence
        velocity_presence = self._check_velocity_presence(generated_tokens)
        results['detailed_checks']['velocity_presence'] = velocity_presence

        if velocity_presence < self.config.velocity_presence_threshold:
            self.issue_tracker.report_issue('missing_velocity', 'warning')
            logger.warning(f"Missing velocity tokens: {velocity_presence:.3f}")

        # Check repetition
        repetition_score = self._check_repetition(generated_tokens)
        results['detailed_checks']['repetition'] = repetition_score

        if repetition_score > self.config.repetition_threshold:
            self.issue_tracker.report_issue('excessive_repetition', 'warning')
            logger.warning(f"Excessive repetition detected: {repetition_score:.3f}")

        return results

    def _check_note_pairing(self, tokens: torch.Tensor) -> float:
        """Check note ON/OFF pairing quality."""
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        batch_scores = []
        for seq in tokens:
            note_on_count = ((seq >= 13) & (seq < 141)).sum().item()
            note_off_count = ((seq >= 141) & (seq < 269)).sum().item()

            if note_on_count == 0:
                score = 0.0
            else:
                score = min(note_off_count / note_on_count, 1.0)

            batch_scores.append(score)

        return np.mean(batch_scores)

    def _check_velocity_presence(self, tokens: torch.Tensor) -> float:
        """Check if velocity tokens are being generated."""
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        velocity_mask = ((tokens >= 369) & (tokens < 497))
        velocity_fraction = velocity_mask.float().mean().item()

        return velocity_fraction

    def _check_repetition(self, tokens: torch.Tensor) -> float:
        """Check for excessive repetition in sequences."""
        if tokens.dim() == 1:
            tokens = tokens.unsqueeze(0)

        repetition_scores = []
        for seq in tokens:
            # Check for repeated n-grams
            seq_list = seq.tolist()
            if len(seq_list) < 4:
                repetition_scores.append(0.0)
                continue

            # Count 4-gram repetitions
            ngrams = {}
            for i in range(len(seq_list) - 3):
                ngram = tuple(seq_list[i:i+4])
                ngrams[ngram] = ngrams.get(ngram, 0) + 1

            # Calculate repetition score
            max_repetitions = max(ngrams.values())
            repetition_score = (max_repetitions - 1) / len(seq_list)
            repetition_scores.append(repetition_score)

        return np.mean(repetition_scores)

    def _apply_corrections(self, check_results: Dict) -> List[str]:
        """Apply corrections based on detected issues."""
        corrections = []

        # Critical issues - immediate action
        if check_results.get('checks', {}).get('critical_note_pairing'):
            if self.config.rollback_on_critical and self.last_good_state:
                self._rollback_to_good_state()
                corrections.append("rollback_to_good_state")
                logger.info("Rolled back to last good state due to critical note pairing")

            # Reduce learning rate
            self._reduce_learning_rate()
            corrections.append("reduced_learning_rate")

        # Persistent issues - gradual correction
        if self.issue_tracker.should_correct('note_pairing'):
            # Increase guidance if available
            if hasattr(self.model, 'guidance_strength'):
                self.model.guidance_strength *= self.config.guidance_boost
                corrections.append(f"boosted_guidance_to_{self.model.guidance_strength:.3f}")

            self.issue_tracker.reset_issue('note_pairing')

        if self.issue_tracker.should_correct('missing_velocity'):
            # Could adjust loss weights if configurable
            corrections.append("flagged_missing_velocity")
            logger.info("Persistent missing velocity issue detected")

        if self.issue_tracker.should_correct('excessive_repetition'):
            # Could increase repetition penalty
            corrections.append("flagged_excessive_repetition")
            logger.info("Persistent repetition issue detected")

        # Save good state if no critical issues
        if not any('critical' in str(check_results.get('checks', {}).get(k, ''))
                  for k in check_results.get('checks', {})):
            if check_results.get('checks', {}).get('note_pairing_score', 0) > 0.8:
                self._save_good_state()

        return corrections

    def _reduce_learning_rate(self):
        """Reduce learning rate for all parameter groups."""
        for param_group in self.optimizer.param_groups:
            param_group['lr'] *= self.config.learning_rate_reduction
            logger.info(f"Reduced learning rate to {param_group['lr']:.6f}")

    def _save_good_state(self):
        """Save current model state as good checkpoint."""
        state = {
            'batch': self.batch_count,
            'model_state': self.model.state_dict().copy(),
            'optimizer_state': self.optimizer.state_dict().copy()
        }
        self.checkpoint_states.append(state)
        self.last_good_state = state

    def _rollback_to_good_state(self):
        """Rollback model to last good state."""
        if self.last_good_state:
            self.model.load_state_dict(self.last_good_state['model_state'])
            self.optimizer.load_state_dict(self.last_good_state['optimizer_state'])
            logger.info(f"Rolled back to batch {self.last_good_state['batch']}")

    def _save_log(self, results: Dict):
        """Save monitoring results to log file."""
        log_entry = {
            'timestamp': datetime.now().isoformat(),
            'batch': self.batch_count,
            'results': results,
            'issues': dict(self.issue_tracker.issues)
        }

        # Append to log file
        with open(self.log_path, 'a') as f:
            f.write(json.dumps(log_entry) + '\n')

    def get_summary(self) -> Dict:
        """Get summary of monitoring state."""
        return {
            'batch_count': self.batch_count,
            'active_issues': dict(self.issue_tracker.issues),
            'corrections_applied': len(self.correction_history),
            'checkpoints_saved': len(self.checkpoint_states),
            'recent_performance': {
                'avg_loss': np.mean([h['loss'] for h in self.performance_history])
                           if self.performance_history else 0,
                'avg_note_pairing': np.mean([h['note_pairing'] for h in self.performance_history])
                                   if self.performance_history else 0
            }
        }