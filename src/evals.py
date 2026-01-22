import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, roc_auc_score, confusion_matrix
from scipy import stats
import maptlotlib.pyplot as plt

class ModelEvaluator:
    def __init__(self, config):
        self.config = config
        
    def calculate_metrics(self, y_true, y_pred, y_pred_proba=None):
        metrics = {
            'accuracy': accuracy_score(y_true, y_pred),
            'precision': precision_score(y_true, y_pred, average='macro', zero_division=0),
            'recall': recall_score(y_true, y_pred, average='macro', zero_division=0),
            'f1': f1_score(y_true, y_pred, average='macro', zero_division=0)
        }
        if y_pred_proba is not None and len(np.unique(y_true)) > 1:
            try:
                metrics['roc_auc'] = roc_auc_score(y_true, y_pred_proba[:, 1])
            except:
                metrics['roc_auc'] = 0.5
        cm = confusion_matrix(y_true, y_pred)
        metrics['confusion_matrix'] = cm
        return metrics
    
    def calculate_performance_drift(self, reference_metrics, current_metrics):
        drift_scores = {}
        for metric in ['accuracy', 'precision', 'recall', 'f1']:
            if metric in reference_metrics and metric in current_metrics:
                drift = reference_metrics[metric] - current_metrics[metric]
                drift_scores[f'{metric}_drift'] = drift
                drift_scores[f'{metric}_relative_drift'] = drift / reference_metrics[metric] if reference_metrics[metric] != 0 else 0
        return drift_scores
    
    def statistical_significance_test(self, reference_scores, current_scores):
        if len(reference_scores) > 1 and len(current_scores) > 1:
            t_stat, p_value = stats.ttest_ind(reference_scores, current_scores)
            return t_stat, p_value
        return None, None
    
    def detect_performance_degradation(self, metrics_history, window_size=5):
        if len(metrics_history) < window_size * 2:
            return False, {}
        recent_metrics = metrics_history[-window_size:]
        older_metrics = metrics_history[-window_size*2:-window_size]
        degradation = {}
        for metric in ['accuracy', 'f1']:
            if metric in recent_metrics[0] and metric in older_metrics[0]:
                recent_avg = np.mean([m[metric] for m in recent_metrics])
                older_avg = np.mean([m[metric] for m in older_metrics])
                degradation[f'{metric}_degradation'] = older_avg - recent_avg
        significant_degradation = any(d > 0.05 for d in degradation.values())
        return significant_degradation, degradation
    
    def generate_evaluation_report(self, metrics_history, drift_history):
        report = {
            'overview': {
                'total_windows': len(metrics_history),
                'windows_with_drift': len([d for d in drift_history if d['drift_features'] > 0]),
                'average_accuracy': np.mean([m.get('accuracy', 0) for m in metrics_history]),
                'average_f1': np.mean([m.get('f1', 0) for m in metrics_history])
            },
            'performance_trend': self._calculate_performance_trend(metrics_history),
            'drift_correlation': self._calculate_drift_correlation(metrics_history, drift_history)
        }
        return report
    
    def _calculate_performance_trend(self, metrics_history):
        if len(metrics_history) < 2:
            return {'trend': 'insufficient_data', 'slope': 0}
        windows = np.arange(len(metrics_history))
        accuracies = [m.get('accuracy', 0) for m in metrics_history]
        slope, intercept = np.polyfit(windows, accuracies, 1)
        if slope < -0.01:
            trend = 'declining'
        elif slope > 0.01:
            trend = 'improving'
        else:
            trend = 'stable'
        return {'trend': trend, 'slope': slope, 'intercept': intercept}
    
    def _calculate_drift_correlation(self, metrics_history, drift_history):
        if len(metrics_history) != len(drift_history):
            return {'correlation': 0, 'p_value': 1}
        drift_counts = [d['drift_features'] for d in drift_history]
        accuracies = [m.get('accuracy', 0) for m in metrics_history]
        if len(set(drift_counts)) > 1 and len(set(accuracies)) > 1:
            correlation, p_value = stats.pearsonr(drift_counts, accuracies)
            return {'correlation': correlation, 'p_value': p_value}
        return {'correlation': 0, 'p_value': 1}