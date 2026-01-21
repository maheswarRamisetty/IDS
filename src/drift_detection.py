import numpy as np
import pandas as pd
from scipy.stats import ks_2samp
import warnings
warnings.filterwarnings('ignore')

class DriftDetector:
    def __init__(self, config):
        self.config = config
        self.drift_history = []
        
    def calculate_psi(self, expected, actual, buckets=10):
        expected = expected[~np.isnan(expected)]
        actual = actual[~np.isnan(actual)]
        breakpoints = np.percentile(expected, np.linspace(0, 100, buckets + 1))
        expected_counts = np.histogram(expected, bins=breakpoints)[0]
        actual_counts = np.histogram(actual, bins=breakpoints)[0]
        expected_proportions = expected_counts / len(expected)
        actual_proportions = actual_counts / len(actual)
        psi_values = np.where((expected_proportions == 0) | (actual_proportions == 0),
                              0,
                              (actual_proportions - expected_proportions) * 
                              np.log(actual_proportions / expected_proportions))
        return np.sum(psi_values)
    
    def ks_test(self, reference, current):
        statistic, p_value = ks_2samp(reference, current)
        return statistic, p_value
    
    def detect_univariate_drift(self, reference_data, current_data, features):
        drift_results = {}
        for feature in features:
            ref_vals = reference_data[feature].dropna().values
            curr_vals = current_data[feature].dropna().values
            if len(ref_vals) < 10 or len(curr_vals) < 10:
                continue
            ks_stat, ks_pvalue = self.ks_test(ref_vals, curr_vals)
            psi_value = self.calculate_psi(ref_vals, curr_vals)
            drift_results[feature] = {
                'ks_statistic': ks_stat,
                'ks_pvalue': ks_pvalue,
                'psi': psi_value,
                'ks_drift': ks_pvalue < self.config.drift.ks_threshold,
                'psi_drift': psi_value > self.config.drift.psi_threshold
            }
        return drift_results
    
    def detect_multivariate_drift(self, reference_data, current_data, method='kl_divergence'):
        if method == 'kl_divergence':
            return self._calculate_kl_divergence(reference_data, current_data)
        elif method == 'wasserstein':
            return self._calculate_wasserstein(reference_data, current_data)
        else:
            raise ValueError(f"Unsupported method: {method}")
    
    def _calculate_kl_divergence(self, P, Q):
        eps = 1e-10
        P = P + eps
        Q = Q + eps
        return np.sum(P * np.log(P / Q))
    
    def _calculate_wasserstein(self, P, Q):
        from scipy.stats import wasserstein_distance
        return wasserstein_distance(P.flatten(), Q.flatten())
    
    def track_drift(self, window_id, drift_results, performance_metrics):
        drift_record = {
            'window_id': window_id,
            'timestamp': pd.Timestamp.now(),
            'drift_features': len([f for f in drift_results if drift_results[f]['ks_drift']]),
            'avg_ks_statistic': np.mean([drift_results[f]['ks_statistic'] for f in drift_results]),
            'avg_psi': np.mean([drift_results[f]['psi'] for f in drift_results]),
            'performance_metrics': performance_metrics
        }
        self.drift_history.append(drift_record)
        return drift_record
    
    def generate_drift_report(self, drift_results):
        report = {
            'total_features_checked': len(drift_results),
            'features_with_ks_drift': [],
            'features_with_psi_drift': [],
            'critical_features': []
        }
        for feature, results in drift_results.items():
            if results['ks_drift']:
                report['features_with_ks_drift'].append(feature)
            if results['psi_drift']:
                report['features_with_psi_drift'].append(feature)
            if results['ks_drift'] and results['psi_drift']:
                report['critical_features'].append(feature)
        report['num_ks_drift'] = len(report['features_with_ks_drift'])
        report['num_psi_drift'] = len(report['features_with_psi_drift'])
        report['num_critical'] = len(report['critical_features'])
        return report