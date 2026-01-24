import numpy as np
from scipy import stats
import pandas as pd
from collections import deque
import warnings
from abc import abstractclassmethod
import logging
warnings.filterwarnings('ignore')


@abstractclassmethod
class Drift:
    def __init__(self):
        pass

class DriftDetector:
    def __init__(self, config, logger=None):
        self.config = config
        self.window_size = config['drift_detection']['window_size']
        self.drift_threshold_ks = config['drift_detection']['drift_threshold_ks']
        self.drift_threshold_psi = config['drift_detection']['drift_threshold_psi']
        self.n_features_drift = config['drift_detection']['n_features_drift']
        
        self.reference_data = None
        self.feature_names = None
        self.drift_history = []
        self.drift_alerts = []
        self.logger = logger or logging.getLogger(__name__)
        
    def set_reference_data(self, data, feature_names):
        self.reference_data = data
        self.feature_names = feature_names
        self.logger.info(f"Reference data set with shape: {data.shape}")
        self.logger.info(f"Number of features: {len(feature_names)}")
        
    def kolmogorov_smirnov_test(self, reference, current):
        try:
            ks_statistic, p_value = stats.ks_2samp(reference, current)
            return ks_statistic, p_value
        except Exception as e:
            self.logger.warning(f"KS test failed: {str(e)}")
            return 0, 1.0
    
    def calculate_psi(self, reference, current, bins=20):
        try:
            ref_min, ref_max = np.min(reference), np.max(reference)
            
            if ref_min == ref_max:
                ref_min -= 0.1
                ref_max += 0.1
            
            bin_edges = np.linspace(ref_min, ref_max, bins + 1)
            
            ref_counts, _ = np.histogram(reference, bins=bin_edges)
            curr_counts, _ = np.histogram(current, bins=bin_edges)
            
            ref_proportions = (ref_counts + 0.001) / (len(reference) + bins * 0.001)
            curr_proportions = (curr_counts + 0.001) / (len(current) + bins * 0.001)
            
            psi = 0
            for i in range(len(ref_proportions)):
                if ref_proportions[i] > 0 and curr_proportions[i] > 0:
                    psi += (curr_proportions[i] - ref_proportions[i]) * np.log(curr_proportions[i] / ref_proportions[i])
            
            return psi
        except Exception as e:
            self.logger.warning(f"PSI calculation failed: {str(e)}")
            return 0
    
    def detect_drift(self, current_data):
        if self.reference_data is None:
            raise ValueError("Reference data not set. Call set_reference_data() first.")
        
        if current_data.shape[1] != self.reference_data.shape[1]:
            raise ValueError(f"Feature dimension mismatch: current {current_data.shape[1]}, reference {self.reference_data.shape[1]}")
        
        drift_results = {
            'ks_drift': False,
            'psi_drift': False,
            'ks_scores': {},
            'psi_scores': {},
            'drift_features': [],
            'drift_magnitude': 0,
            'ks_statistics': [],
            'psi_values': []
        }
        
        ks_scores = []
        psi_scores = []
        significant_features = []
        
        for i, feature in enumerate(self.feature_names):
            if i >= current_data.shape[1]:
                continue
                
            ref_feature = self.reference_data[:, i]
            curr_feature = current_data[:, i]
            
            try:
                ks_stat, ks_p = self.kolmogorov_smirnov_test(ref_feature, curr_feature)
                psi = self.calculate_psi(ref_feature, curr_feature)
                
                ks_scores.append(ks_stat)
                psi_scores.append(psi)
                
                drift_results['ks_scores'][feature] = ks_stat
                drift_results['psi_scores'][feature] = psi
                drift_results['ks_statistics'].append(ks_stat)
                drift_results['psi_values'].append(psi)
                
                if ks_stat > self.drift_threshold_ks:
                    significant_features.append(feature)
                    
            except Exception as e:
                self.logger.warning(f"Error processing feature {feature}: {str(e)}")
                continue
        
        drift_results['drift_features'] = significant_features
        
        if len(significant_features) >= self.n_features_drift:
            drift_results['ks_drift'] = True
        
        if psi_scores:
            avg_psi = np.mean(psi_scores)
            drift_results['drift_magnitude'] = avg_psi
            
            if avg_psi > self.drift_threshold_psi:
                drift_results['psi_drift'] = True
        else:
            drift_results['drift_magnitude'] = 0
        
        if drift_results['ks_drift'] or drift_results['psi_drift']:
            alert = {
                'timestamp': pd.Timestamp.now().strftime("%Y-%m-%d %H:%M:%S"),
                'batch': len(self.drift_history),
                'ks_drift': drift_results['ks_drift'],
                'psi_drift': drift_results['psi_drift'],
                'drift_magnitude': drift_results['drift_magnitude'],
                'drift_features_count': len(significant_features),
                'top_drift_features': significant_features[:5]
            }
            self.drift_alerts.append(alert)
            
            self.logger.warning(f"DRIFT DETECTED! KS: {drift_results['ks_drift']}, PSI: {drift_results['psi_drift']}")
            self.logger.warning(f"Drift magnitude: {drift_results['drift_magnitude']:.4f}")
            self.logger.warning(f"Features with drift: {len(significant_features)}")
        
        self.drift_history.append(drift_results)
        return drift_results
    
    def simulate_intentional_drift(self, batch_data, batch_idx):
        """Simulate intentional drift for testing"""
        if batch_idx >= 15 and batch_idx < 25:
            self.logger.info(f"Simulating MAJOR drift in batch {batch_idx}")
            drift_factor = np.random.uniform(1.5, 3.0, batch_data.shape)
            batch_data = batch_data * drift_factor
            batch_data = batch_data + np.random.normal(0, 2, batch_data.shape)
        
        elif batch_idx >= 35 and batch_idx < 40:
            self.logger.info(f"Simulating MODERATE drift in batch {batch_idx}")
            selected_features = np.random.choice(batch_data.shape[1], 
                                                size=batch_data.shape[1]//3, 
                                                replace=False)
            for feat in selected_features:
                batch_data[:, feat] = batch_data[:, feat] * np.random.uniform(1.2, 1.8)
        
        return batch_data
    
    def detect_drift_streaming(self, X_data, n_batches=50):
        self.logger.info("Starting streaming drift detection...")
        
        batch_size = len(X_data) // n_batches
        if batch_size < 100:
            batch_size = 100
            n_batches = len(X_data) // batch_size
        
        self.logger.info(f"Processing {n_batches} batches of size {batch_size}")
        
        all_results = []
        
        for i in range(n_batches):
            start_idx = i * batch_size
            end_idx = start_idx + batch_size
            
            if end_idx > len(X_data):
                break
            
            current_batch = X_data[start_idx:end_idx].copy()
            
            current_batch = self.simulate_intentional_drift(current_batch, i)
            
            drift_result = self.detect_drift(current_batch)
            drift_result['batch'] = i
            all_results.append(drift_result)
            
            if i % 5 == 0:
                self.logger.info(f"Processed batch {i}/{n_batches}")
        
        self.logger.info("Streaming drift detection completed")
        return all_results
    
    def get_drift_summary(self):
        total_ks_drifts = len([d for d in self.drift_history if d['ks_drift']])
        total_psi_drifts = len([d for d in self.drift_history if d['psi_drift']])
        total_drifts = len([d for d in self.drift_history if d['ks_drift'] or d['psi_drift']])
        
        if self.drift_history:
            magnitudes = [d['drift_magnitude'] for d in self.drift_history if d['drift_magnitude'] > 0]
            avg_magnitude = np.mean(magnitudes) if magnitudes else 0
            max_magnitude = np.max(magnitudes) if magnitudes else 0
        else:
            avg_magnitude = max_magnitude = 0
        
        summary = {
            'total_batches': len(self.drift_history),
            'total_ks_drifts': total_ks_drifts,
            'total_psi_drifts': total_psi_drifts,
            'total_drifts': total_drifts,
            'drift_rate': total_drifts / len(self.drift_history) if self.drift_history else 0,
            'avg_drift_magnitude': avg_magnitude,
            'max_drift_magnitude': max_magnitude,
            'total_alerts': len(self.drift_alerts),
            'drift_alerts': self.drift_alerts
        }
        
        self.logger.info("=" * 50)
        self.logger.info("DRIFT DETECTION SUMMARY")
        self.logger.info("=" * 50)
        for key, value in summary.items():
            if key != 'drift_alerts':
                self.logger.info(f"  {key}: {value}")
        
        if self.drift_alerts:
            self.logger.info("\nRecent drift alerts:")
            for alert in self.drift_alerts[-5:]:
                self.logger.info(f"  Batch {alert['batch']}: Magnitude {alert['drift_magnitude']:.4f}")
        
        self.logger.info("=" * 50)
        
        return summary
    
    def save_detector(self, path):
        import joblib
        joblib.dump({
            'reference_data': self.reference_data,
            'feature_names': self.feature_names,
            'drift_history': self.drift_history,
            'drift_alerts': self.drift_alerts,
            'config': self.config
        }, path)
        self.logger.info(f"Drift detector saved to {path}")
    
    def load_detector(self, path):
        import joblib
        saved_data = joblib.load(path)
        self.reference_data = saved_data['reference_data']
        self.feature_names = saved_data['feature_names']
        self.drift_history = saved_data['drift_history']
        self.drift_alerts = saved_data['drift_alerts']
        self.logger.info(f"Drift detector loaded from {path}")
        return self