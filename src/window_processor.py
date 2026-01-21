import pandas as pd
import numpy as np
import os
from scipy import stats

class WindowProcessor:
    def __init__(self, config):
        self.config = config
        self.window_id = 0
        
    def create_chronological_windows(self, data, window_size, step_size):
        windows = []
        start_idx = 0
        while start_idx + window_size <= len(data):
            window_data = data.iloc[start_idx:start_idx + window_size].copy()
            window_data['window_id'] = self.window_id
            windows.append(window_data)
            start_idx += step_size
            self.window_id += 1
        return windows
    
    def save_windows(self, windows, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        for i, window in enumerate(windows):
            window.to_csv(f"{output_dir}/window_{i}.csv", index=False)
    
    def load_window(self, window_path):
        return pd.read_csv(window_path)
    
    def extract_window_features(self, window_data, target_column=None):
        features = {}
        numeric_cols = window_data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if col == target_column:
                continue
            col_data = window_data[col].dropna()
            if len(col_data) > 0:
                features[f'{col}_mean'] = np.mean(col_data)
                features[f'{col}_std'] = np.std(col_data)
                features[f'{col}_skew'] = stats.skew(col_data)
                features[f'{col}_kurtosis'] = stats.kurtosis(col_data)
                features[f'{col}_min'] = np.min(col_data)
                features[f'{col}_max'] = np.max(col_data)
                features[f'{col}_median'] = np.median(col_data)
        if target_column in window_data.columns:
            target_counts = window_data[target_column].value_counts()
            for label, count in target_counts.items():
                features[f'label_{label}_count'] = count
            features['total_samples'] = len(window_data)
        return pd.Series(features)
    
    def process_all_windows(self, data_path, window_size, step_size, target_column):
        if isinstance(data_path, str):
            data = pd.read_csv(data_path)
        else:
            data = data_path
        windows = self.create_chronological_windows(data, window_size, step_size)
        window_features = []
        for window in windows:
            features = self.extract_window_features(window, target_column)
            window_features.append(features)
        return pd.DataFrame(window_features)