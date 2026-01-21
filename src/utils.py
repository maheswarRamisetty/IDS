import pandas as pd
import numpy as np
import json
import yaml
import pickle
import os
import logging
from datetime import datetime

class DataUtils:
    @staticmethod
    def ensure_directory(path):
        os.makedirs(path, exist_ok=True)
    
    @staticmethod
    def save_pickle(obj, filepath):
        with open(filepath, 'wb') as f:
            pickle.dump(obj, f)
    
    @staticmethod
    def load_pickle(filepath):
        with open(filepath, 'rb') as f:
            return pickle.load(f)
    
    @staticmethod
    def save_json(data, filepath):
        with open(filepath, 'w') as f:
            json.dump(data, f, indent=2)
    
    @staticmethod
    def load_json(filepath):
        with open(filepath, 'r') as f:
            return json.load(f)
    
    @staticmethod
    def generate_timestamp():
        return datetime.now().strftime("%Y%m%d_%H%M%S")
    
    @staticmethod
    def calculate_correlation_matrix(data, threshold=0.8):
        corr_matrix = data.corr().abs()
        upper = corr_matrix.where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
        to_drop = [column for column in upper.columns if any(upper[column] > threshold)]
        return to_drop, corr_matrix
    
    @staticmethod
    def detect_outliers_iqr(data, column, threshold=1.5):
        Q1 = data[column].quantile(0.25)
        Q3 = data[column].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - threshold * IQR
        upper_bound = Q3 + threshold * IQR
        outliers = data[(data[column] < lower_bound) | (data[column] > upper_bound)]
        return outliers, lower_bound, upper_bound
    
    @staticmethod
    def calculate_class_balance(data, target_column):
        class_counts = data[target_column].value_counts()
        class_proportions = class_counts / len(data)
        return class_counts, class_proportions

class Logger:
    def __init__(self, log_dir='logs', log_name='app.log'):
        self.log_dir = log_dir
        DataUtils.ensure_directory(log_dir)
        self.logger = logging.getLogger(__name__)
        self.logger.setLevel(logging.INFO)
        file_handler = logging.FileHandler(f"{log_dir}/{log_name}")
        file_handler.setLevel(logging.INFO)
        console_handler = logging.StreamHandler()
        console_handler.setLevel(logging.INFO)
        formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
        file_handler.setFormatter(formatter)
        console_handler.setFormatter(formatter)
        self.logger.addHandler(file_handler)
        self.logger.addHandler(console_handler)
    
    def info(self, message):
        self.logger.info(message)
    
    def warning(self, message):
        self.logger.warning(message)
    
    def error(self, message):
        self.logger.error(message)
    
    def critical(self, message):
        self.logger.critical(message)

class AlertSystem:
    def __init__(self, alert_path='results/alerts/'):
        self.alert_path = alert_path
        DataUtils.ensure_directory(alert_path)
        self.alerts = []
    
    def add_alert(self, alert_type, message, severity='warning', data=None):
        alert = {
            'timestamp': datetime.now().isoformat(),
            'type': alert_type,
            'severity': severity,
            'message': message,
            'data': data or {}
        }
        self.alerts.append(alert)
        if severity in ['critical', 'error']:
            self.save_immediate_alert(alert)
        return alert
    
    def save_immediate_alert(self, alert):
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        alert_file = f"{self.alert_path}/alert_{timestamp}.json"
        DataUtils.save_json(alert, alert_file)
    
    def save_all_alerts(self):
        alert_file = f"{self.alert_path}/alerts_summary.json"
        DataUtils.save_json(self.alerts, alert_file)
    
    def get_alerts_by_severity(self, severity):
        return [alert for alert in self.alerts if alert['severity'] == severity]