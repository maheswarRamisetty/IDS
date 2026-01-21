import yaml
import json
from dataclasses import dataclass

@dataclass
class Config:
    data: dict
    model: dict
    training: dict
    drift: dict
    evaluation: dict
    logging: dict
    results: dict
    visualization: dict

class ConfigLoader:
    def __init__(self, config_path="../config/config.yaml"):
        self.config_path = config_path
        self.config = self._load_config()
    
    def _load_config(self):
        with open(self.config_path, 'r') as file:
            config_dict = yaml.safe_load(file)
        return Config(**config_dict)
    
    def load_drift_thresholds(self, thresholds_path="../config/drift_thresholds.json"):
        with open(thresholds_path, 'r') as file:
            return json.load(file)
    
    def get_path(self, path_key):
        paths = {
            'raw': self.config.data.raw_path,
            'processed': self.config.data.processed_path,
            'splits': self.config.data.splits_path,
            'window': self.config.data.window_path,
            'models': self.config.model.model_path,
            'plots': self.config.results.plots_path,
            'metrics': self.config.results.metrics_path,
            'logs': self.config.logging.log_path
        }
        return paths.get(path_key)
    