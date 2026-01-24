import yaml
import os
import json
import numpy as np
import pandas as pd
from datetime import datetime
import logging
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

def setup_logging(log_dir, log_file, log_level="INFO"):
    os.makedirs(log_dir, exist_ok=True)
    log_path = os.path.join(log_dir, log_file)
    
    logging.basicConfig(
        level=getattr(logging, log_level),
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.FileHandler(log_path),
            logging.StreamHandler()
        ]
    )
    
    logger = logging.getLogger(__name__)
    return logger

def load_config(config_path='../config/config.yaml'):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def save_figure(fig, filename, figs_dir="../figs"):
    os.makedirs(figs_dir, exist_ok=True)
    filepath = os.path.join(figs_dir, filename)
    fig.savefig(filepath, dpi=300, bbox_inches='tight')
    plt.close(fig)
    return filepath

def save_results(results, filename, results_dir="../results"):
    os.makedirs(results_dir, exist_ok=True)
    filepath = os.path.join(results_dir, filename)
    
    if filename.endswith('.json'):
        with open(filepath, 'w') as f:
            json.dump(results, f, indent=4)
    elif filename.endswith('.csv'):
        if isinstance(results, dict):
            pd.DataFrame([results]).to_csv(filepath, index=False)
        else:
            pd.DataFrame(results).to_csv(filepath, index=False)
    
    return filepath

def create_timestamp():
    return datetime.now().strftime("%Y%m%d_%H%M%S")

def batch_generator(data, batch_size=1000):
    n_samples = len(data)
    for i in range(0, n_samples, batch_size):
        yield data[i:i + batch_size]

def calculate_metrics(y_true, y_pred):
    from sklearn.metrics import precision_score, recall_score, f1_score, accuracy_score
    
    metrics = {
        'accuracy': accuracy_score(y_true, y_pred),
        'precision': precision_score(y_true, y_pred, average='weighted', zero_division=0),
        'recall': recall_score(y_true, y_pred, average='weighted', zero_division=0),
        'f1_score': f1_score(y_true, y_pred, average='weighted', zero_division=0)
    }
    
    return metrics

def log_execution_time(start_time, end_time, process_name, logger):
    duration = end_time - start_time
    logger.info(f"{process_name} completed in {duration:.2f} seconds")
    return duration

def check_file_exists(filepath, logger):
    exists = os.path.exists(filepath)
    if exists:
        logger.info(f"File exists: {filepath}")
    else:
        logger.warning(f"File not found: {filepath}")
    return exists