import pandas as pd
import numpy as np
import os
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

class DataPreprocessor:
    def __init__(self, config):
        self.config = config
        self.label_encoder = LabelEncoder()
        self.scaler = StandardScaler()
        
    def load_data(self, filepath):
        return pd.read_csv(filepath)
    
    def save_data(self, data, filepath):
        data.to_csv(filepath, index=False)
    
    def handle_missing_values(self, data):
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        for col in numeric_cols:
            if data[col].isnull().any():
                data[col] = data[col].fillna(data[col].median())
        return data
    
    def remove_duplicates(self, data):
        return data.drop_duplicates()
    
    def encode_labels(self, data, label_column):
        data[label_column] = self.label_encoder.fit_transform(data[label_column])
        return data
    
    def scale_features(self, data, exclude_columns=None):
        if exclude_columns is None:
            exclude_columns = []
        feature_columns = [col for col in data.columns if col not in exclude_columns]
        scaled_features = self.scaler.fit_transform(data[feature_columns])
        data[feature_columns] = scaled_features
        return data
    
    def split_data(self, data, target_column, test_size=0.2):
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        return train_test_split(X, y, test_size=test_size, random_state=42, stratify=y)
    
    def preprocess_pipeline(self, input_path, output_path):
        data = self.load_data(input_path)
        data = self.handle_missing_values(data)
        data = self.remove_duplicates(data)
        if self.config.training.target_column in data.columns:
            data = self.encode_labels(data, self.config.training.target_column)
        data = self.scale_features(data, exclude_columns=[self.config.training.target_column])
        self.save_data(data, output_path)
        return data