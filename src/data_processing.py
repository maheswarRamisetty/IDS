import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.model_selection import train_test_split
import joblib
import os
import logging

class DataPreprocessor:
    def __init__(self, config, logger=None):
        self.config = config
        self.scaler = StandardScaler()
        self.label_encoder = LabelEncoder()
        self.logger = logger or logging.getLogger(__name__)
        
    def load_and_merge_data(self, data_dir, file_list):
        all_dfs = []
        
        for file_name in file_list:
            file_path = os.path.join(data_dir, file_name)
            if os.path.exists(file_path):
                self.logger.info(f"Loading {file_name}")
                try:
                    df = pd.read_csv(file_path, encoding='utf-8')
                    self.logger.info(f"Loaded {file_name}: {df.shape}")
                    
                    if ' Label' in df.columns:
                        df = df.rename(columns={' Label': 'Label'})
                    
                    if 'label' in df.columns:
                        df = df.rename(columns={'label': 'Label'})
                    
                    all_dfs.append(df)
                    
                except Exception as e:
                    self.logger.error(f"Error loading {file_name}: {str(e)}")
                    try:
                        df = pd.read_csv(file_path, encoding='cp1252')
                        self.logger.info(f"Loaded with cp1252 encoding: {file_name}: {df.shape}")
                        
                        if ' Label' in df.columns:
                            df = df.rename(columns={' Label': 'Label'})
                        
                        all_dfs.append(df)
                    except Exception as e2:
                        self.logger.error(f"Failed to load {file_name} with any encoding: {str(e2)}")
            else:
                self.logger.warning(f"File not found: {file_path}")
        
        if not all_dfs:
            raise ValueError("No data files could be loaded")
        
        combined_df = pd.concat(all_dfs, ignore_index=True)
        self.logger.info(f"Combined data shape: {combined_df.shape}")
        
        return combined_df
    
    def clean_data(self, df):
        self.logger.info("Starting data cleaning...")
        
        original_shape = df.shape
        self.logger.info(f"Original shape: {original_shape}")
        
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        
        numeric_cols = df.select_dtypes(include=[np.number]).columns
        df[numeric_cols] = df[numeric_cols].fillna(df[numeric_cols].median())
        
        df = df.replace('Infinity', 1e6)
        df = df.replace('infinity', 1e6)
        
        for col in df.select_dtypes(include=['object']).columns:
            if col != self.config['data']['target_column']:
                try:
                    df[col] = pd.to_numeric(df[col], errors='coerce')
                except:
                    df.drop(columns=[col], inplace=True)
        
        df = df.dropna()
        
        self.logger.info(f"Final shape after cleaning: {df.shape}")
        self.logger.info(f"Removed {original_shape[0] - df.shape[0]} rows")
        
        return df
        
    def preprocess_target(self, df):
        target_col = self.config['data']['target_column']
        if target_col in df.columns:
            self.logger.info("Encoding target variable...")

            df[target_col] = (
                df[target_col]
                .astype(str)
                .str.encode("latin1", errors="ignore")
                .str.decode("utf-8", errors="ignore")
                .str.replace("�", "-", regex=False)
                .str.strip()
            )

            unique_labels = df[target_col].unique()
            self.logger.info(f"Unique labels found: {unique_labels}")

            df[target_col] = self.label_encoder.fit_transform(df[target_col])

            self.logger.info("Label mapping:")
            for i, label in enumerate(self.label_encoder.classes_):
                safe_label = label.encode("utf-8", "ignore").decode("utf-8")
                self.logger.info(f"  {i}: {safe_label}")

        return df

    
    def extract_features(self, df):
        features = []
        feature_config = self.config['data']['features']
        
        if 'numerical_columns' in feature_config:
            numerical_cols = [col for col in feature_config['numerical_columns'] if col in df.columns]
            features.extend(numerical_cols)
            self.logger.info(f"Selected {len(numerical_cols)} numerical features")
        
        if 'categorical_columns' in feature_config:
            categorical_cols = [col for col in feature_config['categorical_columns'] if col in df.columns]
            features.extend(categorical_cols)
            self.logger.info(f"Selected {len(categorical_cols)} categorical features")
        
        X = df[features]
        y = df[self.config['data']['target_column']] if self.config['data']['target_column'] in df.columns else None
        
        self.logger.info(f"Final feature count: {len(features)}")
        
        return X, y, features
    
    def scale_features(self, X_train, X_test=None):
        self.logger.info("Scaling features...")
        X_train_scaled = self.scaler.fit_transform(X_train)
        
        if X_test is not None:
            X_test_scaled = self.scaler.transform(X_test)
            return X_train_scaled, X_test_scaled
        
        return X_train_scaled
    
    def save_preprocessor(self, path):
        os.makedirs(os.path.dirname(path), exist_ok=True)
        joblib.dump({
            'scaler': self.scaler,
            'label_encoder': self.label_encoder
        }, path)
        self.logger.info(f"Preprocessor saved to {path}")
    
    def load_preprocessor(self, path):
        preprocessor = joblib.load(path)
        self.scaler = preprocessor['scaler']
        self.label_encoder = preprocessor['label_encoder']
        return self
    
    def prepare_data(self, save_path=None):
        self.logger.info("Starting data preparation pipeline...")
        
        df = self.load_and_merge_data(
            self.config['data']['raw_path'],
            self.config['data']['files']
        )
        
        df = self.clean_data(df)
        df = self.preprocess_target(df)
        
        X, y, features = self.extract_features(df)
        
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            test_size=self.config['model']['test_size'],
            random_state=self.config['model']['random_state'],
            stratify=y
        )
        
        X_train_scaled, X_test_scaled = self.scale_features(X_train, X_test)
        
        if save_path:
            os.makedirs(save_path, exist_ok=True)
            
            pd.DataFrame(X_train_scaled, columns=features).to_csv(
                f"{save_path}/X_train.csv", index=False
            )
            pd.DataFrame(y_train, columns=[self.config['data']['target_column']]).to_csv(
                f"{save_path}/y_train.csv", index=False
            )
            pd.DataFrame(X_test_scaled, columns=features).to_csv(
                f"{save_path}/X_test.csv", index=False
            )
            pd.DataFrame(y_test, columns=[self.config['data']['target_column']]).to_csv(
                f"{save_path}/y_test.csv", index=False
            )
            
            pd.DataFrame(features, columns=['feature_names']).to_csv(
                f"{save_path}/feature_names.csv", index=False
            )
            
            self.save_preprocessor(f"{save_path}/preprocessor.joblib")
            
            self.logger.info(f"Data saved to {save_path}")
        
        return X_train_scaled, X_test_scaled, y_train, y_test, features