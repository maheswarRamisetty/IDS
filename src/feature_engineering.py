import pandas as pd
import numpy as np
from sklearn.feature_selection import SelectKBest, f_classif, mutual_info_classif
from sklearn.decomposition import PCA
import joblib

class FeatureEngineer:
    def __init__(self, config):
        self.config = config
        self.selected_features = None
        self.pca = None
        
    def create_interaction_features(self, data):
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        if len(numeric_cols) >= 2:
            for i in range(len(numeric_cols)-1):
                for j in range(i+1, len(numeric_cols)):
                    col1 = numeric_cols[i]
                    col2 = numeric_cols[j]
                    data[f'{col1}_x_{col2}'] = data[col1] * data[col2]
        return data
    
    def create_polynomial_features(self, data, degree=2):
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            for d in range(2, degree+1):
                data[f'{col}^{d}'] = data[col] ** d
        return data
    
    def create_statistical_features(self, data, window_size=10):
        numeric_cols = data.select_dtypes(include=[np.number]).columns.tolist()
        for col in numeric_cols:
            data[f'{col}_rolling_mean'] = data[col].rolling(window=window_size, min_periods=1).mean()
            data[f'{col}_rolling_std'] = data[col].rolling(window=window_size, min_periods=1).std()
        return data.fillna(method='bfill')
    
    def select_features_anova(self, X, y, k=20):
        selector = SelectKBest(score_func=f_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        self.selected_features = X.columns[selector.get_support()]
        return pd.DataFrame(X_selected, columns=self.selected_features), selector
    
    def select_features_mutual_info(self, X, y, k=20):
        selector = SelectKBest(score_func=mutual_info_classif, k=min(k, X.shape[1]))
        X_selected = selector.fit_transform(X, y)
        self.selected_features = X.columns[selector.get_support()]
        return pd.DataFrame(X_selected, columns=self.selected_features), selector
    
    def apply_pca(self, X, n_components=0.95):
        self.pca = PCA(n_components=n_components)
        X_pca = self.pca.fit_transform(X)
        return pd.DataFrame(X_pca, columns=[f'PC{i+1}' for i in range(X_pca.shape[1])])
    
    def feature_engineering_pipeline(self, data, target_column, use_pca=False):
        X = data.drop(target_column, axis=1)
        y = data[target_column]
        X = self.create_interaction_features(X)
        X = self.create_polynomial_features(X, degree=2)
        X = self.create_statistical_features(X, window_size=5)
        X_selected, selector = self.select_features_anova(X, y, k=30)
        if use_pca:
            X_final = self.apply_pca(X_selected)
        else:
            X_final = X_selected
        X_final[target_column] = y.reset_index(drop=True)
        return X_final, selector