import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from xgboost import XGBClassifier
from lightgbm import LGBMClassifier
from sklearn.svm import SVC
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, GridSearchCV
import joblib

class ModelTrainer:
    def __init__(self, config):
        self.config = config
        self.models = {}
        self.best_model = None
        self.best_score = 0
        
    def initialize_models(self):
        return {
            'random_forest': RandomForestClassifier(
                n_estimators=self.config.training.n_estimators,
                max_depth=self.config.training.max_depth,
                random_state=self.config.training.random_state
            ),
            'xgboost': XGBClassifier(
                n_estimators=self.config.training.n_estimators,
                max_depth=self.config.training.max_depth,
                learning_rate=self.config.training.learning_rate,
                subsample=self.config.training.subsample,
                random_state=self.config.training.random_state,
                use_label_encoder=False,
                eval_metric='logloss'
            ),
            'lightgbm': LGBMClassifier(
                n_estimators=self.config.training.n_estimators,
                max_depth=self.config.training.max_depth,
                learning_rate=self.config.training.learning_rate,
                subsample=self.config.training.subsample,
                random_state=self.config.training.random_state
            ),
            'svm': SVC(
                probability=True,
                random_state=self.config.training.random_state
            ),
            'logistic': LogisticRegression(
                max_iter=1000,
                random_state=self.config.training.random_state
            )
        }
    
    def train_models(self, X_train, y_train):
        models = self.initialize_models()
        for name, model in models.items():
            model.fit(X_train, y_train)
            self.models[name] = model
            cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='f1_macro')
            avg_score = np.mean(cv_scores)
            if avg_score > self.best_score:
                self.best_score = avg_score
                self.best_model = model
        return self.models
    
    def hyperparameter_tuning(self, X_train, y_train, model_name='xgboost'):
        if model_name == 'xgboost':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [3, 5, 7, 10],
                'learning_rate': [0.01, 0.05, 0.1, 0.2],
                'subsample': [0.6, 0.8, 1.0]
            }
            base_model = XGBClassifier(random_state=42, use_label_encoder=False, eval_metric='logloss')
        elif model_name == 'random_forest':
            param_grid = {
                'n_estimators': [50, 100, 200],
                'max_depth': [5, 10, 15, None],
                'min_samples_split': [2, 5, 10],
                'min_samples_leaf': [1, 2, 4]
            }
            base_model = RandomForestClassifier(random_state=42)
        grid_search = GridSearchCV(base_model, param_grid, cv=5, scoring='f1_macro', n_jobs=-1)
        grid_search.fit(X_train, y_train)
        return grid_search.best_estimator_, grid_search.best_score_
    
    def save_model(self, model, filepath):
        joblib.dump(model, filepath)
    
    def load_model(self, filepath):
        return joblib.load(filepath)
    
    def get_feature_importance(self, model, feature_names):
        if hasattr(model, 'feature_importances_'):
            importance = model.feature_importances_
            feature_importance_df = pd.DataFrame({
                'feature': feature_names,
                'importance': importance
            }).sort_values('importance', ascending=False)
            return feature_importance_df
        return None