from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, accuracy_score, roc_auc_score, confusion_matrix
from sklearn.model_selection import cross_val_score
from xgboost import XGBClassifier
import joblib
import numpy as np
import pandas as pd
import logging
from sklearn.multiclass import OneVsRestClassifier

class IDSModel:
    def __init__(self, config, logger=None):
        self.config = config
        self.model = None
        self.model_type = config['model']['model_type']
        self.logger = logger or logging.getLogger(__name__)
        
    def build_model(self):
        if self.model_type == 'random_forest':
            self.logger.info("Building Random Forest model...")
            self.model = RandomForestClassifier(
                n_estimators=self.config['model']['random_forest']['n_estimators'],
                max_depth=self.config['model']['random_forest']['max_depth'],
                min_samples_split=self.config['model']['random_forest']['min_samples_split'],
                min_samples_leaf=self.config['model']['random_forest']['min_samples_leaf'],
                n_jobs=self.config['model']['random_forest']['n_jobs'],
                random_state=self.config['model']['random_state'],
                class_weight=self.config['model']['random_forest'].get('class_weight', None)
            )
        elif self.model_type == 'xgboost':
            self.logger.info("Building XGBoost model...")
            self.model = XGBClassifier(
                n_estimators=self.config['model']['xgboost']['n_estimators'],
                max_depth=self.config['model']['xgboost']['max_depth'],
                learning_rate=self.config['model']['xgboost']['learning_rate'],
                subsample=self.config['model']['xgboost']['subsample'],
                colsample_bytree=self.config['model']['xgboost']['colsample_bytree'],
                random_state=self.config['model']['random_state'],
                use_label_encoder=False,
                eval_metric='mlogloss'
            )
        else:
            raise ValueError(f"Unknown model type: {self.model_type}")
        
        return self.model
    
    def train(self, X_train, y_train):
        self.logger.info(f"Training {self.model_type} model...")
        self.build_model()
        self.model.fit(X_train, y_train)
        self.logger.info(f"Model training completed")
        return self.model
    
    def evaluate(self, X_test, y_test):
        self.logger.info("Evaluating model...")
        y_pred = self.model.predict(X_test)
        
        accuracy = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, zero_division=0)
        cm = confusion_matrix(y_test, y_pred)
        
        self.logger.info(f"Accuracy: {accuracy:.4f}")
        self.logger.info(f"Confusion Matrix shape: {cm.shape}")
        
        try:
            if hasattr(self.model, "predict_proba"):
                y_pred_proba = self.model.predict_proba(X_test)
                
                if y_pred_proba.shape[1] > 2:
                    auc = roc_auc_score(y_test, y_pred_proba, multi_class='ovr', average='weighted')
                else:
                    auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                
                self.logger.info(f"AUC-ROC: {auc:.4f}")
            else:
                auc = None
                self.logger.info("Model doesn't support probability predictions")
        except Exception as e:
            self.logger.warning(f"AUC-ROC calculation failed: {str(e)}")
            auc = None
        
        metrics = {
            'accuracy': float(accuracy),
            'classification_report': report,
            'confusion_matrix': cm.tolist(),
            'auc_roc': float(auc) if auc is not None else None
        }
        
        return metrics
    
    def cross_validate(self, X, y, cv=5):
        self.logger.info(f"Performing {cv}-fold cross-validation...")
        scores = cross_val_score(self.model, X, y, cv=cv, scoring='accuracy')
        self.logger.info(f"Cross-validation scores: {scores}")
        self.logger.info(f"Mean CV accuracy: {np.mean(scores):.4f} (+/- {np.std(scores):.4f})")
        return scores
    
    def save_model(self, path):
        joblib.dump(self.model, path)
        self.logger.info(f"Model saved to {path}")
    
    def load_model(self, path):
        self.model = joblib.load(path)
        self.logger.info(f"Model loaded from {path}")
        return self
    
    def get_feature_importance(self, feature_names):
        if hasattr(self.model, 'feature_importances_'):
            importances = self.model.feature_importances_
            indices = np.argsort(importances)[::-1]
            
            importance_df = pd.DataFrame({
                'feature': [feature_names[i] for i in indices],
                'importance': importances[indices]
            })
            
            self.logger.info("Top 10 important features:")
            for i, row in importance_df.head(10).iterrows():
                self.logger.info(f"  {row['feature']}: {row['importance']:.4f}")
            
            return importance_df
        else:
            self.logger.warning("Model doesn't have feature_importances_ attribute")
            return None