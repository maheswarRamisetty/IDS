import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, roc_curve, auc, roc_auc_score
import logging

class IDSEvaluator:
    def __init__(self, config, logger=None):
        self.config = config
        self.logger = logger or logging.getLogger(__name__)
        plt.style.use('seaborn-v0_8-darkgrid')
        sns.set_palette("husl")
        
    def plot_confusion_matrix(self, y_true, y_pred, labels=None, save_path=None):
        cm = confusion_matrix(y_true, y_pred)
        
        fig, ax = plt.subplots(figsize=(10, 8))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=labels, yticklabels=labels,
                   cbar_kws={'label': 'Count'})
        plt.title('Confusion Matrix', fontsize=16, fontweight='bold')
        plt.ylabel('True Label', fontsize=12)
        plt.xlabel('Predicted Label', fontsize=12)
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Confusion matrix saved to {save_path}")
        
        return fig, cm
    
    def plot_roc_curve(self, y_true, y_pred_proba, save_path=None):
        try:
            if len(np.unique(y_true)) > 2:
                fpr = {}
                tpr = {}
                roc_auc = {}
                
                fig, ax = plt.subplots(figsize=(10, 8))
                
                for i in range(y_pred_proba.shape[1]):
                    fpr[i], tpr[i], _ = roc_curve(y_true == i, y_pred_proba[:, i])
                    roc_auc[i] = auc(fpr[i], tpr[i])
                    plt.plot(fpr[i], tpr[i], label=f'Class {i} (AUC = {roc_auc[i]:.4f})')
            else:
                fpr, tpr, thresholds = roc_curve(y_true, y_pred_proba)
                roc_auc_val = auc(fpr, tpr)
                
                fig, ax = plt.subplots(figsize=(10, 8))
                plt.plot(fpr, tpr, color='darkorange', lw=3, 
                        label=f'ROC curve (AUC = {roc_auc_val:.4f})')
                roc_auc = roc_auc_val
            
            plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--', alpha=0.7)
            plt.fill_between(fpr, tpr if isinstance(tpr, np.ndarray) else tpr[0], alpha=0.2, color='darkorange')
            plt.xlim([0.0, 1.0])
            plt.ylim([0.0, 1.05])
            plt.xlabel('False Positive Rate', fontsize=12)
            plt.ylabel('True Positive Rate', fontsize=12)
            plt.title('Receiver Operating Characteristic (ROC) Curve', fontsize=16, fontweight='bold')
            plt.legend(loc="lower right", fontsize=12)
            plt.grid(True, alpha=0.3)
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"ROC curve saved to {save_path}")
            
            return fig, fpr, tpr, roc_auc
        except Exception as e:
            self.logger.error(f"ROC curve generation failed: {str(e)}")
            return None, None, None, None
    
    def plot_drift_history(self, drift_history, save_path=None):
        if not drift_history:
            self.logger.warning("No drift history available")
            return None, None
        
        batches = range(len(drift_history))
        ks_drifts = [1 if d.get('ks_drift', False) else 0 for d in drift_history]
        psi_drifts = [1 if d.get('psi_drift', False) else 0 for d in drift_history]
        magnitudes = [d.get('drift_magnitude', 0) for d in drift_history]
        
        fig, axes = plt.subplots(3, 1, figsize=(14, 12))
        
        axes[0].plot(batches, ks_drifts, 'r-', linewidth=2, label='KS Drift', alpha=0.8)
        axes[0].fill_between(batches, ks_drifts, alpha=0.3, color='red')
        axes[0].set_ylabel('KS Drift Detection', fontsize=12)
        axes[0].set_ylim(-0.1, 1.1)
        axes[0].legend(loc='upper right', fontsize=11)
        axes[0].set_title('Data Drift Detection Over Time', fontsize=16, fontweight='bold')
        axes[0].grid(True, alpha=0.3)
        
        axes[1].plot(batches, psi_drifts, 'b-', linewidth=2, label='PSI Drift', alpha=0.8)
        axes[1].fill_between(batches, psi_drifts, alpha=0.3, color='blue')
        axes[1].set_ylabel('PSI Drift Detection', fontsize=12)
        axes[1].set_ylim(-0.1, 1.1)
        axes[1].legend(loc='upper right', fontsize=11)
        axes[1].grid(True, alpha=0.3)
        
        axes[2].plot(batches, magnitudes, 'g-', linewidth=2, label='Drift Magnitude', alpha=0.8)
        axes[2].fill_between(batches, magnitudes, alpha=0.3, color='green')
        axes[2].set_ylabel('Drift Magnitude', fontsize=12)
        axes[2].set_xlabel('Batch Number', fontsize=12)
        axes[2].legend(loc='upper right', fontsize=11)
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        
        if save_path:
            fig.savefig(save_path, dpi=300, bbox_inches='tight')
            self.logger.info(f"Drift history plot saved to {save_path}")
        
        return fig, magnitudes
    
    def plot_feature_importance(self, model, feature_names, top_n=20, save_path=None):
        try:
            if hasattr(model, 'feature_importances_'):
                importances = model.feature_importances_
            elif hasattr(model, 'coef_'):
                importances = np.abs(model.coef_[0])
            else:
                self.logger.warning("Model doesn't have feature importance attribute")
                return None, None
            
            top_n = min(top_n, len(importances), len(feature_names))
            if top_n <= 0:
                self.logger.warning(f"Cannot plot feature importance: top_n={top_n}")
                return None, None
            
            indices = np.argsort(importances)[-top_n:]
            
            fig, ax = plt.subplots(figsize=(12, 10))
            bars = plt.barh(range(top_n), importances[indices], align='center', 
                           color='steelblue', edgecolor='black', linewidth=0.8)
            
            plt.yticks(range(top_n), [feature_names[i] for i in indices], fontsize=11)
            plt.xlabel('Feature Importance', fontsize=12)
            plt.title(f'Top {top_n} Feature Importances', fontsize=16, fontweight='bold')
            
            for i, bar in enumerate(bars):
                width = bar.get_width()
                plt.text(width + 0.001, bar.get_y() + bar.get_height()/2,
                        f'{width:.4f}', ha='left', va='center', fontsize=10)
            
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Feature importance plot saved to {save_path}")
            
            return fig, importances
        except Exception as e:
            self.logger.error(f"Feature importance plot failed: {str(e)}")
            return None, None
    
    def plot_drift_statistics(self, drift_detector, save_path=None):
        try:
            if not drift_detector.drift_history:
                self.logger.warning("No drift history available for statistics")
                return None
            
            ks_stats = []
            psi_vals = []
            
            for drift in drift_detector.drift_history:
                if 'ks_statistics' in drift and drift['ks_statistics']:
                    ks_stats.append(np.mean(drift['ks_statistics']))
                if 'psi_values' in drift and drift['psi_values']:
                    psi_vals.append(np.mean(drift['psi_values']))
            
            if not ks_stats and not psi_vals:
                self.logger.warning("No drift statistics available")
                return None
            
            fig, axes = plt.subplots(2, 2, figsize=(14, 12))
            
            if ks_stats:
                axes[0, 0].hist(ks_stats, bins=20, color='red', alpha=0.7, edgecolor='black')
                axes[0, 0].set_xlabel('KS Statistic', fontsize=11)
                axes[0, 0].set_ylabel('Frequency', fontsize=11)
                axes[0, 0].set_title('Distribution of KS Statistics', fontsize=13)
                axes[0, 0].grid(True, alpha=0.3)
            
            if psi_vals:
                axes[0, 1].hist(psi_vals, bins=20, color='blue', alpha=0.7, edgecolor='black')
                axes[0, 1].set_xlabel('PSI Value', fontsize=11)
                axes[0, 1].set_ylabel('Frequency', fontsize=11)
                axes[0, 1].set_title('Distribution of PSI Values', fontsize=13)
                axes[0, 1].grid(True, alpha=0.3)
            
            if ks_stats:
                axes[1, 0].plot(ks_stats, 'r-', alpha=0.7, linewidth=1.5)
                axes[1, 0].set_xlabel('Batch', fontsize=11)
                axes[1, 0].set_ylabel('KS Statistic', fontsize=11)
                axes[1, 0].set_title('KS Statistics Over Time', fontsize=13)
                axes[1, 0].grid(True, alpha=0.3)
            
            if psi_vals:
                axes[1, 1].plot(psi_vals, 'b-', alpha=0.7, linewidth=1.5)
                axes[1, 1].set_xlabel('Batch', fontsize=11)
                axes[1, 1].set_ylabel('PSI Value', fontsize=11)
                axes[1, 1].set_title('PSI Values Over Time', fontsize=13)
                axes[1, 1].grid(True, alpha=0.3)
            
            plt.suptitle('Drift Detection Statistics', fontsize=16, fontweight='bold')
            plt.tight_layout()
            
            if save_path:
                fig.savefig(save_path, dpi=300, bbox_inches='tight')
                self.logger.info(f"Drift statistics plot saved to {save_path}")
            
            return fig
        except Exception as e:
            self.logger.error(f"Drift statistics plot failed: {str(e)}")
            return None