import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import numpy as np
import plotly.graph_objects as go
import plotly.express as px
from plotly.subplots import make_subplots

class DataVisualizer:
    def __init__(self, config):
        self.config = config
        self.set_style()
    
    def set_style(self):
        plt.style.use(self.config.visualization.style)
        sns.set_palette(self.config.visualization.color_palette)
        
    def plot_feature_distribution(self, data, feature, reference_data=None, title=None):
        fig, ax = plt.subplots(figsize=(self.config.visualization.figsize_width, 
                                        self.config.visualization.figsize_height))
        ax.hist(data[feature].dropna(), bins=50, alpha=0.7, label='Current', density=True)
        if reference_data is not None:
            ax.hist(reference_data[feature].dropna(), bins=50, alpha=0.7, 
                   label='Reference', density=True, histtype='step', linewidth=2)
        ax.set_xlabel(feature)
        ax.set_ylabel('Density')
        ax.set_title(title or f'Distribution of {feature}')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
    
    def plot_drift_metrics(self, drift_history, metric='avg_ks_statistic'):
        fig, ax = plt.subplots(figsize=(self.config.visualization.figsize_width, 
                                        self.config.visualization.figsize_height))
        windows = [d['window_id'] for d in drift_history]
        values = [d[metric] for d in drift_history]
        ax.plot(windows, values, marker='o', linewidth=2)
        ax.axhline(y=self.config.drift.ks_threshold, color='r', linestyle='--', 
                  label='Drift Threshold')
        ax.set_xlabel('Window ID')
        ax.set_ylabel(metric.replace('_', ' ').title())
        ax.set_title(f'{metric.replace("_", " ").title()} Over Time')
        ax.legend()
        ax.grid(True, alpha=0.3)
        return fig
    
    def plot_performance_trend(self, metrics_history):
        fig, axes = plt.subplots(2, 2, figsize=(self.config.visualization.figsize_width * 1.5, 
                                               self.config.visualization.figsize_height * 1.5))
        axes = axes.flatten()
        metrics_to_plot = ['accuracy', 'precision', 'recall', 'f1']
        for idx, metric in enumerate(metrics_to_plot):
            if idx < len(axes):
                values = [m.get(metric, 0) for m in metrics_history]
                axes[idx].plot(range(len(values)), values, marker='o', linewidth=2)
                axes[idx].set_xlabel('Window')
                axes[idx].set_ylabel(metric.title())
                axes[idx].set_title(f'{metric.title()} Trend')
                axes[idx].grid(True, alpha=0.3)
                if len(values) > 1:
                    z = np.polyfit(range(len(values)), values, 1)
                    p = np.poly1d(z)
                    axes[idx].plot(range(len(values)), p(range(len(values))), 
                                  "r--", alpha=0.5, label='Trend')
                    axes[idx].legend()
        plt.tight_layout()
        return fig
    
    def plot_correlation_heatmap(self, data, title="Feature Correlation Matrix"):
        fig, ax = plt.subplots(figsize=(self.config.visualization.figsize_width * 1.2, 
                                        self.config.visualization.figsize_height * 1.2))
        numeric_cols = data.select_dtypes(include=[np.number]).columns
        corr_matrix = data[numeric_cols].corr()
        mask = np.triu(np.ones_like(corr_matrix, dtype=bool))
        cmap = sns.diverging_palette(230, 20, as_cmap=True)
        sns.heatmap(corr_matrix, mask=mask, cmap=cmap, center=0,
                   square=True, linewidths=.5, cbar_kws={"shrink": .5},
                   ax=ax)
        ax.set_title(title)
        return fig
    
    def plot_feature_importance(self, feature_importance_df, top_n=20):
        fig, ax = plt.subplots(figsize=(self.config.visualization.figsize_width, 
                                        self.config.visualization.figsize_height))
        top_features = feature_importance_df.head(top_n)
        ax.barh(range(len(top_features)), top_features['importance'])
        ax.set_yticks(range(len(top_features)))
        ax.set_yticklabels(top_features['feature'])
        ax.set_xlabel('Importance')
        ax.set_title(f'Top {top_n} Feature Importance')
        ax.invert_yaxis()
        return fig
    
    def plot_interactive_drift_dashboard(self, drift_history, metrics_history):
        fig = make_subplots(
            rows=2, cols=2,
            subplot_titles=('KS Statistic Over Time', 'PSI Over Time',
                          'Accuracy Trend', 'Drift vs Performance'),
            specs=[[{'secondary_y': True}, {'secondary_y': False}],
                   [{'secondary_y': False}, {'secondary_y': True}]]
        )
        windows = [d['window_id'] for d in drift_history]
        ks_stats = [d['avg_ks_statistic'] for d in drift_history]
        psi_vals = [d['avg_psi'] for d in drift_history]
        accuracies = [m.get('accuracy', 0) for m in metrics_history]
        drift_counts = [d['drift_features'] for d in drift_history]
        fig.add_trace(
            go.Scatter(x=windows, y=ks_stats, name='KS Statistic', mode='lines+markers'),
            row=1, col=1
        )
        fig.add_trace(
            go.Scatter(x=windows, y=psi_vals, name='PSI', mode='lines+markers'),
            row=1, col=2
        )
        fig.add_trace(
            go.Scatter(x=windows, y=accuracies, name='Accuracy', mode='lines+markers'),
            row=2, col=1
        )
        fig.add_trace(
            go.Bar(x=windows, y=drift_counts, name='Drift Features', opacity=0.5),
            row=2, col=2
        )
        fig.update_layout(height=800, showlegend=True, title_text="Drift Monitoring Dashboard")
        return fig