import pandas as pd
import numpy as np
import sys
import os
from pathlib import Path
sys.path.append(str(Path(__file__).parent))
from src.config_loader import ConfigLoader
from src.data_preprocessing import DataPreprocessor
from src.feature_engineering import FeatureEngineer
from src.model_training import ModelTrainer
from src.drift_detection import DriftDetector
from src.window_processor import WindowProcessor
from src.evaluation import ModelEvaluator
from src.utils import Logger, AlertSystem, DataUtils
from src.visualization import DataVisualizer

class IDSDriftPipeline:
    def __init__(self):
        self.config_loader = ConfigLoader()
        self.config = self.config_loader.config
        self.logger = Logger(self.config.logging.log_path, 'pipeline.log')
        self.alerts = AlertSystem(self.config.results.alerts_path)
        self.utils = DataUtils()
        self.preprocessor = DataPreprocessor(self.config)
        self.feature_engineer = FeatureEngineer(self.config)
        self.model_trainer = ModelTrainer(self.config)
        self.drift_detector = DriftDetector(self.config)
        self.window_processor = WindowProcessor(self.config)
        self.evaluator = ModelEvaluator(self.config)
        self.visualizer = DataVisualizer(self.config)
        self.reference_data = None
        self.current_model = None
        self.metrics_history = []
        self.drift_history = []
        self.setup_directories()
    
    def setup_directories(self):
        directories = [
            self.config.data.raw_path,
            self.config.data.processed_path,
            self.config.data.splits_path,
            self.config.data.window_path,
            self.config.model.model_path,
            self.config.model.feature_importance_path,
            self.config.logging.log_path,
            self.config.results.plots_path,
            self.config.results.metrics_path,
            self.config.results.reports_path,
            self.config.results.alerts_path
        ]
        for directory in directories:
            self.utils.ensure_directory(directory)
    
    def run_data_preprocessing(self, input_file, output_file):
        self.logger.info("Starting data preprocessing...")
        processed_data = self.preprocessor.preprocess_pipeline(input_file, output_file)
        self.logger.info(f"Data preprocessing completed. Saved to {output_file}")
        return processed_data
    
    def run_feature_engineering(self, input_file, output_file):
        self.logger.info("Starting feature engineering...")
        processed_data = pd.read_csv(input_file)
        engineered_data, feature_selector = self.feature_engineer.feature_engineering_pipeline(processed_data, self.config.training.target_column)
        engineered_data.to_csv(output_file, index=False)
        feature_importance_path = f"{self.config.model.feature_importance_path}/feature_selector.pkl"
        self.utils.save_pickle(feature_selector, feature_importance_path)
        self.logger.info(f"Feature engineering completed. Saved to {output_file}")
        return engineered_data
    
    def train_initial_model(self, train_data_path):
        self.logger.info("Training initial model...")
        train_data = pd.read_csv(train_data_path)
        X_train = train_data.drop(self.config.training.target_column, axis=1)
        y_train = train_data[self.config.training.target_column]
        models = self.model_trainer.train_models(X_train, y_train)
        initial_model_path = f"{self.config.model.model_path}/{self.config.model.initial_model}"
        self.model_trainer.save_model(self.model_trainer.best_model, initial_model_path)
        feature_importance = self.model_trainer.get_feature_importance(self.model_trainer.best_model, X_train.columns)
        if feature_importance is not None:
            feature_importance.to_csv(f"{self.config.model.feature_importance_path}/feature_importance.csv", index=False)
        self.current_model = self.model_trainer.best_model
        self.reference_data = train_data
        self.logger.info(f"Initial model trained and saved to {initial_model_path}")
        return models
    
    def process_windows_and_detect_drift(self, test_data_path):
        self.logger.info("Processing windows and detecting drift...")
        test_data = pd.read_csv(test_data_path)
        windows = self.window_processor.create_chronological_windows(test_data, self.config.drift.window_size, self.config.drift.step_size)
        window_metrics = []
        for i, window in enumerate(windows):
            self.logger.info(f"Processing window {i+1}/{len(windows)}")
            X_window = window.drop(self.config.training.target_column, axis=1)
            y_window = window[self.config.training.target_column]
            if self.current_model is None:
                self.logger.error("No model loaded for prediction")
                continue
            y_pred = self.current_model.predict(X_window)
            y_pred_proba = self.current_model.predict_proba(X_window)
            window_metrics_result = self.evaluator.calculate_metrics(y_window, y_pred, y_pred_proba)
            window_metrics_result['window_id'] = i
            if self.reference_data is not None:
                drift_results = self.drift_detector.detect_univariate_drift(self.reference_data, window, X_window.columns)
                drift_record = self.drift_detector.track_drift(i, drift_results, window_metrics_result)
                self.drift_history.append(drift_record)
                drift_report = self.drift_detector.generate_drift_report(drift_results)
                if drift_report['num_critical'] > 0:
                    self.alerts.add_alert('critical_drift', f"Critical drift detected in window {i}: {drift_report['num_critical']} features", 'critical', drift_report)
            self.metrics_history.append(window_metrics_result)
            window_metrics.append(window_metrics_result)
            if i % 10 == 0:
                self.save_intermediate_results(i)
        metrics_df = pd.DataFrame(window_metrics)
        metrics_path = f"{self.config.results.metrics_path}/window_metrics.csv"
        metrics_df.to_csv(metrics_path, index=False)
        self.alerts.save_all_alerts()
        self.logger.info(f"Window processing completed. Metrics saved to {metrics_path}")
        return metrics_df
    
    def evaluate_performance_degradation(self):
        self.logger.info("Evaluating performance degradation...")
        if len(self.metrics_history) < 10:
            self.logger.warning("Insufficient data for performance degradation analysis")
            return None
        degradation, details = self.evaluator.detect_performance_degradation(self.metrics_history, window_size=5)
        if degradation:
            self.alerts.add_alert('performance_degradation', f"Significant performance degradation detected: {details}", 'warning', details)
            self.logger.warning(f"Performance degradation detected: {details}")
        else:
            self.logger.info("No significant performance degradation detected")
        return degradation, details
    
    def retrain_model_if_needed(self, new_data_path, drift_threshold=5):
        self.logger.info("Checking if retraining is needed...")
        recent_drift_count = sum(1 for d in self.drift_history[-5:] if d['drift_features'] > 0)
        if recent_drift_count >= drift_threshold:
            self.logger.info("Retraining model due to persistent drift...")
            new_data = pd.read_csv(new_data_path)
            X_new = new_data.drop(self.config.training.target_column, axis=1)
            y_new = new_data[self.config.training.target_column]
            self.current_model.fit(X_new, y_new)
            retrained_model_path = f"{self.config.model.model_path}/{self.config.model.retrained_model}"
            self.model_trainer.save_model(self.current_model, retrained_model_path)
            self.alerts.add_alert('model_retrained', f"Model retrained due to persistent drift in {recent_drift_count} recent windows", 'info', {'recent_drift_windows': recent_drift_count})
            self.logger.info(f"Model retrained and saved to {retrained_model_path}")
            return True
        self.logger.info("No retraining needed at this time")
        return False
    
    def generate_reports(self):
        self.logger.info("Generating reports...")
        evaluation_report = self.evaluator.generate_evaluation_report(self.metrics_history, self.drift_history)
        report_path = f"{self.config.results.reports_path}/evaluation_report.json"
        self.utils.save_json(evaluation_report, report_path)
        summary_md = self._generate_summary_markdown(evaluation_report)
        summary_path = f"{self.config.results.reports_path}/executive_summary.md"
        with open(summary_path, 'w') as f:
            f.write(summary_md)
        self.logger.info(f"Reports generated and saved to {self.config.results.reports_path}")
        return evaluation_report
    
    def _generate_summary_markdown(self, evaluation_report):
        summary = f"""# Intrusion Detection System - Drift Analysis Report

- **Total Windows Processed**: {evaluation_report['overview']['total_windows']}
- **Windows with Drift Detected**: {evaluation_report['overview']['windows_with_drift']}
- **Average Accuracy**: {evaluation_report['overview']['average_accuracy']:.3f}
- **Average F1 Score**: {evaluation_report['overview']['average_f1']:.3f}

- **Trend**: {evaluation_report['performance_trend']['trend']}
- **Slope**: {evaluation_report['performance_trend']['slope']:.4f}

- **Correlation Coefficient**: {evaluation_report['drift_correlation']['correlation']:.3f}
- **P-value**: {evaluation_report['drift_correlation']['p_value']:.4f}

"""
        if evaluation_report['performance_trend']['trend'] == 'declining':
            summary += "- Consider model retraining or adaptation\n"
        if evaluation_report['drift_correlation']['correlation'] < -0.3:
            summary += "- Strong negative correlation between drift and performance detected\n"
        return summary
    
    def save_intermediate_results(self, window_id):
        intermediate_path = f"{self.config.results.metrics_path}/intermediate_window_{window_id}.csv"
        if len(self.metrics_history) > 0:
            pd.DataFrame(self.metrics_history).to_csv(intermediate_path, index=False)
    
    def run_full_pipeline(self, raw_data_path):
        try:
            self.logger.info("Starting full IDS Drift Detection Pipeline")
            processed_path = f"{self.config.data.processed_path}/{self.config.data.preprocessed_file}"
            self.run_data_preprocessing(raw_data_path, processed_path)
            features_path = f"{self.config.data.processed_path}/{self.config.data.features_file}"
            self.run_feature_engineering(processed_path, features_path)
            train_path = f"{self.config.data.splits_path}/{self.config.data.train_file}"
            test_path = f"{self.config.data.splits_path}/{self.config.data.test_file}"
            if not os.path.exists(train_path):
                self.logger.warning(f"Train file not found at {train_path}")
                return
            self.train_initial_model(train_path)
            self.process_windows_and_detect_drift(test_path)
            self.evaluate_performance_degradation()
            report = self.generate_reports()
            self.logger.info("Full pipeline completed successfully")
            return report
        except Exception as e:
            self.logger.error(f"Pipeline failed: {str(e)}")
            self.alerts.add_alert('pipeline_failure', f"Pipeline failed: {str(e)}", 'error')
            raise

if __name__ == "__main__":
    pipeline = IDSDriftPipeline()
    raw_data_file = "data/raw/network_traffic.csv"
    if os.path.exists(raw_data_file):
        report = pipeline.run_full_pipeline(raw_data_file)
        print("Pipeline execution completed.")
    else:
        print(f"Raw data file not found: {raw_data_file}")