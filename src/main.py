import os
import sys
import time
import numpy as np
import pandas as pd
from datetime import datetime
from pathlib import Path

sys.path.append('./src')

from utils import setup_logging, load_config, save_figure, save_results
from data_processing import DataPreprocessor
from train import IDSModel
from drift_detect import DriftDetector
from evals import IDSEvaluator

def main():
    start_time = time.time()
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    config = load_config('../config/config.yaml')
    
    logger = setup_logging(
        config['paths']['logs_dir'],
        f"ids_pipeline_{timestamp}.log",
        config['logging']['log_level']
    )
    
    logger.info("=" * 70)
    logger.info("NETWORK INTRUSION DETECTION SYSTEM WITH DRIFT DETECTION")
    logger.info("=" * 70)
    logger.info(f"Start Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    logger.info(f"Configuration loaded")
    
    os.makedirs(config['paths']['figs_dir'], exist_ok=True)
    os.makedirs(config['paths']['results_dir'], exist_ok=True)
    os.makedirs(config['paths']['logs_dir'], exist_ok=True)
    os.makedirs(os.path.dirname(config['paths']['model_save']), exist_ok=True)
    os.makedirs(os.path.dirname(config['paths']['drift_detector_save']), exist_ok=True)
    
    raw_files = list(Path(config['data']['raw_path']).glob('*.csv'))
    logger.info(f"Found {len(raw_files)} CSV files in {config['data']['raw_path']}")
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 1: DATA PREPROCESSING")
    logger.info("=" * 70)
    
    preprocess_start = time.time()
    
    preprocessor = DataPreprocessor(config, logger)
    
    try:
        X_train_scaled, X_test_scaled, y_train, y_test, features = preprocessor.prepare_data(
            save_path=config['data']['processed_path']
        )
        
        logger.info(f"Training data shape: {X_train_scaled.shape}")
        logger.info(f"Test data shape: {X_test_scaled.shape}")
        logger.info(f"Number of features: {len(features)}")
        logger.info(f"Feature names (first 5): {features[:5]}")
        
        unique_labels, counts = np.unique(y_train, return_counts=True)
        logger.info(f"Classes in training data: {len(unique_labels)}")
        for label, count in zip(unique_labels, counts):
            logger.info(f"  Class {label}: {count} samples ({count/len(y_train)*100:.2f}%)")
        
    except Exception as e:
        logger.error(f"Error during data preparation: {str(e)}")
        import traceback
        traceback.print_exc()
        return
    
    preprocess_end = time.time()
    logger.info(f"Data preprocessing completed in {preprocess_end - preprocess_start:.2f} seconds")
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 2: MODEL TRAINING")
    logger.info("=" * 70)
    
    training_start = time.time()
    
    ids_model = IDSModel(config, logger)
    ids_model.train(X_train_scaled, y_train.values.ravel())
    
    logger.info("Model evaluation on test set:")
    metrics = ids_model.evaluate(X_test_scaled, y_test.values.ravel())
    
    ids_model.save_model(config['paths']['model_save'])
    preprocessor.save_preprocessor(config['paths']['scaler_save'])
    
    training_end = time.time()
    logger.info(f"Model training completed in {training_end - training_start:.2f} seconds")
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 3: DRIFT DETECTION SETUP")
    logger.info("=" * 70)
    
    drift_start = time.time()
    
    drift_detector = DriftDetector(config, logger)
    
    reference_samples = min(5000, len(X_train_scaled))
    logger.info(f"Using {reference_samples} samples as reference data")
    drift_detector.set_reference_data(
        X_train_scaled[:reference_samples], 
        features
    )
    
    logger.info("\nStarting drift detection with intentional drift simulation...")
    
    n_batches = 50
    drift_results = drift_detector.detect_drift_streaming(X_test_scaled, n_batches)
    
    drift_summary = drift_detector.get_drift_summary()
    drift_detector.save_detector(config['paths']['drift_detector_save'])
    
    drift_end = time.time()
    logger.info(f"Drift detection completed in {drift_end - drift_start:.2f} seconds")
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 4: VISUALIZATION AND ANALYSIS")
    logger.info("=" * 70)
    
    viz_start = time.time()
    
    evaluator = IDSEvaluator(config, logger)
    
    logger.info("Generating visualizations...")
    
    sample_size = min(5000, len(X_test_scaled))
    y_pred = ids_model.model.predict(X_test_scaled[:sample_size])
    y_true = y_test.values[:sample_size].ravel()
    
    fig_cm, cm = evaluator.plot_confusion_matrix(
        y_true, y_pred, 
        labels=['Benign', 'Malicious'],
        save_path=f"{config['paths']['figs_dir']}/confusion_matrix_{timestamp}.png"
    )
    
    if hasattr(ids_model.model, "predict_proba"):
        try:
            y_pred_proba = ids_model.model.predict_proba(X_test_scaled[:sample_size])
            fig_roc = evaluator.plot_roc_curve(
                y_true, y_pred_proba
            )
            if fig_roc[0] is not None:
                save_figure(fig_roc[0], f"roc_curve_{timestamp}.png", config['paths']['figs_dir'])
        except Exception as e:
            logger.warning(f"ROC curve generation failed: {str(e)}")

    if hasattr(ids_model.model, 'feature_importances_'):
        fig_feat = evaluator.plot_feature_importance(
            ids_model.model, features, top_n=15
        )
        if fig_feat[0] is not None:
            save_figure(fig_feat[0], f"feature_importance_{timestamp}.png", config['paths']['figs_dir'])
        
        fig_drift, magnitudes = evaluator.plot_drift_history(
            drift_results,
            save_path=f"{config['paths']['figs_dir']}/drift_history_{timestamp}.png"
        )
        
    fig_stats = evaluator.plot_drift_statistics(
        drift_detector,
        save_path=f"{config['paths']['figs_dir']}/drift_statistics_{timestamp}.png"
    )
    
    if hasattr(ids_model.model, 'feature_importances_'):
        fig_feat, importances = evaluator.plot_feature_importance(
            ids_model.model, features, top_n=20,
            save_path=f"{config['paths']['figs_dir']}/feature_importance_{timestamp}.png"
        )
    
    viz_end = time.time()
    logger.info(f"Visualization completed in {viz_end - viz_start:.2f} seconds")
    
    logger.info("\n" + "=" * 70)
    logger.info("PHASE 5: RESULTS SAVING")
    logger.info("=" * 70)
    
    final_results = {
        'timestamp': timestamp,
        'data_processing': {
            'training_samples': X_train_scaled.shape[0],
            'test_samples': X_test_scaled.shape[0],
            'features_count': len(features),
            'n_classes': len(np.unique(y_train)),
            'processing_time': preprocess_end - preprocess_start
        },
        'model_performance': metrics,
        'drift_detection': drift_summary,
        'execution_times': {
            'total': time.time() - start_time,
            'preprocessing': preprocess_end - preprocess_start,
            'training': training_end - training_start,
            'drift_detection': drift_end - drift_start,
            'visualization': viz_end - viz_start
        }
    }
    
    results_file = save_results(
        final_results, 
        f"results_{timestamp}.json",
        config['paths']['results_dir']
    )
    
    drift_alerts_file = save_results(
        drift_detector.drift_alerts,
        f"drift_alerts_{timestamp}.json",
        config['paths']['results_dir']
    )
    
    logger.info(f"Results saved to: {results_file}")
    logger.info(f"Drift alerts saved to: {drift_alerts_file}")
    
    total_time = time.time() - start_time
    
    logger.info("\n" + "=" * 70)
    logger.info("PIPELINE EXECUTION SUMMARY")
    logger.info("=" * 70)
    logger.info(f"Total execution time: {total_time:.2f} seconds")
    logger.info(f"Model accuracy: {metrics['accuracy']:.4f}")
    logger.info(f"Number of features: {len(features)}")
    logger.info(f"Total drift alerts: {drift_summary['total_alerts']}")
    logger.info(f"Drift rate: {drift_summary['drift_rate']:.2%}")
    logger.info(f"Figures saved to: {config['paths']['figs_dir']}")
    logger.info(f"Logs saved to: {config['paths']['logs_dir']}")
    logger.info(f"Results saved to: {config['paths']['results_dir']}")
    logger.info("=" * 70)
    
    print("\n" + "=" * 70)
    print("NETWORK IDS WITH DRIFT DETECTION - EXECUTION COMPLETE")
    print("=" * 70)
    print(f"✓ Total time: {total_time:.2f} seconds")
    print(f"✓ Model accuracy: {metrics['accuracy']:.4f}")
    print(f"✓ Number of features: {len(features)}")
    print(f"✓ Drift alerts detected: {drift_summary['total_alerts']}")
    print(f"✓ Drift rate: {drift_summary['drift_rate']:.2%}")
    print(f"✓ Figures saved in: {config['paths']['figs_dir']}")
    print(f"✓ Logs saved in: {config['paths']['logs_dir']}")
    print(f"✓ Results saved in: {config['paths']['results_dir']}")
    print("=" * 70)

if __name__ == "__main__":
    main()