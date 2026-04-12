"""
Model Training Pipeline for Credit Risk Prediction.

Trains 4 models, compares performance, and saves the best one.
Uses F1 score as primary metric (better for imbalanced data).
"""

import pandas as pd
import numpy as np
import os
import json
import joblib
import warnings
warnings.filterwarnings('ignore')

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, classification_report, confusion_matrix
)
from xgboost import XGBClassifier


def evaluate_model(model, X_test, y_test):
    """Compute all evaluation metrics."""
    y_pred = model.predict(X_test)
    y_prob = model.predict_proba(X_test)[:, 1]

    metrics = {
        'accuracy': round(accuracy_score(y_test, y_pred), 4),
        'precision': round(precision_score(y_test, y_pred, zero_division=0), 4),
        'recall': round(recall_score(y_test, y_pred, zero_division=0), 4),
        'f1_score': round(f1_score(y_test, y_pred, zero_division=0), 4),
        'roc_auc': round(roc_auc_score(y_test, y_prob), 4),
    }

    return metrics, y_pred, y_prob


def get_feature_importance(model, feature_names):
    """Extract feature importance from the model."""
    if hasattr(model, 'feature_importances_'):
        importance = model.feature_importances_
    elif hasattr(model, 'coef_'):
        importance = np.abs(model.coef_[0])
    else:
        return None

    feat_imp = pd.DataFrame({
        'feature': feature_names,
        'importance': importance
    }).sort_values('importance', ascending=False)

    return feat_imp


def train_models(X_train, y_train, X_test, y_test):
    """Train multiple models and return results."""

    # Calculate class weight ratio for XGBoost
    neg_count = (y_train == 0).sum()
    pos_count = (y_train == 1).sum()
    scale_ratio = neg_count / pos_count

    models = {
        'logistic_regression': LogisticRegression(
            max_iter=1000, random_state=42, class_weight='balanced'
        ),
        'random_forest': RandomForestClassifier(
            n_estimators=300, max_depth=12, min_samples_split=10,
            random_state=42, class_weight='balanced', n_jobs=-1
        ),
        'gradient_boosting': GradientBoostingClassifier(
            n_estimators=300, max_depth=5, learning_rate=0.1,
            min_samples_split=10, subsample=0.8, random_state=42
        ),
        'xgboost': XGBClassifier(
            n_estimators=300, max_depth=6, learning_rate=0.1,
            scale_pos_weight=scale_ratio, subsample=0.8,
            colsample_bytree=0.8, random_state=42,
            eval_metric='logloss', verbosity=0
        ),
    }

    results = {}

    for name, model in models.items():
        print(f"\n{'='*55}")
        print(f"  Training: {name}")
        print(f"{'='*55}")

        model.fit(X_train, y_train)
        metrics, y_pred, y_prob = evaluate_model(model, X_test, y_test)

        results[name] = {
            'model': model,
            'metrics': metrics,
            'y_pred': y_pred,
            'y_prob': y_prob
        }

        for metric_name, value in metrics.items():
            print(f"  {metric_name:<12}: {value:.4f}")

        print(f"\n{classification_report(y_test, y_pred, target_names=['Paid', 'Default'])}")

        cm = confusion_matrix(y_test, y_pred)
        print(f"  Confusion Matrix:")
        print(f"    TN={cm[0][0]:>5}  FP={cm[0][1]:>5}")
        print(f"    FN={cm[1][0]:>5}  TP={cm[1][1]:>5}")

    return results


def select_best_model(results):
    """Select best model by F1 score (best for imbalanced classification)."""
    best_name = max(results, key=lambda k: results[k]['metrics']['f1_score'])

    print(f"\n{'='*55}")
    print(f"  BEST MODEL: {best_name}")
    print(f"{'='*55}")
    for metric, value in results[best_name]['metrics'].items():
        print(f"  {metric:<12}: {value:.4f}")

    return best_name, results[best_name]


def run_training():
    """Main training pipeline."""
    base_dir = os.path.dirname(os.path.dirname(__file__))
    processed_dir = os.path.join(base_dir, 'data', 'processed')
    model_dir = os.path.join(base_dir, 'models')
    os.makedirs(model_dir, exist_ok=True)

    # Load processed data
    print("=" * 55)
    print("  CREDIT RISK MODEL TRAINING")
    print("=" * 55)

    X_train = pd.read_csv(os.path.join(processed_dir, 'X_train.csv'))
    X_test = pd.read_csv(os.path.join(processed_dir, 'X_test.csv'))
    y_train = pd.read_csv(os.path.join(processed_dir, 'y_train.csv')).values.ravel()
    y_test = pd.read_csv(os.path.join(processed_dir, 'y_test.csv')).values.ravel()

    print(f"\nTraining: {X_train.shape[0]} samples | Test: {X_test.shape[0]} samples")
    print(f"Features: {X_train.shape[1]}")
    print(f"Default rate - Train: {y_train.mean():.2%} | Test: {y_test.mean():.2%}")

    # Train all models
    results = train_models(X_train, y_train, X_test, y_test)

    # Select best
    best_name, best_result = select_best_model(results)
    best_model = best_result['model']

    # Save best model
    model_path = os.path.join(model_dir, 'best_model.pkl')
    joblib.dump(best_model, model_path)
    print(f"\nModel saved: {model_path}")

    # Save metadata
    metadata = {
        'model_name': best_name,
        'metrics': best_result['metrics'],
        'feature_names': list(X_train.columns),
        'training_samples': int(X_train.shape[0]),
        'test_samples': int(X_test.shape[0]),
        'default_rate': float(y_train.mean()),
    }
    with open(os.path.join(model_dir, 'model_metadata.json'), 'w') as f:
        json.dump(metadata, f, indent=2)

    # Feature importance
    feat_imp = get_feature_importance(best_model, X_train.columns)
    if feat_imp is not None:
        print(f"\nTop 10 Important Features:")
        for _, row in feat_imp.head(10).iterrows():
            bar = '#' * int(row['importance'] * 50)
            print(f"  {row['feature']:<30} {row['importance']:.4f}  {bar}")
        feat_imp.to_csv(os.path.join(model_dir, 'feature_importance.csv'), index=False)

    # Save model comparison
    comparison = {name: res['metrics'] for name, res in results.items()}
    comp_df = pd.DataFrame(comparison).T
    comp_df.to_csv(os.path.join(model_dir, 'model_comparison.csv'))

    print(f"\nAll Models Comparison:")
    print(comp_df.to_string())

    return best_model, results


if __name__ == "__main__":
    run_training()
