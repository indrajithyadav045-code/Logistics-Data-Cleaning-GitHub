"""
predictive_modeling.py
======================
Predictive machine learning models for delivery delay risk classification
and transit duration regression for the Week 6 Logistics Project.
"""
from __future__ import annotations

import sys
import json
from pathlib import Path
import numpy as np
import pandas as pd

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, Ridge
from sklearn.ensemble import (
    RandomForestClassifier,
    HistGradientBoostingClassifier,
    HistGradientBoostingRegressor,
)
from sklearn.metrics import (
    roc_auc_score,
    average_precision_score,
    f1_score,
    precision_score,
    recall_score,
    accuracy_score,
    mean_absolute_error,
    mean_squared_error,
    r2_score,
)

sys.path.insert(0, str(Path(__file__).resolve().parents[2] / "week6" / "config"))
from project_config import DATA_PROCESSED, DATA_RESULTS, RANDOM_STATE, TEST_SIZE


def train_predictive_models(df: pd.DataFrame | None = None) -> dict:
    """Train, validate, and compare predictive models for delay risk and transit duration.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Processed orders dataset.

    Returns
    -------
    dict: Complete evaluation metrics, feature importances, and model comparisons.
    """
    print("\n--- [Pillar 3: Predictive Machine Learning Modeling] ---")
    if df is None:
        input_path = DATA_PROCESSED / "consolidated_logistics_orders.csv"
        df = pd.read_csv(input_path, low_memory=False)

    feature_cols = [
        "distance_km",
        "carrier_dispatch_days",
        "product_weight_g",
        "product_volume_cm3",
        "total_freight",
        "total_price",
        "freight_ratio",
        "is_interstate",
        "purchase_dow",
        "purchase_hour",
        "purchase_month",
        "purchase_is_weekend",
    ]

    # Add region dummies
    region_dummies = pd.get_dummies(df[["seller_region", "customer_region"]], drop_first=True)
    X = pd.concat([df[feature_cols], region_dummies], axis=1)
    
    # Target variables
    y_class = df["is_delayed"].values
    y_reg = df["delivery_days"].values

    # Train/Test Split (80/20 stratified)
    X_train, X_test, y_class_train, y_class_test, y_reg_train, y_reg_test = train_test_split(
        X, y_class, y_reg, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_class
    )
    print(f"Dataset split: Train = {X_train.shape[0]:,} rows, Test = {X_test.shape[0]:,} rows")
    print(f"Class distribution: {np.mean(y_class_train)*100:.2f}% positive (delayed)")

    # Standardize numerical features for linear baseline
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # -----------------------------------------------------------------------
    # Task A: Delivery Delay Classification (SLA Breach Risk)
    # -----------------------------------------------------------------------
    print("\n--- Training Classification Models (Delivery Delay Risk) ---")
    models_cls = {
        "Logistic Regression (Baseline)": LogisticRegression(
            max_iter=1000, class_weight="balanced", random_state=RANDOM_STATE
        ),
        "Random Forest Classifier": RandomForestClassifier(
            n_estimators=100, max_depth=12, min_samples_leaf=10,
            class_weight="balanced", random_state=RANDOM_STATE, n_jobs=-1
        ),
        "HistGradientBoosting Classifier": HistGradientBoostingClassifier(
            max_iter=120, max_depth=8, min_samples_leaf=20,
            class_weight="balanced", random_state=RANDOM_STATE
        ),
    }

    cls_results = []
    cls_predictions = {"y_true": y_class_test.tolist()}

    for name, model in models_cls.items():
        print(f"Training {name}...")
        X_tr = X_train_scaled if "Logistic" in name else X_train
        X_te = X_test_scaled if "Logistic" in name else X_test

        model.fit(X_tr, y_class_train)
        y_prob = model.predict_proba(X_te)[:, 1]
        y_pred = (y_prob >= 0.50).astype(int)

        roc = float(roc_auc_score(y_class_test, y_prob))
        pr_auc = float(average_precision_score(y_class_test, y_prob))
        f1 = float(f1_score(y_class_test, y_pred))
        prec = float(precision_score(y_class_test, y_pred, zero_division=0))
        rec = float(recall_score(y_class_test, y_pred))
        acc = float(accuracy_score(y_class_test, y_pred))

        cls_predictions[name] = y_prob.tolist()

        res = {
            "model": name,
            "roc_auc": roc,
            "pr_auc": pr_auc,
            "f1_score": f1,
            "precision": prec,
            "recall": rec,
            "accuracy": acc,
        }
        cls_results.append(res)
        print(f"  {name:35s} | ROC-AUC: {roc:.4f} | PR-AUC: {pr_auc:.4f} | Recall: {rec:.4f} | F1: {f1:.4f}")

    cls_df = pd.DataFrame(cls_results)
    cls_path = DATA_RESULTS / "classification_model_metrics.csv"
    cls_df.to_csv(cls_path, index=False)

    # -----------------------------------------------------------------------
    # Task B: Delivery Duration Regression (Days)
    # -----------------------------------------------------------------------
    print("\n--- Training Regression Models (Total Delivery Duration Days) ---")
    models_reg = {
        "Ridge Regression (Baseline)": Ridge(alpha=1.0, random_state=RANDOM_STATE),
        "HistGradientBoosting Regressor": HistGradientBoostingRegressor(
            max_iter=120, max_depth=8, min_samples_leaf=20, random_state=RANDOM_STATE
        ),
    }

    reg_results = []
    best_reg_model = None

    for name, model in models_reg.items():
        print(f"Training {name}...")
        X_tr = X_train_scaled if "Ridge" in name else X_train
        X_te = X_test_scaled if "Ridge" in name else X_test

        model.fit(X_tr, y_reg_train)
        y_pred_reg = model.predict(X_te)

        mae = float(mean_absolute_error(y_reg_test, y_pred_reg))
        rmse = float(np.sqrt(mean_squared_error(y_reg_test, y_pred_reg)))
        r2 = float(r2_score(y_reg_test, y_pred_reg))

        res = {
            "model": name,
            "mae_days": mae,
            "rmse_days": rmse,
            "r2_score": r2,
        }
        reg_results.append(res)
        print(f"  {name:35s} | MAE: {mae:.2f}d | RMSE: {rmse:.2f}d | R²: {r2:.4f}")
        if "HistGradientBoosting" in name:
            best_reg_model = model

    reg_df = pd.DataFrame(reg_results)
    reg_path = DATA_RESULTS / "regression_model_metrics.csv"
    reg_df.to_csv(reg_path, index=False)

    # -----------------------------------------------------------------------
    # Feature Importance (from Random Forest Classifier)
    # -----------------------------------------------------------------------
    rf_model = models_cls["Random Forest Classifier"]
    importances = rf_model.feature_importances_
    feat_names = X_train.columns.tolist()
    feat_df = pd.DataFrame({
        "feature": feat_names,
        "importance": importances
    }).sort_values("importance", ascending=False).reset_index(drop=True)
    feat_path = DATA_RESULTS / "feature_importance.csv"
    feat_df.to_csv(feat_path, index=False)
    print("\nTop 10 Feature Importances (Random Forest):")
    print(feat_df.head(10).to_string(index=False))

    all_metrics = {
        "classification_metrics": cls_results,
        "regression_metrics": reg_results,
        "top_features": feat_df.head(10).to_dict(orient="records"),
    }
    with open(DATA_RESULTS / "predictive_model_metrics.json", "w") as f:
        json.dump(all_metrics, f, indent=2)

    return all_metrics


if __name__ == "__main__":
    train_predictive_models()
