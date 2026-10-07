"""Trains the ML engine's classifiers and saves them, with their evaluation metrics, as reproducible artifacts.

Run from the repository root:
    python -m src.train

Writes:
    models/random_forest.joblib   the exact default model the app trains on startup
    models/decision_tree.joblib   the single-tree baseline, trained on the same split
    models/metrics.json           accuracy, cross-validation, per-career report, confusion matrix, versions
    data/benchmark_dataset.csv    the synthetic training cohort

The app does not load these files: it retrains the identical model in memory on startup (under a second), which
avoids pickle incompatibilities when the deployed scikit-learn version differs from the one that saved them.
"""

import datetime
import json
import os
import platform
from typing import Any, Dict

import joblib
import numpy as np
import sklearn
from sklearn.metrics import classification_report, confusion_matrix

from src.knowledge_base import CAREER_UNIVERSE
from src.ml_engine import (
    DEFAULT_MAX_DEPTH, DEFAULT_N_ESTIMATORS, DEFAULT_SAMPLES_PER_CLASS, FEATURE_NAMES, MODEL_LABELS,
    compare_classifiers, generate_benchmark_dataset, train_custom_classifier,
)
from src.paths import DATA_DIR, MODELS_DIR


def train_and_save(models_dir: str = MODELS_DIR, data_dir: str = DATA_DIR) -> Dict[str, Any]:
    """Trains both classifiers on the default benchmark cohort, saves them and returns the metrics written."""
    os.makedirs(models_dir, exist_ok=True)
    df = generate_benchmark_dataset(samples_per_class=DEFAULT_SAMPLES_PER_CLASS)
    df.to_csv(os.path.join(data_dir, "benchmark_dataset.csv"), index=False)

    careers = list(CAREER_UNIVERSE)
    cv = compare_classifiers(df, n_estimators=DEFAULT_N_ESTIMATORS, max_depth=DEFAULT_MAX_DEPTH).set_index("Model")
    models: Dict[str, Any] = {}
    for model_type in ["random_forest", "decision_tree"]:
        res = train_custom_classifier(df, n_estimators=DEFAULT_N_ESTIMATORS, max_depth=DEFAULT_MAX_DEPTH, model_type=model_type)
        if "error" in res:
            raise RuntimeError(f"{model_type} training failed: {res['error']}")
        joblib.dump(res["model"], os.path.join(models_dir, f"{model_type}.joblib"))

        y_pred = res["model"].predict(res["X_test"])
        labels = list(range(len(careers)))
        report = classification_report(res["y_test"], y_pred, labels=labels, target_names=careers, output_dict=True, zero_division=0)
        importances = sorted(zip(FEATURE_NAMES, res["model"].feature_importances_), key=lambda x: x[1], reverse=True)
        label = MODEL_LABELS[model_type]
        models[model_type] = {
            "file": f"models/{model_type}.joblib",
            "hyperparameters": {"n_estimators": DEFAULT_N_ESTIMATORS if model_type == "random_forest" else None,
                                "max_depth": DEFAULT_MAX_DEPTH, "random_state": 42},
            "train_accuracy": round(float(res["train_acc"]), 4),
            "test_accuracy": round(float(res["test_acc"]), 4),
            "cv_5fold_mean": round(float(cv.loc[label, "CV Mean"]), 4),
            "cv_5fold_std": round(float(cv.loc[label, "CV Std"]), 4),
            "per_career": {c: {k: round(float(report[c][k]), 4) for k in ("precision", "recall", "f1-score")} for c in careers},
            "confusion_matrix": {"labels": careers, "rows_true_cols_predicted": confusion_matrix(res["y_test"], y_pred, labels=labels).tolist()},
            "top_features": [{"feature": f, "importance": round(float(v), 4)} for f, v in importances[:10]],
        }

    metrics = {
        "generated_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "versions": {"python": platform.python_version(), "scikit_learn": sklearn.__version__, "numpy": np.__version__, "joblib": joblib.__version__},
        "dataset": {
            "file": "data/benchmark_dataset.csv",
            "source": "synthetic cohort from generate_benchmark_dataset() (seed 42), sampled around each career's benchmark skills",
            "samples_per_class": DEFAULT_SAMPLES_PER_CLASS,
            "rows": int(len(df)),
            "classes": careers,
            "features": FEATURE_NAMES,
            "split": {"test_size": 0.2, "stratified": True, "random_state": 42,
                      "train_rows": int(round(len(df) * 0.8)), "test_rows": int(len(df) - round(len(df) * 0.8))},
        },
        "caveat": "The cohort is generated from the same benchmark matrix the rule and fuzzy engines use, so these "
                  "scores show internal consistency, not accuracy on real students.",
        "models": models,
    }
    with open(os.path.join(models_dir, "metrics.json"), "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2)
    return metrics


if __name__ == "__main__":
    m = train_and_save()
    for model_type, info in m["models"].items():
        print(f"{MODEL_LABELS[model_type]:14s} test {info['test_accuracy']:.1%} | 5-fold CV {info['cv_5fold_mean']:.1%} "
              f"(± {info['cv_5fold_std']:.1%}) -> {info['file']}")
    print(f"Dataset: {m['dataset']['rows']} rows -> {m['dataset']['file']}; metrics -> models/metrics.json")
