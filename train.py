import argparse
import json
import os
from dataclasses import dataclass
from typing import Tuple

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
from sklearn import datasets
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.model_selection import StratifiedKFold, cross_val_score, train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier


ARTIFACTS_DIR = os.path.join(os.path.dirname(__file__), "artifacts")


@dataclass
class Dataset:
    X: np.ndarray
    y: np.ndarray
    class_names: Tuple[str, ...]


def load_dataset(name: str) -> Dataset:
    name = name.lower()
    if name == "breast_cancer":
        data = datasets.load_breast_cancer()
    elif name == "digits":
        data = datasets.load_digits()
    elif name == "iris":
        data = datasets.load_iris()
    else:
        raise ValueError(f"Unsupported dataset: {name}")

    X = data.data
    y = data.target
    class_names = tuple(data.target_names) if hasattr(data, "target_names") else tuple(str(i) for i in np.unique(y))
    return Dataset(X=X, y=y, class_names=class_names)


def build_pipeline(model_name: str) -> Pipeline:
    model_name = model_name.lower()
    if model_name == "logreg":
        model = LogisticRegression(max_iter=5000, n_jobs=None)
        steps = [("scaler", StandardScaler()), ("clf", model)]
    elif model_name == "rf":
        model = RandomForestClassifier(
            n_estimators=300,
            max_depth=None,
            min_samples_split=2,
            min_samples_leaf=1,
            n_jobs=-1,
            random_state=42,
        )
        steps = [("clf", model)]
    elif model_name == "gbdt":
        model = GradientBoostingClassifier(random_state=42)
        steps = [("clf", model)]
    else:
        raise ValueError(f"Unsupported model: {model_name}")

    return Pipeline(steps)


def ensure_artifacts_dir() -> str:
    os.makedirs(ARTIFACTS_DIR, exist_ok=True)
    return ARTIFACTS_DIR


def plot_and_save_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray, class_names: Tuple[str, ...], out_path: str) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False, ax=ax,
                xticklabels=class_names, yticklabels=class_names)
    ax.set_title("Confusion Matrix")
    ax.set_xlabel("Predicted label")
    ax.set_ylabel("True label")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def plot_and_save_roc(y_true: np.ndarray, y_proba: np.ndarray, out_path: str) -> None:
    if y_proba.ndim == 2 and y_proba.shape[1] == 2:
        scores = y_proba[:, 1]
    else:
        # Not binary or probabilities not available
        return

    fpr, tpr, _ = roc_curve(y_true, scores)
    auc = roc_auc_score(y_true, scores)
    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"ROC AUC = {auc:.3f}")
    ax.plot([0, 1], [0, 1], linestyle="--", color="gray")
    ax.set_xlabel("False Positive Rate")
    ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve")
    ax.legend(loc="lower right")
    plt.tight_layout()
    fig.savefig(out_path)
    plt.close(fig)


def main() -> None:
    parser = argparse.ArgumentParser(description="Train and evaluate a classifier with sklearn")
    parser.add_argument("--dataset", type=str, default="breast_cancer", choices=["breast_cancer", "digits", "iris"], help="Dataset to use")
    parser.add_argument("--model", type=str, default="rf", choices=["logreg", "rf", "gbdt"], help="Model to use")
    parser.add_argument("--test-size", type=float, default=0.2, help="Test size fraction")
    parser.add_argument("--cv", type=int, default=5, help="Number of CV folds")
    parser.add_argument("--random-state", type=int, default=42, help="Random state for reproducibility")
    args = parser.parse_args()

    dataset = load_dataset(args.dataset)
    X, y = dataset.X, dataset.y

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=args.test_size, stratify=y, random_state=args.random_state
    )

    pipeline = build_pipeline(args.model)

    cv = StratifiedKFold(n_splits=args.cv, shuffle=True, random_state=args.random_state)
    cv_scores = cross_val_score(pipeline, X_train, y_train, cv=cv, scoring="accuracy", n_jobs=-1)

    pipeline.fit(X_train, y_train)
    y_pred = pipeline.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred)
    report = classification_report(y_test, y_pred, target_names=dataset.class_names, zero_division=0)

    proba_supported = hasattr(pipeline[-1], "predict_proba")
    y_proba = pipeline.predict_proba(X_test) if proba_supported else None

    ensure_artifacts_dir()
    metrics = {
        "dataset": args.dataset,
        "model": args.model,
        "test_accuracy": float(accuracy),
        "cv_accuracy_mean": float(np.mean(cv_scores)),
        "cv_accuracy_std": float(np.std(cv_scores)),
        "n_train": int(len(y_train)),
        "n_test": int(len(y_test)),
    }
    with open(os.path.join(ARTIFACTS_DIR, "metrics.json"), "w") as f:
        json.dump(metrics, f, indent=2)
    with open(os.path.join(ARTIFACTS_DIR, "classification_report.txt"), "w") as f:
        f.write(report)

    plot_and_save_confusion_matrix(
        y_true=y_test,
        y_pred=y_pred,
        class_names=dataset.class_names,
        out_path=os.path.join(ARTIFACTS_DIR, "confusion_matrix.png"),
    )

    if y_proba is not None and len(np.unique(y)) == 2:
        plot_and_save_roc(
            y_true=y_test,
            y_proba=y_proba,
            out_path=os.path.join(ARTIFACTS_DIR, "roc_curve.png"),
        )

    joblib.dump(pipeline, os.path.join(ARTIFACTS_DIR, "model.joblib"))

    print("==== Results ====")
    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:\n")
    print(report)


if __name__ == "__main__":
    main()
