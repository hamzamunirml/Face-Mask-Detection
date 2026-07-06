"""
Phase 6 - Model Evaluation
Evaluates trained models on the test set using Accuracy, Precision, Recall,
F1-Score, Confusion Matrix, and ROC Curve. Generates a comparison report.
"""

import os
import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    confusion_matrix, roc_curve, auc, classification_report
)
from tensorflow.keras.models import load_model

from data_preprocessing import get_data_generators

MODELS_DIR = "../models"
RESULTS_DIR = "../results"
MODEL_FILES = {
    "custom_cnn": "custom_cnn_best.h5",
    "mobilenetv2": "mobilenetv2_best.h5",
    "resnet50": "resnet50_best.h5",
}


def evaluate_model(model_name, model_path, test_gen):
    print(f"\n{'='*60}\nEvaluating: {model_name}\n{'='*60}")
    model = load_model(model_path)

    test_gen.reset()
    y_true = test_gen.classes
    y_pred_probs = model.predict(test_gen).ravel()
    y_pred = (y_pred_probs >= 0.5).astype(int)

    metrics = {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision": precision_score(y_true, y_pred),
        "recall": recall_score(y_true, y_pred),
        "f1_score": f1_score(y_true, y_pred),
    }

    print(json.dumps(metrics, indent=2))
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=list(test_gen.class_indices.keys())))

    # Confusion Matrix
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=test_gen.class_indices.keys(),
                yticklabels=test_gen.class_indices.keys())
    plt.title(f"{model_name} - Confusion Matrix")
    plt.ylabel("True Label")
    plt.xlabel("Predicted Label")
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_confusion_matrix.png"))
    plt.close()

    # ROC Curve
    fpr, tpr, _ = roc_curve(y_true, y_pred_probs)
    roc_auc = auc(fpr, tpr)
    plt.figure(figsize=(5, 4))
    plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.3f}")
    plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
    plt.title(f"{model_name} - ROC Curve")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.legend()
    plt.tight_layout()
    plt.savefig(os.path.join(RESULTS_DIR, f"{model_name}_roc_curve.png"))
    plt.close()

    metrics["auc"] = roc_auc
    return metrics


def compare_models():
    os.makedirs(RESULTS_DIR, exist_ok=True)
    _, _, test_gen = get_data_generators()

    all_metrics = {}
    for model_name, filename in MODEL_FILES.items():
        model_path = os.path.join(MODELS_DIR, filename)
        if os.path.exists(model_path):
            all_metrics[model_name] = evaluate_model(model_name, model_path, test_gen)
        else:
            print(f"Skipping {model_name}: {model_path} not found. Train it first.")

    # Save comparison report
    report_path = os.path.join(RESULTS_DIR, "model_comparison_report.json")
    with open(report_path, "w") as f:
        json.dump(all_metrics, f, indent=2)
    print(f"\nSaved comparison report: {report_path}")

    # Print comparison table
    print(f"\n{'Model':<15}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}{'F1-Score':<12}{'AUC':<10}")
    for name, m in all_metrics.items():
        print(f"{name:<15}{m['accuracy']:<12.4f}{m['precision']:<12.4f}{m['recall']:<12.4f}{m['f1_score']:<12.4f}{m['auc']:<10.4f}")

    if all_metrics:
        best_model = max(all_metrics, key=lambda k: all_metrics[k]["accuracy"])
        print(f"\nBest performing model: {best_model}")

    return all_metrics


if __name__ == "__main__":
    compare_models()
