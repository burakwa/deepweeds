from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix, f1_score


@dataclass(slots=True)
class MetricsResult:
    macro_f1: float
    accuracy: float
    per_class: dict[str, dict]
    confusion: list[list[int]]


def compute_metrics(y_true, y_pred, class_names: list[str]) -> MetricsResult:
    y_true, y_pred = np.asarray(y_true), np.asarray(y_pred)
    labels = list(range(len(class_names)))
    report = classification_report(y_true, y_pred, labels=labels, target_names=class_names,
                                   output_dict=True, zero_division=0)
    return MetricsResult(
        macro_f1=float(f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)),
        accuracy=float(accuracy_score(y_true, y_pred)),
        per_class={name: report[name] for name in class_names},
        confusion=confusion_matrix(y_true, y_pred, labels=labels).tolist(),
    )