"""Raporlama çıktıları: grafikler, metrics.json, classification report."""
from __future__ import annotations

from datetime import datetime
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from PIL import Image
from sklearn.metrics import classification_report

from deepweeds.config import save_json
from deepweeds.metrics import MetricsResult


def _save_fig(fig: plt.Figure, path: Path, dpi: int = 150) -> None:
    """Grafik dosyasını kaydeder; üst klasör yoksa oluşturur."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=dpi)
    plt.close(fig)


def save_class_distribution(counts: pd.Series, class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8, 4))
    ax.bar(class_names, counts.values, color="seagreen")
    ax.set_title("Sınıf dağılımı (train)")
    ax.set_ylabel("görüntü sayısı")
    ax.tick_params(axis="x", rotation=45)
    for i, v in enumerate(counts.values):
        ax.text(i, v, str(int(v)), ha="center", va="bottom", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, path)


def save_training_curves(history: list[dict], path: Path) -> None:
    df = pd.DataFrame(history)
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    axes[0].plot(df["epoch"], df["train_loss"], label="train")
    axes[0].plot(df["epoch"], df["val_loss"], label="val")
    axes[0].set_title("Loss"); axes[0].set_xlabel("epoch"); axes[0].legend()
    axes[1].plot(df["epoch"], df["val_macro_f1"], label="val macro-F1")
    axes[1].plot(df["epoch"], df["val_accuracy"], label="val accuracy")
    axes[1].set_ylim(0, 1); axes[1].set_title("Başarı"); axes[1].set_xlabel("epoch"); axes[1].legend()
    fig.tight_layout()
    _save_fig(fig, path)


def save_confusion_matrix(cm: list[list[int]], class_names: list[str], path: Path) -> None:
    fig, ax = plt.subplots(figsize=(8.5, 7))
    sns.heatmap(np.asarray(cm), annot=True, fmt="d", cmap="Blues",
                xticklabels=class_names, yticklabels=class_names, ax=ax)
    ax.set_xlabel("Tahmin"); ax.set_ylabel("Gerçek"); ax.set_title("Val karışıklık matrisi")
    fig.tight_layout()
    _save_fig(fig, path)


def save_per_class_f1(per_class: dict[str, dict], path: Path) -> None:
    names = list(per_class)
    f1s = [per_class[n]["f1-score"] for n in names]
    sup = [per_class[n]["support"] for n in names]
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.barh(names, f1s, color="steelblue")
    ax.set_xlim(0, 1); ax.invert_yaxis()
    ax.set_title("Val sınıf bazlı F1")
    for i, (f, s) in enumerate(zip(f1s, sup)):
        ax.text(f + 0.01, i, f"{f:.3f} (n={s})", va="center", fontsize=8)
    fig.tight_layout()
    _save_fig(fig, path)


def save_misclassified(val_df: pd.DataFrame, y_true: np.ndarray, y_pred: np.ndarray,
                       y_prob: np.ndarray, class_names: list[str], path: Path, k: int = 12) -> None:
    conf = y_prob.max(1)
    wrong = np.where(y_true != y_pred)[0]
    if len(wrong) == 0:
        return
    wrong = wrong[np.argsort(-conf[wrong])][:k]
    cols = 4
    rows = int(np.ceil(len(wrong) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(3 * cols, 3.4 * rows))
    axes = np.atleast_1d(axes).ravel()
    for ax, i in zip(axes, wrong):
        ax.imshow(Image.open(val_df["_path"].iloc[i]))
        ax.set_title(f"G: {class_names[y_true[i]]}\nT: {class_names[y_pred[i]]} ({conf[i]:.2f})",
                     fontsize=8)
        ax.axis("off")
    for ax in axes[len(wrong):]:
        ax.axis("off")
    fig.suptitle("En güvenilir hatalar (val)")
    fig.tight_layout()
    _save_fig(fig, path, dpi=140)


def save_all(run_dir: Path, metrics: MetricsResult, history: list[dict], class_names: list[str],
             val_df: pd.DataFrame, y_true, y_pred, y_prob,
             extra: dict | None = None) -> Path:
    reports_dir = run_dir / "reports"
    reports_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(metrics.confusion, class_names, reports_dir / "confusion_matrix.png")
    save_per_class_f1(metrics.per_class, reports_dir / "per_class_f1.png")
    save_training_curves(history, reports_dir / "training_curves.png")
    save_misclassified(val_df, y_true, y_pred, y_prob, class_names,
                       reports_dir / "misclassified_grid.png")
    (reports_dir / "classification_report.txt").write_text(
        classification_report(y_true, y_pred, labels=list(range(len(class_names))),
                              target_names=class_names, digits=4, zero_division=0),
        encoding="utf-8")
    payload = {"macro_f1": metrics.macro_f1, "accuracy": metrics.accuracy,
               "per_class": metrics.per_class, "confusion": metrics.confusion,
               "n_val": int(len(y_true)), "generated_at": datetime.now().isoformat(timespec="seconds"),
               **(extra or {})}
    save_json(run_dir / "metrics.json", payload)
    return reports_dir