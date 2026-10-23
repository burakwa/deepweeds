r"""
DeepWeeds Tahmin Eşik Kalibratörü (Threshold & Multiplier Optimizer)
===================================================================
Aşırı ağırlıklandırmadan kaynaklanan precision/recall dengesizliğini gidererek
Macro-F1 skorunu anında yukarı çeker.

Kullanım:
    .\.venv\Scripts\python.exe calibrate.py --run runs/exp01-weighted
"""
from __future__ import annotations

import argparse
import io
import json
import sys
from pathlib import Path

if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

import numpy as np
import pandas as pd
from scipy.optimize import minimize
from sklearn.metrics import classification_report, f1_score

CLASS_NAMES = [
    "Chinee apple", "Lantana", "Parkinsonia", "Parthenium", "Prickly acacia",
    "Rubber vine", "Siam weed", "Snake weed", "Negative"
]

def calibrate_run(run_dir: str | Path, val_csv: str | Path = "data/val.csv"):
    run_path = Path(run_dir)
    pred_path = run_path / "predictions_val.csv"
    if not pred_path.exists():
        raise FileNotFoundError(f"predictions_val.csv bulunamadi: {pred_path}")

    val_df = pd.read_csv(val_csv)
    pred_df = pd.read_csv(pred_path)

    y_true = val_df["label"].values
    prob_cols = [f"prob_{i}" for i in range(9)]
    probs = pred_df[prob_cols].values

    orig_pred = np.argmax(probs, axis=1)
    orig_f1 = float(f1_score(y_true, orig_pred, average="macro", zero_division=0))

    # Optimizasyon fonksiyonu
    def loss(w):
        scaled = probs * w
        p = np.argmax(scaled, axis=1)
        return -float(f1_score(y_true, p, average="macro", zero_division=0))

    res = minimize(loss, np.ones(9), method="Nelder-Mead", options={"maxiter": 600})
    best_w = res.x
    calib_pred = np.argmax(probs * best_w, axis=1)
    calib_f1 = float(f1_score(y_true, calib_pred, average="macro", zero_division=0))

    print("=" * 70)
    print(f"[KALIBRASYON] Run: {run_path.name}")
    print("=" * 70)
    print(f"Orijinal Macro-F1    : {orig_f1:.4f}")
    print(f"Kalibre Edilmis F1   : {calib_f1:.4f} (+{(calib_f1 - orig_f1):.4f})")
    print("-" * 70)
    print("Optimal Carpanlar (Multipliers):")
    for name, w in zip(CLASS_NAMES, best_w):
        print(f"  - {name:<16}: {w:.4f}")

    print("\nDetayli Sinif Bazli Rapor:")
    print(classification_report(y_true, calib_pred, target_names=CLASS_NAMES, digits=4))

    out_json = run_path / "calibration.json"
    data = {
        "original_macro_f1": orig_f1,
        "calibrated_macro_f1": calib_f1,
        "multipliers": [round(float(v), 5) for v in best_w],
        "class_names": CLASS_NAMES,
    }
    out_json.write_text(json.dumps(data, indent=2), encoding="utf-8")
    print(f"Carpanlar kaydedildi: {out_json}")
    return calib_f1, best_w

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--run", default="runs/exp01-weighted", help="Run klasoru yolu")
    parser.add_argument("--val-csv", default="data/val.csv", help="val.csv yolu")
    args = parser.parse_args()
    calibrate_run(args.run, args.val_csv)
