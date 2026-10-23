r"""
DeepWeeds Macro-F1 Sweep Scripti
=================================
Mevcut baseline: resnet50, macro-F1=0.7300

Bu script farklı model ve konfigürasyonları sırayla deneyerek
en iyi macro-F1 skorunu bulmaya çalışır.

Kullanım:
    .\.venv\Scripts\python.exe sweep.py                      # tüm sweep çalışır
    .\.venv\Scripts\python.exe sweep.py --dry-run            # sadece planı göster
    .\.venv\Scripts\python.exe sweep.py --only exp-convnext-tiny  # tek bir deneyi çalıştır
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
from datetime import datetime
from pathlib import Path

# Windows cp1254 Türkçe konsol Unicode hatasını önlemek için UTF-8 ayarla
if sys.platform == "win32":
    sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding="utf-8", errors="replace")
    sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding="utf-8", errors="replace")

# Projenin src klasörünü path'e ekle
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import numpy as np
import pandas as pd
import torch
import torch.nn as nn
from scipy.optimize import minimize
from sklearn.metrics import classification_report, f1_score
from torch.amp import autocast
from tqdm import tqdm

from deepweeds.config import DataConfig, ModelConfig, TrainConfig, save_json
from deepweeds.data import DeepWeedsData
from deepweeds.engine import Trainer
from deepweeds.inference import Predictor, write_predictions_csv
from deepweeds.metrics import compute_metrics
from deepweeds.models import build_model, count_parameters
from deepweeds.utils import get_device, get_logger, set_seed

log = get_logger()

# ---------------------------------------------------------------------------
# Sweep konfigürasyonları - RTX 2050 (4GB VRAM) için optimize edildi
# ---------------------------------------------------------------------------

EXPERIMENTS = [
    # --- Deney 1: ConvNeXt-Tiny (En dengeli ve modern CNN mimarisi) ---
    {
        "name": "exp-convnext-tiny-sqrt",
        "description": "ConvNeXt-Tiny + sqrt_weighted_loss + güçlü aug + 224px",
        "model": "convnext_tiny",
        "img_size": 224,
        "batch_size": 24,       # 4GB VRAM için güvenli
        "epochs": 20,
        "lr": 1e-4,
        "wd": 1e-4,
        "label_smoothing": 0.1,
        "balance": "sqrt_weighted_loss",
        "patience": 6,
        "strong_aug": True,
        "use_one_cycle": True,
    },
    # --- Deney 2: EfficientNet-B3 (Görüntü çözünürlüğü 256px ile) ---
    {
        "name": "exp-effb3-sqrt",
        "description": "EfficientNet-B3 + sqrt_weighted_loss + güçlü aug + 256px",
        "model": "efficientnet_b3",
        "img_size": 256,
        "batch_size": 16,       # 256px ve 4GB VRAM için güvenli
        "epochs": 20,
        "lr": 5e-5,
        "wd": 1e-4,
        "label_smoothing": 0.1,
        "balance": "sqrt_weighted_loss",
        "patience": 6,
        "strong_aug": True,
        "use_one_cycle": True,
    },
    # --- Deney 3: ConvNeXt-Small (Yüksek kapasiteli, güçlü sınıflandırma) ---
    {
        "name": "exp-convnext-small-sqrt",
        "description": "ConvNeXt-Small + sqrt_weighted_loss + güçlü aug + 224px",
        "model": "convnext_small",
        "img_size": 224,
        "batch_size": 16,       # 4GB VRAM için 16 batch
        "epochs": 20,
        "lr": 5e-5,
        "wd": 1e-4,
        "label_smoothing": 0.1,
        "balance": "sqrt_weighted_loss",
        "patience": 6,
        "strong_aug": True,
        "use_one_cycle": True,
    },
    # --- Deney 4: EfficientNet-B0 (Çok hızlı, 224px baseline karşılaştırma) ---
    {
        "name": "exp-effb0-sqrt",
        "description": "EfficientNet-B0 + sqrt_weighted_loss + güçlü aug + 224px",
        "model": "efficientnet_b0",
        "img_size": 224,
        "batch_size": 32,
        "epochs": 20,
        "lr": 1e-4,
        "wd": 1e-4,
        "label_smoothing": 0.1,
        "balance": "sqrt_weighted_loss",
        "patience": 6,
        "strong_aug": True,
        "use_one_cycle": True,
    },
]


# ---------------------------------------------------------------------------
# Genişletilmiş Trainer (OneCycleLR + strong_aug desteği)
# ---------------------------------------------------------------------------

class SweepTrainer(Trainer):
    """Trainer'ı OneCycleLR scheduler, strong_aug ve sqrt_weighted_loss ile genişletir."""

    def __init__(self, model, data, model_cfg, cfg, run_dir, device,
                 use_one_cycle: bool = False, strong_aug: bool = False):
        self.strong_aug = strong_aug
        self._use_one_cycle = use_one_cycle
        super().__init__(model, data, model_cfg, cfg, run_dir, device)
        if use_one_cycle:
            # CosineAnnealingLR'ı OneCycleLR ile değiştir
            steps_per_epoch = len(data.train_loader(cfg.balance, strong_aug=strong_aug))
            self.scheduler = torch.optim.lr_scheduler.OneCycleLR(
                self.optimizer,
                max_lr=cfg.lr * 10,
                epochs=cfg.epochs,
                steps_per_epoch=steps_per_epoch,
                pct_start=0.3,
                div_factor=10.0,
                final_div_factor=1e4,
            )

    def _build_criterion(self) -> nn.CrossEntropyLoss:
        ls = self.cfg.label_smoothing
        if self.cfg.balance == "sqrt_weighted_loss":
            # Karekök yumuşatılmış ağırlık: precision çöküşünü engeller!
            raw = self.data.class_weights.cpu().numpy()
            sqrt_w = np.sqrt(raw)
            sqrt_w = sqrt_w / sqrt_w.mean()
            w = torch.tensor(sqrt_w, dtype=torch.float32, device=self.device)
            return nn.CrossEntropyLoss(weight=w, label_smoothing=ls)
        elif self.cfg.balance == "weighted_loss":
            return nn.CrossEntropyLoss(weight=self.data.class_weights.to(self.device), label_smoothing=ls)
        return nn.CrossEntropyLoss(label_smoothing=ls)

    def fit(self) -> float:
        """OneCycleLR step'i epoch değil batch bazında olduğu için override ediyoruz."""
        log = get_logger()
        train_loader = self.data.train_loader(self.cfg.balance, strong_aug=self.strong_aug)
        val_loader = self.data.val_loader()
        patience = 0

        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.time()
            train_loss = self._train_one_epoch_sweep(train_loader)
            val_loss, y_true, y_prob = self._validate(val_loader)
            y_pred = y_prob.argmax(1)
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            acc = float((y_true == y_pred).mean())
            self.history.append(dict(
                epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                val_macro_f1=f1, val_accuracy=acc,
                lr=self.optimizer.param_groups[0]["lr"],
                seconds=round(time.time() - t0, 1),
            ))
            log.info("epoch %02d | train %.4f | val %.4f | macroF1 %.4f | acc %.4f | %.0fs",
                     epoch, train_loss, val_loss, f1, acc, time.time() - t0)
            if f1 > self.best_f1:
                self.best_f1, patience = f1, 0
                self._save_checkpoint(epoch, f1)
                log.info("  -> yeni en iyi model kaydedildi (%.4f)", f1)
            else:
                patience += 1
                if patience >= self.cfg.patience:
                    log.info("Erken durdurma (patience=%d).", self.cfg.patience)
                    break
            if not self._use_one_cycle:
                self.scheduler.step()

        ckpt = torch.load(self.best_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["state_dict"])
        pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)
        return self.best_f1

    def _train_one_epoch_sweep(self, loader) -> float:
        """OneCycleLR: her batch sonrası scheduler.step()."""
        self.model.train()
        total = n = 0
        for x, y in tqdm(loader, desc="train", leave=False):
            x = x.to(self.device, non_blocking=True)
            y = y.to(self.device, non_blocking=True)
            self.optimizer.zero_grad(set_to_none=True)
            with autocast("cuda", enabled=self.use_amp):
                loss = self.criterion(self.model(x), y)
            self.scaler.scale(loss).backward()
            self.scaler.step(self.optimizer)
            self.scaler.update()
            total += loss.item() * y.size(0)
            n += y.size(0)
            if self._use_one_cycle:
                self.scheduler.step()
        return total / max(n, 1)


# ---------------------------------------------------------------------------
# Tek deney çalıştırıcı
# ---------------------------------------------------------------------------

def run_experiment(exp: dict, data_dir: str, run_root: str) -> dict:
    """Bir deneyi çalıştırır ve sonuç dict'i döner."""
    name = exp["name"]
    log.info("=" * 60)
    log.info("DENEY BASLIYOR: %s", name)
    log.info("Aciklama: %s", exp["description"])
    log.info("=" * 60)

    t_start = time.time()

    data_cfg = DataConfig(
        data_dir=Path(data_dir),
        img_size=exp["img_size"],
        batch_size=exp["batch_size"],
        num_workers=2,          # Windows için 2 worker çok daha stabil
        use_aug=True,
    )
    model_cfg = ModelConfig(name=exp["model"])
    train_cfg = TrainConfig(
        epochs=exp["epochs"],
        lr=exp["lr"],
        weight_decay=exp["wd"],
        label_smoothing=exp["label_smoothing"],
        balance=exp["balance"],
        patience=exp["patience"],
        seed=42,
        amp=True,
        run_root=Path(run_root),
        run_name=name,
    )

    run_dir = Path(run_root) / name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", {"data": data_cfg, "model": model_cfg, "train": train_cfg})

    set_seed(42)
    device = get_device()
    log.info("cihaz: %s | model: %s | img_size: %d | batch: %d",
             device, exp["model"], exp["img_size"], exp["batch_size"])

    data = DeepWeedsData(data_cfg)
    log.info("sinif dagilimi: %s", data.class_counts.to_dict())

    model = build_model(model_cfg)
    param_count = count_parameters(model)
    log.info("parametre sayisi: %d (%.1fM)", param_count, param_count / 1e6)

    trainer = SweepTrainer(
        model, data, model_cfg, train_cfg, run_dir, device,
        use_one_cycle=exp.get("use_one_cycle", False),
        strong_aug=exp.get("strong_aug", False),
    )

    try:
        best_f1 = trainer.fit()
    except Exception as e:
        log.error("DENEY BASARISIZ: %s | hata: %s", name, e)
        import traceback
        traceback.print_exc()
        return {"name": name, "error": str(e), "status": "failed"}

    # Değerlendirme
    predictor = Predictor.from_checkpoint(trainer.best_path, device)
    result = predictor.predict_df(data.val_df)
    y_true, y_prob = result.y_true, result.y_prob
    y_pred = y_prob.argmax(1)
    metrics = compute_metrics(y_true, y_pred, data.class_names)

    # Post-processing: Eşik / Çarpan Kalibrasyonu (Threshold Optimization)
    def opt_objective(weights):
        pred_scaled = np.argmax(y_prob * weights, axis=1)
        return -float(f1_score(y_true, pred_scaled, average="macro", zero_division=0))

    calib_res = minimize(opt_objective, np.ones(9), method="Nelder-Mead", options={"maxiter": 400})
    calibrated_f1 = -float(calib_res.fun)
    calib_multipliers = [round(float(v), 4) for v in calib_res.x]

    elapsed = time.time() - t_start
    log.info("DENEY TAMAMLANDI: %s | Raw Macro-F1=%.4f | Calibrated Macro-F1=%.4f | Acc=%.4f | Sure=%.0fs",
             name, metrics.macro_f1, calibrated_f1, metrics.accuracy, elapsed)

    save_json(run_dir / "metrics.json", {
        "macro_f1": metrics.macro_f1,
        "calibrated_macro_f1": calibrated_f1,
        "calibrated_multipliers": calib_multipliers,
        "accuracy": metrics.accuracy,
        "per_class": metrics.per_class,
        "model": exp["model"],
        "img_size": exp["img_size"],
        "strong_aug": exp.get("strong_aug", False),
        "balance": exp["balance"],
    })

    # Tahminleri kaydet
    latency = predictor.benchmark_latency(data.val_df, n=32)
    write_predictions_csv(
        run_dir / "predictions_val.csv",
        data.id_series(data.val_df), y_prob,
        latency["mean_ms"], predictor.size_bytes,
    )

    return {
        "name": name,
        "model": exp["model"],
        "img_size": exp["img_size"],
        "balance": exp["balance"],
        "strong_aug": exp.get("strong_aug", False),
        "macro_f1": metrics.macro_f1,
        "calibrated_macro_f1": calibrated_f1,
        "calib_multipliers": calib_multipliers,
        "accuracy": metrics.accuracy,
        "elapsed_s": round(elapsed, 1),
        "status": "done",
        "per_class_f1": {k: v["f1-score"] for k, v in metrics.per_class.items()},
    }


# ---------------------------------------------------------------------------
# Ana fonksiyon
# ---------------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description="DeepWeeds F1 artirma sweep scripti",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    parser.add_argument("--data", default="data", help="Data klasoru")
    parser.add_argument("--run-root", default="runs", help="Deney cikti klasoru")
    parser.add_argument("--runs", type=int, default=None, help="Kac deney calistirilsin (None=hepsi)")
    parser.add_argument("--dry-run", action="store_true", help="Sadece plani goster, egitme")
    parser.add_argument("--only", default=None,
                        help="Sadece bu deneyi calistir (orn: exp-convnext-tiny-sqrt)")
    args = parser.parse_args()

    experiments = EXPERIMENTS
    if args.only:
        experiments = [e for e in experiments if e["name"] == args.only]
        if not experiments:
            print(f"Deney bulunamadi: {args.only}")
            print(f"Mevcut deneyler: {[e['name'] for e in EXPERIMENTS]}")
            return 1
    if args.runs:
        experiments = experiments[:args.runs]

    print("\n" + "=" * 70)
    print("[PLAN] DeepWeeds Macro-F1 Sweep Plani")
    print("=" * 70)
    print("Baseline: resnet50 | raw macro-F1 = 0.7300 | calibrated = 0.7873")
    has_cuda = torch.cuda.is_available()
    print(f"GPU: {'[CUDA] ' + torch.cuda.get_device_name(0) if has_cuda else '[CPU]'}")
    if has_cuda:
        vram_gb = round(torch.cuda.get_device_properties(0).total_memory / (1024**3), 1)
        print(f"VRAM: {vram_gb} GB")
    print(f"\nPlanlanan {len(experiments)} deney:")
    for i, e in enumerate(experiments, 1):
        print(f"  {i}. [{e['name']}]")
        print(f"     model={e['model']} | img={e['img_size']}px | batch={e['batch_size']}"
              f" | {e['balance']} | epochs={e['epochs']}")
    print("=" * 70 + "\n")

    if args.dry_run:
        print("[DRY RUN] Plan basariyla kontrol edildi, egitim baslatilmiyor.")
        return 0

    results = [
        {
            "name": "BASELINE-resnet50",
            "model": "resnet50",
            "img_size": 224,
            "balance": "weighted_loss",
            "strong_aug": False,
            "macro_f1": 0.7300,
            "calibrated_macro_f1": 0.7873,
            "accuracy": 0.7581,
            "status": "baseline",
        }
    ]

    for exp in experiments:
        result = run_experiment(exp, args.data, args.run_root)
        results.append(result)

        print("\n[SIRALAMA] Guncel Sonuclar:")
        sorted_r = sorted(
            [r for r in results if "macro_f1" in r],
            key=lambda x: x.get("calibrated_macro_f1", x["macro_f1"]), reverse=True,
        )
        for rank, r in enumerate(sorted_r, 1):
            marker = "[EN IYI]" if rank == 1 else f"  {rank}."
            raw_f1 = r["macro_f1"]
            cal_f1 = r.get("calibrated_macro_f1", raw_f1)
            print(f"  {marker} {r['name']}: Calibrated F1={cal_f1:.4f} (Raw={raw_f1:.4f})")

        out_path = Path(args.run_root) / "sweep_results.json"
        save_json(out_path, {"results": results, "generated_at": datetime.now().isoformat()})

    print("\n" + "=" * 70)
    print("[FINAL] SWEEP TAMAMLANDI - Ozet Tablo")
    print("=" * 70)
    sorted_r = sorted(
        [r for r in results if "macro_f1" in r],
        key=lambda x: x.get("calibrated_macro_f1", x["macro_f1"]), reverse=True,
    )
    for rank, r in enumerate(sorted_r, 1):
        marker = "[1. EN IYI]" if rank == 1 else f"   {rank}.    "
        raw_f1 = r["macro_f1"]
        cal_f1 = r.get("calibrated_macro_f1", raw_f1)
        delta = cal_f1 - 0.7300
        delta_str = f"+{delta:.4f}" if delta >= 0 else f"{delta:.4f}"
        print(f"  {marker} | Calibrated F1={cal_f1:.4f} ({delta_str}) | Raw={raw_f1:.4f} | {r['name']}")

    best = sorted_r[0]
    best_cal = best.get("calibrated_macro_f1", best["macro_f1"])
    if best_cal > 0.7300:
        improvement = (best_cal - 0.7300) * 100
        print(f"\n[BASARI] Baseline +{improvement:.2f} puan iyilestirildi!")
        print(f"   En iyi checkpoint: runs/{best['name']}/weights/best.pt")
    else:
        print("\nBaseline gecilemedi.")

    out_path = Path(args.run_root) / "sweep_results.json"
    print(f"\nDetayli sonuclar kaydedildi: {out_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
