from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch

from deepweeds.config import (CLASS_NAMES, DataConfig, ModelConfig, TrainConfig,
                              build_label_map, save_json)
from deepweeds.data import DeepWeedsData
from deepweeds.engine import Trainer
from deepweeds.inference import Predictor, write_predictions_csv
from deepweeds.metrics import compute_metrics
from deepweeds.models import build_model, count_parameters
from deepweeds.reports import save_all, save_class_distribution
from deepweeds.utils import get_device, get_logger, set_seed

log = get_logger()


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument("--data", default="data")
    p.add_argument("--image-dir", default=None)
    p.add_argument("--image-col", default=None)
    p.add_argument("--label-col", default=None)
    p.add_argument("--img-size", type=int, default=224)
    p.add_argument("--batch-size", type=int, default=32)
    p.add_argument("--workers", type=int, default=4)


def _data_cfg(args) -> DataConfig:
    return DataConfig(
        data_dir=Path(args.data), image_dir=Path(args.image_dir) if args.image_dir else None,
        image_col=args.image_col, label_col=args.label_col,
        img_size=args.img_size, batch_size=args.batch_size, num_workers=args.workers)


def cmd_train(args) -> int:
    data_cfg = _data_cfg(args)
    model_cfg = ModelConfig(name=args.model)
    train_cfg = TrainConfig(epochs=args.epochs, lr=args.lr, weight_decay=args.wd,
                            label_smoothing=args.label_smoothing, balance=args.balance,
                            patience=args.patience, seed=args.seed, amp=args.amp,
                            deterministic=args.deterministic, run_root=Path(args.run_root),
                            run_name=args.run_name)

    run_name = train_cfg.run_name or datetime.now().strftime("%Y%m%d-%H%M%S") \
        + f"-{model_cfg.name}-{train_cfg.balance}"
    run_dir = train_cfg.run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(run_dir / "config.json", {"data": data_cfg, "model": model_cfg, "train": train_cfg})
    set_seed(train_cfg.seed, train_cfg.deterministic)
    device = get_device()
    log.info("cihaz: %s | run: %s", device, run_dir)

    data = DeepWeedsData(data_cfg)
    log.info("sınıf dağılımı (train): %s", data.class_counts.to_dict())
    save_class_distribution(data.class_counts, data.class_names,
                            run_dir / "reports" / "class_distribution.png")

    model = build_model(model_cfg)
    log.info("model: %s | parametre: %d", model_cfg.name, count_parameters(model))
    trainer = Trainer(model, data, model_cfg, train_cfg, run_dir, device)
    best_f1 = trainer.fit()

    predictor = Predictor.from_checkpoint(trainer.best_path, device)
    y_true, y_prob = predictor.predict_df(data.val_df)
    y_pred = y_prob.argmax(1)
    metrics = compute_metrics(y_true, y_pred, data.class_names)
    latency = predictor.benchmark_latency(data.val_df, n=64)
    reports_dir = save_all(run_dir, metrics, trainer.history, data.class_names,
                           data.val_df, y_true, y_pred, y_prob,
                           extra={"checkpoint": str(predictor.checkpoint_path),
                                  "model_size_bytes": predictor.size_bytes,
                                  "latency": latency, "balance": train_cfg.balance,
                                  "seed": train_cfg.seed})
    write_predictions_csv(run_dir / "predictions_val.csv",
                          data.id_series(data.val_df), y_prob,
                          latency["mean_ms"], predictor.size_bytes)
    log.info("EN İYİ val macro-F1 = %.4f | acc = %.4f", metrics.macro_f1, metrics.accuracy)
    log.info("gecikme: %s", latency)
    log.info("Raporlar: %s", reports_dir)
    log.info("Uyarı: test.csv yalnızca TÜM seçimler bittikten sonra, 'deepweeds predict --split test' ile kullanılmalı.")
    return 0


def cmd_predict(args) -> int:
    data = DeepWeedsData(_data_cfg(args))
    predictor = Predictor.from_checkpoint(args.run)
    df = data.val_df if args.split == "val" else data.test_df
    probs = predictor.predict_df(df)
    latency = predictor.benchmark_latency(df, n=64)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)
    write_predictions_csv(out, data.id_series(df), probs, latency["mean_ms"], predictor.size_bytes)
    log.info("yazıldı: %s | gecikme: %s", out, latency)
    if args.split == "val":
        metrics = compute_metrics(df["_label"].values, probs.argmax(1), data.class_names)
        log.info("val macro-F1 = %.4f", metrics.macro_f1)
    return 0


def cmd_evaluate(args) -> int:
    truth, preds = pd.read_csv(args.truth), pd.read_csv(args.predictions)
    label_col = next((c for c in ("label", "class", "target", "species") if c in truth.columns),
                     truth.columns[-1])
    label2id = build_label_map(truth[label_col])
    y_true = truth[label_col].map(label2id).values
    y_pred = preds["prediction"].values
    metrics = compute_metrics(y_true, y_pred, list(CLASS_NAMES))
    from deepweeds.reports import save_confusion_matrix
    out_dir = Path(args.out_dir); out_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(metrics.confusion, list(CLASS_NAMES), out_dir / "confusion_matrix.png")
    save_json(out_dir / "metrics.json", {"macro_f1": metrics.macro_f1,
                                         "accuracy": metrics.accuracy,
                                         "per_class": metrics.per_class,
                                         "confusion": metrics.confusion})
    log.info("macro-F1 = %.4f | acc = %.4f | çıktılar: %s",
             metrics.macro_f1, metrics.accuracy, out_dir)
    return 0


def cmd_api(args) -> int:
    import uvicorn
    from deepweeds.api import create_app
    app = create_app(args.model, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(prog="deepweeds", description="DeepWeeds T03 araç kutusu")
    sub = parser.add_subparsers(dest="command", required=True)

    p = sub.add_parser("train", help="Model eğit; rapor ve ağırlıkları kaydet")
    _add_data_args(p)
    p.add_argument("--model", default="resnet50")
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument("--balance", default="weighted_loss",
                   choices=["none", "weighted_loss", "balanced_sampler"])
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-root", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic", action="store_true")
    p.set_defaults(func=cmd_train)

    p = sub.add_parser("predict", help="Val/test tahminlerini yarışma formatında yaz")
    _add_data_args(p)
    p.add_argument("--run", required=True, help="weights/best.pt yolu")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--output", default="predictions.csv")
    p.set_defaults(func=cmd_predict)

    p = sub.add_parser("evaluate", help="truth + predictions CSV'den metrik ve grafik üret")
    p.add_argument("--truth", required=True)
    p.add_argument("--predictions", required=True)
    p.add_argument("--out-dir", default="reports")
    p.set_defaults(func=cmd_evaluate)

    p = sub.add_parser("api", help="FastAPI + tarayıcı demosu başlat")
    p.add_argument("--model", required=True, help="weights/best.pt yolu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None)
    p.set_defaults(func=cmd_api)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())