from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from deepweeds.config import (
    CLASS_NAMES,
    DataConfig,
    ModelConfig,
    TrainConfig,
    build_label_map,
    save_json,
)
from deepweeds.data import DeepWeedsData
from deepweeds.engine import Trainer
from deepweeds.inference import Predictor, write_predictions_csv
from deepweeds.metrics import compute_metrics
from deepweeds.models import SUGGESTED_MODELS, build_model, count_parameters
from deepweeds.reports import save_all, save_class_distribution
from deepweeds.utils import get_device, get_logger, set_seed

log = get_logger()


# ---------------------------------------------------------------------------
# Ortak yardımcılar
# ---------------------------------------------------------------------------


def _add_data_args(p: argparse.ArgumentParser) -> None:
    p.add_argument(
        "--data", default="data", help="CSV'lerin bulunduğu klasör (varsayılan: data)"
    )
    p.add_argument(
        "--image-dir", default=None, help="Görüntülerin ayrı bir klasörde olduğu durum"
    )
    p.add_argument("--image-col", default=None, help="CSV'deki görüntü yolu kolonu adı")
    p.add_argument("--label-col", default=None, help="CSV'deki etiket kolonu adı")
    p.add_argument(
        "--img-size",
        type=int,
        default=224,
        help="Giriş görüntü boyutu (varsayılan: 224)",
    )
    p.add_argument(
        "--batch-size", type=int, default=32, help="Batch boyutu (varsayılan: 32)"
    )
    p.add_argument(
        "--workers",
        type=int,
        default=4,
        help="DataLoader worker sayısı (varsayılan: 4)",
    )


def _data_cfg(args) -> DataConfig:
    return DataConfig(
        data_dir=Path(args.data),
        image_dir=Path(args.image_dir) if args.image_dir else None,
        image_col=args.image_col,
        label_col=args.label_col,
        img_size=args.img_size,
        batch_size=args.batch_size,
        num_workers=args.workers,
    )


# ---------------------------------------------------------------------------
# Komutlar
# ---------------------------------------------------------------------------


def cmd_train(args) -> int:
    data_cfg = _data_cfg(args)
    model_cfg = ModelConfig(name=args.model)
    train_cfg = TrainConfig(
        epochs=args.epochs,
        lr=args.lr,
        weight_decay=args.wd,
        label_smoothing=args.label_smoothing,
        balance=args.balance,
        patience=args.patience,
        seed=args.seed,
        amp=args.amp,
        deterministic=args.deterministic,
        run_root=Path(args.run_root),
        run_name=args.run_name,
    )

    run_name = train_cfg.run_name or (
        datetime.now().strftime("%Y%m%d-%H%M%S")
        + f"-{model_cfg.name}-{train_cfg.balance}"
    )
    run_dir = train_cfg.run_root / run_name
    run_dir.mkdir(parents=True, exist_ok=True)
    save_json(
        run_dir / "config.json",
        {"data": data_cfg, "model": model_cfg, "train": train_cfg},
    )
    set_seed(train_cfg.seed, train_cfg.deterministic)
    device = get_device()
    log.info("cihaz: %s | run: %s", device, run_dir)

    data = DeepWeedsData(data_cfg)
    log.info("sınıf dağılımı (train): %s", data.class_counts.to_dict())
    save_class_distribution(
        data.class_counts,
        data.class_names,
        run_dir / "reports" / "class_distribution.png",
    )

    model = build_model(model_cfg)
    log.info("model: %s | parametre: %d", model_cfg.name, count_parameters(model))
    trainer = Trainer(model, data, model_cfg, train_cfg, run_dir, device)
    best_f1 = trainer.fit()

    # --- eğitim sonrası değerlendirme ---
    predictor = Predictor.from_checkpoint(trainer.best_path, device)
    result = predictor.predict_df(data.val_df)  # PredictResult(y_true, y_prob)
    y_true, y_prob = result.y_true, result.y_prob
    y_pred = y_prob.argmax(1)
    metrics = compute_metrics(y_true, y_pred, data.class_names)
    latency = predictor.benchmark_latency(data.val_df, n=64)
    reports_dir = save_all(
        run_dir,
        metrics,
        trainer.history,
        data.class_names,
        data.val_df,
        y_true,
        y_pred,
        y_prob,
        extra={
            "checkpoint": str(predictor.checkpoint_path),
            "model_size_bytes": predictor.size_bytes,
            "latency": latency,
            "balance": train_cfg.balance,
            "seed": train_cfg.seed,
        },
    )
    write_predictions_csv(
        run_dir / "predictions_val.csv",
        data.id_series(data.val_df),
        y_prob,
        latency["mean_ms"],
        predictor.size_bytes,
    )
    log.info(
        "EN İYİ val macro-F1 = %.4f | acc = %.4f", metrics.macro_f1, metrics.accuracy
    )
    log.info("gecikme: %s", latency)
    log.info("Raporlar: %s", reports_dir)
    log.info(
        "Uyarı: test.csv yalnızca TÜM seçimler bittikten sonra,"
        " 'deepweeds predict --split test' ile kullanılmalı."
    )
    return 0


def cmd_predict(args) -> int:
    import json

    data = DeepWeedsData(_data_cfg(args))
    predictor = Predictor.from_checkpoint(args.run)
    df = data.val_df if args.split == "val" else data.test_df
    result = predictor.predict_df(df)  # PredictResult(y_true, y_prob)
    y_true, y_prob = result.y_true, result.y_prob
    latency = predictor.benchmark_latency(df, n=64)
    out = Path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    # Otomatik kalibrasyon dosyası kontrolü
    multipliers = None
    calib_file = Path(args.calibration) if getattr(args, "calibration", None) else None
    if calib_file is None:
        cand1 = Path(args.run).parent.parent / "calibration.json"
        cand2 = Path(args.run).parent / "calibration.json"
        if cand1.exists():
            calib_file = cand1
        elif cand2.exists():
            calib_file = cand2

    if calib_file and calib_file.exists():
        try:
            calib_data = json.loads(calib_file.read_text(encoding="utf-8"))
            multipliers = calib_data.get("multipliers")
            log.info("Kalibrasyon çarpanları uygulandı: %s", calib_file)
        except Exception as e:
            log.warning("Kalibrasyon dosyası okunamadı: %s", e)

    write_predictions_csv(
        out,
        data.id_series(df),
        y_prob,
        latency["mean_ms"],
        predictor.size_bytes,
        multipliers=multipliers,
    )
    log.info("yazıldı: %s | gecikme: %s", out, latency)
    if args.split == "val":
        pred_labels = (
            (y_prob * np.asarray(multipliers)).argmax(1)
            if multipliers
            else y_prob.argmax(1)
        )
        metrics = compute_metrics(y_true, pred_labels, data.class_names)
        log.info("val macro-F1 = %.4f", metrics.macro_f1)
    return 0


def cmd_evaluate(args) -> int:
    truth, preds = pd.read_csv(args.truth), pd.read_csv(args.predictions)
    label_col = next(
        (c for c in ("label", "class", "target", "species") if c in truth.columns),
        truth.columns[-1],
    )
    label2id = build_label_map(truth[label_col])
    y_true = truth[label_col].map(label2id).values
    y_pred = preds["prediction"].values
    metrics = compute_metrics(y_true, y_pred, list(CLASS_NAMES))
    from deepweeds.reports import save_confusion_matrix

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    save_confusion_matrix(
        metrics.confusion, list(CLASS_NAMES), out_dir / "confusion_matrix.png"
    )
    save_json(
        out_dir / "metrics.json",
        {
            "macro_f1": metrics.macro_f1,
            "accuracy": metrics.accuracy,
            "per_class": metrics.per_class,
            "confusion": metrics.confusion,
        },
    )
    log.info(
        "macro-F1 = %.4f | acc = %.4f | çıktılar: %s",
        metrics.macro_f1,
        metrics.accuracy,
        out_dir,
    )
    return 0


def cmd_api(args) -> int:
    import uvicorn

    from deepweeds.api import create_app

    app = create_app(args.model, device=args.device)
    uvicorn.run(app, host=args.host, port=args.port)
    return 0


def cmd_streamlit(args) -> int:
    import subprocess
    import sys

    project_root = Path(__file__).parent.parent.parent
    streamlit_app = project_root / "src" / "deepweeds" / "streamlit_app.py"

    if not streamlit_app.exists():
        print(f"Hata: Streamlit uygulaması bulunamadı: {streamlit_app}")
        return 1

    cmd = [
        sys.executable,
        "-m",
        "streamlit",
        "run",
        str(streamlit_app),
        "--server.port",
        str(args.port),
        "--server.address",
        args.host,
    ]

    if args.model:
        cmd.extend(["--", "--model", args.model])

    print("🚀 DeepWeeds Streamlit başlatılıyor...")
    print(f"   Adres: http://{args.host}:{args.port}")
    if args.model:
        print(f"   Model: {args.model}")
    print()

    try:
        subprocess.run(cmd, cwd=project_root, check=True)
    except subprocess.CalledProcessError as e:
        print(f"Hata: Streamlit başlatılamadı (kod: {e.returncode})")
        return e.returncode
    except KeyboardInterrupt:
        print("\n👋 Streamlit kapatıldı.")
    return 0


# ---------------------------------------------------------------------------
# CLI tanımı
# ---------------------------------------------------------------------------


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(
        prog="deepweeds",
        description="DeepWeeds T03 araç kutusu",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    sub = parser.add_subparsers(dest="command", required=True)

    # --- train ---
    p = sub.add_parser(
        "train",
        help="Model eğit; rapor ve ağırlıkları kaydet",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_data_args(p)
    p.add_argument(
        "--model",
        default="resnet50",
        help=f"timm model adı. Önerilen: {', '.join(SUGGESTED_MODELS)}",
    )
    p.add_argument("--epochs", type=int, default=25)
    p.add_argument("--lr", type=float, default=1e-4)
    p.add_argument("--wd", type=float, default=1e-4)
    p.add_argument("--label-smoothing", type=float, default=0.1)
    p.add_argument(
        "--balance",
        default="weighted_loss",
        choices=["none", "weighted_loss", "balanced_sampler"],
    )
    p.add_argument("--patience", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--run-root", default="runs")
    p.add_argument("--run-name", default=None)
    p.add_argument("--aug", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--amp", action=argparse.BooleanOptionalAction, default=True)
    p.add_argument("--deterministic", action="store_true")
    p.set_defaults(func=cmd_train)

    # --- predict ---
    p = sub.add_parser(
        "predict",
        help="Val/test tahminlerini yarışma formatında yaz",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    _add_data_args(p)
    p.add_argument("--run", required=True, help="weights/best.pt yolu")
    p.add_argument("--split", choices=["val", "test"], default="test")
    p.add_argument("--output", default="predictions.csv")
    p.add_argument(
        "--calibration",
        default=None,
        help="calibration.json yolu (varsayılan: run klasöründen otomatik algılanır)",
    )
    p.set_defaults(func=cmd_predict)

    # --- evaluate ---
    p = sub.add_parser(
        "evaluate",
        help="truth + predictions CSV'den metrik ve grafik üret",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--truth", required=True, help="Gerçek etiketlerin bulunduğu CSV")
    p.add_argument("--predictions", required=True, help="Tahminlerin bulunduğu CSV")
    p.add_argument("--out-dir", default="reports")
    p.set_defaults(func=cmd_evaluate)

    # --- api ---
    p = sub.add_parser(
        "api",
        help="FastAPI + tarayıcı demosu başlat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument("--model", required=True, help="weights/best.pt yolu")
    p.add_argument("--host", default="127.0.0.1")
    p.add_argument("--port", type=int, default=8000)
    p.add_argument("--device", default=None, help="cuda / cpu (varsayılan: otomatik)")
    p.set_defaults(func=cmd_api)

    # --- streamlit ---
    p = sub.add_parser(
        "streamlit",
        help="Streamlit arayüzünü başlat",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--model",
        default=None,
        help="weights/best.pt yolu (varsayılan: runs/exp-effb3-sqrt/weights/best.pt)",
    )
    p.add_argument(
        "--port", type=int, default=8501, help="Streamlit port (varsayılan: 8501)"
    )
    p.add_argument(
        "--host", default="localhost", help="Streamlit host (varsayılan: localhost)"
    )
    p.add_argument("--browser", action="store_true", help="Tarayıcıyı otomatik aç")
    p.set_defaults(func=cmd_streamlit)

    args = parser.parse_args(argv)
    return args.func(args)


if __name__ == "__main__":
    raise SystemExit(main())
