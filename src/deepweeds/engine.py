from __future__ import annotations

import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
import timm
import torch
import torch.nn as nn
from sklearn.metrics import f1_score
from torch.amp import GradScaler, autocast
from tqdm import tqdm

from deepweeds.config import ModelConfig, TrainConfig, jsonable
from deepweeds.data import DeepWeedsData
from deepweeds.utils import get_logger


class Trainer:
    def __init__(self, model: nn.Module, data: DeepWeedsData, model_cfg: ModelConfig,
                 cfg: TrainConfig, run_dir: Path, device: torch.device):
        self.model = model.to(device)
        self.data, self.model_cfg, self.cfg = data, model_cfg, cfg
        self.run_dir, self.device = Path(run_dir), device
        self.weights_dir = self.run_dir / "weights"
        self.weights_dir.mkdir(parents=True, exist_ok=True)
        self.best_path = self.weights_dir / "best.pt"
        self.best_f1 = -1.0
        self.history: list[dict] = []

        self.criterion = self._build_criterion()
        self.eval_criterion = nn.CrossEntropyLoss()
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=cfg.lr,
                                           weight_decay=cfg.weight_decay)
        self.scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self.optimizer, T_max=cfg.epochs)
        self.use_amp = cfg.amp and device.type == "cuda"
        self.scaler = GradScaler("cuda", enabled=self.use_amp)

    def _build_criterion(self) -> nn.CrossEntropyLoss:
        ls = self.cfg.label_smoothing
        if self.cfg.balance == "weighted_loss":  # ağırlıklar yalnızca train'den
            return nn.CrossEntropyLoss(weight=self.data.class_weights.to(self.device),
                                       label_smoothing=ls)
        return nn.CrossEntropyLoss(label_smoothing=ls)

    def fit(self) -> float:
        log = get_logger()
        train_loader = self.data.train_loader(self.cfg.balance)
        val_loader = self.data.val_loader()
        patience = 0
        for epoch in range(1, self.cfg.epochs + 1):
            t0 = time.perf_counter()
            train_loss = self._train_one_epoch(train_loader)
            val_loss, y_true, y_prob = self._validate(val_loader)
            y_pred = y_prob.argmax(1)
            f1 = float(f1_score(y_true, y_pred, average="macro", zero_division=0))
            acc = float((y_true == y_pred).mean())
            self.history.append(dict(epoch=epoch, train_loss=train_loss, val_loss=val_loss,
                                     val_macro_f1=f1, val_accuracy=acc,
                                     lr=self.optimizer.param_groups[0]["lr"],
                                     seconds=round(time.perf_counter() - t0, 1)))
            log.info("epoch %02d | train %.4f | val %.4f | macroF1 %.4f | acc %.4f | %.0fs",
                     epoch, train_loss, val_loss, f1, acc, time.perf_counter() - t0)
            if f1 > self.best_f1:
                self.best_f1, patience = f1, 0
                self._save_checkpoint(epoch, f1)
                log.info("  → yeni en iyi model kaydedildi (%.4f)", f1)
            else:
                patience += 1
                if patience >= self.cfg.patience:
                    log.info("Erken durdurma (patience=%d).", self.cfg.patience)
                    break
            self.scheduler.step()

        ckpt = torch.load(self.best_path, map_location=self.device, weights_only=True)
        self.model.load_state_dict(ckpt["state_dict"])
        pd.DataFrame(self.history).to_csv(self.run_dir / "history.csv", index=False)
        return self.best_f1

    def _train_one_epoch(self, loader) -> float:
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
        return total / max(n, 1)

    @torch.no_grad()
    def _validate(self, loader):
        self.model.eval()
        total = n = 0
        ys, ps = [], []
        for x, y in loader:
            x = x.to(self.device, non_blocking=True)
            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(x)
            logits = logits.float()
            loss = self.eval_criterion(logits, y.to(self.device))
            total += loss.item() * y.size(0)
            n += y.size(0)
            ys.append(y.numpy())
            ps.append(torch.softmax(logits, dim=1).cpu().numpy())
        return total / max(n, 1), np.concatenate(ys), np.vstack(ps)

    @torch.no_grad()
    def predict(self, loader):
        self.model.eval()
        ys, ps = [], []
        for x, y in loader:
            with autocast("cuda", enabled=self.use_amp):
                logits = self.model(x.to(self.device, non_blocking=True))
            ps.append(torch.softmax(logits.float(), dim=1).cpu().numpy())
            ys.append(y.numpy())
        return np.concatenate(ys), np.vstack(ps)

    def _save_checkpoint(self, epoch: int, f1: float) -> None:
        ckpt = {
            "state_dict": {k: v.detach().cpu().clone() for k, v in self.model.state_dict().items()},
            "class_names": self.data.class_names,
            "label2id": {str(k): int(v) for k, v in self.data.label2id.items()},
            "config": jsonable({"model": self.model_cfg, "train": self.cfg, "data": self.data.cfg}),
            "epoch": epoch, "val_macro_f1": f1,
            "torch": torch.__version__, "timm": timm.__version__,
            "saved_at": datetime.now().isoformat(timespec="seconds"),
        }
        tmp = self.best_path.with_suffix(".tmp")
        torch.save(ckpt, tmp)
        tmp.replace(self.best_path)