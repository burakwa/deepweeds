from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch
from PIL import Image
from torch.utils.data import DataLoader, Dataset, WeightedRandomSampler
from torchvision import transforms as T

from deepweeds.config import CLASS_NAMES, DataConfig, NUM_CLASSES, build_label_map

IMAGE_EXTENSIONS = {".jpg", ".jpeg", ".png"}
IMAGENET_MEAN = (0.485, 0.456, 0.406)
IMAGENET_STD = (0.229, 0.224, 0.225)


def build_train_transform(img_size: int, use_aug: bool = True, strong_aug: bool = False) -> T.Compose:
    if not use_aug:
        return build_eval_transform(img_size)
    if strong_aug:
        # Güçlü augmentation - az örnekli sınıflar için idealdir
        return T.Compose([
            T.RandomResizedCrop(img_size, scale=(0.5, 1.0), ratio=(0.75, 1.33), antialias=True),
            T.RandomHorizontalFlip(p=0.5),
            T.RandomVerticalFlip(p=0.2),
            T.RandomRotation(degrees=15),
            T.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.05),
            T.RandomGrayscale(p=0.02),
            T.GaussianBlur(kernel_size=3, sigma=(0.1, 2.0)),
            T.ToTensor(),
            T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
            T.RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3)),
        ])
    return T.Compose([
        T.RandomResizedCrop(img_size, scale=(0.6, 1.0), antialias=True),
        T.RandomHorizontalFlip(),
        T.RandomVerticalFlip(p=0.2),
        T.ColorJitter(0.3, 0.3, 0.3, 0.05),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        T.RandomErasing(p=0.15, scale=(0.02, 0.15)),
    ])


def build_eval_transform(img_size: int) -> T.Compose:
    return T.Compose([
        T.Resize(int(img_size * 1.14), antialias=True),   # biraz büyük resize → center crop
        T.CenterCrop(img_size),
        T.ToTensor(),
        T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
    ])


def build_tta_transforms(img_size: int) -> list[T.Compose]:
    """Test Time Augmentation için 5 farklı transform döner (orijinal + 4 flip/crop varyant)."""
    base_resize = int(img_size * 1.14)
    return [
        T.Compose([
            T.Resize(base_resize, antialias=True), T.CenterCrop(img_size),
            T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        T.Compose([
            T.Resize(base_resize, antialias=True), T.CenterCrop(img_size),
            T.RandomHorizontalFlip(p=1.0),
            T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        T.Compose([
            T.Resize(base_resize, antialias=True), T.CenterCrop(img_size),
            T.RandomVerticalFlip(p=1.0),
            T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        T.Compose([
            T.Resize(int(img_size * 1.05), antialias=True), T.FiveCrop(img_size),
            T.Lambda(lambda crops: crops[0]),   # sol-üst köşe
            T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
        T.Compose([
            T.Resize(int(img_size * 1.05), antialias=True), T.FiveCrop(img_size),
            T.Lambda(lambda crops: crops[4]),   # merkez
            T.ToTensor(), T.Normalize(IMAGENET_MEAN, IMAGENET_STD),
        ]),
    ]


class DeepWeedsDataset(Dataset):
    """Beklenen kolonlar: `_path` (str) ve `_label` (int, test için -1)."""

    def __init__(self, df: pd.DataFrame, transform: T.Compose):
        self.df = df.reset_index(drop=True)
        self.transform = transform

    def __len__(self) -> int:
        return len(self.df)

    def __getitem__(self, idx: int):
        row = self.df.iloc[idx]
        image = Image.open(row["_path"]).convert("RGB")
        return self.transform(image), int(row["_label"])


class DeepWeedsData:
    """CSV'leri yükler, kolonları ve görüntü yollarını çözer, DataLoader üretir."""

    IMG_CANDIDATES = ("image", "image_path", "filepath", "path", "file", "filename", "img")
    LABEL_CANDIDATES = ("label", "class", "species", "target", "y", "category")

    def __init__(self, cfg: DataConfig):
        self.cfg = cfg
        self.data_dir = Path(cfg.data_dir)
        self.train_df = self._read("train.csv", required=True)
        self.val_df = self._read("val.csv", required=True)
        self.test_df = self._read("test.csv", required=False)

        self.image_col = cfg.image_col or self._detect_col(self.train_df, self.IMG_CANDIDATES, "görüntü")
        self.label_col = cfg.label_col or self._detect_col(self.train_df, self.LABEL_CANDIDATES, "etiket")
        self.label2id = build_label_map(self.train_df[self.label_col])
        if sorted(self.label2id.values()) != list(range(NUM_CLASSES)):
            raise ValueError(f"9 sınıf bekleniyordu, bulunan: {sorted(self.label2id.values())}")
        self.class_names = list(CLASS_NAMES)

        self._index_images()
        self.train_df = self._prepare(self.train_df, require_labels=True)
        self.val_df = self._prepare(self.val_df, require_labels=True)
        if self.test_df is not None:
            self.test_df = self._prepare(self.test_df, require_labels=False)

        counts = self.train_df["_label"].value_counts().reindex(range(NUM_CLASSES), fill_value=0)
        self.class_counts = counts.astype(int)
        weights = self.class_counts.sum() / (NUM_CLASSES * np.maximum(self.class_counts.values, 1))
        self.class_weights = torch.tensor(weights, dtype=torch.float32)

    # --- yükleme ve algılama -------------------------------------------------
    def _read(self, filename: str, required: bool) -> pd.DataFrame | None:
        path = self.data_dir / filename
        if not path.exists():
            if required:
                raise FileNotFoundError(f"Bulunamadı: {path} (data_dir doğru mu?)")
            return None
        return pd.read_csv(path)

    def _detect_col(self, df: pd.DataFrame, candidates: tuple[str, ...], kind: str) -> str:
        lower = {c.lower(): c for c in df.columns}
        for cand in candidates:
            if cand in lower:
                return lower[cand]
        for c in df.columns:
            if df[c].astype(str).str.contains(r"\.(jpe?g|png)\s*$", case=False, regex=True).any():
                return c
        raise ValueError(f"{kind} kolonu bulunamadı. DataConfig'e image_col/label_col girin. Kolonlar: {list(df.columns)}")

    def _index_images(self) -> None:
        root = Path(self.cfg.image_dir) if self.cfg.image_dir else self.data_dir
        self._by_name = {p.name: p for p in root.rglob("*") if p.suffix.lower() in IMAGE_EXTENSIONS}

    def _resolve(self, rel: str) -> Path:
        p = Path(rel)
        if p.is_absolute() and p.exists():
            return p
        cand = self.data_dir / p
        if cand.exists():
            return cand
        hit = self._by_name.get(p.name)
        if hit:
            return hit
        raise KeyError(f"Görüntü bulunamadı: {rel}")

    def _prepare(self, df: pd.DataFrame, require_labels: bool) -> pd.DataFrame:
        out = df.copy()
        paths, missing = [], []
        for rel in out[self.image_col].astype(str):
            try:
                paths.append(str(self._resolve(rel)))
            except KeyError:
                missing.append(rel)
        if missing:
            raise FileNotFoundError(f"{len(missing)} görüntü bulunamadı. Örnek: {missing[:5]}")
        out["_path"] = paths
        out["_label"] = (out[self.label_col].map(self.label2id).astype(int)
                         if require_labels else -1)
        return out

    # --- erişimler ------------------------------------------------------------
    def id_series(self, df: pd.DataFrame) -> pd.Series:
        col = "id" if "id" in df.columns else self.image_col
        return df[col].astype(str)

    @property
    def sample_weights(self) -> np.ndarray:
        return 1.0 / self.class_counts.values[self.train_df["_label"].values]

    def train_loader(self, balance: str = "none", strong_aug: bool = False) -> DataLoader:
        ds = DeepWeedsDataset(self.train_df, build_train_transform(self.cfg.img_size, self.cfg.use_aug, strong_aug=strong_aug))
        if balance == "balanced_sampler":
            sampler = WeightedRandomSampler(
                torch.tensor(self.sample_weights, dtype=torch.double),
                num_samples=len(ds), replacement=True)
            return DataLoader(ds, self.cfg.batch_size, sampler=sampler, shuffle=False,
                              num_workers=self.cfg.num_workers, pin_memory=True,
                              persistent_workers=self.cfg.num_workers > 0, drop_last=True)
        return DataLoader(ds, self.cfg.batch_size, shuffle=True,
                          num_workers=self.cfg.num_workers, pin_memory=True,
                          persistent_workers=self.cfg.num_workers > 0, drop_last=True)

    def val_loader(self) -> DataLoader:
        ds = DeepWeedsDataset(self.val_df, build_eval_transform(self.cfg.img_size))
        return DataLoader(ds, self.cfg.batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=True)

    def test_loader(self) -> DataLoader:
        if self.test_df is None:
            raise FileNotFoundError("test.csv yok; --split test kullanılamaz.")
        ds = DeepWeedsDataset(self.test_df, build_eval_transform(self.cfg.img_size))
        return DataLoader(ds, self.cfg.batch_size, shuffle=False,
                          num_workers=self.cfg.num_workers, pin_memory=True)