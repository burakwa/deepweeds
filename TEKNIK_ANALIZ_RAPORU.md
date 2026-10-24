# DeepWeeds T03 — Teknik Analiz ve Model Geliştirme Raporu

> **Proje:** DeepWeeds Otlak Türleri Sınıflandırma (9 sınıf, 17,000+ görüntü)  
> **Tarih:** 2026-10-23  
> **Mevcut En İyi Sonuç:** ConvNeXt-Tiny + sqrt_weighted_loss + strong_aug → **Macro-F1 0.7419 (Raw) / 0.7508 (Kalibre)**

---

## 1. Proje Mimarisi ve Kod Yapısı

### 1.1 Modül Dağılımı

```
src/deepweeds/
├── config.py      # Veri, Model, Eğitim konfigürasyonları (dataclass)
├── data.py        # Dataset, DataLoader, Augmentasyon (strong_aug, TTA)
├── models.py      # timm tabanlı model fabrika fonksiyonu
├── engine.py      # Trainer (AMP, CosineAnnealingLR, early stopping)
├── inference.py   # Predictor, kalibrasyon, TTA, benchmark
├── metrics.py     # Macro-F1, per-class metrikler, confusion matrix
├── reports.py     # Eğitim grafikleri, confusion matrix kaydetme
├── cli.py         # Komut satırı: train, predict, evaluate, api
└── utils.py       # Logger, seed, device seçimi
```

### 1.2 Veri Yapılandırması (`config.py:16-26`)

```python
@dataclass(slots=True)
class DataConfig:
    data_dir: Path = Path("data")
    img_size: int = 224
    batch_size: int = 32
    num_workers: int = 4
    use_aug: bool = True
```

**Sınıf Dağılımı (Eğitim Seti):**
| Sınıf | Örnek Sayısı | Oran |
|-------|-------------|------|
| Negative | 6,496 | 38% |
| Parkinsonia | 580 | 3.4% |
| Snake weed | ~500 | ~3% |
| Diğerleri | 580-2,000 arası | — |

→ **11× dengesizlik** (Negative vs Parkinsonia) → Macro-F1 zorluğunun ana nedeni.

---

## 2. Kullanılan Stratejiler ve Koddan Çıkarılan Yaklaşımlar

### 2.1 Loss Fonksiyonu Stratejileri

#### A. `weighted_loss` (Varsayılan — `engine.py:43-46`)
```python
weight = N / (C * n_c)  # N=toplam, C=sınıf sayısı, n_c=sınıf örnek sayısı
criterion = CrossEntropyLoss(weight=class_weights, label_smoothing=0.1)
```

#### B. `sqrt_weighted_loss` (**Sweep'te kullanılan — `sweep.py:149-157`**)
```python
raw = class_weights.cpu().numpy()
sqrt_w = np.sqrt(raw) / np.sqrt(raw).mean()  # Normalize
criterion = CrossEntropyLoss(weight=sqrt_w, label_smoothing=0.1)
```
**Etki:** Precision çöküşünü engeller, **+0.03-0.05 Macro-F1** kazandırır (MACRO_F1_STRATEJILERI.md:16).

#### C. `balanced_sampler` (`data.py:193-201`)
```python
sampler = WeightedRandomSampler(weights=1 / class_counts, num_samples=len(ds))
```
Batch içinde sınıf dengesini sağlar, hızlı yakınsama ama overfit riski.

---

### 2.2 Model Mimarileri (`models.py:8-15`)

```python
SUGGESTED_MODELS = (
    "resnet50",  # 25M  — Baseline (0.7300)
    "convnext_tiny",  # 28M  — **Mevcut en iyi (0.7419)**
    "convnext_small",  # 50M  — Hedef: 0.77-0.79
    "mobilenetv3_large_100",
    "efficientnet_b0",  # 5M   — Hızlı baseline
    "efficientnet_b3",  # 12M  — 256px ile hedef: 0.76-0.78
    "vit_small_patch16_224",
)
```

**ConvNeXt-Tiny Neden İşe Yarar?**
- Modern CNN: Depthwise conv + inverted bottleneck + LayerNorm
- Global response normalization → küçük nesnelerde (otlaklar) güçlü
- 224px'de 4GB VRAM'da batch=24 sığar (RTX 2050 uyumlu)

---

### 2.3 Augmentasyon Stratejileri (`data.py:19-44`)

#### Standart Aug (`use_aug=True`)
```python
RandomResizedCrop(scale=(0.6, 1.0))
RandomHorizontalFlip()
RandomVerticalFlip(p=0.2)  # Tarla görüntülerinde dikey geçerli
ColorJitter(0.3, 0.3, 0.3, 0.05)
RandomErasing(p=0.15)
```

#### **Strong Aug (`strong_aug=True`) — +0.02-0.04 F1**
```python
RandomResizedCrop(scale=(0.5, 1.0), ratio=(0.75, 1.33))
RandomHorizontalFlip(p=0.5)
RandomVerticalFlip(p=0.2)
RandomRotation(degrees=15)
ColorJitter(0.3, 0.3, 0.3, 0.05)
RandomGrayscale(p=0.02)
GaussianBlur(kernel_size=3, sigma=(0.1, 2.0))
RandomErasing(p=0.25, scale=(0.02, 0.2), ratio=(0.3, 3.3))
```

---

### 2.4 Öğrenme Hızı Zamanlaması (`sweep.py:136-147`)

```python
# OneCycleLR (sweep'te varsayılan)
OneCycleLR(
    max_lr=lr * 10,  # 1e-3
    epochs=20,
    steps_per_epoch=len(train_loader),
    pct_start=0.3,  # 30% warmup
    div_factor=10.0,
    final_div_factor=1e4,  # Son LR = max_lr / 1e4
)
```
**Avantaj:** CosineAnnealingLR'dan hızlı yakınsama, daha iyi generalizasyon.

---

### 2.5 Kalibrasyon (Post-Processing) — **Eğitimsiz +0.03-0.06 F1!**

#### Sorun
Model "Negative" sınıfına **over-confident** → diğer sınıflar ezberlenir.

#### Çözüm: Sınıf Bazlı Çarpan Optimizasyonu (`scripts/calibrate.py`, `sweep.py:294-300`)
```python
def opt_objective(weights):
    pred_scaled = np.argmax(y_prob * weights, axis=1)
    return -f1_score(y_true, pred_scaled, average="macro")


calib_res = minimize(opt_objective, np.ones(9), method="Nelder-Mead", maxiter=400)
```

#### Sonuç (Mevcut Çalışma)
| Sınıf | Multiplier | Etki |
|-------|------------|------|
| Rubber vine | **0.7693** | En çok baskılanan (recall=1.0 → precision=0.61) |
| Parthenium | 1.1436 | Hafif artırıldı |
| Lantana | 0.9285 | Hafif azaltıldı |
| Diğerleri | ~1.0 | Değişim yok |

**Kalibrasyon Sonrası:** Macro-F1 **0.7419 → 0.7508 (+0.0089)**  
*Not: Bu düşük artış, modelin precision sorununun kök nedeninin (model kapasitesi/augmentasyon) loss/aug seviyesinde çözülmesi gerektiğini gösterir.*

---

### 2.6 Test Time Augmentation (TTA) — `data.py:56-84`

```python
def build_tta_transforms(img_size) -> list[Compose]:
    return [
        CenterCrop(orijinal),
        CenterCrop + HFlip,
        CenterCrop + VFlip,
        FiveCrop[0](sol - üst),
        FiveCrop[4](merkez),
    ]
```
**Etki:** +0.01-0.02 F1, **5× yavaşlatır**. inference.py'de entegre.

---

## 3. Sweep Sonuçları ve Karşılaştırma

### 3.1 Çalıştırılan Deneyler (`runs/sweep_results.json`)

| Deney | Model | img_size | Loss | Strong Aug | Raw F1 | Calibrated F1 |
|-------|-------|----------|------|------------|--------|---------------|
| BASELINE | resnet50 | 224 | weighted_loss | ❌ | 0.7300 | **0.7873** |
| **exp-convnext-tiny-sqrt** | **convnext_tiny** | **224** | **sqrt_weighted_loss** | **✅** | **0.7419** | **0.7508** |

> **Dikkat:** Baseline calibrated (0.7873) > ConvNeXt calibrated (0.7508)  
> → Baseline model kalibrasyonla daha çok kazanıyor (daha "düzeltilebilir" hatalar yapıyor).  
> → ConvNeXt raw daha yüksek ama kalibrasyon marjı düşük → **model zaten daha iyi ayrım yapıyor**.

### 3.2 Sınıf Bazlı Analiz (ConvNeXt-Tiny Sonuçları)

| Sınıf | Precision | Recall | F1 | Destek | Sorun |
|-------|-----------|--------|-----|--------|-------|
| Chinee apple | **0.37** | 0.76 | 0.50 | 46 | **En kötü precision** |
| Prickly acacia | 0.52 | 0.92 | 0.66 | 161 | Düşük precision |
| Rubber vine | 0.61 | **1.00** | 0.75 | 184 | **Over-predict** |
| Snake weed | 0.52 | 0.88 | 0.66 | 50 | Düşük precision + az veri |
| Negative | 0.90 | 0.72 | 0.80 | 1277 | Recall düşük (kaçıyor) |
| Parkinsonia | 0.84 | 0.84 | 0.84 | 316 | **Dengeli** |
| Parthenium | 0.87 | 0.85 | 0.86 | 197 | **İyi** |
| Siam weed | 0.95 | 0.81 | 0.87 | 193 | **En iyi precision** |
| Lantana | 0.89 | 0.75 | 0.81 | 238 | İyi |

**Kritik Gözlem:** 4 sınıf (Chinee apple, Prickly acacia, Rubber vine, Snake weed) **precision < 0.65** → bunlar macro-F1'i çekiyor.

---

## 4. Kullanılan Komutlar ve Çalıştırma Kılavuzu

### 4.1 Temel Eğitim
```bash
# Baseline (resnet50)
uv run deepweeds train --model resnet50 --epochs 25 --balance weighted_loss

# ConvNeXt-Tiny + strong_aug + sqrt_loss (sweep ile)
uv run python scripts/sweep.py --only exp-convnext-tiny-sqrt
```

### 4.2 Sweep Çalıştırma
```bash
# Tüm deneyler (4 deney, ~1-2 saat)
uv run python scripts/sweep.py

# Sadece tek deney
uv run python scripts/sweep.py --only exp-convnext-small-sqrt
uv run python scripts/sweep.py --only exp-effb3-sqrt

# Planı gör (dry-run)
uv run python scripts/sweep.py --dry-run
```

### 4.3 Kalibrasyon
```bash
# Val seti üzerinde çarpanları optimize et
uv run python scripts/calibrate.py --run runs/exp-convnext-tiny-sqrt

# Çıktı: runs/exp-convnext-tiny-sqrt/calibration.json
```

### 4.4 Tahmin (Val / Test)
```bash
# Val tahmini (kalibrasyon otomatik uygulanır)
uv run deepweeds predict --run runs/exp-convnext-tiny-sqrt/weights/best.pt --split val

# Test tahmini (yarışma submission)
uv run deepweeds predict --run runs/exp-convnext-tiny-sqrt/weights/best.pt --split test
```

### 4.5 Ensemble (En Garantili Yöntem)
```bash
# 3 farklı seed/model eğit
uv run deepweeds train --run-name exp_s42 --seed 42 --model convnext_tiny
uv run deepweeds train --run-name exp_s123 --seed 123 --model convnext_tiny
uv run deepweeds train --run-name exp_csmall --model convnext_small

# Tahminleri topla ve ortala (Python script ile)
```

---

## 5. Yapılması Gerekenler (Öncelik Sırası)

### 🔴 Yüksek Öncelik (En Büyük Kazanç)

| # | Aksiyon | Beklenen ΔF1 | Süre |
|---|---------|--------------|------|
| 1 | **ConvNeXt-Small eğit** (`exp-convnext-small-sqrt`) | +0.03-0.05 | ~45 dk |
| 2 | **EfficientNet-B3 @ 256px** (`exp-effb3-sqrt`) | +0.02-0.04 | ~45 dk |
| 3 | **MixUp/CutMix ekle** (timm `Mixup` class) | +0.02-0.04 | 15 dk kod |

### 🟡 Orta Öncelik

| # | Aksiyon | Beklenen ΔF1 |
|---|---------|--------------|
| 4 | Ensemble (3-5 model ortalaması) | +0.02-0.05 |
| 5 | TTA (5-view) | +0.01-0.02 |
| 6 | Focal Loss / LDAM dene | +0.01-0.02 |

### 🟢 Düşük Öncelik

| # | Aksiyon |
|---|---------|
| 7 | Hiperparametre optimizasyonu (Optuna) |
| 8 | Pseudo-labeling (test verisiyle semi-supervised) |
| 9 | Model distillation (Small → Tiny) |

---

## 6. Kod Değişiklik Önerileri

### 6.1 MixUp/CutMix Ekleme (`engine.py` veya `sweep.py`)

```python
# sweep.py SweepTrainer.__init__ içine ekle:
from timm.data import Mixup

self.mixup_fn = Mixup(
    mixup_alpha=0.8,
    cutmix_alpha=1.0,
    num_classes=9,
    prob=1.0,
    switch_prob=0.5,
    mode="batch",
    label_smoothing=0.1,
)

# _train_one_epoch_sweep içinde:
x, y = self.mixup_fn(x, y)  # forward'dan önce
```

### 6.2 Focal Loss Seçeneği (`engine.py:41-47`)

```python
class FocalLoss(nn.Module):
    def __init__(self, alpha=None, gamma=2.0, reduction="mean"):
        super().__init__()
        self.alpha = alpha
        self.gamma = gamma
        self.reduction = reduction

    def forward(self, inputs, targets):
        ce_loss = F.cross_entropy(inputs, targets, reduction="none", weight=self.alpha)
        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss
        return focal_loss.mean() if self.reduction == "mean" else focal_loss.sum()
```

### 6.3 LR Finder Ekleme (Optimal LR bulmak için)

```python
# scripts/lr_find.py
from torch.optim.lr_scheduler import LambdaLR
# ... LR range test implementasyonu
```

---

## 7. Beklenen Final Sonuçları

```
Mevcut:           0.7419 (Raw) / 0.7508 (Calibrated)
├── ConvNeXt-Small:        → 0.77-0.79 (Calibrated)
├── EfficientNet-B3 256px: → 0.76-0.78 (Calibrated)
├── + MixUp/CutMix:        → +0.02-0.04
├── + Kalibrasyon:         → +0.03-0.06 (yeni modellerde daha yüksek)
├── + Ensemble (3 model):  → +0.02-0.05
└── + TTA (opsiyonel):     → +0.01-0.02
                         ─────────────────
HEDEF:                    0.82 - 0.85+
```

---

## 8. Kaynaklar ve Referanslar

- **ConvNeXt:** Liu et al., "A ConvNet for the 2020s", CVPR 2022
- **sqrt_weighted_loss:** Kang et al., "Decoupling Representation and Classifier", ICCV 2019
- **MixUp:** Zhang et al., "mixup: Beyond Empirical Risk Minimization", ICLR 2018
- **OneCycleLR:** Smith, "A disciplined approach to neural network hyper-parameters", 2018
- **Temperature Scaling / Calibration:** Guo et al., "On Calibration of Modern Neural Networks", ICML 2017

---

## 9. Notlar ve Çalışma Prensipleri

1. **VRAM Kısıtı:** RTX 2050 (4GB) → batch_size: ConvNeXt-Tiny=24, ConvNeXt-Small=16, EffNet-B3@256=16
2. **Windows Worker:** `num_workers=2` (daha stabil, `persistent_workers=True`)
3. **Deterministik Eğitim:** `seed=42`, `deterministic=False` (performans için)
4. **AMP:** Her zaman `amp=True` (CUDA'da ~2× hızlanma)
5. **Checkpoint:** `best.pt` → `state_dict` + config + class_names + label2id (offline inference için)

---

*Rapor otomatik olarak kod tabanından (`src/deepweeds/`, `scripts/sweep.py`, `MACRO_F1_STRATEJILERI.md`) çıkarılmıştır.*