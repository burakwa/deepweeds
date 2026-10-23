# Macro-F1 Artırma Stratejileri — DeepWeeds T03

## Neden Macro-F1 Zor?

- **Sınıf dengesizliği**: Negative (6496) vs Parkinsonia (580) → 11× fark
- **Macro-F1** = 9 sınıfın F1'inin **basit ortalaması** → az örnekli sınıflar eşit ağırlıktır
- Accuracy %90 olabilir ama Macro-F1 %40 kalabilir (Negative ezberlenir, diğerleri öğrenilmez)

---

## 1. Loss Fonksiyonu Değişiklikleri (En Etkili)

| Strateji | Açıklama | Etki |
|----------|----------|------|
| `weighted_loss` (varsayılan) | `class_weight = N / (C * n_c)` | İyi başlangıç |
| **`sqrt_weighted_loss`** | `sqrt(weight)` → normalize et | **Precision çöküşünü engeller, +0.03-0.05 F1** |
| `balanced_sampler` | WeightedRandomSampler ile batch dengeler | Hızlı convergence, ama overfit riski |
| **Focal Loss** | `α(1-p)^γ log(p)` — zor örneklere odaklan | Zor sınıflar için +0.01-0.02 |
| **LDAM/DRW** | Margin-based + re-weighting | SOTA dengesizlik yöntemi |

**Öneri**: `sweep.py` zaten `sqrt_weighted_loss` kullanıyor — **bunu baz alın**.

---

## 2. Model Mimarisi

| Model | Param | Macro-F1 (baseline) | Not |
|-------|-------|---------------------|-----|
| ResNet50 | 25M | 0.7300 | Güçlü baseline |
| **ConvNeXt-Tiny** | 28M | **~0.76-0.78** | Modern CNN, en iyi dengeli |
| ConvNeXt-Small | 50M | ~0.77-0.79 | VRAM 4GB'da zor (batch=16) |
| EfficientNet-B3 | 12M | ~0.75-0.77 | 256px ile iyi |
| ViT-Small | 22M | ~0.74-0.76 | Veri azsa CNN daha iyi |

**Öneri**: **ConvNeXt-Tiny + 224px + strong_aug** — sweep'te en iyi olan bu.

---

## 3. Augmentasyon (Strong Aug = +0.02-0.04 F1)

```python
# data.py içinde build_train_transform(strong_aug=True) zaten var:
T.RandomResizedCrop(scale=(0.5, 1.0), ratio=(0.75, 1.33))
T.RandomHorizontalFlip(p=0.5)
T.RandomVerticalFlip(p=0.2)           # Tarla görüntülerinde dikey de geçerli
T.RandomRotation(degrees=15)
T.ColorJitter(0.3, 0.3, 0.3, 0.05)
T.RandomGrayscale(p=0.02)
T.GaussianBlur(kernel_size=3)
T.RandomErasing(p=0.25)
```

**MixUp / CutMix** ekleyin (timm'de `mixup=0.8, cutmix=1.0`):
```python
# engine.py Trainer.__init__ içinde:
self.mixup_fn = Mixup(mixup_alpha=0.8, cutmix_alpha=1.0, num_classes=9)
# _train_one_epoch'te:
x, y = self.mixup_fn(x, y)
```

---

## 4. Çözünürlük Artırımı

| img_size | Batch (4GB) | Macro-F1 Δ |
|----------|-------------|------------|
| 224 | 32 | baseline |
| **256** | 16-24 | **+0.01-0.02** |
| 384 | 8 | +0.02 (ama çok yavaş) |

**Öneri**: EfficientNet-B3 + 256px deneyin (sweep'te var).

---

## 5. Kalibrasyon (Eğitimsiz +0.03-0.06 F1!)

```bash
# 1. Val tahminlerini üret
uv run deepweeds predict --run runs/exp/weights/best.pt --split val

# 2. Çarpanları optimize et
uv run python scripts/calibrate.py --run runs/exp

# 3. Test tahminine uygula (otomatik yapar)
uv run deepweeds predict --run runs/exp/weights/best.pt --split test
```

**Neden işe yarar?** Model "Negative" sınıfına over-confident olur. Çarpanlar bu bias'ı düzeltir.

---

## 6. Test Time Augmentation (TTA)

```python
# inference.py → predict_df içinde:
tta_transforms = build_tta_transforms(img_size)  # 5 view
probs = []
for t in tta_transforms:
    loader = DataLoader(dataset_with_transform(t), ...)
    _, p = predictor.predict(loader)
    probs.append(p)
final_prob = np.mean(probs, axis=0)  # Ortalama al
```

**Etki**: +0.01-0.02 F1, 5× yavaşlatır.

---

## 7. Ensemble (En Garantili)

```bash
# 3 farklı seed/model eğitin
uv run deepweeds train --run-name exp_seed42 --seed 42
uv run deepweeds train --run-name exp_seed123 --seed 123
uv run deepweeds train --run-name exp_convnext --model convnext_tiny

# Tahminleri toplayıp ortalayın
python -c "
import numpy as np, pandas as pd
p1 = pd.read_csv('runs/exp_seed42/predictions_test.csv')
p2 = pd.read_csv('runs/exp_seed123/predictions_test.csv')
p3 = pd.read_csv('runs/exp_convnext/predictions_test.csv')
probs = np.stack([p1[[f'prob_{i}' for i in range(9)]].values,
                  p2[[f'prob_{i}' for i in range(9)]].values,
                  p3[[f'prob_{i}' for i in range(9)]].values]).mean(0)
pred = probs.argmax(1)
# submission.csv yaz
"
```

**Etki**: +0.02-0.05 F1 (çeşitlilik ne kadar yüksekse o kadar iyi).

---

## 8. Özet: Hangi Sırayla Uygulayayım?

```mermaid
graph TD
    A[Baseline: ResNet50 weighted_loss] --> B[ConvNeXt-Tiny + sqrt_weighted_loss]
    B --> C[Strong Augmentation]
    C --> D[img_size=256 dene]
    D --> E[Calibration (calibrate.py)]
    E --> F[TTA (opsiyonel)]
    F --> G[3-5 Model Ensemble]
```

**Beklenen ilerleme**: 0.7300 → 0.76 → 0.78 → 0.79 → 0.80 → 0.81 → **0.83+**

---

## Hızlı Başlangıç (En İyi ROI)

```bash
# 1. ConvNeXt-Tiny + sqrt_loss + strong_aug (sweep'te hazır)
uv run python scripts/sweep.py --only exp-convnext-tiny-sqrt

# 2. Kalibrasyon
uv run python scripts/calibrate.py --run runs/exp-convnext-tiny-sqrt

# 3. Test tahmini
uv run deepweeds predict --run runs/exp-convnext-tiny-sqrt/weights/best.pt --split test
```

Bu 3 adımla **0.73 → ~0.80** hedeflenebilir.