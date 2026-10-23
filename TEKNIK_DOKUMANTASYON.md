# DeepWeeds T03 — Teknik Dokümantasyon

## Proje Mimarisi

DeepWeeds, tarla fotoğraflarından 8 yabancı ot türü + "Negative" (hiçbiri) sınıfını ayıran bir görüntü sınıflandırma sistemidir. Sistem **uçtan uca (end-to-end)** çalışır: veri yükleme → eğitim → değerlendirme → tahmin → servis (API).

### Klasör Yapısı

```
deepweeds/
├── src/deepweeds/           # Ana paket
│   ├── __init__.py          # Versiyon ve entry point
│   ├── config.py            # Yapılandırma (dataclass'lar, etiket eşleme)
│   ├── models.py            # Model oluşturma (timm kütüphanesi)
│   ├── data.py              # Veri setleri, DataLoader, augmentasyon
│   ├── engine.py            # Eğitim döngüsü (Trainer)
│   ├── metrics.py           # Metrik hesaplama (macro-F1, confusion matrix)
│   ├── reports.py           # Grafik ve rapor üretimi
│   ├── inference.py         # Çıkarım (Predictor, CSV yazma)
│   ├── api.py               # FastAPI servisi + tarayıcı demo
│   ├── cli.py               # Komut satırı arayüzü (train/predict/evaluate/api)
│   └── utils.py             # Yardımcılar (seed, device, logger)
├── scripts/
│   ├── sweep.py             # Hiperparametre/model arama (bkz. alt bölüm)
│   └── calibrate.py         # Tahmin kalibrasyonu (bkz. alt bölüm)
├── tests/                   # Birim testleri (pytest)
├── data/                    # Veri seti (CSV + images/)
│   ├── train.csv / val.csv / test.csv
│   └── dataset_card.json    # Veri seti meta bilgisi
├── runs/                    # Eğitim çıktıları (otomatik oluşur)
│   └── <run_name>/
│       ├── weights/best.pt  # En iyi model ağırlıkları
│       ├── history.csv      # Epoch bazlı loss/F1 geçmişi
│       ├── metrics.json     # Final metrikler
│       ├── predictions_val.csv
│       └── reports/         # PNG grafikler + classification_report.txt
├── notebooks/               # EDA not defterleri
├── pyproject.toml           # Bağımlılıklar ve proje meta verisi
└── README.md                # Kullanım kılavuzu
```

---

## Veri Akışı

```
train.csv / val.csv / test.csv
         │
         ▼
DeepWeedsData (data.py)
  - CSV okur, kolonları algılar (path, label)
  - Görselleri diskten bulur (_index_images)
  - Etiketleri 0-8 arası indekslere eşler (build_label_map)
  - Sınıf ağırlıklarını hesaplar (class_weights)
  - Train/Val/Test DataLoader üretir
         │
         ▼
Trainer (engine.py)
  - Model: timm.create_model (ResNet50, ConvNeXt, EfficientNet, ViT...)
  - Kayıp: CrossEntropy (label_smoothing=0.1) + opsiyonel sınıf ağırlığı
  - Optimizer: AdamW (lr=1e-4, wd=1e-4)
  - Scheduler: CosineAnnealingLR (veya OneCycleLR sweep'te)
  - AMP (karma hassasiyet) destekli
  - Early stopping: val macro-F1 bazlı (patience=5)
         │
         ▼
Predictor (inference.py)
  - Checkpoint'ten modeli yükler (offline, pretrained=False)
  - Tek görüntü (predict_pil) veya batch (predict_df) çıkarım
  - TTA (Test Time Augmentation) opsiyonu
  - Benchmark gecikme ölçümü
         │
         ▼
Yarışma Formatı CSV (write_predictions_csv)
  id,prediction,prob_0..prob_8,latency_ms,bytes
```

---

## Komutlar

```bash
# 1) Eğitim
uv run deepweeds train --run-name exp01 --model resnet50 --epochs 25

# 2) Val/Test tahmini (yarışma formatı CSV)
uv run deepweeds predict --run runs/exp01/weights/best.pt --split val --output pred_val.csv
uv run deepweeds predict --run runs/exp01/weights/best.pt --split test --output predictions.csv

# 3) Demo / API (tarayıcı: http://127.0.0.1:8000)
uv run deepweeds api --model runs/exp01/weights/best.pt

# 4) Değerlendirme (truth + pred CSV → metrik + grafik)
uv run deepweeds evaluate --truth data/val.csv --predictions pred_val.csv --out-dir reports
```

---

## Çıktı Dosyaları ve Anlamları

| Dosya | İçerik | Ne Anlama Gelir |
|-------|--------|-----------------|
| `weights/best.pt` | Model ağırlıkları + config + class_names | En iyi val macro-F1 veren epoch. **Bu dosya çıkarım için kullanılır.** |
| `history.csv` | epoch, train_loss, val_loss, val_macro_f1, val_accuracy, lr, seconds | Eğitim dinamikleri. Overfitting varsa train_loss ↓ ama val_loss ↑ olur. |
| `metrics.json` | macro_f1, accuracy, per_class, confusion, n_val, generated_at | **Final başarı özeti.** `macro_f1` yarışma metriğidir. |
| `predictions_val.csv` | id, prediction, prob_0..prob_8, latency_ms, bytes | Val seti için sınıf olasılıkları. `prediction = argmax(prob)`. |
| `reports/confusion_matrix.png` | Isı haritası (gerçek vs tahmin) | Hangi sınıflar birbiriyle karıştırılıyor? |
| `reports/per_class_f1.png` | Yatay bar grafiği (F1 per class + support) | Az örnekli sınıflar (Parkinsonia vb.) ne kadar iyi öğrenildi? |
| `reports/training_curves.png` | Loss ve Accuracy/F1 eğrileri | Eğitim stabil mi? Erken durdurma doğru mu çalıştı? |
| `reports/misclassified_grid.png` | En güvenli yanlış tahminler (görsel) | Modelin "emin olduğu" hataları görselleştirir. |
| `reports/classification_report.txt` | sklearn classification_report (precision/recall/f1/support) | Sınıf bazlı detaylı metrikler. |

### Önemli Metrikler

- **Macro-F1** → 9 sınıfın F1 skorunun **aritmetik ortalaması**. Az örnekli sınıflar (Parkinsonia, Siam weed) ile çok örnekli sınıflar (Negative) **eşit ağırlıkta** değerlendirilir. Yarışma başarı kriteridir.
- **Accuracy** → Doğru tahmin / toplam. Sınıf dengesizliğinde yanıltıcı olabilir (Negative dominant).
- **Per-class F1** → Her sınıf için ayrı F1. Hangi ot türünde zayıf kaldığımızı gösterir.

---

## `scripts/sweep.py` — Hiperparametre/Model Arama

**Ne işe yarar?**  
Farklı model mimarileri, çözünürlükler, augmentasyon stratejileri ve loss fonksiyonlarını **sırayla eğitip** val macro-F1 skorlarını karşılaştırır. Baseline (ResNet50, 0.7300) üzerine kuran deneyler tasarlanmıştır.

### Deneylerin Ortak Özellikleri
- `sqrt_weighted_loss`: Sınıf ağırlıklarının karekökü alınıp normalize edilir → aşırı ağırlıklandırmadan kaynaklı precision düşüşünü engeller.
- `strong_aug`: RandomVerticalFlip, Rotation, GaussianBlur, RandomErasing gibi güçlü augmentasyon.
- `OneCycleLR`: CosineAnnealing yerine batch-bazlı OneCycleLR (max_lr = 10× base_lr).
- RTX 2050 (4 GB VRAM) için batch size ve img_size optimize edilmiştir.

### Çalıştırma
```bash
# Planı gör (eğitmez)
uv run python scripts/sweep.py --dry-run

# Tüm deneyleri sırayla çalıştır
uv run python scripts/sweep.py

# Sadece belirli bir deney
uv run python scripts/sweep.py --only exp-convnext-tiny-sqrt

# İlk 2 deneyi çalıştır
uv run python scripts/sweep.py --runs 2
```

### Çıktı
- Her deney için `runs/<exp_name>/` klasörü oluşur.
- `runs/sweep_results.json` → Tüm deneylerin özet tablosu (raw + calibrated macro-F1).
- Konsolda canlı sıralama ve en iyi model yolu yazdırılır.

---

## `scripts/calibrate.py` — Tahmin Eşik / Çarpan Optimizasyonu

**Ne işe yarar?**  
Eğitilmiş bir modelin val setindeki **softmax olasılıkları** (`predictions_val.csv`) üzerine **sınıf bazlı çarpanlar (multipliers)** uygulayarak macro-F1'i artırır. Modeli yeniden eğitmez; sadece çıkarımdaki karar sınırını kaydırır.

### Mantık
- Sınıf dengesizliğinde model "Negative" sınıfına biaslı olabilir.
- `prob_c * multiplier_c` formülü ile olasılıkları yeniden ağırlar.
- Nelder-Mead optimizasyonu ile macro-F1'i maksimize eden 9 çarpan bulunur.
- Sonuç `calibration.json` olarak kaydedilir.

### Çalıştırma
```bash
uv run python scripts/calibrate.py --run runs/exp01-weighted
```

### Çıktı Örneği
```
Orijinal Macro-F1    : 0.7300
Kalibre Edilmis F1   : 0.7873 (+0.0573)

Optimal Carpanlar:
  - Chinee apple     : 1.12
  - Lantana          : 0.98
  - Parkinsonia      : 1.45   <-- Az örnekli sınıf boost edildi
  - Negative         : 0.85   <-- Dominant sınıf baskılandı
```

### Kullanım
`predict` komutu `--calibration` argümanıyla veya run klasöründeki `calibration.json`'ı otomatik bulup test tahminlerine uygular.

```bash
uv run deepweeds predict --run runs/exp01-weighted/weights/best.pt --split test
# Otomatik calibration.json kullanır (varsa)
```

---

## Model Seçimi ve Strateji Önerileri

| Senaryo | Öneri |
|---------|-------|
| Hızlı baseline | `resnet50` + `weighted_loss` + 224px |
| En iyi macro-F1 | `convnext_tiny` + `sqrt_weighted_loss` + `strong_aug` + 224px |
| Yüksek çözünürlük | `efficientnet_b3` + 256px (batch=16) |
| Çok hızlı deneme | `efficientnet_b0` + 224px (batch=32) |

**Kalibrasyon her zaman uygulanmalıdır** — modeli yeniden eğitmeden +0.03–0.06 macro-F1 kazandırır.

---

## Sık Karşılaşılan Sorunlar

| Sorun | Çözüm |
|-------|-------|
| `CUDA out of memory` | `batch_size` azaltın (16→8), `img_size` küçültün (256→224), `amp=True` olduğundan emin olun. |
| `predictions_val.csv` yok | Önce `train` komutu çalışmalı, veya `predict --split val` ile üretin. |
| Etiketler eşleşmiyor | `dataset_card.json` sınıf sırası ile CSV etiketleri aynı mı? `build_label_map` hata mesajını okuyun. |
| Macro-F1 düşük | `calibrate.py` çalıştırın → `predict` tekrar çalıştırın. `strong_aug` + `sqrt_weighted_loss` deneyin. |
| API `python-multipart` hatası | `uv pip install python-multipart` |

---

## Lisans ve Kaynaklar

- Veri seti: [DeepWeeds](https://github.com/AlexOlsen/DeepWeeds) — CC BY 4.0
- Kod: Bu proje — MIT lisansı (veya proje lisansına göre)
- Kütüphaneler: PyTorch, timm, FastAPI, Albumentations (augmentation), scikit-learn (metrics)