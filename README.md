# 🌱 DeepWeeds T03 — Tarla Fotoğraflarından Yabancı Ot Sınıflandırma

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-API%20%26%20Demo-teal?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Streamlit-Web%20UI-orange?style=flat-square&logo=streamlit&logoColor=white" alt="Streamlit">
  <img src="https://img.shields.io/badge/Package%20Manager-uv-purple?style=flat-square" alt="uv">
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square" alt="License">
</p>

**Takım:** `BM_TAKIMKODU_T03` · **Tema:** `T03` · Tarım Teknolojileri · **Sürüm:** `0.1.0`

---

## 📋 İçindekiler
1. [Bu Proje Ne Yapar?](#1-bu-proje-ne-yapar)
2. [En İyi Sonuçlar (Leaderboard)](#2-en-iyi-sonuçlar-leaderboard)
3. [Yöntem Özeti](#3-yöntem-özeti)
4. [Gereksinimler ve Kurulum](#4-gereksinimler-ve-kurulum)
5. [Hızlı Başlangıç (5 Dakika)](#5-hızlı-başlangıç-5-dakika)
6. [Komutlar ve Parametreler](#6-komutlar-ve-parametreler)
7. [Web Arayüzleri (Demo)](#7-web-arayüzleri-demo)
8. [Otomatik Üretilen Çıktılar](#8-otomatik-üretilen-çıktılar)
9. [Proje Yapısı](#9-proje-yapısı)
10. [Sorun Giderme (FAQ)](#10-sorun-giderme-faq)
11. [Kaynaklar ve Lisanslar](#11-kaynaklar-ve-lisanslar)

---

## 1. Bu Proje Ne Yapar?

Bir tarlada çekilen bitki fotoğrafına bakıp, fotoğraftaki bitkinin **8 zararlı yabancı ot türünden hangisi olduğunu** (veya bu otlardan hiçbirinin bulunmadığını) tahmin eden **uçtan uca görüntü sınıflandırma sistemidir**.

> **Örnek Çıktı:** `"Lantana" → %91 · "Negative" → %4 · diğerleri → %5`

Sistem üç ana bileşenden oluşur:

| Bileşen | Açıklama |
|---------|----------|
| **🎓 Eğitim (`train`)** | Derin öğrenme modeli (EfficientNet-B3), ~12.000 etiketli tarla fotoğrafıyla eğitilir. |
| **🔮 Tahmin (`predict`)** | Eğitilmiş model, yarışma formatında `predictions.csv` üretir (otomatik kalibrasyon uygulanır). |
| **🌐 Demo / API (`api`, `streamlit`)** | Tarayıcıdan fotoğraf yükleyip anlık tahmin alan yerel arayüzler (internetsiz çalışır). |

### 🌿 Hedef Sınıflar (9 Sınıf)

| # | Türkçe Adı | Bilimsel / İngilizce Adı |
|:---:|------------|---------------------------|
| **0** | Hint hünnabı | Chinee apple |
| **1** | Ağaç minesi | Lantana |
| **2** | Kudüs dikeni | Parkinsonia |
| **3** | Partenyum otu | Parthenium |
| **4** | Dikenli akasya | Prickly acacia |
| **5** | Kauçuk sarmaşığı | Rubber vine |
| **6** | Siyam otu | Siam weed |
| **7** | Yılan otu | Snake weed |
| **8** | Hedef otlardan hiçbiri yok | **Negative** |

> 📌 **Başarı Ölçütü:** **Macro-F1** — her sınıfın F1 skoru eşit ağırlıkla ortalanır. Bu, az örnekli sınıfların (ör. Parkinsonia, Snake weed) en büyük sınıf kadar önem taşıdığını garanti eder.

---

## 2. En İyi Sonuçlar (Leaderboard)

| Model | Strateji | Görüntü Boyutu | Macro-F1 (Val) | Kalibreli Macro-F1 | Doğruluk | Eğitim Süresi |
|-------|----------|----------------|----------------|-------------------|----------|---------------|
| 🏆 **EfficientNet-B3** | **sqrt_weighted_loss** + Strong Aug | **256×256** | **0.8181** | **0.8568** | **0.8591** | ~40 dk (RTX 3060) |
| ResNet50 | weighted_loss | 224×224 | 0.7300 | 0.7873 | 0.7581 | ~25 dk |

> **Not:** Kalibrasyon (temperature scaling benzeri per-sınıf çarpanlar) validation setinde **~%4.8 macro-F1 artışı** sağlar. Test tahminlerinde **otomatik uygulanır**.

### Sınıf Bazında F1 (exp-effb3-sqrt, kalibreli)
| Sınıf | F1 | Support |
|-------|-----|---------|
| Parkinsonia | **0.942** | 316 |
| Parthenium | **0.897** | 197 |
| Siam weed | **0.902** | 193 |
| Lantana | **0.887** | 238 |
| Rubber vine | **0.881** | 184 |
| Negative | **0.863** | 1277 |
| Prickly acacia | **0.729** | 161 |
| Chinee apple | **0.660** | 46 |
| Snake weed | **0.601** | 50 |

---

## 3. Yöntem Özeti

| Bileşen | Seçim / Yaklaşım |
|---------|------------------|
| **Model** | `EfficientNet-B3` (timm), ImageNet ön-eğitimli, son katman 9 sınıfa uyarlanmış |
| **Kayıp Fonksiyonu** | **Sqrt Weighted Cross-Entropy** (ağırlıklar `√(1/freq)` formülüyle *train* setinden hesaplanır) + Label Smoothing (`0.1`) |
| **Sınıf Dengesizliği** | `sqrt_weighted_loss` (varsayılan en iyi), alternatif: `weighted_loss`, `balanced_sampler`, `none` |
| **Veri Artırma** | Sadece train'de: `RandomResizedCrop` (%60–100) + Yatay Çevirme + `ColorJitter` + `RandAugment` (strong_aug=True) |
| **Optimizasyon** | `AdamW` (lr=`5e-5`, wd=`1e-4`) + Cosine Annealing LR + **AMP** (Karma Hassasiyet) |
| **Model Seçimi** | Erken durdurma (Early Stopping) **val macro-F1**'e göre; en iyi ağırlık `best.pt` olarak kaydedilir |
| **Kalibrasyon** | Validation setinde per-sınıf çarpanlar öğrenilir, test tahminlerinde otomatik uygulanır |
| **Test Verisi** | Eğitim ve model seçim süreçlerinde **asla kullanılmaz**; yalnızca nihai `predict --split test` aşamasında bir kez işlenir |

---

## 4. Gereksinimler ve Kurulum

### 🖥️ Sistem Gereksinimleri
| Bileşen | Minimum | Önerilen |
|---------|---------|----------|
| **İşletim Sistemi** | Windows 10/11, Linux, macOS | Linux / WSL2 |
| **Python** | 3.12+ | 3.12 |
| **GPU** | CPU çalışır (yavaş) | NVIDIA GPU (CUDA 12.4+, 8GB+ VRAM) |
| **RAM** | 8 GB | 16+ GB |
| **Disk** | 5 GB boş | 10+ GB SSD |

### 📦 Paket Yöneticisi: `uv` (Hızlı, Modern)
Bu proje [`uv`](https://docs.astral.sh/uv/) kullanır — `pip`/`venv`/`pipx` alternatifi, 10-100x hızlı.

```bash
# Windows (PowerShell)
powershell -c "irm https://astral.sh/uv/install.ps1 | iex"

# Linux / macOS / WSL
curl -LsSf https://astral.sh/uv/install.sh | sh

# Kurulumu doğrula
uv --version
```

> **Neden uv?** Sanal ortam oluşturur, bağımlılıkları çözer ve projeyi izole eder — `venv` + `pip` + `pip-tools` tek araçta.

---

## 5. Hızlı Başlangıç (5 Dakika)

### Adım 1: Depoyu Klonlayın
```bash
git clone https://github.com/burakwa/deepweeds
cd deepweeds
```

### Adım 2: Bağımlılıkları Kurun (İlk kez ~2-3 dk)
```bash
# uv.pyproject.toml'daki torch/torchvision CUDA 12.4 wheel'lerini indirir
uv sync
```

### Adım 3: Kurulumu Doğrulayın
```bash
# 3 testin geçtiğini görün (saniyeler sürer)
uv run pytest -q
```
> ✅ `108 passed` görürseniz kurulum tamam.

---

### 🎯 Senaryo A: Hazır Model ile Hemen Tahmin Yapın (Eğitimsiz)

En iyi modelimiz (`exp-effb3-sqrt`, **Macro-F1: 0.818**) zaten `runs/` klasöründe var.

```bash
# 1) Test seti için yarışma formatında predictions.csv üret
uv run deepweeds predict --run runs/exp-effb3-sqrt/weights/best.pt --split test --output predictions.csv

# 2) Sonuçları kontrol et
head predictions.csv
```
> **Çıktı:** `predictions.csv` — yarışma gönderim formatında hazır.

---

### 🎓 Senaryo B: Kendi Modelinizi Eğitin

```bash
# EfficientNet-B3 + sqrt_weighted_loss + strong aug (en iyi konfigürasyon)
uv run deepweeds train \
  --model efficientnet_b3 \
  --img-size 256 \
  --balance sqrt_weighted_loss \
  --aug \
  --epochs 20 \
  --lr 5e-5 \
  --patience 6 \
  --run-name my-exp-effb3-sqrt
```

| Parametre | Açıklama | Önerilen |
|-----------|----------|----------|
| `--model` | timm model adı | `efficientnet_b3`, `convnext_tiny`, `resnet50` |
| `--img-size` | Giriş çözünürlüğü | 256 (EffB3), 224 (ResNet) |
| `--balance` | Sınıf dengesizliği stratejisi | `sqrt_weighted_loss` ⭐ |
| `--aug` / `--no-aug` | Strong augmentation | `--aug` (varsayılan) |
| `--epochs` | Maks epoch | 20-30 |
| `--lr` | Öğrenme oranı | `5e-5` (EffB3), `1e-4` (ResNet) |
| `--patience` | Early stopping bekleme | 5-8 |
| `--run-name` | Çıktı klasör adı | Kendi isim verirsiniz |

> **Eğitim süresi:** RTX 3060 12GB'de ~40 dakika, RTX 2050'de ~1 saat. Tüm loglar, grafikler, ağırlıklar `runs/my-exp-effb3-sqrt/` altına kaydedilir.

---

### 🔮 Senaryo C: Kendi Eğittiğiniz Model ile Tahmin

```bash
# Kendi modelinizle test tahmini (kalibrasyon otomatik algılanır)
uv run deepweeds predict --run runs/my-exp-effb3-sqrt/weights/best.pt --split test --output my_predictions.csv
```

---

## 6. Komutlar ve Parametreler

```bash
# Tüm komutları listele
uv run deepweeds --help

# Belirli bir komutun yardımını al
uv run deepweeds train --help
uv run deepweeds predict --help
```

### 6.1 `train` — Model Eğit
```bash
uv run deepweeds train [OPTIONS]
```
| Seçenek | Varsayılan | Açıklama |
|---------|------------|----------|
| `--data` | `data` | CSV dosyalarının klasörü |
| `--model` | `resnet50` | timm model adı (`efficientnet_b3`, `convnext_tiny`, vb.) |
| `--img-size` | `224` | Giriş görüntü boyutu |
| `--batch-size` | `32` | Batch boyutu (VRAM'e göre ayarlayın) |
| `--balance` | `weighted_loss` | `none` \| `weighted_loss` \| `sqrt_weighted_loss` \| `balanced_sampler` |
| `--epochs` | `25` | Maksimum epoch |
| `--lr` | `1e-4` | Öğrenme oranı |
| `--wd` | `1e-4` | Weight decay |
| `--label-smoothing` | `0.1` | Label smoothing |
| `--patience` | `5` | Early stopping bekleme |
| `--seed` | `42` | Rastgele tohum |
| `--run-root` | `runs` | Çıktı kök klasörü |
| `--run-name` | `auto` | Çalıştırma adı (yoksa zaman damgası) |
| `--aug` / `--no-aug` | `True` | Strong augmentation aç/kapa |
| `--amp` / `--no-amp` | `True` | Mixed precision (AMP) |
| `--deterministic` | `False` | Tam deterministik mod (daha yavaş) |

---

### 6.2 `predict` — Tahmin Üret (Val / Test)
```bash
uv run deepweeds predict --run PATH/TO/best.pt [OPTIONS]
```
| Seçenek | Varsayılan | Açıklama |
|---------|------------|----------|
| `--run` | **zorunlu** | `weights/best.pt` dosyasının yolu |
| `--split` | `test` | `val` veya `test` |
| `--output` | `predictions.csv` | Çıktı CSV yolu |
| `--calibration` | `auto` | `calibration.json` yolu (yoksa run klasöründen otomatik bulunur) |
| `--data` | `data` | Veri kök klasörü |

> **Önemli:** `--split test` **yalnızca** model seçimi bittikten **son** kullanılmalıdır. Validation üzerinde iterasyon yaparken `--split val` kullanın.

---

### 6.3 `evaluate` — Metrik Hesapla
```bash
uv run deepweeds evaluate --truth ground_truth.csv --predictions preds.csv --out-dir reports
```
Gerçek etiketler + tahminler CSV'sinden macro-F1, confusion matrix, per-class metrikler üretir.

---

### 6.4 `api` — FastAPI + Tarayıcı Demosu
```bash
uv run deepweeds api --model runs/exp-effb3-sqrt/weights/best.pt
```
- **Adres:** http://127.0.0.1:8000
- **Swagger UI:** http://127.0.0.1:8000/docs
- **Endpoint:** `POST /predict` (multipart/form-data: `file`)
- **Çıktı:** JSON `{ "predictions": [{"class": "Lantana", "probability": 0.91}, ...] }`

---

## 7. Web Arayüzleri (Demo)

### 7.1 Streamlit Arayüzü (Önerilen — Daha Görsel)
```bash
# Varsayılan en iyi model ile başlar
uv run deepweeds streamlit

# Kendi modelinizle
uv run deepweeds streamlit --model runs/my-exp/weights/best.pt

# Farklı portta
uv run deepweeds streamlit --port 8502
```
**Özellikler:**
- 📸 Görsel yükle (drag & drop)
- 🎯 Anlık tahmin + güven skoru
- 📊 Tüm sınıflar için olasılık çubuk grafiği
- ⚙️ TTA (Test-Time Augmentation) aç/kapa
- 📱 Mobil uyumlu, responsive tasarım
- 🔄 Model yeniden yükleme butonu

> **Adres:** http://localhost:8501 (varsayılan)

---

### 7.2 FastAPI + Basit HTML Demo
```bash
uv run deepweeds api --model runs/exp-effb3-sqrt/weights/best.pt
```
**Özellikler:**
- ⚡ Hızlı, hafif REST API
- 📖 Otomatik Swagger/OpenAPI dokümantasyonu (`/docs`)
- 🖥️ Basit tek sayfalık test arayüzü (`/`)
- 🔧 Programatik entegrasyon için hazır

---

## 8. Otomatik Üretilen Çıktılar

Her `train` çalıştırması şu yapıyı oluşturur:

```
runs/<run-name>/
├── config.json              # Eğitim konfigürasyonu (tam yeniden üretilebilirlik)
├── weights/
│   └── best.pt              # En iyi model ağırlıkları (macro-F1'e göre)
├── calibration.json         # Per-sınıf kalibrasyon çarpanları
├── history.csv              # Epoch başına loss/metric geçmişi
├── metrics.json             # Final validation metrikleri
├── predictions_val.csv      # Validation seti tahminleri (yarışma formatı)
└── reports/
    ├── training_curves.png  # Loss / Macro-F1 / LR eğrileri
    ├── confusion_matrix.png # Normalize confusion matrix
    ├── per_class_f1.png     # Sınıf bazında F1 çubuk grafiği
    ├── class_distribution.png # Train/Val sınıf dağılımı
    ├── misclassified_grid.png # Yanlış sınıflandırılan örnekler (grid)
    └── classification_report.txt # sklearn classification_report
```

> **Yeniden Eğitim:** `config.json` + `best.pt` ile tam yeniden üretim mümkündür.

---

## 9. Proje Yapısı

```
deepweeds/
├── data/                    # Veri seti (train.csv, val.csv, test.csv)
│   └── LICENSE_NOTICE.md
├── runs/                    # Eğitim çıktıları (gitignore'da)
│   ├── exp-effb3-sqrt/      # 🏆 En iyi model
│   ├── sweep_results.json   # Tüm deney özeti
│   └── ...
├── scripts/                 # Yardımcı scriptler
│   ├── sweep.py             # Hiperparametre arama
│   ├── run_api.py           # API başlatıcı
│   ├── run_streamlit.py     # Streamlit başlatıcı
│   └── calibrate.py         # Kalibrasyon hesapla
├── src/deepweeds/           # Ana kaynak kodu
│   ├── __init__.py
│   ├── cli.py               # 🎯 Ana CLI giriş noktası
│   ├── config.py            # Yapılandırma sınıfları
│   ├── data.py              # Veri yükleme / augmentation
│   ├── models.py            # Model oluşturma (timm wrapper)
│   ├── engine.py            # Eğitim döngüsü (Trainer)
│   ├── inference.py         # Tahmin + benchmark + TTA
│   ├── metrics.py           # Macro-F1, per-class metrikler
│   ├── reports.py           # Grafik/rapor üretimi
│   ├── api.py               # FastAPI uygulaması
│   ├── api_v2.py            # API v2 (geliştirme)
│   ├── streamlit_app.py     # 🌐 Streamlit arayüzü
│   └── utils.py             # Yardımcılar (logger, seed, device)
├── tests/                   # Pytest testleri (unit + smoke)
├── notebooks/               # EDA / Baseline not defterleri
├── pyproject.toml           # Proje meta + bağımlılıklar (uv)
├── uv.lock                  # Kilit dosyası (reproducible builds)
├── .python-version          # Python 3.12
└── README.md                # Bu dosya
```

---

## 10. Sorun Giderme (FAQ)

### ❌ `uv sync` hata veriyor / torch indirmiyor
```bash
# Cache'i temizle ve tekrar dene
uv cache clean
uv sync --reinstall
```
> `pyproject.toml`'da `torch` ve `torchvision` için `pytorch` indexi (CUDA 12.4) tanımlıdır. Eğer CPU-only istiyorsanız `[[tool.uv.index]]` bölümünü kaldırın.

---

### ❌ `CUDA out of memory` (VRAM yetersiz)
```bash
# Batch size küçültün
uv run deepweeds train --batch-size 8 --model efficientnet_b3 ...

# Veya gradient accumulation (kodda desteklenmiyor, manuel epoch artırarak simüle edilebilir)
```
| GPU VRAM | Önerilen Batch (EffB3, 256×256) |
|----------|----------------------------------|
| 6 GB     | 8-12                             |
| 8 GB     | 16                               |
| 12+ GB   | 32                               |

---

### ❌ `ModuleNotFoundError: No module named 'deepweeds'`
```bash
# uv ortamında çalıştığınızdan emin olun
uv run python -c "import deepweeds; print('OK')"
```

---

### ❌ Streamlit / API "Model checkpoint bulunamadı" hatası
```bash
# Doğru yol verirsiniz
uv run deepweeds streamlit --model runs/exp-effb3-sqrt/weights/best.pt

# Varsayılan: runs/exp-effb3-sqrt/weights/best.pt
```

---

### ❌ Windows'da `uv` komutu tanınmıyor
```powershell
# Terminali yeniden başlatın (PATH güncellenir)
# Veya manuel ekleyin:
$env:PATH += ";$env:USERPROFILE\.local\bin"
```

---

### 🐛 Detaylı Loglar İçin
```bash
# DEBUG seviyesi loglar
uv run deepweeds train ... 2>&1 | tee train.log
```

---

## 11. Kaynaklar ve Lisanslar

| Kaynak | Link |
|--------|------|
| **DeepWeeds Veri Seti** | [Kaggle - DeepWeeds](https://www.kaggle.com/datasets/andrewmvd/deep-weeds) |
| **timm (Model Kütüphanesi)** | [github.com/huggingface/pytorch-image-models](https://github.com/huggingface/pytorch-image-models) |
| **PyTorch** | [pytorch.org](https://pytorch.org/) |
| **FastAPI** | [fastapi.tiangolo.com](https://fastapi.tiangolo.com/) |
| **Streamlit** | [streamlit.io](https://streamlit.io/) |
| **uv** | [docs.astral.sh/uv](https://docs.astral.sh/uv/) |

### Lisans
- **Kod:** MIT License (veya proje lisansınıza göre)
- **Veri Seti:** CC BY 4.0 — [DeepWeeds Lisans](data/LICENSE_NOTICE.md)
- **Ön-eğitilmiş Ağırlıklar (ImageNet):** İlgili model sahiplerinin lisanslarına tabidir (timm aracılığıyla).

---

## 🤝 Katkı ve İletişim

- **Sorun bildirimi:** [GitHub Issues](https://github.com/burakwa/deepweeds/issues)
- **Öneri / PR:** Memnuniyetle incelenir
- **E-posta:** burakkilnnc@gmail.com

---

<p align="center">
  <strong>🌿 DeepWeeds T03 — Tarım Teknolojileri Yarışması</strong><br>
  <em>EfficientNet-B3 • sqrt_weighted_loss • Macro-F1: 0.818 → 0.857 (kalibreli)</em>
</p>