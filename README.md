# 🌱 DeepWeeds T03 — Tarla Fotoğraflarından Yabancı Ot Sınıflandırma

<p align="center">
  <img src="https://img.shields.io/badge/Python-3.12%2B-blue?style=flat-square&logo=python&logoColor=white" alt="Python Version">
  <img src="https://img.shields.io/badge/PyTorch-Deep%20Learning-red?style=flat-square&logo=pytorch&logoColor=white" alt="PyTorch">
  <img src="https://img.shields.io/badge/FastAPI-API%20%26%20Demo-teal?style=flat-square&logo=fastapi&logoColor=white" alt="FastAPI">
  <img src="https://img.shields.io/badge/Package%20Manager-uv-purple?style=flat-square" alt="uv">
  <img src="https://img.shields.io/badge/License-CC%20BY%204.0-lightgrey?style=flat-square" alt="License">
</p>

**Takım:** `BM_TAKIMKODU_T03` · **Tema:** `T03` · Tarım Teknolojileri · **Sürüm:** `0.1.0`

---

## 📋 İçindekiler
1. [Bu Proje Ne Yapar?](#1-bu-proje-ne-yapar)
2. [Yöntem Özeti](#2-yöntem-özeti)
3. [Gereksinimler ve Kurulum](#3-gereksinimler-ve-kurulum)
4. [Hızlı Başlangıç](#4-hızlı-başlangıç)
5. [Komutlar ve Parametreler](#5-komutlar-ve-parametreler)
6. [Otomatik Üretilen Çıktılar](#6-otomatik-üretilen-çıktılar)
7. [Kontrollü Deneyler](#7-kontrollü-deneyler)
8. [Proje Yapısı](#8-proje-yapısı)
9. [Sınırlar ve Etik Çerçeve](#9-sınırlar-ve-etik-çerçeve)
10. [Sorun Giderme](#10-sorun-giderme)
11. [Kaynaklar ve Lisanslar](#11-kaynaklar-ve-lisanslar)

---

## 1. Bu proje ne yapar?

Bir tarlada çekilen bitki fotoğrafına bakıp, fotoğraftaki bitkinin **8 zararlı yabancı ot türünden hangisi olduğunu** (ya da bu otlardan hiçbirinin bulunmadığını) tahmin eden uçtan uca bir görüntü sınıflandırma sistemidir. Çıktı olarak 9 sınıfın her biri için bir olasılık üretir ve en yüksek olasılıklı sınıfı bildirir.

> **Örnek Çıktı:** `"Lantana" → %91 · "Negative" → %4 · diğer → %5`

Sistem üç ana bileşenden oluşur:
1. **Eğitim (`train`)** — Derin öğrenme modeli (ResNet50), ~12.000 etiketli tarla fotoğrafıyla eğitilir.
2. **Tahmin (`predict`)** — Eğitilmiş model, yarışma formatında tahmin dosyası (`predictions.csv`) üretir.
3. **Demo / API (`api`)** — Tarayıcı üzerinden fotoğraf yükleyip canlı tahmin yapılan yerel web arayüzü (internetsiz çalışır) ve FastAPI tabanlı programatik servis.

### 🌿 Hedef Sınıflar

| # | Türkçe Adı | Bilimsel / İngilizce Adı |
|:---:|---|---|
| **0** | Hint hünnabı | Chinee apple |
| **1** | Ağaç minesi | Lantana |
| **2** | Kudüs dikeni | Parkinsonia |
| **3** | Partenyum otu | Parthenium |
| **4** | Dikenli akasya | Prickly acacia |
| **5** | Kauçuk sarmaşığı | Rubber vine |
| **6** | Siyam otu | Siam weed |
| **7** | Yılan otu | Snake weed |
| **8** | Hedef otlardan hiçbiri yok | Negative |

> 📌 **Başarı Ölçütü:** **Macro-F1** (her sınıfın F1 skoru eşit ağırlıkla ortalanır). Bu metrik, az örnekli sınıfların da (örn. Parkinsonia) en büyük sınıf kadar önem taşıdığını garanti eder.

---

## 2. Yöntem özeti

| Bileşen | Seçim / Yaklaşım |
|---|---|
| **Model** | `ResNet50`, ImageNet ön-eğitimli (`timm`), son katman 9 sınıfa uyarlanmıştır |
| **Kayıp Fonksiyonu** | Sınıf ağırlıklı Cross-Entropy (ağırlıklar **yalnızca train'den**) + Label Smoothing (`0.1`) |
| **Sınıf Dengesizliği** | `weighted_loss` (varsayılan) veya `balanced_sampler` veya `none` |
| **Veri Artırma (Augmentation)** | Sadece train'de, anlık: `RandomResizedCrop` (%60–100) + Yatay Çevirme + `ColorJitter` |
| **Optimizasyon** | `AdamW` ($\text{lr}=1\text{e-4}$, $\text{wd}=1\text{e-4}$) + Cosine LR + Karma Hassasiyet (`AMP`) |
| **Model Seçimi** | Erken durdurma (Early Stopping) **val macro-F1** metriğine göre; en iyi ağırlık `best.pt` olarak saklanır |
| **Test Verisi** | Eğitim ve model seçim süreçlerinde **asla** kullanılmaz; yalnızca nihai tahmin aşamasında bir kez işleme alınır |

---

## 3. Gereksinimler ve kurulum

* **Gereksinimler:** Python 3.12+, [uv](https://docs.astral.sh/uv/), NVIDIA GPU (önerilir; CPU desteği de mevcuttur ancak yavaştır).

```bash
# 1) Depoyu klonlayın
git clone https://github.com/burakwa/deepweeds
cd deepweeds

# 2) Bağımlılıkları kurun (torch, torchvision cu124 wheel'leri pyproject'te tanımlıdır)
uv sync

# 3) Kurulumu doğrulayın (3 testin geçtiğini görün)
uv run pytest -q

# 1) Modeli Eğit (RTX 2050'de ~1 saat; tüm loglar ve raporlar runs/ altına kaydedilir)
uv run deepweeds train --run-name exp01 --model resnet50

# 2) Test Tahminlerini Üret (Yalnızca model seçim aşaması tamamlandıktan sonra!)
uv run deepweeds predict --run runs/exp01/weights/best.pt --split test --output predictions.csv

# 3) Arayüzü / Demoyu Başlat → Tarayıcıda aç: [http://127.0.0.1:8000](http://127.0.0.1:8000)
uv run deepweeds api --model runs/exp01/weights/best.pt
