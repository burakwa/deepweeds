# DeepWeeds (T03): Yabani Ot Sınıflandırma

Avustralya'da tarlada çekilmiş 256×256 renkli fotoğraflardan **9 sınıflı** yabani ot tanıma. 5 günlük yarışma projesi. Bu dosya, ne yaptığımızı, neden yaptığımızı ve sıradaki adımları tek yerde toplar.

---

## 1. Görev ve kurallar

| Madde | Değer |
|---|---|
| Görev | 9 sınıflı görüntü sınıflandırma (8 yabani ot türü + Negative) |
| **Ana metrik** | **macro-F1** (karar bununla verilir; accuracy sadece raporlanır) |
| Seed | `20261019` (dataset kartından) |
| Veri | 12.109 train · 2.662 val · 2.738 test |
| Lisans | CC BY 4.0 (kaynak: AlexOlsen/DeepWeeds) |

**Sınıflar:** 0 Chinee apple · 1 Lantana · 2 Parkinsonia · 3 Parthenium · 4 Prickly acacia · 5 Rubber vine · 6 Siam weed · 7 Snake weed · 8 Negative

### Kurallar (ihlal edilmez)

- **Test etiketine asla bakmayız.** Kart, test etiketlerinin kaynaktan geri kazanılabildiğini söylüyor. Bunu kullanmıyoruz. Test sadece en sonda tahmin üretmek için kullanılır.
- Val, eğitime karıştırılmaz. Tüm kararlar val macro-F1 ile verilir.
- Ek etiketleme yapılırsa yalnızca eğitim görüntülerinde, kaynak ve yöntem kaydıyla yapılır.
- Tür bilgisi hastalık veya ilaç dozu etiketi **değildir**. Veri Avustralya'dandır, Türkiye tarımına genelleme iddiası için yerel veri gerekir.

---

## 2. Veri hakkında bildiklerimiz

- **Dosyalar:** `train.csv`, `val.csv`, `test.csv`, `images/`, dataset kartı (`.json`).
- **CSV sütunları:** `id`, `path`, `label`, `group_id`, `source_id`, `sha256`, `split`. `group_id` çekim günü + kamera bilgisidir (ör. `20160928_0`).
- **Test etiketsizdir** (`label` sütunu boş). Test için sadece tahmin üretebiliriz, skoru jüri hesaplar. Tek ölçüm aracımız **val**.
- **Sızıntı kontrolü temiz:** 0 kopya görsel, split'ler arası 0 ortak `group_id`. Val skoru train'in dürüst bir vekili.
- **Test'te `group_id` bilgisi görünmüyor** (set sayımı tek değer verdi, büyük olasılıkla boş). Test'in gün/kamera yapısını bilemiyoruz.
- **Kameralar (0-3) hem train'de hem val'de var.** Asıl kayma çekim günü ve ışık koşulları (karanlık/aydınlık, karmaşık arka plan).
- **Val küçük:** 14 grup. Sınıf 0 (46) ve 7 (50) için çok az örnek var, bu sınıfların val F1'i gürültülü. Küçük farklara fazla anlam yüklemeyiz.

### Sınıf dağılımı (kaymaya dikkat)

| Sınıf | train | val | test (karttan) |
|---|---|---|---|
| 0 Chinee apple | 798 | **46** | 281 |
| 1 Lantana | 630 | 238 | 196 |
| 2 Parkinsonia | 580 | 316 | 135 |
| 3 Parthenium | 772 | 197 | **53** |
| 4 Prickly acacia | 814 | 161 | **87** |
| 5 Rubber vine | 667 | 184 | 158 |
| 6 Siam weed | 495 | 193 | **386** |
| 7 Snake weed | 857 | **50** | 109 |
| 8 Negative | 6496 (%54) | 1277 (%48) | 1333 (%49) |

Split'ler arası dağılım ciddi farklı. Bu yüzden "Negative'i bastır" gibi sınıf önselini ayarlayan hileler riskli.

---

## 3. Ortam

### Yerel (analiz, rapor, küçük testler)

```bash
uv init deepweeds --python 3.12
cd deepweeds
uv venv --python 3.12
uv add torch torchvision --index pytorch=https://download.pytorch.org/whl/cu124   # GPU yoksa: uv add torch torchvision
uv add timm scikit-learn pandas numpy matplotlib seaborn pillow ipykernel jupyter tqdm
uv run python -m ipykernel install --user --name deepweeds --display-name "Python (deepweeds)"
```

Yerel makine: RTX 2050 (4 GB VRAM). 256px eğitimde VRAM yetmiyor ve çok yavaş kalıyor. Bu yüzden **eğitim Colab'da yapılır.**

### Eğitim makinesi: Google Colab (T4)

1. `data/` içeriğini (`images/`, CSV'ler, kart json'u) `data.zip` yapıp Drive'a (`Drive/deepweeds/`) yükle.
2. Colab'da *Runtime → Change runtime type → T4 GPU*.
3. Veriyi Drive'dan **yerel diske** aç (Drive'dan doğrudan okumak yavaş), `pip install timm`.
4. Yollar: veri ve cache yerel diskte (`/content/...`), checkpoint ve sonuçlar Drive'da (oturum kopsa kaybolmasın).

### Önerilen klasör yapısı

```
deepweeds/
├── data/            # images/, train.csv, val.csv, test.csv, dataset kartı
├── notebooks/       # 01_eda_baseline.ipynb
├── outputs/         # checkpoint'ler, olasılık matrisleri, grafikler
└── README.md
```

---

## 4. 5 günlük plan

| Gün | İş | Durum |
|---|---|---|
| **1** | Keşif, sızıntı kontrolü, hızlı veri hattı, baseline (EfficientNet-B0, düz CE) | Devam ediyor |
| 2 | Sınıf dengesizliği (class weights / sampler / focal loss), ışık-renk augmentation, `GroupKFold(group_id)` ile ikinci doğrulama | Bekliyor |
| 3 | Daha güçlü model, TTA, olasılık kalibrasyonu | Bekliyor |
| 4 | Geliştirme fikri: belirsiz sonuçları uzmana yönlendirme + hızlı çıkarım ölçümü | Bekliyor |
| 5 | Final eğitim (tam veri ile), test olasılıkları ve tahminleri, kullanıcı senaryosu, sınırlar, rapor | Bekliyor |

### Gün 1: notebook hücre sırası (`01_eda_baseline.ipynb`)

| Hücre | İçerik | Neden |
|---|---|---|
| 1 | Import, seed (karttan), `DATA_ROOT`, cihaz | Tekrarlanabilirlik |
| 2 | CSV'leri yükle, yolları çöz, **kart sayılarıyla doğrula** | Eksik/yanlış veri varsa hemen yakala |
| 3 | Sınıf dağılımı (train/val/test) | Kaymayı gör |
| 4 | Örnek görseller | Işık/arka plan çeşitliliğini gör, augmentation'ı buna göre seç |
| 5 | Sızıntı + grup analizi | Val'in güvenilir olduğunu kanıtla |
| 6 | Hızlı veri hattı: tüm görseller uint8 dizide, **augmentation GPU'da** | Worker'sız, hızlı, düşük RAM |
| 7 | Baseline eğitimi (EfficientNet-B0, AdamW + OneCycle, AMP) | Ölçülebilir başlangıç noktası |
| 8 | Loss/metrik grafikleri, en iyi epoch işareti | Over/underfitting teşhisi |
| 9 | Sınıf bazlı rapor + confusion matrix | Zorunlu çıktı, yarının odağı |

**Temel tasarım kararları**

- **EfficientNet-B0, ImageNet ön eğitimli:** küçük veriyle sıfırdan eğitmekten çok daha iyi, hafif olduğu için hızlı çıkarım fikrine de uyuyor.
- **IMG = 256:** görseller zaten 256×256, yeniden boyutlandırma bilgi kaybettirmez.
- **Düz CrossEntropy:** baseline dengesizliği bilerek çözmüyor, Gün 2'de farkı ölçeceğiz.
- **Augmentation:** RandomResizedCrop, flip, rotasyon, parlaklık/kontrast/doygunluk. Çekim günü ve ışık kaymasına karşı ana savunma.
- **En iyi epoch val macro-F1'e göre kaydedilir.** Overfitting'e karşı otomatik koruma.
- **Hız:** aşamalı çözünürlük (160→208→256), fused AdamW, veri GPU'da.

### Over/underfitting nasıl okunur

| Gözlem | Anlam | Yapılacak |
|---|---|---|
| Train loss düşüyor, val loss yükseliyor | Overfitting | Augmentation artır, weight decay artır, epoch azalt |
| İkisi de yüksek ve düz | Underfitting | Daha büyük model, daha uzun eğitim, lr ayarı |
| İkisi birlikte düşüyor, yakın | Sağlıklı | Devam |
| Val macro-F1 sert zıplıyor | Val küçük/gürültülü | Tek epoch'a değil eğilime bak |

Not: Train'de güçlü augmentation olduğundan train loss'un val loss'tan yüksek çıkması normaldir.

---

## 5. Teslim edilecekler (kontrol listesi)

- [ ] Test için **dokuz sınıfın olasılıkları** ve sınıf tahmini (`2738 × 9` matris + `argmax`). Dosya formatı yarışma şartnamesinden netleştirilecek.
- [ ] **Sınıf bazlı sonuçlar** (precision, recall, F1, confusion matrix)
- [ ] **Kullanıcı senaryosu:** sistemi kim, hangi durumda, nasıl kullanır
- [ ] **Geliştirme fikri**, seçilip ölçülmüş olarak. Adaylar: düşük donanımda hızlı çıkarım · ışık ve bulanıklık dayanıklılığı · belirsiz sonuçları uzmana yönlendirme · internetsiz tarla demosu
- [ ] **Sınırlar bölümü:** tür bilgisi hastalık/ilaç etiketi değil, veri Avustralya'dan (Türkiye'ye genelleme iddiası yok), test dağılımı val'den farklı, val küçük

---

## 6. Sık karşılaşılan sorunlar

| Sorun | Çözüm |
|---|---|
| Eğitim çok yavaş, VRAM dolu | Yerelde Brave ve diğer GPU kullanan süreçleri kapat, ya da Colab'a geç |
| `Widgets require us to download supporting files` (VS Code) | **Cancel**'a bas, zararsız. Kalıcı: `os.environ["HF_HUB_DISABLE_PROGRESS_BARS"] = "1"` (timm importundan önce) |
| `CUDA out of memory` | Batch size'ı 64'ten 32'ye indir |
| Bulunamayan görsel sayısı ≠ 0 | CSV yol sütunu ile `images/` yapısını kontrol et |
| Colab oturumu kopar | Veri/cache yeniden açılır, checkpoint Drive'dan okunur |