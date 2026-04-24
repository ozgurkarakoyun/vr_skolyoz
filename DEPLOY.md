# 🚀 Schroth VR — Railway Deploy Rehberi

## 📁 Proje Yapısı (Tam)

```
schroth-vr/
├── app.py                    # Ana Flask uygulaması
├── schroth_analyzer.py       # Schroth postür analiz motoru
├── scoliosis_engine.py       # Orijinal T/TL/L açı motoru (model_point4.pt)
├── download_models.py        # Startup model indirici
├── requirements.txt          # Python bağımlılıkları (sabit sürümler)
├── Procfile                  # Gunicorn + eventlet başlatıcı
├── railway.toml              # Railway konfigürasyonu
├── .gitignore
├── .github/
│   └── workflows/
│       └── deploy.yml        # Otomatik deploy (GitHub Actions)
├── models/
│   ├── .gitkeep              # Klasörü Git'e dahil et
│   ├── model_point4.pt       # ← Kendi modelinizi buraya koyun
│   └── yolo26n-pose.pt       # ← Otomatik indirilir
└── templates/
    ├── index.html            # Ana sayfa
    ├── phone.html            # Terapist / kamera ekranı
    └── quest.html            # Meta Quest WebXR ekranı
```

---

## 🔧 1. İlk Kurulum (1 Seferlik)

### GitHub Repo Oluştur
```bash
# Proje klasörüne gir
cd schroth-vr

# Git başlat
git init
git add .
git commit -m "feat: Schroth VR v1.0 — ilk deploy"

# GitHub'da yeni repo oluştur (github.com → New repository)
# Repo adı: schroth-vr

git remote add origin https://github.com/KULLANICI_ADINIZ/schroth-vr.git
git push -u origin main
```

### Railway Proje Oluştur
1. [railway.app](https://railway.app) → **New Project**
2. **Deploy from GitHub repo** seçin
3. `schroth-vr` reposunu seçin
4. **Deploy** butonuna tıklayın

---

## 🔑 2. Environment Variables

Railway dashboard → Projeniz → **Variables** sekmesi:

| Değişken | Değer | Açıklama |
|----------|-------|----------|
| `SECRET_KEY` | `guclu-rastgele-sifre-123!` | Flask secret key |
| `MODEL_PATH` | `models/yolo26n-pose.pt` | Pose model yolu |
| `SCOL_MODEL_PATH` | `models/model_point4.pt` | Skolyoz model yolu |
| `PORT` | (Railway otomatik ayarlar) | — |

> ⚠️ `SECRET_KEY` mutlaka değiştirin! En az 32 karakter rastgele bir değer kullanın.

---

## 📦 3. model_point4.pt Nasıl Yüklenir?

Model dosyaları `.gitignore`'da olduğu için Git'e push edilmez. İki seçenek:

### Seçenek A — Railway Volume (Önerilen)
```
Railway Dashboard → Projeniz → Settings → Volumes
→ "Add Volume" → Mount path: /app/models
```
Sonra Railway'in terminal özelliği ile modeli yükleyin:
```bash
# Railway CLI ile
railway run bash
# Terminal açıldığında:
# /app/models/ klasörüne model dosyasını kopyalayın
```

### Seçenek B — Environment Variable ile URL
`app.py` ve `scoliosis_engine.py` dosyalarında model URL'den indirme eklenebilir.
Modeli Google Drive veya Dropbox'a yükleyin, `SCOL_MODEL_URL` env var'ı ile çekin.

### Seçenek C — Model Olmadan Çalış
`model_point4.pt` yoksa sistem otomatik olarak **pose keypoint tahminini** kullanır.
Arayüzde `(tahmin)` etiketi görünür. T/TL/L açıları yaklaşık değer verir.

---

## 🔒 4. HTTPS Konfigürasyonu

### Railway Otomatik HTTPS
Railway deploy edilen her proje için otomatik HTTPS sertifikası oluşturur:
```
https://schroth-vr-production-XXXX.railway.app
```

> ✅ Meta Quest tarayıcısı ve telefon kamerasına erişim için HTTPS **zorunludur**.
> Railway bunu otomatik sağlar — ekstra ayar gerekmez.

### Özel Domain (İsteğe Bağlı)
```
Railway Dashboard → Settings → Networking → Custom Domain
→ schroth.klinigim.com gibi bir subdomain ekleyin
→ DNS'de CNAME kaydı oluşturun
```

---

## 🌐 5. WebXR + HTTPS Neden Gerekli?

| Özellik | HTTP | HTTPS |
|---------|------|-------|
| Kamera erişimi | ❌ Engellenir | ✅ Çalışır |
| WebXR immersive-vr | ❌ Engellenir | ✅ Çalışır |
| Socket.IO WSS | ❌ Güvensiz | ✅ Güvenli |
| Meta Quest tarayıcı | ❌ Çoğu özellik bloklu | ✅ Tam destek |

---

## 📱 6. Kullanım Kılavuzu

### Adım 1 — Ana Sayfayı Açın
```
https://sizin-railway-url.railway.app
```
"Yeni Seans" → 6 haneli ID oluşur (örn: `KA7XP2`)

### Adım 2 — Telefonu Hazırlayın
- Aynı URL'yi terapistin telefonunda açın
- "Telefon Modunu Aç" → Seans ID otomatik eşleşir
- "🎥 Başlat" → Arka kamera açılır

### Adım 3 — Meta Quest'i Hazırlayın
- Quest tarayıcısında `https://sizin-url.railway.app/quest?session=KA7XP2`
- "VR Modunu Başlat" → Stereo görüntüye geçer

### Adım 4 — Hasta Pozisyonlandırma
```
Hasta                    Telefon
sırt kameraya dönük  ←→  1-2 metre mesafe
                          hafif yukarıdan açı
```

---

## 🔄 7. Güncelleme / Yeniden Deploy

```bash
# Değişiklik yaptıktan sonra
git add .
git commit -m "fix: açıklama"
git push origin main
# → Railway otomatik yeniden deploy eder (~2 dakika)
```

---

## 🐛 8. Sık Karşılaşılan Sorunlar

### "Kamera erişimi sağlanamadı"
→ HTTPS üzerinden açtığınızdan emin olun (`https://` ile başlamalı)

### "immersive-vr desteklenmiyor"
→ Meta Quest tarayıcısında açtığınızdan emin olun
→ Quest Settings → Browser → WebXR aktif olmalı

### Socket bağlantısı kesiliyor
→ Railway ücretsiz planda uyku modu olabilir
→ `/health` endpoint'ine düzenli ping atın (UptimeRobot ile ücretsiz yapılabilir)

### Model bulunamadı hatası
→ `SCOL_MODEL_PATH` env var'ını kontrol edin
→ Model yok ise pose tahmin moduna geçer (normal davranış)

### Build hatası (libGL eksik)
→ `railway.toml` içindeki nixPkgs listesi `libGL` içermeli (zaten ekli)

---

## 📊 9. Performans Notları

| Bileşen | Gecikme |
|---------|---------|
| Telefon → Backend (frame) | ~50-100ms |
| YOLO Pose analizi (CPU) | ~100-300ms |
| model_point4.pt analizi (CPU) | ~80-200ms |
| Backend → Quest (analiz) | ~20-50ms |
| **Toplam döngü** | **~250-650ms** |

> Railway ücretsiz plan CPU limitli olduğundan 8 FPS önerilir.
> Ücretli plan ile 15+ FPS mümkündür.

---

## 🔗 Önemli Linkler

- Railway Dashboard: https://railway.app
- Railway Docs: https://docs.railway.app
- WebXR Emulator (Chrome): https://chrome.google.com/webstore/detail/webxr-api-emulator
- Meta Quest Browser WebXR: Settings → Experimental Features → WebXR aktivasyonu
