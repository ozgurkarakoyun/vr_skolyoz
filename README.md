# Schroth VR — Kurulum & Dağıtım Rehberi

## Proje Yapısı
```
schroth-vr/
├── app.py              # Flask backend + AI analiz
├── requirements.txt
├── Procfile
├── railway.toml
├── templates/
│   ├── index.html      # Ana sayfa (cihaz seçim)
│   ├── phone.html      # Telefon kamera arayüzü
│   └── quest.html      # Meta Quest WebXR arayüzü
└── models/
    └── pose_model.pt   # YOLOv8 pose modeliniz (opsiyonel)
```

## Railway Dağıtımı

### 1. GitHub'a push edin
```bash
git init
git add .
git commit -m "Initial Schroth VR"
git remote add origin https://github.com/KULLANICI/schroth-vr.git
git push -u origin main
```

### 2. Railway'de yeni proje açın
- railway.app → New Project → Deploy from GitHub
- Bu repo'yu seçin

### 3. Environment Variables (Railway dashboard)
```
SECRET_KEY=guclu-bir-sifre-yazin
MODEL_PATH=models/pose_model.pt   # model varsa
```

### 4. YOLO Modeli (opsiyonel)
- Kendi modelinizi `models/pose_model.pt` olarak ekleyin
- Model yoksa otomatik olarak `yolov8n-pose.pt` indirilir

## Kullanım

### Adım 1 — Ana Sayfayı Açın
```
https://sizin-railway-url.railway.app
```
- "Yeni Seans" butonuna tıklayın → 6 haneli ID oluşur

### Adım 2 — Telefonu Açın
- Aynı URL'yi telefonda açın → "Telefon Modunu Aç"
- Seans ID otomatik eşleşir

### Adım 3 — Quest'i Açın
- Meta Quest tarayıcısında → "Quest Modunu Aç"
- "VR Modunu Başlat" → Stereo görüntüye geçer

## Sistem Gereksinimleri

| Bileşen | Gereksinim |
|---------|-----------|
| Telefon | Android/iOS, arka kamera |
| VR Gözlük | Meta Quest 2/3/Pro |
| Ağ | Her iki cihaz aynı ağda olmalı (Railway üzerinden çalışır) |
| Model | YOLOv8n-pose (otomatik) veya özel model |

## Teknik Notlar

- **WebRTC değil Socket.IO frame relay**: Quest'in tarayıcısında WebRTC P2P bağlantı sorunlu olabilir. Bu nedenle telefon frame'leri backend'e gönderir, backend analiz eder ve sonuçları Socket.IO ile Quest'e iletir.
- **FPS**: 8 FPS ile gönderim yapılır (AI analiz için yeterli)
- **AI Model**: `get_model()` lazy loading ile çalışır, ilk istekte yüklenir

## Geliştirme (Lokal)
```bash
pip install -r requirements.txt
python app.py
# → http://localhost:5000
```
