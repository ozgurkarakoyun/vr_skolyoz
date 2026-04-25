FROM python:3.11-slim

# libGL, libstdc++ ve diğer sistem kütüphaneleri
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgl1 \
    libglib2.0-0 \
    libstdc++6 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY requirements.txt constraints.txt ./

# 1. Headless opencv'yi önce kur
RUN pip install --no-cache-dir opencv-python-headless==4.9.0.80

# 2. Diğer bağımlılıkları kur (ultralytics hariç)
RUN pip install --no-cache-dir -r requirements.txt

# 3. ultralytics'i --no-deps ile kur (opencv-python çekmesin)
RUN pip install --no-cache-dir ultralytics==8.2.18 --no-deps

# 4. Full opencv geldiyse temizle
RUN pip uninstall -y opencv-python || true

# 5. Doğrula
RUN python -c "import cv2; print('[OK] cv2', cv2.__version__)"
RUN python -c "from ultralytics import YOLO; print('[OK] ultralytics OK')"

COPY . .

EXPOSE 8080

CMD ["sh", "-c", "gunicorn --worker-class eventlet -w 1 --bind 0.0.0.0:$PORT --timeout 120 app:app"]
