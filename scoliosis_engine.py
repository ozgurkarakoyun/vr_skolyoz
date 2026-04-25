"""
scoliosis_engine.py
────────────────────────────────────────────────────────────────
Orijinal Skolyoz Analiz Motorundan (model_point4.pt) Alınan Yöntem

Orijinal projedeki algoritma:
  - YOLO model (model_point4.pt) → 4 bounding-box noktası tespit
  - Noktaları Y eksenine göre sırala (yuktan aşağı)
  - Ardışık noktalar arası eğimleri hesapla (calc_slope)
  - Eğimler arası açıları hesapla (calc_angle)
  - T (Thoracic), TL (Thoracolumbar), L (Lumbar) açıları
  - Renk eşikleri: NORMAL / ORTA / KRİTİK

Bu dosya:
  - Orijinal algoritmanın birebir kopyası (api.py'den)
  - Schroth VR projesine entegre edilmiş hali
  - Hem model_point4.pt ile hem de pose model keypoints ile çalışır
"""

import math
import cv2
import numpy as np
import base64
import logging
import os

logger = logging.getLogger(__name__)

# ─── Orijinal renk ve eşik değerleri (api.py'den) ───────────
COLOR_GREEN  = (0, 200, 0)
COLOR_BLUE   = (255, 140, 0)
COLOR_RED    = (0, 0, 220)
COLOR_WHITE  = (255, 255, 255)
COLOR_BLACK  = (0, 0, 0)

ANGLE_HIGH = 40.0   # KRİTİK eşiği
ANGLE_MID  = 20.0   # ORTA eşiği
REQUIRED_POINTS = 4 # model_point4.pt → 4 nokta

# ─── YOLO Model ──────────────────────────────────────────────
_scol_model = None

def get_scoliosis_model():
    global _scol_model
    if _scol_model is None:
        try:
            import torch
            from ultralytics import YOLO

            # PyTorch 2.6+ patch
            _orig_load = torch.load
            torch.load = lambda *a, **kw: _orig_load(
                *a, **{**kw, 'weights_only': False}
            )

            model_path = os.environ.get('SCOL_MODEL_PATH', 'models/model_point4.pt')
            if os.path.exists(model_path):
                _scol_model = YOLO(model_path)
                logger.info(f"Scoliosis model loaded: {model_path}")
            else:
                logger.warning(f"model_point4.pt not found at {model_path}")

            torch.load = _orig_load  # geri yükle
        except Exception as e:
            logger.error(f"Scoliosis model error: {e}")
    return _scol_model

# ─── Orijinal Hesaplama Fonksiyonları (api.py'den birebir) ───

def calc_slope(p1, p2):
    """İki nokta arası eğim — orijinal api.py'den"""
    dx = p2[0] - p1[0]
    return (p2[1] - p1[1]) / dx if dx != 0 else float('inf')

def calc_angle(s1, s2):
    """İki eğim arası açı (derece) — orijinal api.py'den"""
    denom = 1 + s1 * s2
    if denom == 0:
        return 90.0
    return math.degrees(math.atan(abs((s2 - s1) / denom)))

def angle_color(angle):
    """Açı şiddeti rengi — orijinal api.py'den"""
    if angle > ANGLE_HIGH:
        return COLOR_RED
    elif angle > ANGLE_MID:
        return COLOR_BLUE
    return COLOR_GREEN

def angle_label(angle):
    """Açı şiddeti etiketi — orijinal api.py'den"""
    if angle > ANGLE_HIGH:
        return "KRİTİK"
    elif angle > ANGLE_MID:
        return "ORTA"
    return "NORMAL"

def draw_point(image, x, y, color, radius=10):
    """Nokta çizimi — orijinal api.py'den"""
    cv2.circle(image, (x, y), radius, COLOR_WHITE, -1)
    cv2.circle(image, (x, y), radius, COLOR_BLACK, 2)
    cv2.circle(image, (x, y), radius - 3, color, -1)

def draw_angle_label(image, text, x, y, color, font_scale=1.4):
    """Açı etiketi çizimi — orijinal api.py'den"""
    thickness = 2
    (tw, th), baseline = cv2.getTextSize(
        text, cv2.FONT_HERSHEY_SIMPLEX, font_scale, thickness)
    pad = 6
    cv2.rectangle(image,
        (x - pad, y - th - pad),
        (x + tw + pad, y + baseline + pad),
        COLOR_BLACK, -1)
    cv2.putText(image, text, (x, y),
        cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

def draw_summary(image, angles):
    """Sol üst köşe özet kutusu — orijinal api.py'den"""
    lines = ["=== SKOL. ACILARI ==="]
    for name, angle in zip(["T (Thoracic)", "TL", "L (Lumbar)"], angles):
        lines.append(f"{name}: {angle:.1f} [{angle_label(angle)}]")

    font_scale, thickness, padding, line_height = 0.8, 2, 12, 36
    box_w = 320
    box_h = padding * 2 + line_height * len(lines)
    overlay = image.copy()
    cv2.rectangle(overlay, (10, 10), (10 + box_w, 10 + box_h), (30, 30, 30), -1)
    cv2.addWeighted(overlay, 0.6, image, 0.4, 0, image)

    for i, line in enumerate(lines):
        y = 10 + padding + line_height * (i + 1) - 5
        color = COLOR_WHITE if i == 0 else angle_color(angles[i - 1])
        cv2.putText(image, line, (18, y),
            cv2.FONT_HERSHEY_SIMPLEX, font_scale, color, thickness, cv2.LINE_AA)

# ─── Ana Analiz Fonksiyonu ───────────────────────────────────

def analyze_scoliosis_frame(frame: np.ndarray) -> dict | None:
    """
    model_point4.pt ile skolyoz analizi yap.
    Orijinal /analyze endpoint mantığının birebir kopyası.

    Returns:
        {
          angles: {thoracic, thoracolumbar, lumbar},
          labels: {thoracic, thoracolumbar, lumbar},
          points: [[x,y], ...],
          result_image_b64: str,  ← çizilmiş görüntü (base64)
          severity: str,          ← en kötü açı etiketi
          max_angle: float,
        }
    """
    model = get_scoliosis_model()
    if model is None:
        return None

    try:
        # Geçici dosyaya kaydet (YOLO dosya yolu ister)
        tmp_path = "/tmp/scol_input.jpg"
        cv2.imwrite(tmp_path, frame)

        results = model(tmp_path, verbose=False)
        all_boxes = []
        for r in results:
            boxes_np = r.boxes.xywh.cpu().numpy()
            all_boxes.extend(boxes_np.tolist())

        if len(all_boxes) < REQUIRED_POINTS:
            logger.debug(f"Yeterli nokta yok: {len(all_boxes)}/{REQUIRED_POINTS}")
            return None

        # Noktaları Y eksenine göre sırala (yuktan aşağı)
        sorted_boxes = sorted(all_boxes[:REQUIRED_POINTS], key=lambda b: b[1])
        points = [(int(b[0]), int(b[1])) for b in sorted_boxes]

        # Eğim ve açı hesapla — orijinal api.py mantığı
        slopes = [calc_slope(points[i], points[i + 1])
                  for i in range(REQUIRED_POINTS - 1)]
        angles = [
            calc_angle(slopes[0], slopes[1]),
            calc_angle(slopes[1], slopes[2]),
            calc_angle(slopes[2], slopes[3]) if len(slopes) > 2 else 0.0,
        ]

        # Görüntüyü çiz — orijinal api.py çizim kodu
        image = frame.copy()
        segment_colors = [
            angle_color(angles[0]),
            angle_color(angles[0]),
            angle_color(angles[2]),
            angle_color(angles[2]),
        ]
        for i in range(REQUIRED_POINTS - 1):
            cv2.line(image, points[i], points[i + 1],
                     segment_colors[i], 5, cv2.LINE_AA)
        for i, (px, py) in enumerate(points):
            pt_color = segment_colors[i] if i < len(segment_colors) else segment_colors[-1]
            draw_point(image, px, py, pt_color)
        for i, (angle, lbl) in enumerate(zip(angles, ["T", "TL", "L"])):
            px, py = points[i + 1]
            text = f"{lbl}: {angle:.1f}"
            color = angle_color(angle)
            offset_y = -25 if py > 50 else 40
            draw_angle_label(image, text, px + 12, py + offset_y, color)
        draw_summary(image, angles)

        # Base64 encode
        _, buffer = cv2.imencode('.jpg', image)
        img_b64 = base64.b64encode(buffer).decode('utf-8')

        max_angle = max(angles)
        return {
            'angles': {
                'thoracic':      round(angles[0], 2),
                'thoracolumbar': round(angles[1], 2),
                'lumbar':        round(angles[2], 2),
            },
            'labels': {
                'thoracic':      angle_label(angles[0]),
                'thoracolumbar': angle_label(angles[1]),
                'lumbar':        angle_label(angles[2]),
            },
            'points': points,
            'result_image_b64': img_b64,
            'severity': angle_label(max_angle),
            'max_angle': round(max_angle, 1),
        }

    except Exception as e:
        logger.error(f"Scoliosis analysis error: {e}")
        return None
    finally:
        if os.path.exists('/tmp/scol_input.jpg'):
            os.remove('/tmp/scol_input.jpg')


# ─── Pose Keypoint'lerden Scoliosis Tahmini ─────────────────
def estimate_from_pose_keypoints(keypoints: np.ndarray) -> dict | None:
    """
    model_point4.pt yoksa YOLO pose keypoints'lerden
    T/TL/L açılarını tahmin eder.

    COCO keypoints kullanılarak 4 omurga noktası oluşturulur:
      P1 = baş merkezi (kulaklar arası)
      P2 = omuz merkezi
      P3 = kalça merkezi
      P4 = diz merkezi
    """
    try:
        def get_kp(idx):
            pt = keypoints[idx]
            return (float(pt[0]), float(pt[1])) if float(pt[2]) > 0.3 else None

        left_ear   = get_kp(3)
        right_ear  = get_kp(4)
        l_shoulder = get_kp(5)
        r_shoulder = get_kp(6)
        l_hip      = get_kp(11)
        r_hip      = get_kp(12)
        l_knee     = get_kp(13)
        r_knee     = get_kp(14)

        # 4 proxy nokta oluştur
        def midpoint(a, b):
            if a and b:
                return ((a[0]+b[0])/2, (a[1]+b[1])/2)
            return a or b

        p1 = midpoint(left_ear, right_ear)
        p2 = midpoint(l_shoulder, r_shoulder)
        p3 = midpoint(l_hip, r_hip)
        p4 = midpoint(l_knee, r_knee)

        if not all([p1, p2, p3, p4]):
            return None

        points = [
            (int(p1[0]), int(p1[1])),
            (int(p2[0]), int(p2[1])),
            (int(p3[0]), int(p3[1])),
            (int(p4[0]), int(p4[1])),
        ]
        # Y'ye göre sırala
        points = sorted(points, key=lambda p: p[1])

        slopes = [calc_slope(points[i], points[i+1]) for i in range(3)]
        angles = [
            calc_angle(slopes[0], slopes[1]),
            calc_angle(slopes[1], slopes[2]),
            0.0,  # 4. segment yok
        ]

        max_angle = max(angles[:2])
        return {
            'angles': {
                'thoracic':      round(angles[0], 2),
                'thoracolumbar': round(angles[1], 2),
                'lumbar':        round(0.0, 2),
            },
            'labels': {
                'thoracic':      angle_label(angles[0]),
                'thoracolumbar': angle_label(angles[1]),
                'lumbar':        'NORMAL',
            },
            'points': points,
            'result_image_b64': None,
            'severity': angle_label(max_angle),
            'max_angle': round(max_angle, 1),
            'estimated': True,  # gerçek model değil, tahmin
        }
    except Exception as e:
        logger.error(f"Pose estimation error: {e}")
        return None
