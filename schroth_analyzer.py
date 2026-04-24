"""
schroth_analyzer.py
─────────────────────────────────────────────────────────────────
Gelişmiş Schroth Yöntemi Analiz Motoru

Özellikler:
  - 17 COCO keypoint tam analizi
  - Trunk Inclination (gövde eğim açısı)
  - Cobb açısı tahmini (2D proxy)
  - Schroth eğri sınıflaması (3c / 3cp / 4c / 4cp)
  - RAB (Rotational Angular Breathing) tarafı tespiti
  - Egzersiz aşaması yönetimi (Elongation → Derotation → Stabilizasyon)
  - Temporal smoothing (titreme azaltma)
  - Seans istatistikleri
"""

import numpy as np
from collections import deque
from dataclasses import dataclass, field, asdict
from typing import Optional, List, Dict
import time

# ─── COCO Keypoint indeksleri ────────────────────────────────
KP = {
    'nose': 0,
    'left_eye': 1, 'right_eye': 2,
    'left_ear': 3, 'right_ear': 4,
    'left_shoulder': 5, 'right_shoulder': 6,
    'left_elbow': 7, 'right_elbow': 8,
    'left_wrist': 9, 'right_wrist': 10,
    'left_hip': 11, 'right_hip': 12,
    'left_knee': 13, 'right_knee': 14,
    'left_ankle': 15, 'right_ankle': 16,
}

CONF_THRESHOLD = 0.4  # Minimum keypoint güven skoru

# ─── Schroth Egzersiz Fazları ────────────────────────────────
PHASES = [
    {
        'id': 'elongation',
        'name': 'Elongasyon',
        'name_tr': '1. Faz — Boyun Uzatma',
        'duration': 30,
        'instruction_tr': 'Başınızı tavana doğru uzatın. Omurgayı dik tutun.',
        'breath_tr': 'Normal nefes alın, boyunuzu hissedin',
        'color': '#00e5ff',
    },
    {
        'id': 'derotation',
        'name': 'Derotasyon',
        'name_tr': '2. Faz — Derotasyon',
        'duration': 45,
        'instruction_tr': 'Konkav tarafı açın, konveks tarafı bastırın.',
        'breath_tr': 'Konkav tarafa doğru RAB nefesi alın',
        'color': '#7b2fff',
    },
    {
        'id': 'rab_breathing',
        'name': 'RAB Nefes',
        'name_tr': '3. Faz — RAB Nefes',
        'duration': 60,
        'instruction_tr': 'Çökmüş tarafı nefesle doldurun. 4 sayı al, 6 sayı ver.',
        'breath_tr': 'Derin — Tut — Yavaş ver',
        'color': '#2ed573',
    },
    {
        'id': 'stabilization',
        'name': 'Stabilizasyon',
        'name_tr': '4. Faz — İzometrik Stabilizasyon',
        'duration': 30,
        'instruction_tr': 'Düzeltilmiş pozisyonu 10 saniye tutun. Kasları sıkın.',
        'breath_tr': 'Nefesi tutun — Kas kontraksiyonu',
        'color': '#ffa502',
    },
]

# ─── Veri Sınıfları ──────────────────────────────────────────

@dataclass
class KeypointSet:
    left_shoulder: Optional[np.ndarray] = None
    right_shoulder: Optional[np.ndarray] = None
    left_hip: Optional[np.ndarray] = None
    right_hip: Optional[np.ndarray] = None
    left_knee: Optional[np.ndarray] = None
    right_knee: Optional[np.ndarray] = None
    left_ear: Optional[np.ndarray] = None
    right_ear: Optional[np.ndarray] = None
    nose: Optional[np.ndarray] = None

@dataclass
class PostureMetrics:
    # Açılar (derece)
    shoulder_angle: float = 0.0       # Omuz eğimi (+ = sol yüksek)
    hip_angle: float = 0.0            # Kalça eğimi (+ = sol yüksek)
    trunk_inclination: float = 0.0    # Gövde eğimi (lateral)
    cobb_proxy: float = 0.0           # Tahmini Cobb açısı (2D proxy)
    pelvic_tilt: float = 0.0          # Pelvik tilt

    # Pozisyonlar
    lateral_shift_px: float = 0.0     # Gövde lateral kayması (px)
    lateral_shift_pct: float = 0.0    # Gövde genişliğine göre %

    # Sınıflama
    curve_type: str = '3c'
    rab_side: str = 'right'           # RAB nefes tarafı (konkav)
    dominant_curve: str = 'thoracic'  # Baskın eğri bölgesi

    # Skor
    score: float = 0.0
    score_components: Dict = field(default_factory=dict)

    # Talimatlar
    instructions: List[str] = field(default_factory=list)
    phase_instruction: str = ''
    breath_instruction: str = ''

    # Normalleştirme referansı
    body_width_px: float = 1.0


@dataclass
class SessionStats:
    start_time: float = field(default_factory=time.time)
    frame_count: int = 0
    valid_detections: int = 0
    scores: List[float] = field(default_factory=list)
    best_score: float = 0.0
    current_phase_idx: int = 0
    phase_start_time: float = field(default_factory=time.time)
    rep_count: int = 0


# ─── Ana Analiz Sınıfı ───────────────────────────────────────

class SchrothAnalyzer:
    def __init__(self, smoothing_window: int = 5):
        self.smoothing_window = smoothing_window
        self._history: Dict[str, deque] = {
            'shoulder_angle': deque(maxlen=smoothing_window),
            'hip_angle': deque(maxlen=smoothing_window),
            'trunk_inclination': deque(maxlen=smoothing_window),
            'lateral_shift_px': deque(maxlen=smoothing_window),
            'score': deque(maxlen=smoothing_window),
        }
        self.session = SessionStats()
        self._last_valid: Optional[PostureMetrics] = None

    # ─── Public API ──────────────────────────────────────────

    def analyze(self, keypoints_raw: np.ndarray, frame_width: int = 640, frame_height: int = 480) -> Optional[dict]:
        """
        Ana analiz fonksiyonu.
        keypoints_raw: shape (17, 3) — [x, y, conf]
        """
        self.session.frame_count += 1

        kps = self._extract_keypoints(keypoints_raw)
        if not self._validate_keypoints(kps):
            # Geçersiz frame — son geçerli sonucu döndür
            return self._last_valid_dict()

        self.session.valid_detections += 1

        metrics = PostureMetrics()
        metrics.body_width_px = max(
            abs(float(kps.left_shoulder[0]) - float(kps.right_shoulder[0])), 1
        )

        self._compute_angles(kps, metrics, frame_width)
        self._smooth_metrics(metrics)
        self._classify_curve(metrics)
        self._compute_score(metrics)
        self._generate_instructions(metrics, kps)
        self._inject_phase_info(metrics)

        # Seans istatistikleri
        self.session.scores.append(metrics.score)
        if metrics.score > self.session.best_score:
            self.session.best_score = metrics.score
        self._update_phase()

        result = self._metrics_to_dict(metrics, kps)
        self._last_valid = result
        return result

    def get_session_summary(self) -> dict:
        s = self.session
        elapsed = time.time() - s.start_time
        avg_score = np.mean(s.scores) if s.scores else 0
        detection_rate = s.valid_detections / max(s.frame_count, 1) * 100
        return {
            'duration_sec': round(elapsed),
            'frame_count': s.frame_count,
            'detection_rate_pct': round(detection_rate, 1),
            'avg_score': round(avg_score, 1),
            'best_score': round(s.best_score, 1),
            'rep_count': s.rep_count,
            'current_phase': PHASES[s.current_phase_idx]['name_tr'],
        }

    def reset_session(self):
        self.session = SessionStats()
        for key in self._history:
            self._history[key].clear()
        self._last_valid = None

    # ─── Keypoint Extraction ─────────────────────────────────

    def _extract_keypoints(self, kps_raw: np.ndarray) -> KeypointSet:
        def get(idx: int) -> Optional[np.ndarray]:
            if idx >= len(kps_raw):
                return None
            pt = kps_raw[idx]
            if float(pt[2]) < CONF_THRESHOLD:
                return None
            return pt

        return KeypointSet(
            left_shoulder=get(KP['left_shoulder']),
            right_shoulder=get(KP['right_shoulder']),
            left_hip=get(KP['left_hip']),
            right_hip=get(KP['right_hip']),
            left_knee=get(KP['left_knee']),
            right_knee=get(KP['right_knee']),
            left_ear=get(KP['left_ear']),
            right_ear=get(KP['right_ear']),
            nose=get(KP['nose']),
        )

    def _validate_keypoints(self, kps: KeypointSet) -> bool:
        """En az omuz + kalça noktaları olmalı"""
        required = [kps.left_shoulder, kps.right_shoulder,
                    kps.left_hip, kps.right_hip]
        return all(k is not None for k in required)

    # ─── Açı Hesaplamaları ───────────────────────────────────

    def _compute_angles(self, kps: KeypointSet, m: PostureMetrics, frame_width: int):
        ls = kps.left_shoulder
        rs = kps.right_shoulder
        lh = kps.left_hip
        rh = kps.right_hip

        # Omuz eğimi (y ekseninde fark)
        s_dy = float(ls[1]) - float(rs[1])
        s_dx = abs(float(rs[0]) - float(ls[0])) or 1
        m.shoulder_angle = float(np.degrees(np.arctan2(s_dy, s_dx)))

        # Kalça eğimi
        h_dy = float(lh[1]) - float(rh[1])
        h_dx = abs(float(rh[0]) - float(lh[0])) or 1
        m.hip_angle = float(np.degrees(np.arctan2(h_dy, h_dx)))

        # Pelvik tilt (kalça ile omuz farkı)
        m.pelvic_tilt = m.shoulder_angle - m.hip_angle

        # Gövde lateral kayması
        sc_x = (float(ls[0]) + float(rs[0])) / 2  # omuz merkezi
        hc_x = (float(lh[0]) + float(rh[0])) / 2  # kalça merkezi
        m.lateral_shift_px = sc_x - hc_x
        m.lateral_shift_pct = (m.lateral_shift_px / frame_width) * 100

        # Trunk inclination (omuz–kalça açısı, lateral eğim)
        sc_y = (float(ls[1]) + float(rs[1])) / 2
        hc_y = (float(lh[1]) + float(rh[1])) / 2
        trunk_dy = sc_y - hc_y or 1
        trunk_dx = sc_x - hc_x
        m.trunk_inclination = float(np.degrees(np.arctan2(trunk_dx, trunk_dy)))

        # Cobb açısı 2D proxy
        # Omuz eğimi + kalça eğimi toplamının mutlak değeri
        # (gerçek Cobb değil, rölatif gösterge)
        m.cobb_proxy = abs(m.shoulder_angle) + abs(m.hip_angle) * 0.7

        # Dirsek ve diz bilgileri varsa ek analiz
        if kps.left_knee and kps.right_knee:
            lk = kps.left_knee
            rk = kps.right_knee
            k_dy = float(lk[1]) - float(rk[1])
            k_dx = abs(float(rk[0]) - float(lk[0])) or 1
            knee_angle = float(np.degrees(np.arctan2(k_dy, k_dx)))
            # Diz eğimi pelvik tilti destekliyorsa notla
            if abs(knee_angle) > 2:
                m.pelvic_tilt = (m.pelvic_tilt + knee_angle) / 2

    # ─── Temporal Smoothing ──────────────────────────────────

    def _smooth_metrics(self, m: PostureMetrics):
        """Son N frame ortalamasıyla titreşimi azalt"""
        fields = ['shoulder_angle', 'hip_angle', 'trunk_inclination', 'lateral_shift_px']
        for f in fields:
            self._history[f].append(getattr(m, f))
            smoothed = float(np.mean(self._history[f]))
            setattr(m, f, smoothed)

    # ─── Schroth Eğri Sınıflaması ────────────────────────────

    def _classify_curve(self, m: PostureMetrics):
        """
        Klinik Schroth sınıflaması:
        3c  — 3 eğrili, kompansatör lumbar
        3cp — 3 eğrili + pelvik kayma
        4c  — 4 eğrili (double major)
        4cp — 4 eğrili + pelvik kayma
        """
        sa = abs(m.shoulder_angle)
        ha = abs(m.hip_angle)
        ls = abs(m.lateral_shift_px)
        pelvik_shift = ls > 25  # piksel eşiği

        # Dominant eğri yönü
        if m.shoulder_angle > 2:
            m.dominant_curve = 'right_thoracic'
            m.rab_side = 'left'   # konkav = sol
        elif m.shoulder_angle < -2:
            m.dominant_curve = 'left_thoracic'
            m.rab_side = 'right'  # konkav = sağ
        else:
            m.dominant_curve = 'balanced'
            m.rab_side = 'both'

        # 4 eğri: hem omuz hem kalça belirgin eğimliyse
        if sa > 4 and ha > 3:
            m.curve_type = '4cp' if pelvik_shift else '4c'
        elif sa > 4 and ha <= 3:
            m.curve_type = '3cp' if pelvik_shift else '3c'
        elif sa <= 2 and ha <= 2:
            m.curve_type = '3c'  # minimal / normal
        else:
            m.curve_type = '3c'

    # ─── Skor Hesaplama ──────────────────────────────────────

    def _compute_score(self, m: PostureMetrics):
        """
        Weighted postür skoru (0–100)
        Bileşenler:
          40 puan — omuz simetrisi
          30 puan — kalça simetrisi
          20 puan — lateral kayma
          10 puan — trunk inclination
        """
        # Omuz
        s_pen = min(abs(m.shoulder_angle) * 4, 40)
        s_score = 40 - s_pen

        # Kalça
        h_pen = min(abs(m.hip_angle) * 4, 30)
        h_score = 30 - h_pen

        # Lateral kayma (frame genişliğine göre normalize)
        lat_pen = min(abs(m.lateral_shift_pct) * 1.5, 20)
        lat_score = 20 - lat_pen

        # Trunk inclination
        t_pen = min(abs(m.trunk_inclination) * 1.5, 10)
        t_score = 10 - t_pen

        total = max(0.0, s_score + h_score + lat_score + t_score)

        # Smoothing
        self._history['score'].append(total)
        m.score = float(np.mean(self._history['score']))

        m.score_components = {
            'shoulder': round(s_score, 1),
            'hip': round(h_score, 1),
            'lateral': round(lat_score, 1),
            'trunk': round(t_score, 1),
        }

    # ─── Talimat Üretimi ─────────────────────────────────────

    def _generate_instructions(self, m: PostureMetrics, kps: KeypointSet):
        instructions = []
        sa = m.shoulder_angle
        ha = m.hip_angle
        ls = m.lateral_shift_px
        score = m.score

        # --- Omuz düzeltmesi ---
        if sa > 5:
            instructions.append("⬇️ Sol omzunuzu aşağı indirin")
        elif sa > 2:
            instructions.append("↙️ Sol omzunuzu hafifçe indirin")
        elif sa < -5:
            instructions.append("⬇️ Sağ omzunuzu aşağı indirin")
        elif sa < -2:
            instructions.append("↘️ Sağ omzunuzu hafifçe indirin")

        # --- Kalça / pelvis ---
        if ha > 4:
            instructions.append("↔️ Kalçanızı sağa kaydırın")
        elif ha < -4:
            instructions.append("↔️ Kalçanızı sola kaydırın")

        # --- Lateral gövde kayması ---
        if ls > 30:
            instructions.append("⬅️ Gövdenizi sola uzatın")
        elif ls < -30:
            instructions.append("➡️ Gövdenizi sağa uzatın")
        elif 15 < ls <= 30:
            instructions.append("↙️ Hafifçe sola doğru uzanın")
        elif -30 <= ls < -15:
            instructions.append("↘️ Hafifçe sağa doğru uzanın")

        # --- Mükemmel pozisyon ---
        if score >= 85 and not instructions:
            instructions.append("✅ Mükemmel! Bu pozisyonu koruyun")
        elif score >= 70 and not instructions:
            instructions.append("👍 İyi pozisyon — devam edin")

        # --- RAB nefes ---
        rab_side_tr = {
            'left': 'sol', 'right': 'sağ', 'both': 'her iki'
        }.get(m.rab_side, 'konkav')
        instructions.append(f"🫁 {rab_side_tr.capitalize()} tarafa RAB nefesi alın")

        # --- Elongasyon hatırlatıcı ---
        instructions.append("⬆️ Tepeden çekiliyormuş gibi omurgayı uzatın")

        m.instructions = instructions

    # ─── Egzersiz Fazı Yönetimi ──────────────────────────────

    def _inject_phase_info(self, m: PostureMetrics):
        phase = PHASES[self.session.current_phase_idx]
        m.phase_instruction = phase['instruction_tr']
        m.breath_instruction = phase['breath_tr']

    def _update_phase(self):
        phase = PHASES[self.session.current_phase_idx]
        elapsed = time.time() - self.session.phase_start_time
        if elapsed >= phase['duration']:
            next_idx = (self.session.current_phase_idx + 1) % len(PHASES)
            if next_idx == 0:
                self.session.rep_count += 1
            self.session.current_phase_idx = next_idx
            self.session.phase_start_time = time.time()

    def get_current_phase(self) -> dict:
        phase = PHASES[self.session.current_phase_idx]
        elapsed = time.time() - self.session.phase_start_time
        remaining = max(0, phase['duration'] - elapsed)
        progress = min(1.0, elapsed / phase['duration'])
        return {
            **phase,
            'elapsed_sec': round(elapsed),
            'remaining_sec': round(remaining),
            'progress': round(progress, 2),
        }

    # ─── Çıktı Formatı ───────────────────────────────────────

    def _metrics_to_dict(self, m: PostureMetrics, kps: KeypointSet) -> dict:
        def kp_coords(pt):
            if pt is None:
                return None
            return [round(float(pt[0]), 1), round(float(pt[1]), 1), round(float(pt[2]), 3)]

        return {
            # Açılar
            'shoulder_angle': round(m.shoulder_angle, 1),
            'hip_angle': round(m.hip_angle, 1),
            'trunk_inclination': round(m.trunk_inclination, 1),
            'cobb_proxy': round(m.cobb_proxy, 1),
            'pelvic_tilt': round(m.pelvic_tilt, 1),
            'lateral_shift_px': round(m.lateral_shift_px, 1),
            'lateral_shift_pct': round(m.lateral_shift_pct, 1),

            # Sınıflama
            'curve_type': m.curve_type,
            'rab_side': m.rab_side,
            'dominant_curve': m.dominant_curve,

            # Skor
            'score': round(m.score),
            'score_components': m.score_components,

            # Talimatlar
            'instructions': m.instructions,
            'phase_instruction': m.phase_instruction,
            'breath_instruction': m.breath_instruction,

            # Keypoints (overlay için)
            'keypoints': {
                'left_shoulder': kp_coords(kps.left_shoulder),
                'right_shoulder': kp_coords(kps.right_shoulder),
                'left_hip': kp_coords(kps.left_hip),
                'right_hip': kp_coords(kps.right_hip),
                'left_knee': kp_coords(kps.left_knee),
                'right_knee': kp_coords(kps.right_knee),
                'left_ear': kp_coords(kps.left_ear),
                'right_ear': kp_coords(kps.right_ear),
                'nose': kp_coords(kps.nose),
            },

            # Faz bilgisi
            'phase': self.get_current_phase(),

            # Seans özeti
            'session': self.get_session_summary(),
        }

    def _last_valid_dict(self):
        return self._last_valid
