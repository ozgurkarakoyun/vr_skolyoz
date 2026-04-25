"""
app.py — Schroth VR Backend v5
"""
import os
import logging
import base64
import json
from datetime import datetime

# ── Logging ilk önce kurulsun ────────────────────────────────
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s"
)
logger = logging.getLogger(__name__)

# ── Flask ────────────────────────────────────────────────────
from flask import Flask, render_template, request, jsonify, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'schroth-vr-secret-2024')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')

# ── Lazy imports (crash etmesin) ─────────────────────────────
# cv2, numpy, ultralytics — ilk frame gelince yüklenir

def _import_cv2():
    try:
        import cv2
        import numpy as np
        return cv2, np
    except ImportError as e:
        logger.error(f"cv2 import failed: {e}")
        return None, None

# ── Database — hata olursa uygulama çökmez ───────────────────
try:
    from database import (
        create_patient, get_patient, get_all_patients, update_patient, delete_patient,
        create_session, end_session, get_patient_sessions, get_session_by_code,
        get_patient_stats,
    )
    DB_OK = True
    logger.info("Database OK")
except Exception as e:
    logger.error(f"Database init failed: {e} — DB features disabled")
    DB_OK = False
    # Stub fonksiyonlar — uygulama yine de açılır
    def get_all_patients(): return []
    def get_patient(pid): return None
    def create_patient(**kw): return None
    def update_patient(pid, **kw): pass
    def delete_patient(pid): pass
    def create_session(pid, code): return None
    def end_session(code, data): pass
    def get_patient_sessions(pid, limit=20): return []
    def get_session_by_code(code): return None
    def get_patient_stats(pid): return {}

# ── Schroth Analyzer ─────────────────────────────────────────
try:
    from schroth_analyzer import SchrothAnalyzer
    ANALYZER_OK = True
    logger.info("SchrothAnalyzer OK")
except Exception as e:
    logger.error(f"SchrothAnalyzer import failed: {e}")
    ANALYZER_OK = False
    SchrothAnalyzer = None

# ── Scoliosis Engine ─────────────────────────────────────────
try:
    from scoliosis_engine import (
        analyze_scoliosis_frame, estimate_from_pose_keypoints, get_scoliosis_model
    )
    SCOL_OK = True
    logger.info("ScoliosisEngine OK")
except Exception as e:
    logger.error(f"ScoliosisEngine import failed: {e}")
    SCOL_OK = False
    def analyze_scoliosis_frame(f): return None
    def estimate_from_pose_keypoints(k): return None
    def get_scoliosis_model(): return None

# ── PDF Report ───────────────────────────────────────────────
try:
    from pdf_report import generate_pdf
    PDF_OK = True
    logger.info("PDF report OK")
except Exception as e:
    logger.error(f"pdf_report import failed: {e}")
    PDF_OK = False
    def generate_pdf(*a, **kw): return b""

# ─── Pose model (lazy) ───────────────────────────────────────
_pose_model = None

def get_pose_model():
    global _pose_model
    if _pose_model is None:
        try:
            import torch
            from ultralytics import YOLO
            from ultralytics.nn.tasks import PoseModel

            # PyTorch 2.6+ weights_only=True varsayılan — YOLO modelini allowlist'e al
            torch.serialization.add_safe_globals([PoseModel])

            path = os.environ.get('MODEL_PATH', 'models/yolo26n-pose.pt')
            model_file = path if os.path.exists(path) else 'yolo26n-pose.pt'
            _pose_model = YOLO(model_file)
            logger.info(f"Pose model loaded: {model_file}")
        except Exception as e:
            logger.error(f"Pose model load failed: {e}")
    return _pose_model

# ─── Seans havuzu ────────────────────────────────────────────
_analyzers: dict = {}
_session_patients: dict = {}

def get_analyzer(room: str):
    if not ANALYZER_OK:
        return None
    if room not in _analyzers:
        _analyzers[room] = SchrothAnalyzer(smoothing_window=5)
    return _analyzers[room]

# ─── Frame işleme ────────────────────────────────────────────
def process_frame(image_b64: str, room: str) -> dict:
    cv2, np = _import_cv2()
    if cv2 is None or np is None:
        return {}
    try:
        if ',' in image_b64:
            image_b64 = image_b64.split(',')[1]
        nparr = np.frombuffer(base64.b64decode(image_b64), np.uint8)
        frame = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        if frame is None:
            return {}

        h, w = frame.shape[:2]
        analyzer = get_analyzer(room)
        pose_kps = None
        schroth_data = None

        pm = get_pose_model()
        if pm:
            results = pm(frame, verbose=False)
            if results and results[0].keypoints is not None:
                kps_all = results[0].keypoints.data.cpu().numpy()
                if len(kps_all) > 0:
                    if len(kps_all) > 1 and results[0].boxes is not None:
                        boxes = results[0].boxes.xyxy.cpu().numpy()
                        areas = [(b[2]-b[0])*(b[3]-b[1]) for b in boxes]
                        pose_kps = kps_all[int(np.argmax(areas))]
                    else:
                        pose_kps = kps_all[0]
                    if analyzer:
                        schroth_data = analyzer.analyze(pose_kps, w, h)
        else:
            if analyzer:
                schroth_data = _mock_schroth(analyzer, np)

        scol_data = None
        if SCOL_OK:
            sm = get_scoliosis_model()
            if sm is not None:
                scol_data = analyze_scoliosis_frame(frame)
            elif pose_kps is not None:
                scol_data = estimate_from_pose_keypoints(pose_kps)

        combined = {}
        if schroth_data:
            combined.update(schroth_data)
        if scol_data:
            combined['scoliosis'] = {
                'thoracic':      scol_data['angles']['thoracic'],
                'thoracolumbar': scol_data['angles']['thoracolumbar'],
                'lumbar':        scol_data['angles']['lumbar'],
                'labels':        scol_data['labels'],
                'severity':      scol_data['severity'],
                'max_angle':     scol_data['max_angle'],
                'points':        scol_data.get('points', []),
                'estimated':     scol_data.get('estimated', False),
            }
        return combined

    except Exception as e:
        logger.error(f"process_frame error: {e}")
        return {}


def _mock_schroth(analyzer, np):
    import time, math
    t = time.time()
    angle = math.sin(t * 0.3) * 6
    mock_kps = np.array([
        [320,50,0.9],[310,45,0.9],[330,45,0.9],[305,55,0.9],[335,55,0.9],
        [280+angle,150,0.95],[360-angle,150+angle*2,0.95],
        [260,220,0.8],[380,220,0.8],[250,280,0.7],[390,280,0.7],
        [290+angle*.5,300+angle,0.9],[350-angle*.5,300-angle,0.9],
        [290,380,0.8],[350,380,0.8],[290,450,0.7],[350,450,0.7],
    ], dtype=np.float32)
    r = analyzer.analyze(mock_kps, 640, 480)
    if r:
        r['mock'] = True
    return r or {}

# ─── Sayfa routes ─────────────────────────────────────────────
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/phone')
def phone():
    return render_template('phone.html')

@app.route('/quest')
def quest():
    return render_template('quest.html')

@app.route('/report')
def report():
    return render_template('report.html')

@app.route('/therapist')
def therapist():
    return render_template('therapist.html')

@app.route('/patient/<int:pid>')
def patient_detail(pid):
    return render_template('patient_detail.html', patient_id=pid)


# ─── Geçici Model Upload Endpoint ────────────────────────────
# Kullanım: https://sizin-url/admin/upload?key=ADMIN_KEY
# Modeli yükledikten sonra bu endpoint'i kaldırın!
@app.route('/admin/upload', methods=['GET', 'POST'])
def admin_upload():
    admin_key = os.environ.get('ADMIN_KEY', '')
    if not admin_key or request.args.get('key') != admin_key:
        return 'Yetkisiz — ADMIN_KEY env var eksik veya hatalı', 403

    if request.method == 'POST':
        f = request.files.get('model')
        if not f:
            return 'Dosya seçilmedi', 400
        save_dir = os.environ.get('SCOL_MODEL_PATH', '/data/models/model_point4.pt')
        save_dir = os.path.dirname(save_dir)
        os.makedirs(save_dir, exist_ok=True)
        dest = os.path.join(save_dir, f.filename)
        f.save(dest)
        size_mb = os.path.getsize(dest) / 1024 / 1024
        return f'✅ Yüklendi: {dest} ({size_mb:.1f} MB)', 200

    # GET — form göster
    files = []
    model_dir = os.path.dirname(os.environ.get('SCOL_MODEL_PATH', '/data/models/model_point4.pt'))
    if os.path.exists(model_dir):
        files = [(f, round(os.path.getsize(os.path.join(model_dir,f))/1024/1024,1))
                 for f in os.listdir(model_dir)]
    file_list = ''.join(f'<li>{n} — {s} MB</li>' for n, s in files)
    return f"""
    <html><body style="font-family:sans-serif;padding:24px;background:#1a1a2e;color:#e8f4f8">
    <h2>📦 Model Yükle</h2>
    <form method="post" enctype="multipart/form-data">
        <input type="file" name="model" accept=".pt" style="color:#e8f4f8"><br><br>
        <button type="submit" style="padding:10px 20px;background:#00e5ff;border:none;border-radius:8px;cursor:pointer;font-weight:bold">
            Yükle
        </button>
    </form>
    <h3>Mevcut Dosyalar:</h3>
    <ul>{file_list if file_list else '<li>Henüz dosya yok</li>'}</ul>
    </body></html>
    """

# ─── Health check ─────────────────────────────────────────────
@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'db': DB_OK,
        'analyzer': ANALYZER_OK,
        'scol_engine': SCOL_OK,
        'pdf': PDF_OK,
        'active_sessions': len(_analyzers),
    })

# ─── Hasta API ────────────────────────────────────────────────
@app.route('/api/patients', methods=['GET'])
def api_patients():
    return jsonify(get_all_patients())

@app.route('/api/patients', methods=['POST'])
def api_create_patient():
    d = request.json or {}
    if not d.get('name'):
        return jsonify({'error': 'name zorunlu'}), 400
    pid = create_patient(
        name=d['name'],
        birth_year=d.get('birth_year'),
        gender=d.get('gender', '—'),
        diagnosis=d.get('diagnosis', ''),
        curve_type=d.get('curve_type', ''),
        cobb_angle=d.get('cobb_angle', 0),
        risser=d.get('risser', 0),
        notes=d.get('notes', ''),
    )
    return jsonify({'id': pid, 'status': 'created'}), 201

@app.route('/api/patients/<int:pid>', methods=['GET'])
def api_get_patient(pid):
    p = get_patient(pid)
    if not p:
        return jsonify({'error': 'Hasta bulunamadı'}), 404
    return jsonify(p)

@app.route('/api/patients/<int:pid>', methods=['PUT'])
def api_update_patient(pid):
    update_patient(pid, **(request.json or {}))
    return jsonify({'status': 'updated'})

@app.route('/api/patients/<int:pid>', methods=['DELETE'])
def api_delete_patient(pid):
    delete_patient(pid)
    return jsonify({'status': 'deleted'})

@app.route('/api/patients/<int:pid>/sessions', methods=['GET'])
def api_patient_sessions(pid):
    return jsonify(get_patient_sessions(pid))

@app.route('/api/patients/<int:pid>/stats', methods=['GET'])
def api_patient_stats(pid):
    return jsonify(get_patient_stats(pid))

@app.route('/api/sessions/<code>', methods=['GET'])
def api_get_session(code):
    s = get_session_by_code(code)
    if not s:
        return jsonify({'error': 'Seans bulunamadı'}), 404
    return jsonify(s)

@app.route('/api/sessions/<code>/end', methods=['POST'])
def api_end_session(code):
    end_session(code, request.json or {})
    return jsonify({'status': 'ended'})

@app.route('/api/sessions/start', methods=['POST'])
def api_start_session():
    d = request.json or {}
    pid = d.get('patient_id')
    code = d.get('session_code')
    if not pid or not code:
        return jsonify({'error': 'patient_id ve session_code zorunlu'}), 400
    sid = create_session(int(pid), code)
    _session_patients[code] = int(pid)
    return jsonify({'session_db_id': sid, 'status': 'started'})

# ─── PDF routes ───────────────────────────────────────────────
@app.route('/api/patients/<int:pid>/report.pdf')
def api_patient_report(pid):
    if not PDF_OK or not DB_OK:
        return jsonify({'error': 'PDF/DB kullanılamıyor'}), 503
    p = get_patient(pid)
    if not p:
        return jsonify({'error': 'Hasta bulunamadı'}), 404
    sessions = get_patient_sessions(pid, limit=10)
    stats = get_patient_stats(pid)
    session_data = sessions[0] if sessions else {}
    try:
        pdf_bytes = generate_pdf(p, session_data, stats, sessions)
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = (
            f'attachment; filename="schroth_{p["name"].replace(" ","_")}'
            f'_{datetime.now().strftime("%Y%m%d")}.pdf"'
        )
        return response
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<code>/report.pdf')
def api_session_report(code):
    if not PDF_OK:
        return jsonify({'error': 'PDF kullanılamıyor'}), 503
    sess = get_session_by_code(code)
    if not sess:
        return jsonify({'error': 'Seans bulunamadı'}), 404
    pid = sess.get('patient_id')
    p = get_patient(pid) if pid else {'name': 'Bilinmeyen Hasta'}
    stats = get_patient_stats(pid) if pid else {}
    recent = get_patient_sessions(pid, limit=10) if pid else []
    try:
        pdf_bytes = generate_pdf(p, sess, stats, recent)
        response = make_response(pdf_bytes)
        response.headers['Content-Type'] = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="schroth_seans_{code}.pdf"'
        return response
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return jsonify({'error': str(e)}), 500

# ─── Socket.IO ────────────────────────────────────────────────
@socketio.on('connect')
def on_connect():
    logger.info(f"Connected: {request.sid}")

@socketio.on('disconnect')
def on_disconnect():
    logger.info(f"Disconnected: {request.sid}")

@socketio.on('join_room')
def on_join_room(data):
    room = data.get('room', 'default')
    role = data.get('role', 'unknown')
    join_room(room)
    emit('room_joined', {'room': room, 'role': role, 'sid': request.sid})
    emit('peer_joined', {'role': role, 'sid': request.sid}, to=room, include_self=False)

@socketio.on('leave_room')
def on_leave_room(data):
    leave_room(data.get('room', 'default'))

@socketio.on('offer')
def on_offer(data):
    emit('offer', data, to=data.get('room'), include_self=False)

@socketio.on('answer')
def on_answer(data):
    emit('answer', data, to=data.get('room'), include_self=False)

@socketio.on('ice_candidate')
def on_ice_candidate(data):
    emit('ice_candidate', data, to=data.get('room'), include_self=False)

@socketio.on('frame')
def on_frame(data):
    try:
        image_b64 = data.get('image')
        room = data.get('room', 'default')
        if not image_b64:
            return
        analysis = process_frame(image_b64, room)
        pid = _session_patients.get(room)
        if pid:
            analysis['patient_id'] = pid
        emit('analysis', {
            'analysis': analysis,
            'timestamp': datetime.now().isoformat()
        }, to=room)
    except Exception as e:
        logger.error(f"frame event error: {e}")

@socketio.on('reset_session')
def on_reset_session(data):
    room = data.get('room', 'default')
    if room in _analyzers:
        _analyzers[room].reset_session()
    emit('session_reset', {'room': room}, to=room)

@socketio.on('link_patient')
def on_link_patient(data):
    room = data.get('room')
    pid = data.get('patient_id')
    if room and pid:
        _session_patients[room] = int(pid)
        emit('patient_linked', {'room': room, 'patient_id': pid}, to=room)

logger.info("app.py loaded successfully — waiting for gunicorn to bind")

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
