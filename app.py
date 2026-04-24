"""
app.py — Schroth VR Backend v5
────────────────────────────────────────────────────────────────
Aşama 5: Hasta profili + Terapist paneli eklendi
"""
# ── OpenCV headless guard ─────────────────────────────────────
# ultralytics opencv-python (full) kurmuşsa, import öncesi kaldır
import subprocess, sys
try:
    import cv2
except ImportError:
    # full opencv kurulu ve libGL eksik → headless'e geç
    subprocess.check_call([sys.executable, "-m", "pip", "uninstall", "-y", "opencv-python"], 
                          stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
    subprocess.check_call([sys.executable, "-m", "pip", "install", 
                           "opencv-python-headless==4.9.0.80", "--quiet"])
    import cv2
# ─────────────────────────────────────────────────────────────

from flask import Flask, render_template, request, jsonify, send_file, make_response
from flask_socketio import SocketIO, emit, join_room, leave_room
import base64, cv2, numpy as np, os, logging
from datetime import datetime

from schroth_analyzer import SchrothAnalyzer
from scoliosis_engine import analyze_scoliosis_frame, estimate_from_pose_keypoints, get_scoliosis_model
from pdf_report import generate_pdf
from database import (
    create_patient, get_patient, get_all_patients, update_patient, delete_patient,
    create_session, end_session, get_patient_sessions, get_session_by_code,
    get_patient_stats,
)

app = Flask(__name__)
app.config['SECRET_KEY'] = os.environ.get('SECRET_KEY', 'schroth-vr-secret-2024')
socketio = SocketIO(app, cors_allowed_origins="*", async_mode='threading')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Pose model ──────────────────────────────────────────────
_pose_model = None
def get_pose_model():
    global _pose_model
    if _pose_model is None:
        try:
            from ultralytics import YOLO
            path = os.environ.get('MODEL_PATH', 'models/pose_model.pt')
            _pose_model = YOLO(path if os.path.exists(path) else 'yolo26n-pose.pt')
        except Exception as e:
            logger.error(f"Pose model: {e}")
    return _pose_model

# ─── Seans havuzu ────────────────────────────────────────────
_analyzers: dict = {}
_session_patients: dict = {}   # session_code → patient_id

def get_analyzer(room: str) -> SchrothAnalyzer:
    if room not in _analyzers:
        _analyzers[room] = SchrothAnalyzer(smoothing_window=5)
    return _analyzers[room]

# ─── Frame işleme ────────────────────────────────────────────
def process_frame(image_b64: str, room: str) -> dict:
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
                    schroth_data = analyzer.analyze(pose_kps, w, h)
        else:
            schroth_data = _mock_schroth(analyzer)

        scol_data = None
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
        logger.error(f"process_frame: {e}")
        return {}

def _mock_schroth(analyzer):
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
    if r: r['mock'] = True
    return r or {}

# ─── Sayfa Routes ─────────────────────────────────────────────
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
    d = request.json or {}
    update_patient(pid, **d)
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

# ─── Seans API ────────────────────────────────────────────────
@app.route('/api/sessions/<code>', methods=['GET'])
def api_get_session(code):
    s = get_session_by_code(code)
    if not s:
        return jsonify({'error': 'Seans bulunamadı'}), 404
    return jsonify(s)

@app.route('/api/sessions/<code>/end', methods=['POST'])
def api_end_session(code):
    d = request.json or {}
    end_session(code, d)
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

# ─── Diğer ────────────────────────────────────────────────────

# ─── PDF Rapor ────────────────────────────────────────────────
@app.route('/api/patients/<int:pid>/report.pdf')
def api_patient_report(pid):
    """Son seans PDF raporu"""
    p = get_patient(pid)
    if not p:
        return jsonify({'error': 'Hasta bulunamadı'}), 404

    sessions = get_patient_sessions(pid, limit=10)
    stats    = get_patient_stats(pid)
    session_data = sessions[0] if sessions else {}

    try:
        pdf_bytes = generate_pdf(p, session_data, stats, sessions)
        response = make_response(pdf_bytes)
        response.headers['Content-Type']        = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="schroth_{p["name"].replace(" ","_")}_{datetime.now().strftime("%Y%m%d")}.pdf"'
        return response
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/api/sessions/<code>/report.pdf')
def api_session_report(code):
    """Belirli seans PDF raporu"""
    sess = get_session_by_code(code)
    if not sess:
        return jsonify({'error': 'Seans bulunamadı'}), 404

    pid = sess.get('patient_id')
    p   = get_patient(pid) if pid else {'name': 'Bilinmeyen Hasta'}
    stats = get_patient_stats(pid) if pid else {}
    recent = get_patient_sessions(pid, limit=10) if pid else []

    try:
        pdf_bytes = generate_pdf(p, sess, stats, recent)
        response = make_response(pdf_bytes)
        response.headers['Content-Type']        = 'application/pdf'
        response.headers['Content-Disposition'] = f'attachment; filename="schroth_seans_{code}.pdf"'
        return response
    except Exception as e:
        logger.error(f"PDF error: {e}")
        return jsonify({'error': str(e)}), 500

@app.route('/health')
def health():
    return jsonify({
        'status': 'ok',
        'timestamp': datetime.now().isoformat(),
        'active_sessions': len(_analyzers),
        'scol_model': os.path.exists(os.environ.get('SCOL_MODEL_PATH','models/model_point4.pt')),
        'pose_model': os.path.exists(os.environ.get('MODEL_PATH','models/pose_model.pt')),
    })

@app.route('/session/<room>/summary')
def session_summary(room):
    a = _analyzers.get(room)
    return jsonify(a.get_session_summary() if a else {'error': 'not found'}), 200 if a else 404

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
        # Hasta bilgisini ekle
        pid = _session_patients.get(room)
        if pid:
            analysis['patient_id'] = pid
        emit('analysis', {'analysis': analysis, 'timestamp': datetime.now().isoformat()}, to=room)
    except Exception as e:
        logger.error(f"frame event: {e}")

@socketio.on('reset_session')
def on_reset_session(data):
    room = data.get('room', 'default')
    if room in _analyzers:
        _analyzers[room].reset_session()
    emit('session_reset', {'room': room}, to=room)

@socketio.on('link_patient')
def on_link_patient(data):
    """Seansı hastaya bağla"""
    room = data.get('room')
    pid  = data.get('patient_id')
    if room and pid:
        _session_patients[room] = int(pid)
        emit('patient_linked', {'room': room, 'patient_id': pid}, to=room)

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    socketio.run(app, host='0.0.0.0', port=port, debug=False)
