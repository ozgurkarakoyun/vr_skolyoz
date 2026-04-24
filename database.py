"""
database.py
────────────────────────────────────────────────────────────────
Hasta profili ve seans kayıtları için SQLite veritabanı
Railway'deki diğer projelerle aynı /data volume yapısı kullanılır
"""

import sqlite3
import os
import json
import logging
from datetime import datetime
from contextlib import contextmanager

logger = logging.getLogger(__name__)

# Railway Volume veya lokal — /data yoksa app klasörüne düş
_default_data = '/data'
try:
    os.makedirs(_default_data, exist_ok=True)
    DATA_DIR = os.environ.get('DATA_DIR', _default_data)
except PermissionError:
    DATA_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'data')
os.makedirs(DATA_DIR, exist_ok=True)

DB_PATH = os.path.join(DATA_DIR, 'schroth.db')

# ─── Bağlantı ────────────────────────────────────────────────

@contextmanager
def get_db():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    conn.execute("PRAGMA journal_mode=WAL")
    conn.execute("PRAGMA foreign_keys=ON")
    try:
        yield conn
        conn.commit()
    except Exception:
        conn.rollback()
        raise
    finally:
        conn.close()

# ─── Şema ────────────────────────────────────────────────────

def init_db():
    with get_db() as conn:
        conn.executescript("""
        CREATE TABLE IF NOT EXISTS patients (
            id          INTEGER PRIMARY KEY AUTOINCREMENT,
            name        TEXT NOT NULL,
            birth_year  INTEGER,
            gender      TEXT CHECK(gender IN ('K','E','—')) DEFAULT '—',
            diagnosis   TEXT DEFAULT '',
            curve_type  TEXT DEFAULT '',
            cobb_angle  REAL DEFAULT 0,
            risser      INTEGER DEFAULT 0,
            notes       TEXT DEFAULT '',
            created_at  TEXT DEFAULT (datetime('now')),
            updated_at  TEXT DEFAULT (datetime('now'))
        );

        CREATE TABLE IF NOT EXISTS sessions (
            id              INTEGER PRIMARY KEY AUTOINCREMENT,
            patient_id      INTEGER NOT NULL,
            session_code    TEXT NOT NULL,
            started_at      TEXT DEFAULT (datetime('now')),
            ended_at        TEXT,
            duration_sec    INTEGER DEFAULT 0,
            frame_count     INTEGER DEFAULT 0,
            avg_score       REAL DEFAULT 0,
            best_score      REAL DEFAULT 0,
            rep_count       INTEGER DEFAULT 0,
            avg_thoracic    REAL,
            avg_thoracolumbar REAL,
            avg_lumbar      REAL,
            avg_shoulder    REAL,
            avg_hip         REAL,
            trend           TEXT DEFAULT 'stable',
            phase_log       TEXT DEFAULT '[]',
            notes           TEXT DEFAULT '',
            FOREIGN KEY(patient_id) REFERENCES patients(id) ON DELETE CASCADE
        );

        CREATE INDEX IF NOT EXISTS idx_sessions_patient ON sessions(patient_id);
        CREATE INDEX IF NOT EXISTS idx_sessions_code ON sessions(session_code);
        """)
    logger.info(f"DB initialized: {DB_PATH}")

# ─── Hasta CRUD ──────────────────────────────────────────────

def create_patient(name, birth_year=None, gender='—', diagnosis='',
                   curve_type='', cobb_angle=0, risser=0, notes=''):
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO patients (name, birth_year, gender, diagnosis,
                                  curve_type, cobb_angle, risser, notes)
            VALUES (?,?,?,?,?,?,?,?)
        """, (name, birth_year, gender, diagnosis, curve_type, cobb_angle, risser, notes))
        return cur.lastrowid

def get_patient(patient_id):
    with get_db() as conn:
        row = conn.execute("SELECT * FROM patients WHERE id=?", (patient_id,)).fetchone()
        return dict(row) if row else None

def get_all_patients():
    with get_db() as conn:
        rows = conn.execute("""
            SELECT p.*,
                COUNT(s.id) as session_count,
                MAX(s.started_at) as last_session,
                AVG(s.avg_score) as overall_avg_score
            FROM patients p
            LEFT JOIN sessions s ON s.patient_id = p.id
            GROUP BY p.id
            ORDER BY p.name
        """).fetchall()
        return [dict(r) for r in rows]

def update_patient(patient_id, **kwargs):
    allowed = {'name','birth_year','gender','diagnosis','curve_type',
               'cobb_angle','risser','notes'}
    fields = {k: v for k, v in kwargs.items() if k in allowed}
    if not fields:
        return
    fields['updated_at'] = datetime.now().isoformat()
    sets = ', '.join(f"{k}=?" for k in fields)
    vals = list(fields.values()) + [patient_id]
    with get_db() as conn:
        conn.execute(f"UPDATE patients SET {sets} WHERE id=?", vals)

def delete_patient(patient_id):
    with get_db() as conn:
        conn.execute("DELETE FROM patients WHERE id=?", (patient_id,))

# ─── Seans CRUD ──────────────────────────────────────────────

def create_session(patient_id, session_code):
    with get_db() as conn:
        cur = conn.execute("""
            INSERT INTO sessions (patient_id, session_code)
            VALUES (?,?)
        """, (patient_id, session_code))
        return cur.lastrowid

def end_session(session_code, data: dict):
    """Seans tamamlandığında istatistikleri kaydet"""
    phase_log_json = json.dumps(data.get('phase_log', []))
    with get_db() as conn:
        conn.execute("""
            UPDATE sessions SET
                ended_at        = datetime('now'),
                duration_sec    = ?,
                frame_count     = ?,
                avg_score       = ?,
                best_score      = ?,
                rep_count       = ?,
                avg_thoracic    = ?,
                avg_thoracolumbar = ?,
                avg_lumbar      = ?,
                avg_shoulder    = ?,
                avg_hip         = ?,
                trend           = ?,
                phase_log       = ?,
                notes           = ?
            WHERE session_code = ?
        """, (
            data.get('duration', 0),
            data.get('frame_count', 0),
            data.get('avg_score', 0),
            data.get('best_score', 0),
            data.get('rep_count', 0),
            data.get('avg_thoracic'),
            data.get('avg_thoracolumbar'),
            data.get('avg_lumbar'),
            data.get('avg_shoulder'),
            data.get('avg_hip'),
            data.get('trend', 'stable'),
            phase_log_json,
            data.get('notes', ''),
            session_code,
        ))

def get_patient_sessions(patient_id, limit=20):
    with get_db() as conn:
        rows = conn.execute("""
            SELECT * FROM sessions
            WHERE patient_id=?
            ORDER BY started_at DESC
            LIMIT ?
        """, (patient_id, limit)).fetchall()
        result = []
        for r in rows:
            d = dict(r)
            try:
                d['phase_log'] = json.loads(d['phase_log'] or '[]')
            except Exception:
                d['phase_log'] = []
            result.append(d)
        return result

def get_session_by_code(session_code):
    with get_db() as conn:
        row = conn.execute(
            "SELECT * FROM sessions WHERE session_code=?", (session_code,)
        ).fetchone()
        return dict(row) if row else None

def get_patient_stats(patient_id):
    """Hasta için trend ve özet istatistikler"""
    with get_db() as conn:
        row = conn.execute("""
            SELECT
                COUNT(*) as total_sessions,
                SUM(duration_sec) as total_seconds,
                AVG(avg_score) as overall_avg,
                MAX(best_score) as all_time_best,
                SUM(rep_count) as total_reps,
                AVG(avg_thoracic) as avg_thoracic,
                AVG(avg_thoracolumbar) as avg_thoracolumbar,
                AVG(avg_lumbar) as avg_lumbar
            FROM sessions
            WHERE patient_id=? AND ended_at IS NOT NULL
        """, (patient_id,)).fetchone()

        # Son 5 seans skoru (trend için)
        recent = conn.execute("""
            SELECT avg_score FROM sessions
            WHERE patient_id=? AND ended_at IS NOT NULL
            ORDER BY started_at DESC LIMIT 5
        """, (patient_id,)).fetchall()
        recent_scores = [r['avg_score'] for r in recent]

    d = dict(row) if row else {}
    d['recent_scores'] = recent_scores
    if len(recent_scores) >= 2:
        diff = recent_scores[0] - recent_scores[-1]
        d['trend'] = 'improving' if diff > 3 else 'declining' if diff < -3 else 'stable'
    else:
        d['trend'] = 'insufficient_data'
    return d

# ─── Init ────────────────────────────────────────────────────
init_db()
