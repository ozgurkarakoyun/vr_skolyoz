/**
 * exercise_tracker.js
 * ─────────────────────────────────────────────────────────────
 * Schroth VR — Egzersiz Seri Takip Sistemi
 *
 * Özellikler:
 *   - Her egzersiz seansını kayıt altına alır
 *   - Faz süre takibi (Elongasyon/Derotasyon/RAB/Stabilizasyon)
 *   - Set ve tekrar sayısı
 *   - Skor zaman serisi (grafik için)
 *   - LocalStorage'a kalıcı kayıt
 *   - Seans raporu üretimi
 *   - Haftalık trend
 */

class ExerciseTracker {
  constructor() {
    this.STORAGE_KEY = 'schroth_sessions';
    this.MAX_STORED   = 30;   // Son 30 seans saklanır

    this.currentSession = null;
    this.scoreHistory   = [];   // [{t: ms, score: n}, ...]
    this.phaseLog       = [];   // [{name, start, end, avgScore}]
    this._currentPhase  = null;
    this._phaseStart    = null;
    this._phaseScores   = [];

    this.allSessions    = this._load();
  }

  // ─── Seans Yönetimi ───────────────────────────────────────

  startSession(sessionId) {
    this.currentSession = {
      id:          sessionId,
      startTime:   Date.now(),
      endTime:     null,
      frameCount:  0,
      scores:      [],
      phaseLog:    [],
      scolAngles:  { thoracic: [], thoracolumbar: [], lumbar: [] },
      postureData: [],
      completed:   false,
    };
    this.scoreHistory  = [];
    this.phaseLog      = [];
    this._currentPhase = null;
    console.log('[Tracker] Session started:', sessionId);
  }

  endSession() {
    if (!this.currentSession) return null;
    this.currentSession.endTime   = Date.now();
    this.currentSession.completed = true;
    if (this._currentPhase) this._closePhase();

    const report = this._buildReport(this.currentSession);
    this.currentSession.report   = report;

    // Kaydet
    this.allSessions.unshift(this.currentSession);
    if (this.allSessions.length > this.MAX_STORED)
      this.allSessions = this.allSessions.slice(0, this.MAX_STORED);
    this._save();

    console.log('[Tracker] Session ended:', report);
    return report;
  }

  // ─── Frame Güncelle ───────────────────────────────────────

  update(analysis) {
    if (!this.currentSession || !analysis) return;
    const now = Date.now();

    this.currentSession.frameCount++;

    // Skor kaydı
    const score = analysis.score || 0;
    this.currentSession.scores.push(score);
    this.scoreHistory.push({ t: now, score });

    // Postür verisi (her 5 frame'de bir kaydet - hafıza tasarrufu)
    if (this.currentSession.frameCount % 5 === 0) {
      this.currentSession.postureData.push({
        t:          now,
        score,
        shoulder:   analysis.shoulder_angle || 0,
        hip:        analysis.hip_angle       || 0,
        trunk:      analysis.trunk_inclination || 0,
        curve:      analysis.curve_type      || '—',
      });
    }

    // Skolyoz açıları
    const scol = analysis.scoliosis;
    if (scol && !scol.estimated) {
      this.currentSession.scolAngles.thoracic.push(scol.thoracic || 0);
      this.currentSession.scolAngles.thoracolumbar.push(scol.thoracolumbar || 0);
      this.currentSession.scolAngles.lumbar.push(scol.lumbar || 0);
    }

    // Faz takibi
    const phase = analysis.phase;
    if (phase) {
      if (!this._currentPhase || this._currentPhase !== phase.id) {
        if (this._currentPhase) this._closePhase();
        this._currentPhase = phase.id;
        this._phaseName    = phase.name_tr || phase.name;
        this._phaseStart   = now;
        this._phaseScores  = [];
      }
      this._phaseScores.push(score);
    }
  }

  _closePhase() {
    if (!this._currentPhase || !this._phaseStart) return;
    const entry = {
      id:       this._currentPhase,
      name:     this._phaseName,
      start:    this._phaseStart,
      end:      Date.now(),
      duration: Math.round((Date.now() - this._phaseStart) / 1000),
      avgScore: this._phaseScores.length
        ? Math.round(this._phaseScores.reduce((a, b) => a + b, 0) / this._phaseScores.length)
        : 0,
    };
    this.phaseLog.push(entry);
    this.currentSession.phaseLog.push(entry);
    this._currentPhase = null;
    this._phaseScores  = [];
  }

  // ─── Rapor Üretimi ────────────────────────────────────────

  _buildReport(session) {
    const scores       = session.scores;
    const duration     = Math.round(((session.endTime || Date.now()) - session.startTime) / 1000);
    const avgScore     = scores.length ? Math.round(scores.reduce((a,b)=>a+b,0) / scores.length) : 0;
    const bestScore    = scores.length ? Math.round(Math.max(...scores)) : 0;
    const worstScore   = scores.length ? Math.round(Math.min(...scores)) : 0;

    // Skor trendi (ilk yarı vs ikinci yarı)
    const half         = Math.floor(scores.length / 2);
    const firstHalf    = half > 0 ? scores.slice(0, half).reduce((a,b)=>a+b,0)/half : 0;
    const secondHalf   = half > 0 ? scores.slice(half).reduce((a,b)=>a+b,0)/(scores.length-half) : 0;
    const trend        = secondHalf - firstHalf;

    // Skolyoz açı ortalamaları
    const avgT  = _avg(session.scolAngles.thoracic);
    const avgTL = _avg(session.scolAngles.thoracolumbar);
    const avgL  = _avg(session.scolAngles.lumbar);

    // En çok hangi düzeltme yapıldı
    const postureData  = session.postureData;
    const shoulderBias = postureData.length
      ? postureData.reduce((a,b) => a + b.shoulder, 0) / postureData.length : 0;
    const hipBias      = postureData.length
      ? postureData.reduce((a,b) => a + b.hip, 0) / postureData.length : 0;

    return {
      sessionId:  session.id,
      date:       new Date(session.startTime).toLocaleDateString('tr-TR'),
      time:       new Date(session.startTime).toLocaleTimeString('tr-TR', {hour:'2-digit',minute:'2-digit'}),
      duration,
      frameCount: session.frameCount,
      scores: { avg: avgScore, best: bestScore, worst: worstScore },
      trend:  trend > 3 ? 'improving' : trend < -3 ? 'declining' : 'stable',
      trendValue: Math.round(trend),
      scoliosis:  { thoracic: avgT, thoracolumbar: avgTL, lumbar: avgL },
      posture:    { shoulderBias: Math.round(shoulderBias*10)/10, hipBias: Math.round(hipBias*10)/10 },
      phaseLog:   session.phaseLog,
      repCount:   session.phaseLog.filter(p => p.id === 'stabilization').length,
    };
  }

  // ─── İstatistik Sorgular ──────────────────────────────────

  getWeeklyStats() {
    const now    = Date.now();
    const week   = 7 * 24 * 3600 * 1000;
    const recent = this.allSessions.filter(s => s.startTime && (now - s.startTime) < week);

    if (!recent.length) return null;

    const reports = recent.map(s => s.report).filter(Boolean);
    return {
      sessionCount: recent.length,
      totalMinutes: Math.round(recent.reduce((a,s) => a + (s.report?.duration||0), 0) / 60),
      avgScore:     Math.round(reports.reduce((a,r) => a + (r.scores?.avg||0), 0) / (reports.length||1)),
      bestScore:    Math.max(...reports.map(r => r.scores?.best||0)),
      trend:        this._weeklyTrend(reports),
    };
  }

  _weeklyTrend(reports) {
    if (reports.length < 2) return 'insufficient_data';
    const first = reports[reports.length-1].scores?.avg || 0;
    const last  = reports[0].scores?.avg || 0;
    return last > first + 5 ? 'improving' : last < first - 5 ? 'declining' : 'stable';
  }

  getSessionHistory(limit = 10) {
    return this.allSessions
      .slice(0, limit)
      .map(s => s.report)
      .filter(Boolean);
  }

  getCurrentScoreHistory() {
    return this.scoreHistory;
  }

  // ─── Canlı UI Verileri ────────────────────────────────────

  getLiveStats() {
    if (!this.currentSession) return null;
    const scores   = this.currentSession.scores;
    const duration = Math.round((Date.now() - this.currentSession.startTime) / 1000);
    const reps     = this.currentSession.phaseLog.filter(p => p.id === 'stabilization').length;
    return {
      duration,
      frameCount:  this.currentSession.frameCount,
      avgScore:    scores.length ? Math.round(scores.reduce((a,b)=>a+b,0)/scores.length) : 0,
      bestScore:   scores.length ? Math.round(Math.max(...scores)) : 0,
      repCount:    reps,
      phaseCount:  this.phaseLog.length,
    };
  }

  // ─── LocalStorage ─────────────────────────────────────────

  _save() {
    try {
      // Büyük score dizilerini sıkıştır
      const toSave = this.allSessions.map(s => ({
        ...s,
        scores: [], // Ham skorları kaydetme, sadece raporu sakla
      }));
      localStorage.setItem(this.STORAGE_KEY, JSON.stringify(toSave));
    } catch(e) {
      console.warn('[Tracker] Save failed:', e);
    }
  }

  _load() {
    try {
      const raw = localStorage.getItem(this.STORAGE_KEY);
      return raw ? JSON.parse(raw) : [];
    } catch(e) {
      return [];
    }
  }

  clearHistory() {
    this.allSessions = [];
    localStorage.removeItem(this.STORAGE_KEY);
  }
}

// ─── Yardımcı ─────────────────────────────────────────────
function _avg(arr) {
  if (!arr || !arr.length) return 0;
  return Math.round(arr.reduce((a,b)=>a+b,0)/arr.length * 10) / 10;
}

// Global instance
window.exerciseTracker = new ExerciseTracker();
