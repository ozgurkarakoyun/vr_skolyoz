/**
 * audio_engine.js
 * ─────────────────────────────────────────────────────────────
 * Schroth VR — Ses Geri Bildirimi Motoru
 *
 * Özellikler:
 *   - Web Speech API ile Türkçe TTS (text-to-speech)
 *   - AudioContext ile beep/tık sesleri (model yokken de çalışır)
 *   - Öncelik kuyruğu (acil talimatlar önce söylenir)
 *   - Debounce (aynı talimat 8 sn içinde tekrar söylenmez)
 *   - Faz geçiş sesi + nefes ritmi sesi
 *   - Skor milestone sesi (50, 70, 85 üzeri)
 */

class SchrothAudioEngine {
  constructor() {
    this.enabled = false;
    this.ttsEnabled = false;
    this.beepEnabled = true;

    // Web Speech API
    this.synth = window.speechSynthesis || null;
    this.turkishVoice = null;

    // AudioContext (beep sesleri)
    this.audioCtx = null;

    // Kuyruk & debounce
    this.queue = [];
    this.speaking = false;
    this.lastSpoken = {};        // { text: timestamp }
    this.DEBOUNCE_MS = 8000;    // 8 sn aynı cümle tekrar edilmez

    // Nefes metronom
    this.breathInterval = null;
    this.breathPhase = 'in';    // 'in' | 'hold' | 'out'

    // Skor milestone takibi
    this.lastMilestone = 0;

    this._initVoice();
    this._initAudioCtx();
  }

  // ─── Init ─────────────────────────────────────────────────

  _initVoice() {
    if (!this.synth) return;
    const loadVoices = () => {
      const voices = this.synth.getVoices();
      // Türkçe ses ara
      this.turkishVoice =
        voices.find(v => v.lang === 'tr-TR') ||
        voices.find(v => v.lang.startsWith('tr')) ||
        voices.find(v => v.lang.startsWith('en')) || // fallback İngilizce
        null;
      if (this.turkishVoice) {
        this.ttsEnabled = true;
        console.log('[Audio] TTS voice:', this.turkishVoice.name, this.turkishVoice.lang);
      }
    };
    if (this.synth.getVoices().length > 0) loadVoices();
    else this.synth.addEventListener('voiceschanged', loadVoices);
  }

  _initAudioCtx() {
    try {
      this.audioCtx = new (window.AudioContext || window.webkitAudioContext)();
    } catch(e) {
      console.warn('[Audio] AudioContext not available');
    }
  }

  // ─── Public API ───────────────────────────────────────────

  enable() {
    this.enabled = true;
    // iOS/Safari için AudioContext'i kullanıcı etkileşimiyle aç
    if (this.audioCtx?.state === 'suspended') {
      this.audioCtx.resume();
    }
    this.beep('start');
  }

  disable() {
    this.enabled = false;
    this.stopBreath();
    if (this.synth) this.synth.cancel();
    this.queue = [];
  }

  toggle() {
    this.enabled ? this.disable() : this.enable();
    return this.enabled;
  }

  /**
   * Ana güncelleme — her analiz frame'inde çağrılır
   * @param {object} analysis - backend'den gelen analiz verisi
   * @param {object} prevAnalysis - önceki frame verisi
   */
  update(analysis, prevAnalysis) {
    if (!this.enabled || !analysis) return;

    const score = analysis.score || 0;
    const scol = analysis.scoliosis;
    const phase = analysis.phase;

    // ── 1. Skor milestone ──────────────────────────────────
    this._checkScoreMilestone(score);

    // ── 2. Kritik açı uyarısı ─────────────────────────────
    if (scol) this._checkCriticalAngle(scol);

    // ── 3. Postür düzeltme talimatı ───────────────────────
    this._speakPostureInstruction(analysis);

    // ── 4. Faz geçişi ─────────────────────────────────────
    if (phase && prevAnalysis?.phase) {
      if (phase.name !== prevAnalysis.phase.name) {
        this._onPhaseChange(phase);
      }
    }

    // ── 5. Nefes metronom (RAB fazında) ───────────────────
    if (phase?.id === 'rab_breathing') {
      this._startBreathMetronome();
    } else {
      this._stopBreathMetronome();
    }
  }

  // ─── Skor Milestone ───────────────────────────────────────

  _checkScoreMilestone(score) {
    const milestones = [50, 70, 85];
    for (const m of milestones) {
      if (score >= m && this.lastMilestone < m) {
        this.lastMilestone = m;
        if (m === 85) {
          this.beep('milestone_high');
          this._speak('Harika! Mükemmel pozisyon!', 'high');
        } else if (m === 70) {
          this.beep('milestone_mid');
          this._speak('Çok iyi, devam edin', 'normal');
        } else {
          this.beep('milestone_low');
        }
        return;
      }
    }
    // Skor düşerse milestone sıfırla
    if (score < 45 && this.lastMilestone > 0) {
      this.lastMilestone = 0;
    }
  }

  // ─── Kritik Açı Uyarısı ───────────────────────────────────

  _checkCriticalAngle(scol) {
    if (scol.severity === 'KRİTİK' && scol.max_angle > 40) {
      this._speak(
        `Dikkat! ${scol.max_angle.toFixed(0)} derece eğrilik tespit edildi`,
        'urgent'
      );
      this.beep('warning');
    }
  }

  // ─── Postür Talimatı ──────────────────────────────────────

  _speakPostureInstruction(analysis) {
    const instructions = analysis.instructions || [];
    if (!instructions.length) return;

    // İlk talimatı al, emoji'leri temizle
    const raw = instructions[0];
    const clean = raw.replace(/[⬇️⬆️⬅️➡️↔️↙️↘️✅👍🫁📋]/gu, '').trim();
    if (!clean) return;

    // Her talimat için ayrı debounce
    const now = Date.now();
    const last = this.lastSpoken[clean] || 0;
    if (now - last < this.DEBOUNCE_MS) return;

    this._speak(clean, 'normal');
  }

  // ─── Faz Geçişi ───────────────────────────────────────────

  _onPhaseChange(phase) {
    this.beep('phase_change');
    const text = phase.name_tr || phase.name;
    setTimeout(() => {
      this._speak(text, 'high');
      if (phase.instruction_tr) {
        setTimeout(() => this._speak(phase.instruction_tr, 'normal'), 1500);
      }
    }, 400);
  }

  // ─── Nefes Metronom ───────────────────────────────────────

  _startBreathMetronome() {
    if (this.breathInterval) return; // Zaten çalışıyor
    // 4 saniye al, 6 saniye ver
    const sequence = [
      { phase: 'in',   duration: 4000, text: 'Nefes al' },
      { phase: 'hold', duration: 1000, text: 'Tut' },
      { phase: 'out',  duration: 6000, text: 'Nefes ver' },
      { phase: 'rest', duration: 1000, text: null },
    ];
    let idx = 0;

    const tick = () => {
      const step = sequence[idx % sequence.length];
      this.breathPhase = step.phase;
      if (step.text) {
        this.beep(step.phase === 'in' ? 'breath_in' : step.phase === 'out' ? 'breath_out' : 'breath_hold');
        if (this.ttsEnabled && step.text && idx % 3 === 0) { // Her 3 döngüde bir söyle
          this._speak(step.text, 'breath');
        }
      }
      // Fazı UI'a bildir
      window.dispatchEvent(new CustomEvent('breathPhase', { detail: step }));
      idx++;
    };

    tick();
    // Düzensiz zamanlama için recursive setTimeout
    const scheduleNext = () => {
      const step = sequence[idx % sequence.length];
      this.breathInterval = setTimeout(() => {
        tick();
        scheduleNext();
      }, step.duration);
    };
    scheduleNext();
  }

  _stopBreathMetronome() {
    if (!this.breathInterval) return;
    clearTimeout(this.breathInterval);
    this.breathInterval = null;
    this.breathPhase = 'idle';
    window.dispatchEvent(new CustomEvent('breathPhase', { detail: { phase: 'idle' } }));
  }

  // Alias
  startBreath() { this._startBreathMetronome(); }
  stopBreath() { this._stopBreathMetronome(); }

  // ─── TTS Speak ────────────────────────────────────────────

  _speak(text, priority = 'normal') {
    if (!this.enabled || !this.ttsEnabled || !this.synth) return;
    if (!text) return;

    const now = Date.now();
    this.lastSpoken[text] = now;

    const utterance = new SpeechSynthesisUtterance(text);
    if (this.turkishVoice) utterance.voice = this.turkishVoice;
    utterance.lang = 'tr-TR';
    utterance.rate = priority === 'urgent' ? 1.1 : 0.9;
    utterance.pitch = 1.0;
    utterance.volume = priority === 'breath' ? 0.7 : 1.0;

    if (priority === 'urgent') {
      // Acil → mevcut sözü kes ve hemen söyle
      this.synth.cancel();
      this.queue = [];
      this.synth.speak(utterance);
    } else {
      this.queue.push(utterance);
      if (!this.speaking) this._processQueue();
    }
  }

  _processQueue() {
    if (!this.queue.length) { this.speaking = false; return; }
    this.speaking = true;
    const utt = this.queue.shift();
    utt.onend = () => {
      setTimeout(() => this._processQueue(), 300);
    };
    utt.onerror = () => this._processQueue();
    this.synth.speak(utt);
  }

  // ─── Beep Sesleri ─────────────────────────────────────────

  beep(type = 'tick') {
    if (!this.enabled || !this.beepEnabled || !this.audioCtx) return;
    if (this.audioCtx.state === 'suspended') this.audioCtx.resume();

    const ctx = this.audioCtx;
    const now = ctx.currentTime;

    const configs = {
      start:          { freq: [440, 550, 660], dur: 0.12, vol: 0.3, type: 'sine' },
      phase_change:   { freq: [330, 440],       dur: 0.15, vol: 0.4, type: 'sine' },
      milestone_low:  { freq: [523],            dur: 0.15, vol: 0.35, type: 'sine' },
      milestone_mid:  { freq: [523, 659],       dur: 0.15, vol: 0.4,  type: 'sine' },
      milestone_high: { freq: [523, 659, 784],  dur: 0.2,  vol: 0.5,  type: 'sine' },
      warning:        { freq: [220, 220],        dur: 0.3,  vol: 0.6,  type: 'sawtooth' },
      breath_in:      { freq: [330],             dur: 0.1,  vol: 0.2,  type: 'sine' },
      breath_out:     { freq: [220],             dur: 0.1,  vol: 0.2,  type: 'sine' },
      breath_hold:    { freq: [280],             dur: 0.08, vol: 0.15, type: 'sine' },
      tick:           { freq: [800],             dur: 0.05, vol: 0.15, type: 'square' },
    };

    const cfg = configs[type] || configs.tick;

    cfg.freq.forEach((freq, i) => {
      const osc = ctx.createOscillator();
      const gain = ctx.createGain();
      osc.connect(gain);
      gain.connect(ctx.destination);
      osc.type = cfg.type;
      osc.frequency.setValueAtTime(freq, now + i * cfg.dur * 0.8);
      gain.gain.setValueAtTime(0, now + i * cfg.dur * 0.8);
      gain.gain.linearRampToValueAtTime(cfg.vol, now + i * cfg.dur * 0.8 + 0.01);
      gain.gain.exponentialRampToValueAtTime(0.001, now + i * cfg.dur * 0.8 + cfg.dur);
      osc.start(now + i * cfg.dur * 0.8);
      osc.stop(now + i * cfg.dur * 0.8 + cfg.dur + 0.05);
    });
  }

  // ─── Utils ────────────────────────────────────────────────

  get isEnabled() { return this.enabled; }
  get hasTTS() { return this.ttsEnabled; }
  get breathState() { return this.breathPhase; }
}

// Global instance
window.schrothAudio = new SchrothAudioEngine();
