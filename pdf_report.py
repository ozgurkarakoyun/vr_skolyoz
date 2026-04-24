"""
pdf_report.py
────────────────────────────────────────────────────────────────
Schroth VR — Klinik Seans PDF Raporu
reportlab ile A4 formatlı, profesyonel klinik rapor üretir.

Rapor İçeriği:
  - Başlık: Hasta adı, tarih, terapist
  - Hasta bilgileri: yaş, cinsiyet, tanı, eğri tipi, Cobb°, Risser
  - Seans özeti: süre, tekrar, ort. skor, en iyi skor
  - Skolyoz açıları: T / TL / L (şiddet renkleriyle)
  - Schroth postür metrikleri: omuz/kalça/gövde/kayma
  - Faz logu tablosu (Elongasyon, Derotasyon, RAB, Stabilizasyon)
  - Skor trend mini grafiği (son 10 seans)
  - Klinik notlar
  - Alt bilgi: Dr. Özgür Karakoyun imzası
"""

import io
import os
from datetime import datetime

from reportlab.lib.pagesizes import A4
from reportlab.lib.units import mm, cm
from reportlab.lib import colors
from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
from reportlab.lib.enums import TA_CENTER, TA_LEFT, TA_RIGHT
from reportlab.platypus import (
    SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle,
    HRFlowable, KeepTogether
)
from reportlab.graphics.shapes import Drawing, Rect, Line, String, Polygon
from reportlab.graphics import renderPDF
from reportlab.graphics.charts.lineplots import LinePlot
from reportlab.graphics.widgets.markers import makeMarker

# ─── Renk Paleti ─────────────────────────────────────────────
BG_DARK    = colors.HexColor('#050a0f')
ACCENT     = colors.HexColor('#00e5ff')
PURPLE     = colors.HexColor('#7b2fff')
SUCCESS    = colors.HexColor('#2ed573')
WARN       = colors.HexColor('#ffa502')
DANGER     = colors.HexColor('#ff4757')
TEXT_MAIN  = colors.HexColor('#1a2a35')
TEXT_MUTED = colors.HexColor('#5a7a8a')
SURFACE    = colors.HexColor('#f0f7fa')
BORDER     = colors.HexColor('#d0e8f0')

def angle_color(angle):
    if (angle or 0) > 40: return DANGER
    if (angle or 0) > 20: return WARN
    return SUCCESS

def score_color(score):
    if (score or 0) >= 80: return SUCCESS
    if (score or 0) >= 55: return WARN
    return DANGER

def angle_label(angle):
    if (angle or 0) > 40: return 'KRİTİK'
    if (angle or 0) > 20: return 'ORTA'
    return 'NORMAL'

# ─── Stil Tanımları ──────────────────────────────────────────

def make_styles():
    base = getSampleStyleSheet()
    styles = {}

    styles['title'] = ParagraphStyle('title',
        fontName='Helvetica-Bold', fontSize=20,
        textColor=TEXT_MAIN, spaceAfter=2*mm, leading=24)

    styles['subtitle'] = ParagraphStyle('subtitle',
        fontName='Helvetica', fontSize=11,
        textColor=TEXT_MUTED, spaceAfter=4*mm)

    styles['section'] = ParagraphStyle('section',
        fontName='Helvetica-Bold', fontSize=9,
        textColor=TEXT_MUTED, spaceBefore=4*mm, spaceAfter=3*mm,
        letterSpacing=1.5, textTransform='uppercase')

    styles['body'] = ParagraphStyle('body',
        fontName='Helvetica', fontSize=10,
        textColor=TEXT_MAIN, leading=14, spaceAfter=2*mm)

    styles['small'] = ParagraphStyle('small',
        fontName='Helvetica', fontSize=8,
        textColor=TEXT_MUTED, leading=12)

    styles['footer'] = ParagraphStyle('footer',
        fontName='Helvetica', fontSize=8,
        textColor=TEXT_MUTED, alignment=TA_CENTER)

    return styles

# ─── Yardımcı Bileşenler ─────────────────────────────────────

def header_block(patient, session_data, terapist_name='Dr. Özgür Karakoyun'):
    """Başlık bloğu: logo satırı + hasta adı + meta bilgiler"""
    elements = []
    styles = make_styles()

    # Üst bar
    now = datetime.now().strftime('%d.%m.%Y %H:%M')
    header_table = Table([
        [
            Paragraph('<b>SCHROTH VR</b>', ParagraphStyle('logo',
                fontName='Helvetica-Bold', fontSize=14,
                textColor=ACCENT)),
            Paragraph(f'Seans Raporu — {now}', ParagraphStyle('h_date',
                fontName='Helvetica', fontSize=9,
                textColor=TEXT_MUTED, alignment=TA_RIGHT)),
        ]
    ], colWidths=[90*mm, 100*mm])
    header_table.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3*mm),
    ]))
    elements.append(header_table)
    elements.append(HRFlowable(width='100%', thickness=1.5, color=ACCENT, spaceAfter=4*mm))

    # Hasta adı
    name = patient.get('name', '—')
    elements.append(Paragraph(name, styles['title']))

    # Meta satırı
    age = ''
    if patient.get('birth_year'):
        age = f"{datetime.now().year - patient['birth_year']} yaş"
    gender_map = {'K': 'Kadın', 'E': 'Erkek', '—': ''}
    gender = gender_map.get(patient.get('gender','—'), '')
    diagnosis = patient.get('diagnosis', '') or ''

    meta_parts = [x for x in [age, gender, diagnosis] if x]
    elements.append(Paragraph(' · '.join(meta_parts) if meta_parts else '—', styles['subtitle']))

    return elements


def info_grid(patient, session_data, stats):
    """2 kolonlu bilgi ızgarası: sol hasta, sağ seans özeti"""
    styles = make_styles()

    # Sol: Hasta klinik bilgileri
    left_data = [
        ['EĞRİ TİPİ', patient.get('curve_type') or '—'],
        ['COBB AÇISI', f"{patient.get('cobb_angle') or 0}°"],
        ['RİSSER', str(patient.get('risser') or 0)],
        ['TOPLAM SEANS', str(stats.get('total_sessions') or 0)],
    ]

    # Sağ: Bu seans
    dur = session_data.get('duration_sec') or 0
    dur_str = f"{dur//60}dk {dur%60}s"
    right_data = [
        ['SÜRE', dur_str],
        ['TEKRAR', str(session_data.get('rep_count') or 0)],
        ['ORT. SKOR', str(round(session_data.get('avg_score') or 0))],
        ['EN İYİ SKOR', str(round(session_data.get('best_score') or 0))],
    ]

    def mini_table(data, accent_color):
        rows = []
        for label, val in data:
            rows.append([
                Paragraph(label, ParagraphStyle('tbl_lbl',
                    fontName='Helvetica', fontSize=7.5,
                    textColor=TEXT_MUTED, leading=10)),
                Paragraph(f'<b>{val}</b>', ParagraphStyle('tbl_val',
                    fontName='Helvetica-Bold', fontSize=10,
                    textColor=TEXT_MAIN, leading=13)),
            ])
        t = Table(rows, colWidths=[28*mm, 32*mm])
        t.setStyle(TableStyle([
            ('BACKGROUND', (0,0), (-1,-1), SURFACE),
            ('ROWBACKGROUNDS', (0,0), (-1,-1), [SURFACE, colors.white]),
            ('BOX', (0,0), (-1,-1), 0.5, BORDER),
            ('INNERGRID', (0,0), (-1,-1), 0.3, BORDER),
            ('TOPPADDING', (0,0), (-1,-1), 3*mm),
            ('BOTTOMPADDING', (0,0), (-1,-1), 3*mm),
            ('LEFTPADDING', (0,0), (-1,-1), 3*mm),
            ('RIGHTPADDING', (0,0), (-1,-1), 3*mm),
            ('ROUNDEDCORNERS', [4]),
        ]))
        return t

    outer = Table([
        [
            mini_table(left_data, PURPLE),
            mini_table(right_data, ACCENT),
        ]
    ], colWidths=[65*mm, 65*mm], spaceBefore=0)
    outer.setStyle(TableStyle([
        ('VALIGN', (0,0), (-1,-1), 'TOP'),
        ('LEFTPADDING', (0,0), (-1,-1), 0),
        ('RIGHTPADDING', (0,0), (-1,-1), 0),
        ('TOPPADDING', (0,0), (-1,-1), 0),
        ('BOTTOMPADDING', (0,0), (-1,-1), 0),
        ('COLPADDING', (0,0), (0,-1), 3*mm),
    ]))
    return outer


def scol_angles_block(session_data):
    """T / TL / L açı kartları"""
    styles = make_styles()
    elements = []
    elements.append(Paragraph('Skolyoz Açıları', styles['section']))

    t  = session_data.get('avg_thoracic') or 0
    tl = session_data.get('avg_thoracolumbar') or 0
    l  = session_data.get('avg_lumbar') or 0

    def angle_cell(label, val):
        col = angle_color(val)
        lbl = angle_label(val)
        return [
            Paragraph(f'<b>{label}</b>', ParagraphStyle('ac_name',
                fontName='Helvetica-Bold', fontSize=8,
                textColor=TEXT_MUTED, leading=10)),
            Paragraph(f'<b>{val:.1f}°</b>', ParagraphStyle('ac_val',
                fontName='Helvetica-Bold', fontSize=18,
                textColor=col, leading=22)),
            Paragraph(lbl, ParagraphStyle('ac_lbl',
                fontName='Helvetica-Bold', fontSize=8,
                textColor=col, leading=10)),
        ]

    cells = [angle_cell('THORACİK', t), angle_cell('T-LUMBAR', tl), angle_cell('LUMBAR', l)]
    rows = [[c[0] for c in cells], [c[1] for c in cells], [c[2] for c in cells]]

    tbl = Table(rows, colWidths=[58*mm, 58*mm, 58*mm])
    tbl.setStyle(TableStyle([
        ('BACKGROUND', (0,0), (-1,-1), SURFACE),
        ('BOX', (0,0), (0,-1), 0.5, BORDER),
        ('BOX', (1,0), (1,-1), 0.5, BORDER),
        ('BOX', (2,0), (2,-1), 0.5, BORDER),
        ('ALIGN', (0,0), (-1,-1), 'CENTER'),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 3*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 3*mm),
        ('COLPADDING', (1,0), (1,-1), 3*mm),
    ]))
    elements.append(tbl)
    return elements


def posture_block(session_data):
    """Schroth postür metrikleri"""
    styles = make_styles()
    elements = []
    elements.append(Paragraph('Schroth Postür Metrikleri', styles['section']))

    metrics = [
        ('Omuz Açısı',  f"{session_data.get('avg_shoulder') or 0:.1f}°",   'Omuz simetrisi'),
        ('Kalça Açısı', f"{session_data.get('avg_hip') or 0:.1f}°",         'Pelvik hizalanma'),
        ('Ort. Skor',   str(round(session_data.get('avg_score') or 0)),       'Postür kalitesi'),
        ('Tekrar',      str(session_data.get('rep_count') or 0),              'Stabilizasyon tamamlanan'),
    ]

    rows = []
    for name, val, desc in metrics:
        rows.append([
            Paragraph(f'<b>{name}</b>', ParagraphStyle('pm_name',
                fontName='Helvetica-Bold', fontSize=9, textColor=TEXT_MAIN)),
            Paragraph(f'<b>{val}</b>', ParagraphStyle('pm_val',
                fontName='Helvetica-Bold', fontSize=12, textColor=ACCENT)),
            Paragraph(desc, ParagraphStyle('pm_desc',
                fontName='Helvetica', fontSize=8, textColor=TEXT_MUTED)),
        ])

    tbl = Table(rows, colWidths=[45*mm, 30*mm, 100*mm])
    tbl.setStyle(TableStyle([
        ('ROWBACKGROUNDS', (0,0), (-1,-1), [SURFACE, colors.white]),
        ('INNERGRID', (0,0), (-1,-1), 0.3, BORDER),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 3*mm),
        ('RIGHTPADDING', (0,0), (-1,-1), 3*mm),
    ]))
    elements.append(tbl)
    return elements


def phase_log_block(session_data):
    """Faz logu tablosu"""
    styles = make_styles()
    elements = []
    phase_log = session_data.get('phase_log') or []
    if not phase_log:
        return elements

    elements.append(Paragraph('Egzersiz Faz Logu', styles['section']))

    phase_colors = {
        'elongation':    colors.HexColor('#e8f8ff'),
        'derotation':    colors.HexColor('#f3eeff'),
        'rab_breathing': colors.HexColor('#eafff4'),
        'stabilization': colors.HexColor('#fff8ec'),
    }

    header = [
        Paragraph('<b>FAZ</b>', ParagraphStyle('ph_hdr',fontName='Helvetica-Bold',fontSize=8,textColor=TEXT_MUTED)),
        Paragraph('<b>SÜRE</b>', ParagraphStyle('ph_hdr',fontName='Helvetica-Bold',fontSize=8,textColor=TEXT_MUTED,alignment=TA_RIGHT)),
        Paragraph('<b>ORT. SKOR</b>', ParagraphStyle('ph_hdr',fontName='Helvetica-Bold',fontSize=8,textColor=TEXT_MUTED,alignment=TA_RIGHT)),
    ]
    rows = [header]
    row_colors = []

    for p in phase_log[:12]:  # Max 12 faz satırı
        col = phase_colors.get(p.get('id',''), SURFACE)
        rows.append([
            Paragraph(p.get('name') or p.get('id','—'),
                ParagraphStyle('ph_name',fontName='Helvetica',fontSize=9,textColor=TEXT_MAIN)),
            Paragraph(f"{p.get('duration',0)}s",
                ParagraphStyle('ph_dur',fontName='Helvetica-Bold',fontSize=9,textColor=TEXT_MUTED,alignment=TA_RIGHT)),
            Paragraph(str(p.get('avgScore') or p.get('avg_score',0)),
                ParagraphStyle('ph_score',fontName='Helvetica-Bold',fontSize=9,textColor=ACCENT,alignment=TA_RIGHT)),
        ])
        row_colors.append(col)

    tbl = Table(rows, colWidths=[100*mm, 30*mm, 44*mm])
    style_cmds = [
        ('BACKGROUND', (0,0), (-1,0), colors.HexColor('#e0eef4')),
        ('BOX', (0,0), (-1,-1), 0.5, BORDER),
        ('INNERGRID', (0,0), (-1,-1), 0.3, BORDER),
        ('VALIGN', (0,0), (-1,-1), 'MIDDLE'),
        ('TOPPADDING', (0,0), (-1,-1), 2.5*mm),
        ('BOTTOMPADDING', (0,0), (-1,-1), 2.5*mm),
        ('LEFTPADDING', (0,0), (-1,-1), 3*mm),
        ('RIGHTPADDING', (0,0), (-1,-1), 3*mm),
    ]
    for i, col in enumerate(row_colors):
        style_cmds.append(('BACKGROUND', (0, i+1), (-1, i+1), col))
    tbl.setStyle(TableStyle(style_cmds))
    elements.append(tbl)
    return elements


def trend_chart_block(recent_sessions):
    """Son 10 seans skor trend grafiği"""
    styles = make_styles()
    elements = []
    if len(recent_sessions) < 2:
        return elements

    elements.append(Paragraph('Skor Trendi (Son Seanslar)', styles['section']))

    scores = [s.get('avg_score') or 0 for s in reversed(recent_sessions[:10])]
    n = len(scores)
    W, H = 174*mm, 35*mm

    drawing = Drawing(W, H)

    # Arka plan
    drawing.add(Rect(0, 0, W, H, fillColor=SURFACE, strokeColor=BORDER, strokeWidth=0.5))

    # Izgaralar
    for i in range(0, 101, 25):
        y = (i/100) * (H - 8*mm) + 4*mm
        drawing.add(Line(8*mm, y, W-4*mm, y,
                         strokeColor=BORDER, strokeWidth=0.3))
        drawing.add(String(1*mm, y-2*mm, str(i),
                           fontSize=6, fillColor=TEXT_MUTED))

    # Veri çizgisi
    if n >= 2:
        x_step = (W - 12*mm) / (n - 1)
        pts = []
        for i, s in enumerate(scores):
            x = 8*mm + i * x_step
            y = (s/100) * (H - 8*mm) + 4*mm
            pts.append((x, y))

        for i in range(len(pts)-1):
            drawing.add(Line(pts[i][0], pts[i][1], pts[i+1][0], pts[i+1][1],
                             strokeColor=ACCENT, strokeWidth=1.5))

        for x, y in pts:
            s_val = scores[pts.index((x,y))]
            dot_color = score_color(s_val)
            drawing.add(Rect(x-2*mm, y-2*mm, 4*mm, 4*mm,
                             fillColor=dot_color, strokeColor=colors.white, strokeWidth=0.5))

    elements.append(drawing)
    return elements


def notes_block(patient, session_data):
    """Klinik notlar"""
    styles = make_styles()
    elements = []
    notes = (patient.get('notes') or '') + '\n' + (session_data.get('notes') or '')
    notes = notes.strip()
    if not notes:
        return elements

    elements.append(Paragraph('Klinik Notlar', styles['section']))
    elements.append(Paragraph(notes.replace('\n','<br/>'), styles['body']))
    return elements


def footer_block(terapist='Dr. Özgür Karakoyun', title='Ortopedi ve Travmatoloji Uzmanı'):
    """Alt bilgi"""
    styles = make_styles()
    return [
        HRFlowable(width='100%', thickness=0.5, color=BORDER, spaceBefore=6*mm, spaceAfter=3*mm),
        Table([
            [
                Paragraph(f'<b>{terapist}</b><br/>{title}',
                    ParagraphStyle('footer_l', fontName='Helvetica',
                        fontSize=8, textColor=TEXT_MUTED, leading=11)),
                Paragraph(f'Schroth VR Sistemi<br/>{datetime.now().strftime("%d.%m.%Y")}',
                    ParagraphStyle('footer_r', fontName='Helvetica',
                        fontSize=8, textColor=TEXT_MUTED, leading=11,
                        alignment=TA_RIGHT)),
            ]
        ], colWidths=[87*mm, 87*mm]),
        Spacer(1, 2*mm),
        Paragraph('Bu rapor Schroth VR sistemi tarafından otomatik oluşturulmuştur. Klinik karar için terapist değerlendirmesi esastır.',
            ParagraphStyle('disclaimer', fontName='Helvetica', fontSize=7,
                textColor=colors.HexColor('#aaaaaa'), alignment=TA_CENTER)),
    ]

# ─── Ana PDF Üretici ─────────────────────────────────────────

def generate_pdf(patient: dict, session_data: dict, stats: dict,
                 recent_sessions: list = None,
                 terapist: str = 'Dr. Özgür Karakoyun') -> bytes:
    """
    PDF bayt dizisi döndürür.
    Doğrudan Flask response veya dosyaya yazılabilir.
    """
    buf = io.BytesIO()
    doc = SimpleDocTemplate(
        buf, pagesize=A4,
        leftMargin=18*mm, rightMargin=18*mm,
        topMargin=15*mm, bottomMargin=15*mm,
        title=f"Schroth VR — {patient.get('name','Hasta')}",
        author=terapist,
        subject='Schroth Seans Raporu',
    )

    story = []
    styles = make_styles()

    # ── Başlık
    story.extend(header_block(patient, session_data, terapist))
    story.append(Spacer(1, 3*mm))

    # ── Bilgi ızgarası
    story.append(Paragraph('Hasta & Seans Bilgileri', styles['section']))
    story.append(info_grid(patient, session_data, stats))
    story.append(Spacer(1, 5*mm))

    # ── Skolyoz açıları
    story.extend(scol_angles_block(session_data))
    story.append(Spacer(1, 4*mm))

    # ── Postür metrikleri
    story.extend(posture_block(session_data))
    story.append(Spacer(1, 4*mm))

    # ── Skor trendi
    if recent_sessions and len(recent_sessions) >= 2:
        story.extend(trend_chart_block(recent_sessions))
        story.append(Spacer(1, 4*mm))

    # ── Faz logu
    story.extend(phase_log_block(session_data))
    story.append(Spacer(1, 4*mm))

    # ── Notlar
    story.extend(notes_block(patient, session_data))

    # ── Alt bilgi
    story.extend(footer_block(terapist))

    doc.build(story)
    return buf.getvalue()
