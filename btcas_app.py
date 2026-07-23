import json
import os
import tempfile
import uuid
from datetime import datetime

import streamlit as st

from btcas_pipeline import (
    MODEL_PATH,
    CLASS_NAMES,
    process_video,
    summarize,
    build_api_result,
    save_api_report,
)

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="BTCAS Inspection System",
    page_icon="🚆",
    layout="wide",
    initial_sidebar_state="collapsed"
)

# ─────────────────────────────────────────
# STYLES
# ─────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* ── Apple Design System Tokens ── */
:root {
  --apple-blue: #0066cc;
  --apple-blue-hover: #0071e3;
  --apple-ink: #1d1d1f;
  --apple-ink-secondary: #6e6e73;
  --apple-ink-tertiary: #86868b;
  --apple-bg: #f5f5f7;
  --apple-surface: #ffffff;
  --apple-hairline: #d2d2d7;
  --apple-green: #34c759;
  --apple-orange: #ff9f0a;
  --apple-red: #ff3b30;
  --apple-font: 'SF Pro Display', 'SF Pro Text', system-ui, -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
}

html, body, [class*="css"], .stApp {
  font-family: var(--apple-font) !important;
  background-color: var(--apple-bg) !important;
  color: var(--apple-ink) !important;
}

/* ── Header ── */
.btcas-header {
  border-bottom: 1px solid var(--apple-hairline);
  padding: 20px 0;
  margin-bottom: 32px;
}
.btcas-title {
  font-family: var(--apple-font);
  font-size: 28px;
  font-weight: 600;
  color: var(--apple-ink);
  letter-spacing: -0.374px;
  line-height: 1.14;
  margin: 0;
}
.btcas-subtitle {
  font-family: var(--apple-font);
  font-size: 14px;
  font-weight: 400;
  color: var(--apple-ink-secondary);
  letter-spacing: -0.224px;
  margin-top: 6px;
  text-transform: none;
}

/* ── Status Badges ── */
.badge {
  display: inline-block;
  padding: 4px 12px;
  border-radius: 9999px;
  font-family: var(--apple-font);
  font-size: 12px;
  font-weight: 600;
  letter-spacing: -0.12px;
  border: none;
}
.badge-ok     { background: rgba(52,199,89,0.12); color: #248a3d; }
.badge-warn   { background: rgba(255,159,10,0.14); color: #c93400; }
.badge-danger { background: rgba(255,59,48,0.12); color: #d70015; }
.badge-info   { background: rgba(0,102,204,0.10); color: #0066cc; }

/* ── Coach Card ── */
.coach-card {
  background: var(--apple-surface);
  border: 1px solid var(--apple-hairline);
  border-radius: 18px;
  padding: 24px;
  margin-bottom: 16px;
}
.coach-card.maintenance {
  border-top: 2.5px solid var(--apple-red);
}
.coach-id {
  font-family: var(--apple-font);
  font-size: 21px;
  font-weight: 600;
  color: var(--apple-ink);
  letter-spacing: 0.011em;
  line-height: 1.19;
  margin-bottom: 16px;
}

/* ── Inspection Table ── */
.insp-table { width: 100%; border-collapse: collapse; }
.insp-table th {
  text-align: left;
  color: var(--apple-ink-secondary);
  font-family: var(--apple-font);
  font-weight: 400;
  font-size: 12px;
  letter-spacing: -0.12px;
  padding: 8px 12px 8px 0;
  border-bottom: 1px solid rgba(0,0,0,0.06);
  text-transform: none;
}
.insp-table td {
  padding: 10px 12px 10px 0;
  border-bottom: 1px solid rgba(0,0,0,0.04);
  color: var(--apple-ink);
  font-family: var(--apple-font);
  font-size: 14px;
  letter-spacing: -0.224px;
  vertical-align: middle;
}
.insp-table tr:last-child td { border-bottom: none; }

/* ── Summary Bar ── */
.summary-bar {
  background: var(--apple-surface);
  border: 1px solid var(--apple-hairline);
  border-radius: 18px;
  padding: 28px 32px;
  margin-bottom: 28px;
  display: flex;
  gap: 40px;
  align-items: center;
  flex-wrap: wrap;
}
.summary-stat { text-align: center; }
.summary-num {
  font-family: var(--apple-font);
  font-size: 28px;
  font-weight: 600;
  color: var(--apple-blue);
  display: block;
  letter-spacing: -0.374px;
  line-height: 1.14;
}
.summary-label {
  font-family: var(--apple-font);
  font-size: 12px;
  font-weight: 400;
  color: var(--apple-ink-secondary);
  letter-spacing: -0.12px;
  text-transform: none;
  margin-top: 4px;
  display: block;
}

/* ── Server Status ── */
.server-ok {
  color: var(--apple-green);
  font-family: var(--apple-font);
  font-size: 12px;
  letter-spacing: -0.12px;
}
.server-down {
  color: var(--apple-red);
  font-family: var(--apple-font);
  font-size: 12px;
  letter-spacing: -0.12px;
}

/* ── Upload Label ── */
.upload-label {
  font-size: 12px;
  color: var(--apple-ink-secondary);
  letter-spacing: -0.12px;
  margin-bottom: 6px;
  display: block;
}

/* ── Section Divider ── */
.cam-divider {
  font-family: var(--apple-font);
  font-size: 14px;
  font-weight: 600;
  color: var(--apple-ink);
  letter-spacing: -0.224px;
  line-height: 1.29;
  border-bottom: 1px solid var(--apple-hairline);
  padding-bottom: 8px;
  margin: 24px 0 16px 0;
  display: block;
  text-transform: none;
}

/* ── Notes ── */
.limitation-note {
  background: var(--apple-surface);
  border: 1px solid var(--apple-hairline);
  border-radius: 12px;
  padding: 16px 20px;
  font-family: var(--apple-font);
  font-size: 12px;
  color: var(--apple-ink-secondary);
  letter-spacing: -0.12px;
  line-height: 1.6;
  margin-top: 12px;
}
.crop-label {
  font-size: 12px;
  color: var(--apple-ink-tertiary);
  margin-top: 8px;
  font-family: var(--apple-font);
  letter-spacing: -0.12px;
}

/* ── Streamlit Widget Overrides ── */
div.stButton > button[kind="primary"],
div.stButton > button[data-testid="baseButton-primary"] {
  background-color: var(--apple-blue) !important;
  color: #ffffff !important;
  border: none !important;
  border-radius: 9999px !important;
  font-family: var(--apple-font) !important;
  font-size: 17px !important;
  font-weight: 400 !important;
  letter-spacing: -0.374px !important;
  padding: 11px 24px !important;
  transition: transform 0.15s ease, background-color 0.15s ease !important;
}
div.stButton > button[kind="primary"]:hover,
div.stButton > button[data-testid="baseButton-primary"]:hover {
  background-color: var(--apple-blue-hover) !important;
}
div.stButton > button[kind="primary"]:active,
div.stButton > button[data-testid="baseButton-primary"]:active {
  transform: scale(0.97) !important;
}

div.stDownloadButton > button {
  background-color: var(--apple-surface) !important;
  color: var(--apple-blue) !important;
  border: 1px solid var(--apple-blue) !important;
  border-radius: 9999px !important;
  font-family: var(--apple-font) !important;
  font-size: 14px !important;
  font-weight: 400 !important;
  letter-spacing: -0.224px !important;
  padding: 11px 24px !important;
  transition: transform 0.15s ease !important;
}
div.stDownloadButton > button:hover {
  background-color: rgba(0,102,204,0.04) !important;
}
div.stDownloadButton > button:active {
  transform: scale(0.97) !important;
}

[data-testid="stFileUploader"] {
  background: var(--apple-surface);
  border: 1px solid var(--apple-hairline);
  border-radius: 18px;
  padding: 20px;
}
[data-testid="stFileUploader"] section { border: none !important; }

.stProgress > div > div { background-color: var(--apple-blue) !important; }
.stAlert { border-radius: 12px; font-family: var(--apple-font); }
</style>
""", unsafe_allow_html=True)


# ─────────────────────────────────────────
# HELPERS
# ─────────────────────────────────────────
def badge(value, field):
    """Return colored badge HTML based on field + value."""
    good_values = {
        "pipe_status":             "Connected",
        "pipe_support_status":     "Present",
        "surface_status":          "Clean",
        "coupler_status":          "Detected",
        "connecting_wire_status":  "Detected",
        "stairs_status":           "Detected",
        "maintenance_status":      "Normal",
    }
    warn_values = {
        "surface_status":       "Not Clean",
        "maintenance_status":   "Maintenance Required",
    }

    if field in good_values:
        if value == good_values[field]:
            cls = "badge-ok"
        elif field in warn_values and value == warn_values[field]:
            cls = "badge-warn"
        else:
            cls = "badge-danger"
    else:
        cls = "badge-info"

    return f'<span class="badge {cls}">{value}</span>'


def conf_bar(conf):
    """Return confidence as colored mono text."""
    color = "#34c759" if conf >= 0.65 else "#ff9f0a" if conf >= 0.45 else "#ff3b30"
    return f'<span style="color:{color};font-weight:600">{conf:.2f}</span>'


def render_coach_card(coach):
    needs_maint = coach["maintenance_status"] == "Maintenance Required"
    card_class  = "coach-card maintenance" if needs_maint else "coach-card"
    cam_icon    = "◀" if coach["camera_side"] == "LEFT" else "▶"
    ts         = coach.get("timestamp_sec", "—")

    rows = [
        ("Tank ID",             f'<span class="badge badge-info">{coach["tank_id"]}</span>'),
        ("Coach Number",        str(coach.get("coach_number", "—")),                       None),
        ("Camera Side",         f'{cam_icon} {coach["camera_side"]}',                      None),
        ("Timestamp (s)",       str(ts),                                                    None),
        ("Pipe Status",         badge(coach["pipe_status"],         "pipe_status"),         None),
        ("Pipe Support Status", badge(coach["pipe_support_status"], "pipe_support_status"), None),
        ("Surface Status",      badge(coach["surface_status"],      "surface_status"),      None),
        ("Coupler Status",      badge(coach["coupler_status"],      "coupler_status"),      None),
        ("Connecting Wire",     badge(coach["connecting_wire_status"], "connecting_wire_status"), None),
        ("Stairs Status",       badge(coach["stairs_status"],       "stairs_status"),       None),
        ("Detection Confidence",conf_bar(coach["detection_confidence"]),                    None),
        ("Maintenance Status",  badge(coach["maintenance_status"],  "maintenance_status"),  None),
        ("Tank Image Path",     coach.get("tank_image_path") or "—",                        None),
    ]

    rows_html = ""
    for label, value, badge_type in rows:
        if badge_type == "info":
            value_html = f'<span class="badge badge-info">{value}</span>'
        elif badge_type is None and "<span" in str(value):
            value_html = value
        else:
            value_html = str(value)
        rows_html += f"""
        <tr>
            <th>{label}</th>
            <td>{value_html}</td>
        </tr>"""

    html = f"""
    <div class="{card_class}">
        <div class="coach-id">TANK {coach['tank_id']}</div>
        <table class="insp-table">
            <tbody>{rows_html}</tbody>
        </table>
    </div>
    """
    return html


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="btcas-header">
    <p class="btcas-title">BTCAS — Bio Toilet Camera Annotation System</p>
    <p class="btcas-subtitle">Central Railway · LHB Coach Inspection · Standalone Streamlit</p>
</div>
""", unsafe_allow_html=True)


st.markdown(f'<p class="server-ok">● Running locally &nbsp;|&nbsp; Model: {MODEL_PATH}</p>',
            unsafe_allow_html=True)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# VIDEO UPLOAD
# ─────────────────────────────────────────
col1, col2 = st.columns(2)

with col1:
    st.markdown('<span class="cam-divider">◀ LEFT CAMERA FEED</span>', unsafe_allow_html=True)
    left_video = st.file_uploader(
        "Upload left camera video",
        type=["mp4", "avi", "mov"],
        key="left",
        label_visibility="collapsed"
    )
    if left_video:
        st.video(left_video)

with col2:
    st.markdown('<span class="cam-divider">▶ RIGHT CAMERA FEED</span>', unsafe_allow_html=True)
    right_video = st.file_uploader(
        "Upload right camera video",
        type=["mp4", "avi", "mov"],
        key="right",
        label_visibility="collapsed"
    )
    if right_video:
        st.video(right_video)

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# INSPECT BUTTON & SETTINGS
# ─────────────────────────────────────────
with st.expander("⚙️ Performance & Frame Sampling Settings", expanded=False):
    frame_skip_val = st.select_slider(
        "Frame Skip Rate",
        options=[5, 10, 15, 20, 25, 30],
        value=15,
        help="Higher frame skip processes fewer frames per video, drastically speeding up inspection.\n"
             "• Skip 5: ~2,100 frames for 7-min video\n"
             "• Skip 15: ~700 frames for 7-min video (Default — 3x Faster)\n"
             "• Skip 25: ~420 frames for 7-min video (5x Faster)"
    )

can_inspect = left_video and right_video

if not (left_video and right_video):
    st.info("Upload both left and right camera videos to begin inspection.")

if can_inspect:
    if st.button("▶  RUN INSPECTION", use_container_width=True, type="primary"):

        with st.spinner("Processing videos — this may take 1–5 minutes depending on video length..."):
            try:
                job_id = str(uuid.uuid4())[:8]

                with tempfile.TemporaryDirectory(prefix=f"btcas_{job_id}_") as tmp_dir:
                    left_path = os.path.join(tmp_dir, "left.mp4")
                    right_path = os.path.join(tmp_dir, "right.mp4")

                    with open(left_path, "wb") as left_file:
                        left_file.write(left_video.getvalue())

                    with open(right_path, "wb") as right_file:
                        right_file.write(right_video.getvalue())

                    left_reports = process_video(left_path, camera_side="LEFT", frame_skip=frame_skip_val)
                    right_reports = process_video(right_path, camera_side="RIGHT", frame_skip=frame_skip_val)

                    all_reports = left_reports + right_reports
                    all_reports.sort(key=lambda x: (x["camera_side"], x["tank_id"]))

                    left_summary = summarize(left_reports, "LEFT", "L")
                    right_summary = summarize(right_reports, "RIGHT", "R")
                    import math
                    total_coaches_counted = math.ceil((len(left_reports) + len(right_reports)) / 4.0) if (len(left_reports) + len(right_reports)) > 0 else 0

                    ts_now = datetime.now().isoformat()
                    inspection_run_id = f"INSP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{job_id}"

                    result = {
                        "job_id": job_id,
                        "inspection_run_id": inspection_run_id,
                        "timestamp": ts_now,
                        "status": "COMPLETE",
                        "total_tanks": len(all_reports),
                        "left_tanks": len(left_reports),
                        "right_tanks": len(right_reports),
                        "total_coaches_counted": total_coaches_counted,
                        "left_coach_count": left_summary["coach_count"],
                        "right_coach_count": right_summary["coach_count"],
                        "inspection": all_reports,
                        "model_version": "YOLOv8s_v2",
                        "model_classes": CLASS_NAMES,
                        "model_map50_95": 0.459,
                        "known_limitations": {
                            "bio_tank_top_surface_recall": 0.383,
                            "connected_pipe_recall": 0.457,
                            "surface_status_note": "Not Clean = inconclusive, not confirmed dirty",
                            "coach_counting_rule": "Total tanks (left + right) / 4",
                            "maintenance_rule": "Normal = bio tank top surface detected (surface clean)",
                        },
                    }

                    # Auto-save API contract JSON
                    api_result = build_api_result(
                        inspection_run_id=inspection_run_id,
                        processing_timestamp=ts_now,
                        left_reports=left_reports,
                        right_reports=right_reports,
                    )
                    save_api_report(api_result, output_dir=".")

                    st.session_state["result"] = result

            except Exception as e:
                st.error(f"Error: {str(e)}")

# ─────────────────────────────────────────
# RESULTS
# ─────────────────────────────────────────
if "result" in st.session_state:
    result = st.session_state["result"]
    inspection = result.get("inspection", [])

    # ── Summary Bar ──
    total         = result.get("total_tanks", 0)
    left_c        = result.get("left_tanks",  0)
    right_c       = result.get("right_tanks", 0)
    coach_count   = result.get("total_coaches_counted", 0)
    l_coach_count = result.get("left_coach_count",  0)
    r_coach_count = result.get("right_coach_count", 0)
    maint         = sum(1 for c in inspection if c["maintenance_status"] == "Maintenance Required")
    normal        = total - maint

    st.markdown(f"""
    <div class="summary-bar">
        <div class="summary-stat">
            <span class="summary-num">{total}</span>
            <span class="summary-label">Total Tanks</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#0066cc">{coach_count}</span>
            <span class="summary-label">Coaches (L:{l_coach_count}·R:{r_coach_count})</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#6e6e73">{left_c}</span>
            <span class="summary-label">Left Tanks</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#6e6e73">{right_c}</span>
            <span class="summary-label">Right Tanks</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#34c759">{normal}</span>
            <span class="summary-label">Normal</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#ff3b30">{maint}</span>
            <span class="summary-label">Maintenance Required</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#86868b;font-size:14px">{result.get('job_id','—')}</span>
            <span class="summary-label">Job ID</span>
        </div>
    </div>
    """, unsafe_allow_html=True)

    # ── Per-coach cards in two columns ──
    left_coaches  = [c for c in inspection if c["camera_side"] == "LEFT"]
    right_coaches = [c for c in inspection if c["camera_side"] == "RIGHT"]

    col_l, col_r = st.columns(2)

    with col_l:
        st.markdown('<span class="cam-divider">◀ LEFT CAMERA — RESULTS</span>', unsafe_allow_html=True)
        if left_coaches:
            for coach in left_coaches:
                st.markdown(render_coach_card(coach), unsafe_allow_html=True)

                ann = coach.get("annotated_frame_path")
                if ann and os.path.exists(ann):
                    st.image(ann, caption=f"Full Frame — {coach['tank_id']}", width=800)
        else:
            st.markdown('<p style="color:#86868b;font-size:14px">No tanks detected on left feed.</p>',
                        unsafe_allow_html=True)

    with col_r:
        st.markdown('<span class="cam-divider">▶ RIGHT CAMERA — RESULTS</span>', unsafe_allow_html=True)
        if right_coaches:
            for coach in right_coaches:
                st.markdown(render_coach_card(coach), unsafe_allow_html=True)

                ann = coach.get("annotated_frame_path")
                if ann and os.path.exists(ann):
                    st.image(ann, caption=f"Full Frame — {coach['tank_id']}", width=800)
        else:
            st.markdown('<p style="color:#86868b;font-size:14px">No tanks detected on right feed.</p>',
                        unsafe_allow_html=True)

    # ── Model Info Note ──
    cls_str = ", ".join(f"{i}:{n}" for i, n in result.get("model_classes", CLASS_NAMES).items())
    run_id = result.get("inspection_run_id", result.get('job_id','—'))
    st.markdown(f"""
    <div class="limitation-note">
        ⚠ MODEL {result.get('model_version','YOLOv8s_v2')} — 7 CLASSES:<br>
        &nbsp;&nbsp;· {cls_str}<br>
        &nbsp;&nbsp;· Status: {result.get('status','—')} &nbsp;|&nbsp; Run: {run_id}<br>
        &nbsp;&nbsp;· Coach count: {result.get('total_coaches_counted',0)} (total tanks left+right / 4)<br>
        &nbsp;&nbsp;· Maintenance rule: Normal = bio tank top surface detected (surface clean → no maintenance)<br>
        &nbsp;&nbsp;· Surface recall ~38%, Connected Pipe recall ~45% — "Not Connected" may be missed detection, not confirmed absence
    </div>
    """, unsafe_allow_html=True)

    # ── Download Buttons ──
    st.markdown("<br>", unsafe_allow_html=True)
    col_d1, col_d2 = st.columns(2)
    with col_d1:
        st.download_button(
            label="⬇  Download Full Report (JSON)",
            data=json.dumps(result, indent=2),
            file_name=f"btcas_full_{result.get('job_id','export')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    with col_d2:
        api_result = build_api_result(
            inspection_run_id=run_id,
            processing_timestamp=result.get("timestamp", ""),
            left_reports=[c for c in inspection if c["camera_side"] == "LEFT"],
            right_reports=[c for c in inspection if c["camera_side"] == "RIGHT"],
        )
        st.download_button(
            label="⬇  Download API Contract (JSON)",
            data=json.dumps(api_result, indent=2),
            file_name=f"btcas_api_report_{result.get('job_id','export')}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
