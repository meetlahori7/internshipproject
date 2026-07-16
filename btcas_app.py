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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700&display=swap');

html, body, [class*="css"], .stApp {
    font-family: 'Inter', sans-serif;
    background-color: #010102 !important;
    color: #f7f8f8 !important;
}

/* Header */
.btcas-header {
    border-bottom: 2px solid #23252a;
    padding-bottom: 12px;
    margin-bottom: 28px;
}
.btcas-title {
    font-family: 'Inter', sans-serif;
    font-size: 22px;
    font-weight: 600;
    color: #5e6ad2;
    letter-spacing: -0.4px;
    margin: 0;
}
.btcas-subtitle {
    font-size: 12px;
    color: #8a8f98;
    letter-spacing: 0.08em;
    text-transform: uppercase;
    margin-top: 4px;
}

/* Status badges */
.badge {
    display: inline-block;
    padding: 3px 10px;
    border-radius: 9999px;
    font-family: 'Inter', sans-serif;
    font-size: 12px;
    font-weight: 500;
    letter-spacing: 0.05em;
}
.badge-ok      { background: #162a1a; color: #27a644; border: 1px solid #27a644; }
.badge-warn    { background: #2e2210; color: #d4af37; border: 1px solid #d4af37; }
.badge-danger  { background: #2a1616; color: #f44336; border: 1px solid #f44336; }
.badge-info    { background: #141b2c; color: #5e6ad2; border: 1px solid #5e6ad2; }

/* Coach card */
.coach-card {
    background: #0f1011;
    border: 1px solid #23252a;
    border-left: 4px solid #5e6ad2;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 12px;
}
.coach-card.maintenance {
    border-left-color: #f44336;
}
.coach-id {
    font-family: 'Inter', sans-serif;
    font-size: 20px;
    font-weight: 600;
    color: #f7f8f8;
    margin-bottom: 12px;
    letter-spacing: -0.2px;
}

/* Table */
.insp-table {
    width: 100%;
    border-collapse: collapse;
    font-size: 13px;
}
.insp-table th {
    text-align: left;
    color: #8a8f98;
    font-weight: 400;
    font-size: 11px;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    padding: 6px 12px 6px 0;
    border-bottom: 1px solid #23252a;
}
.insp-table td {
    padding: 8px 12px 8px 0;
    border-bottom: 1px solid #141516;
    color: #d0d6e0;
    font-family: 'Inter', sans-serif;
    font-size: 13px;
    vertical-align: middle;
}
.insp-table tr:last-child td { border-bottom: none; }

/* Summary bar */
.summary-bar {
    background: #0f1011;
    border: 1px solid #23252a;
    border-radius: 12px;
    padding: 24px;
    margin-bottom: 24px;
    display: flex;
    gap: 40px;
    align-items: center;
}
.summary-stat { text-align: center; }
.summary-num {
    font-family: 'Inter', sans-serif;
    font-size: 28px;
    font-weight: 600;
    color: #5e6ad2;
    display: block;
    letter-spacing: -1.0px;
}
.summary-label {
    font-size: 11px;
    color: #8a8f98;
    text-transform: uppercase;
    letter-spacing: 0.08em;
}

/* Server status */
.server-ok   { color: #27a644; font-family: ui-monospace, monospace; font-size: 12px; }
.server-down { color: #f44336; font-family: ui-monospace, monospace; font-size: 12px; }

/* Upload zone */
.upload-label {
    font-size: 11px;
    color: #8a8f98;
    text-transform: uppercase;
    letter-spacing: 0.08em;
    margin-bottom: 6px;
    display: block;
}

/* Divider */
.cam-divider {
    font-family: 'Inter', sans-serif;
    font-size: 11px;
    color: #8a8f98;
    text-transform: uppercase;
    letter-spacing: 0.1em;
    border-bottom: 1px solid #23252a;
    padding-bottom: 6px;
    margin: 20px 0 14px 0;
}

/* Limitation note */
.limitation-note {
    background: #0f1011;
    border: 1px solid #23252a;
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 11px;
    color: #8a8f98;
    font-family: 'Inter', sans-serif;
    margin-top: 8px;
}

/* Crop image */
.crop-label {
    font-size: 11px;
    color: #666;
    margin-top: 8px;
    font-family: 'IBM Plex Mono', monospace;
}
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
    color = "#27a644" if conf >= 0.65 else "#d4af37" if conf >= 0.45 else "#f44336"
    return f'<span style="color:{color};font-family:ui-monospace,monospace">{conf:.2f}</span>'


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
    <p class="btcas-title">▶ BTCAS — BIO TOILET CAMERA ANNOTATION SYSTEM</p>
    <p class="btcas-subtitle">Central Railway · LHB Coach Inspection · Standalone Streamlit Inspection</p>
</div>
""", unsafe_allow_html=True)


st.markdown(f'<p class="server-ok">● RUNNING LOCALLY &nbsp;|&nbsp; Model: {MODEL_PATH}</p>',
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
# INSPECT BUTTON
# ─────────────────────────────────────────
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

                    left_reports = process_video(left_path, camera_side="LEFT")
                    right_reports = process_video(right_path, camera_side="RIGHT")

                    all_reports = left_reports + right_reports
                    all_reports.sort(key=lambda x: (x["camera_side"], x["tank_id"]))

                    left_summary = summarize(left_reports, "LEFT", "L")
                    right_summary = summarize(right_reports, "RIGHT", "R")
                    total_coaches_counted = max(
                        left_summary["coach_count"], right_summary["coach_count"]
                    )

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
                            "coach_counting_rule": "Coupler + Stairs co-occurrence (fallback: tank tracks)",
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
            <span class="summary-num" style="color:#5e6ad2">{coach_count}</span>
            <span class="summary-label">Coaches (L:{l_coach_count}·R:{r_coach_count})</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#8a8f98">{left_c}</span>
            <span class="summary-label">Left Tanks</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#8a8f98">{right_c}</span>
            <span class="summary-label">Right Tanks</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#27a644">{normal}</span>
            <span class="summary-label">Normal</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#f44336">{maint}</span>
            <span class="summary-label">Maintenance Required</span>
        </div>
        <div class="summary-stat">
            <span class="summary-num" style="color:#62666d;font-size:14px">{result.get('job_id','—')}</span>
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
            st.markdown('<p style="color:#444;font-size:13px">No tanks detected on left feed.</p>',
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
            st.markdown('<p style="color:#444;font-size:13px">No tanks detected on right feed.</p>',
                        unsafe_allow_html=True)

    # ── Model Info Note ──
    cls_str = ", ".join(f"{i}:{n}" for i, n in result.get("model_classes", CLASS_NAMES).items())
    run_id = result.get("inspection_run_id", result.get('job_id','—'))
    st.markdown(f"""
    <div class="limitation-note">
        ⚠ MODEL {result.get('model_version','YOLOv8s_v2')} — 7 CLASSES:<br>
        &nbsp;&nbsp;· {cls_str}<br>
        &nbsp;&nbsp;· Status: {result.get('status','—')} &nbsp;|&nbsp; Run: {run_id}<br>
        &nbsp;&nbsp;· Coach count: {result.get('total_coaches_counted',0)} (coupler + stairs co-occurrence, fallback: tank tracks)<br>
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
