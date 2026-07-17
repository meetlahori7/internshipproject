"""
BTCAS Test App — Video + Photo inspection
Reuses btcas_pipeline.py directly. Drop in same folder, run:
    streamlit run test_app.py
"""

import json
import os
import tempfile
import uuid
from datetime import datetime

import cv2
import numpy as np
import streamlit as st
from PIL import Image

# Reuse GLM's pipeline — no changes to it
from btcas_pipeline import (
    MODEL_PATH,
    CLASS_NAMES,
    CLASS_INDEX,
    CONF_THRESHOLD,
    IOU_THRESHOLD,
    CROPPED_TANKS_DIR,
    process_video,
    summarize,
    build_api_result,
    save_api_report,
    _load_model,
    _run_inference,
)

CLS_TANK    = CLASS_INDEX["bio tank front"]
CLS_SURFACE = CLASS_INDEX["bio tank top surface"]
CLS_PIPE    = CLASS_INDEX["connected pipe"]
CLS_SUPPORT = CLASS_INDEX["pipesupport"]

CLASS_COLORS_BGR = {
    "Stairs":               (180, 255, 0  ),
    "bio tank front":       (0,   140, 255),
    "cbc_coupler":          (255, 180, 0  ),
    "bio tank top surface": (80,  255, 0  ),
    "connected pipe":       (60,  60,  255),
    "connecting_wire":      (0,   220, 255),
    "pipesupport":          (255, 0,   180),
}

# ─────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────
st.set_page_config(
    page_title="BTCAS Test — Video + Photo",
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
.btcas-header { border-bottom: 2px solid #23252a; padding-bottom: 12px; margin-bottom: 28px; }
.btcas-title  { font-family:'Inter',sans-serif; font-size:22px; font-weight:600; color:#5e6ad2; letter-spacing:-0.4px; margin:0; }
.btcas-subtitle { font-size:12px; color:#8a8f98; letter-spacing:0.08em; text-transform:uppercase; margin-top:4px; }
.badge { display:inline-block; padding:3px 10px; border-radius:9999px; font-family:'Inter',sans-serif; font-size:12px; font-weight:500; }
.badge-ok     { background:#162a1a; color:#27a644; border:1px solid #27a644; }
.badge-warn   { background:#2e2210; color:#d4af37; border:1px solid #d4af37; }
.badge-danger { background:#2a1616; color:#f44336; border:1px solid #f44336; }
.badge-info   { background:#141b2c; color:#5e6ad2; border:1px solid #5e6ad2; }
.coach-card   { background:#0f1011; border:1px solid #23252a; border-left:4px solid #5e6ad2; border-radius:12px; padding:24px; margin-bottom:12px; }
.coach-card.maintenance { border-left-color:#f44336; }
.coach-id { font-family:'Inter',sans-serif; font-size:20px; font-weight:600; color:#f7f8f8; margin-bottom:12px; letter-spacing:-0.2px; }
.insp-table { width:100%; border-collapse:collapse; font-size:13px; }
.insp-table th { text-align:left; color:#8a8f98; font-weight:400; font-size:11px; text-transform:uppercase; letter-spacing:0.08em; padding:6px 12px 6px 0; border-bottom:1px solid #23252a; }
.insp-table td { padding:8px 12px 8px 0; border-bottom:1px solid #141516; color:#d0d6e0; font-family:'Inter',sans-serif; font-size:13px; vertical-align:middle; }
.insp-table tr:last-child td { border-bottom:none; }
.summary-bar { background:#0f1011; border:1px solid #23252a; border-radius:12px; padding:24px; margin-bottom:24px; }
.summary-num { font-family:'Inter',sans-serif; font-size:28px; font-weight:600; color:#5e6ad2; display:block; letter-spacing:-1.0px; }
.summary-label { font-size:11px; color:#8a8f98; text-transform:uppercase; letter-spacing:0.08em; }
.cam-divider { font-family:'Inter',sans-serif; font-size:11px; color:#8a8f98; text-transform:uppercase; letter-spacing:0.1em; border-bottom:1px solid #23252a; padding-bottom:6px; margin:20px 0 14px 0; display:block; }
.limitation-note { background:#0f1011; border:1px solid #23252a; border-radius:8px; padding:10px 14px; font-size:11px; color:#8a8f98; font-family:'Inter',sans-serif; margin-top:8px; }
.photo-note { background:#0f1011; border:1px solid #23252a; border-radius:8px; padding:8px 12px; font-size:11px; color:#8a8f98; font-family:'Inter',sans-serif; margin:8px 0; }
.det-pill { display:inline-block; margin:2px 4px 2px 0; padding:2px 8px; border-radius:9999px; font-family:'Inter',sans-serif; font-size:11px; background:#141516; border:1px solid #23252a; color:#d0d6e0; }
</style>
""",unsafe_allow_html=True)


# ─────────────────────────────────────────
# PHOTO-MODE ALGORITHMS
# ─────────────────────────────────────────

def annotate_image(frame_bgr: np.ndarray, boxes) -> np.ndarray:
    """Draw bounding boxes + labels on frame."""
    out  = frame_bgr.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for b in boxes:
        cls_name = CLASS_NAMES.get(b.cls, str(b.cls))
        color    = CLASS_COLORS_BGR.get(cls_name, (200, 200, 200))
        x1,y1,x2,y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(out, (x1,y1), (x2,y2), color, 2)
        label = f"{cls_name} {b.conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(out, (x1, y1-lh-6), (x1+lw+4, y1), color, -1)
        cv2.putText(out, label, (x1+2, y1-3), font, 0.45, (0,0,0), 1)
    return out


def derive_status_photo(boxes) -> dict:
    """
    Derive inspection status from single image.
    Surface absence = "Not Clean" (aligned with API contract).
    Pipe/support absence = "Inconclusive" (cannot confirm absence from 1 frame).
    """
    detected = {CLASS_NAMES[b.cls]: b.conf for b in boxes if b.cls in CLASS_NAMES}

    pipe_status          = "Connected"           if "connected pipe"       in detected else "Inconclusive"
    pipe_support_status  = "Present"             if "pipesupport"          in detected else "Inconclusive"
    surface_status       = "Clean"               if "bio tank top surface" in detected else "Not Clean"
    coupler_status       = "Detected"            if "cbc_coupler"          in detected else "Not Detected"
    wire_status          = "Detected"            if "connecting_wire"      in detected else "Not Detected"
    stairs_status        = "Detected"            if "Stairs"               in detected else "Not Detected"

    anchor = ["bio tank front", "connected pipe", "bio tank top surface", "pipesupport"]
    conf_vals = [detected[c] for c in anchor if c in detected]
    detection_confidence = round(sum(conf_vals)/len(conf_vals), 3) if conf_vals else 0.0

    maintenance_status = (
        "Maintenance Required" if surface_status == "Not Clean" else "Normal"
    )

    return {
        "pipe_status":            pipe_status,
        "pipe_support_status":    pipe_support_status,
        "surface_status":         surface_status,
        "coupler_status":         coupler_status,
        "connecting_wire_status": wire_status,
        "stairs_status":          stairs_status,
        "detection_confidence":   detection_confidence,
        "maintenance_status":     maintenance_status,
        "detected_classes":       detected,
    }


def save_photo_crop(frame_bgr: np.ndarray, boxes, label: str) -> str:
    """Crop best bio_tank_front detection and save."""
    tank_boxes = [b for b in boxes if b.cls == CLS_TANK]
    if not tank_boxes:
        return ""
    best = max(tank_boxes, key=lambda b: b.conf)
    side_dir = os.path.join(CROPPED_TANKS_DIR, "photo")
    os.makedirs(side_dir, exist_ok=True)
    h, w = frame_bgr.shape[:2]
    x1 = max(0, int(best.x1)); y1 = max(0, int(best.y1))
    x2 = min(w, int(best.x2)); y2 = min(h, int(best.y2))
    if x2 <= x1 or y2 <= y1:
        return ""
    crop  = frame_bgr[y1:y2, x1:x2]
    fpath = os.path.join(side_dir, f"{label}.jpg")
    cv2.imwrite(fpath, crop)
    return fpath


# ─────────────────────────────────────────
# SHARED UI HELPERS
# ─────────────────────────────────────────
def badge(value, field):
    good = {
        "pipe_status": "Connected", "pipe_support_status": "Present",
        "surface_status": "Clean", "coupler_status": "Detected",
        "connecting_wire_status": "Detected", "stairs_status": "Detected",
        "maintenance_status": "Normal",
    }
    warn_vals = {"Inconclusive", "Not Clean"}
    danger_vals = {"Not Connected", "Absent", "Maintenance Required", "Not Detected"}

    if field in good and value == good[field]:
        cls = "badge-ok"
    elif value in warn_vals:
        cls = "badge-warn"
    elif value in danger_vals:
        cls = "badge-danger"
    else:
        cls = "badge-info"
    return f'<span class="badge {cls}">{value}</span>'


def conf_color(conf):
    c = "#27a644" if conf >= 0.65 else "#d4af37" if conf >= 0.45 else "#f44336"
    return f'<span style="color:{c};font-family:ui-monospace,monospace">{conf:.3f}</span>'


def render_card(r):
    maint    = r["maintenance_status"] == "Maintenance Required"
    cls      = "coach-card maintenance" if maint else "coach-card"
    cam_icon = "◀" if r.get("camera_side","") == "LEFT" else "▶"

    rows = [
        ("Tank ID",             f'<span class="badge badge-info">{r["tank_id"]}</span>'),
        ("Coach Number",        str(r.get("coach_number","—"))),
        ("Camera Side",         f'{cam_icon} {r.get("camera_side","—")}'),
        ("Timestamp (s)",       str(r.get("timestamp_sec","—"))),
        ("Pipe Status",         badge(r["pipe_status"],                    "pipe_status")),
        ("Pipe Support Status", badge(r["pipe_support_status"],            "pipe_support_status")),
        ("Surface Status",      badge(r["surface_status"],                 "surface_status")),
        ("Coupler Status",      badge(r.get("coupler_status","—"),         "coupler_status")),
        ("Connecting Wire",     badge(r.get("connecting_wire_status","—"), "connecting_wire_status")),
        ("Stairs Status",       badge(r.get("stairs_status","—"),          "stairs_status")),
        ("Detection Confidence",conf_color(r["detection_confidence"])),
        ("Maintenance Status",  badge(r["maintenance_status"],             "maintenance_status")),
        ("Tank Image Path",
         f'<span style="color:#555;font-size:11px">{r.get("tank_image_path") or "—"}</span>'),
    ]

    rows_html = "".join(
        f"<tr><th>{l}</th><td>{v}</td></tr>" for l, v in rows
    )
    return f"""
    <div class="{cls}">
        <div class="coach-id">TANK {r['tank_id']}</div>
        <table class="insp-table"><tbody>{rows_html}</tbody></table>
    </div>"""


# ─────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────
st.markdown("""
<div class="btcas-header">
    <p class="btcas-title">▶ BTCAS TEST — VIDEO + PHOTO INSPECTION</p>
    <p class="btcas-subtitle">Central Railway · LHB Coach · Test Mode · YOLOv8s v2</p>
</div>
""", unsafe_allow_html=True)

try:
    _load_model()
    st.markdown(
        f'<p style="font-family:ui-monospace,monospace;font-size:12px;color:#27a644">'
        f'● MODEL LOADED — {MODEL_PATH} &nbsp;|&nbsp; conf={CONF_THRESHOLD} &nbsp;'
        f'iou={IOU_THRESHOLD} &nbsp;classes={len(CLASS_NAMES)}</p>',
        unsafe_allow_html=True
    )
except Exception as e:
    st.error(f"Model load failed: {e}")
    st.stop()

st.markdown("<br>", unsafe_allow_html=True)

# ─────────────────────────────────────────
# TABS
# ─────────────────────────────────────────
tab_video, tab_photo = st.tabs([
    "🎞  VIDEO MODE  (Full Pipeline)",
    "📷  PHOTO MODE  (Single Frame Debug)"
])


# ═══════════════════════════════════════════════════════
# VIDEO MODE
# ═══════════════════════════════════════════════════════
with tab_video:
    st.markdown('<span class="cam-divider">Upload synchronized left + right camera videos</span>',
                unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**◀ LEFT CAMERA**")
        lv = st.file_uploader("Left video", type=["mp4","avi","mov"],
                              key="vl", label_visibility="collapsed")
        if lv: st.video(lv)

    with c2:
        st.markdown("**▶ RIGHT CAMERA**")
        rv = st.file_uploader("Right video", type=["mp4","avi","mov"],
                              key="vr", label_visibility="collapsed")
        if rv: st.video(rv)

    if not (lv and rv):
        st.info("Upload both videos to run the full inspection pipeline.")

    if lv and rv:
        if st.button("▶  RUN VIDEO INSPECTION", use_container_width=True,
                     type="primary", key="btn_vid"):
            try:
                jid = uuid.uuid4().hex[:8]

                # Progress bar + status text for real-time feedback
                progress_bar = st.progress(0)
                status_text = st.empty()

                def make_progress_cb(label, offset=0.0, weight=0.5):
                    """Create a progress callback for one video (LEFT or RIGHT)."""
                    def cb(current_frame, total_frames):
                        pct = current_frame / max(1, total_frames)
                        overall = offset + pct * weight
                        progress_bar.progress(min(overall, 1.0))
                        status_text.markdown(
                            f'<span style="font-family:ui-monospace,monospace;'
                            f'font-size:12px;color:#5e6ad2">'
                            f'⏳ {label}: frame {current_frame}/{total_frames} '
                            f'({int(pct*100)}%)</span>',
                            unsafe_allow_html=True,
                        )
                    return cb

                with tempfile.TemporaryDirectory(prefix=f"btcas_{jid}_") as tmp:
                    lp = os.path.join(tmp, "left.mp4")
                    rp = os.path.join(tmp, "right.mp4")
                    with open(lp,"wb") as f: f.write(lv.getvalue())
                    with open(rp,"wb") as f: f.write(rv.getvalue())

                    status_text.markdown(
                        '<span style="font-family:ui-monospace,monospace;'
                        'font-size:12px;color:#5e6ad2">'
                        '⏳ Processing LEFT camera...</span>',
                        unsafe_allow_html=True,
                    )
                    lr = process_video(lp, camera_side="LEFT",
                                       progress_callback=make_progress_cb("LEFT", 0.0, 0.5))

                    status_text.markdown(
                        '<span style="font-family:ui-monospace,monospace;'
                        'font-size:12px;color:#5e6ad2">'
                        '⏳ Processing RIGHT camera...</span>',
                        unsafe_allow_html=True,
                    )
                    rr = process_video(rp, camera_side="RIGHT",
                                       progress_callback=make_progress_cb("RIGHT", 0.5, 0.5))

                progress_bar.progress(1.0)
                status_text.markdown(
                    '<span style="font-family:ui-monospace,monospace;'
                    'font-size:12px;color:#27a644">'
                    '✓ Processing complete!</span>',
                    unsafe_allow_html=True,
                )

                ls = summarize(lr, "LEFT", "L")
                rs = summarize(rr, "RIGHT", "R")
                ts_now = datetime.now().isoformat()
                run_id = f"INSP_{datetime.now().strftime('%Y%m%d_%H%M%S')}_{jid}"

                st.session_state["vres"] = {
                    "job_id":                jid,
                    "inspection_run_id":     run_id,
                    "timestamp":             ts_now,
                    "status":                "COMPLETE",
                    "inspection":            lr + rr,
                    "total_tanks":           len(lr)+len(rr),
                    "left_tanks":            len(lr),
                    "right_tanks":           len(rr),
                    "total_coaches_counted": math.ceil((len(lr) + len(rr)) / 4.0) if (len(lr) + len(rr)) > 0 else 0,
                    "left_coach_count":      ls["coach_count"],
                    "right_coach_count":     rs["coach_count"],
                }

                # Auto-save API contract JSON
                api_result = build_api_result(
                    inspection_run_id=run_id,
                    processing_timestamp=ts_now,
                    left_reports=lr,
                    right_reports=rr,
                )
                save_api_report(api_result, output_dir=".")

            except Exception as e:
                st.error(f"Pipeline error: {e}")

    if "vres" in st.session_state:
        res   = st.session_state["vres"]
        insp  = res["inspection"]
        total = res["total_tanks"]
        maint = sum(1 for r in insp if r["maintenance_status"]=="Maintenance Required")
        lcc   = res["left_coach_count"]
        rcc   = res["right_coach_count"]

        st.markdown(f"""
        <div class="summary-bar">
          <div style="display:flex;gap:32px;flex-wrap:wrap;align-items:center">
            <div><span class="summary-num">{total}</span><span class="summary-label">Total Tanks</span></div>
            <div><span class="summary-num" style="color:#8b5cf6">{res['total_coaches_counted']}</span>
                 <span class="summary-label">Coaches (L:{lcc}·R:{rcc})</span></div>
            <div><span class="summary-num" style="color:#8a8f98">{res['left_tanks']}</span>
                 <span class="summary-label">Left Tanks</span></div>
            <div><span class="summary-num" style="color:#8a8f98">{res['right_tanks']}</span>
                 <span class="summary-label">Right Tanks</span></div>
            <div><span class="summary-num" style="color:#27a644">{total-maint}</span>
                 <span class="summary-label">Normal</span></div>
            <div><span class="summary-num" style="color:#f44336">{maint}</span>
                 <span class="summary-label">Maintenance Req.</span></div>
            <div><span class="summary-num" style="font-size:13px;color:#62666d">{res['job_id']}</span>
                 <span class="summary-label">Job ID</span></div>
          </div>
        </div>
        """, unsafe_allow_html=True)

        lc = [r for r in insp if r["camera_side"]=="LEFT"]
        rc = [r for r in insp if r["camera_side"]=="RIGHT"]
        cl, cr = st.columns(2)

        for col, coaches, lbl in [(cl,lc,"◀ LEFT"),(cr,rc,"▶ RIGHT")]:
            with col:
                st.markdown(f'<span class="cam-divider">{lbl} — RESULTS</span>',
                            unsafe_allow_html=True)
                if coaches:
                    for r in coaches:
                        st.markdown(render_card(r), unsafe_allow_html=True)
                        ann = r.get("annotated_frame_path","")
                        if ann and os.path.exists(ann):
                            st.image(ann, caption=f"Full Frame — {r['tank_id']}", width=800)
                else:
                    st.markdown('<p style="color:#444;font-size:13px">No tanks detected.</p>',
                                unsafe_allow_html=True)

        st.markdown(f"""
        <div class="limitation-note">
        ⚠ Status: {res.get('status','—')} &nbsp;|&nbsp; Run: {res.get('inspection_run_id',res.get('job_id','—'))}<br>
        ⚠ Coach counting: Total tanks (left + right) / 4<br>
        ⚠ Maintenance: Normal = bio tank top surface detected (surface clean → no maintenance)<br>
        ⚠ Surface recall ~38% · Pipe recall ~45% — absences may be missed detections not confirmed faults
        </div>""", unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)
        col_d1, col_d2 = st.columns(2)
        with col_d1:
            st.download_button(
                "⬇  Download Full Report (JSON)",
                data=json.dumps(res, indent=2),
                file_name=f"btcas_full_{res['job_id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )
        with col_d2:
            api_result = build_api_result(
                inspection_run_id=res.get("inspection_run_id", res['job_id']),
                processing_timestamp=res.get("timestamp", ""),
                left_reports=lc,
                right_reports=rc,
            )
            st.download_button(
                "⬇  Download API Contract (JSON)",
                data=json.dumps(api_result, indent=2),
                file_name=f"btcas_api_{res['job_id']}_{datetime.now().strftime('%Y%m%d_%H%M')}.json",
                mime="application/json",
                use_container_width=True
            )


# ═══════════════════════════════════════════════════════
# PHOTO MODE
# ═══════════════════════════════════════════════════════
with tab_photo:
    st.markdown('<span class="cam-divider">Upload individual photos — left and/or right camera</span>',
                unsafe_allow_html=True)

    st.markdown("""
    <div class="photo-note">
    ℹ PHOTO MODE is for debugging and single-frame verification only.<br>
    Pipe/support absence = "Inconclusive" — single frame cannot confirm true absence vs occlusion.<br>
    Surface absence = "Not Clean" (aligned with API contract).<br>
    Use VIDEO MODE for production inspection decisions.
    </div>""", unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    c1, c2 = st.columns(2)
    with c1:
        st.markdown("**◀ LEFT CAMERA PHOTOS**")
        lp = st.file_uploader("Left photos", type=["jpg","jpeg","png"],
                              accept_multiple_files=True,
                              key="pl", label_visibility="collapsed")
    with c2:
        st.markdown("**▶ RIGHT CAMERA PHOTOS**")
        rp = st.file_uploader("Right photos", type=["jpg","jpeg","png"],
                              accept_multiple_files=True,
                              key="pr", label_visibility="collapsed")

    if not (lp or rp):
        st.info("Upload at least one photo to begin.")

    if lp or rp:
        if st.button("▶  INSPECT PHOTOS", use_container_width=True,
                     type="primary", key="btn_photo"):

            def proc_photos(photos, camera_side):
                prefix  = "L" if camera_side == "LEFT" else "R"
                results = []
                for i, photo in enumerate(photos, start=1):
                    img_pil = Image.open(photo).convert("RGB")
                    img_bgr = cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)
                    boxes   = _run_inference(img_bgr)
                    ann     = annotate_image(img_bgr, boxes)
                    status  = derive_status_photo(boxes)
                    label   = f"{prefix}{i}"
                    crop    = save_photo_crop(img_bgr, boxes, label)
                    results.append({
                        "tank_id":                label,
                        "bio_tank_number":        label,
                        "camera_side":            camera_side,
                        "timestamp_sec":          "—",
                        "pipe_status":            status["pipe_status"],
                        "pipe_support_status":    status["pipe_support_status"],
                        "surface_status":         status["surface_status"],
                        "coupler_status":         status["coupler_status"],
                        "connecting_wire_status": status["connecting_wire_status"],
                        "stairs_status":          status["stairs_status"],
                        "detection_confidence":   status["detection_confidence"],
                        "maintenance_status":     status["maintenance_status"],
                        "tank_image_path":        crop,
                        "_ann_bgr":               ann,
                        "_detected":              status["detected_classes"],
                    })
                return results

            with st.spinner("Running inference..."):
                lr2 = proc_photos(lp, "LEFT")  if lp else []
                rr2 = proc_photos(rp, "RIGHT") if rp else []
                st.session_state["pres"] = lr2 + rr2

    if "pres" in st.session_state:
        results = st.session_state["pres"]
        if not results:
            st.warning("No results.")
        else:
            lr2 = [r for r in results if r["camera_side"]=="LEFT"]
            rr2 = [r for r in results if r["camera_side"]=="RIGHT"]
            tot = len(results)
            mnt = sum(1 for r in results if r["maintenance_status"]=="Maintenance Required")

            m1,m2,m3,m4 = st.columns(4)
            m1.metric("Total Photos",         tot)
            m2.metric("Left",                 len(lr2))
            m3.metric("Right",                len(rr2))
            m4.metric("Maintenance Required", mnt)
            st.markdown("<br>", unsafe_allow_html=True)

            cl, cr = st.columns(2)
            for col, side_res, lbl in [(cl,lr2,"◀ LEFT"),(cr,rr2,"▶ RIGHT")]:
                with col:
                    st.markdown(f'<span class="cam-divider">{lbl}</span>',
                                unsafe_allow_html=True)
                    for r in side_res:
                        ann_rgb = cv2.cvtColor(r["_ann_bgr"], cv2.COLOR_BGR2RGB)
                        st.image(ann_rgb,
                                 caption=f"Detections — {r['tank_id']}",
                                 use_container_width=True)

                        if r["_detected"]:
                            pills = ""
                            for cn, cf in sorted(r["_detected"].items(), key=lambda x:-x[1]):
                                color = CLASS_COLORS_BGR.get(cn,(150,150,150))
                                hx    = "#{:02x}{:02x}{:02x}".format(color[2],color[1],color[0])
                                pills += (f'<span class="det-pill" style="border-color:{hx};color:{hx}">'
                                          f'{cn} {cf:.2f}</span>')
                            st.markdown(f'<div style="margin-bottom:8px">{pills}</div>',
                                        unsafe_allow_html=True)
                        else:
                            st.markdown(
                                '<p style="color:#444;font-size:12px;margin-bottom:8px">'
                                'No detections above threshold.</p>',
                                unsafe_allow_html=True
                            )

                        st.markdown(render_card(r), unsafe_allow_html=True)

                        ann = r.get("annotated_frame_path","")
                        if ann and os.path.exists(ann):
                            st.image(ann, caption=f"Full Frame — {r['tank_id']}", width=800)

            clean = [{k:v for k,v in r.items() if not k.startswith("_")} for r in results]
            st.markdown("<br>", unsafe_allow_html=True)
            st.download_button(
                "⬇  Download Photo Report (JSON)",
                data=json.dumps({
                    "mode":      "photo",
                    "timestamp": datetime.now().isoformat(),
                    "total":     tot,
                    "inspection": clean,
                }, indent=2),
                file_name=f"btcas_photo_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                mime="application/json",
                use_container_width=True
            )
