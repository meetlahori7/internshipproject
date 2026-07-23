"""
BTCAS Test App — Video + Photo inspection
Reuses btcas_pipeline.py directly. Drop in same folder, run:
    streamlit run test_app.py
"""

import json
import math
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
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;600;700&display=swap');

/* ── Apple Design System Tokens ── */
:root {
  --apple-blue: #0066cc;
  --apple-blue-hover: #0071e3;
  --apple-ink: #1d1d1f;
  --apple-ink-secondary: #6e6e73;
  --apple-ink-tertiary: #86868b;
  --apple-bg: #e4e4e9;
  --apple-surface: #f2f2f6;
  --apple-hairline: #c7c7cc;
  --apple-green: #34c759;
  --apple-orange: #ff9f0a;
  --apple-red: #ff3b30;
  --apple-font: 'SF Pro Display', 'SF Pro Text', system-ui, -apple-system, BlinkMacSystemFont, 'Inter', sans-serif;
}

html, body, [class*="css"], .stApp {
  font-family: var(--apple-font) !important;
  color: var(--apple-ink) !important;
}

.stApp, [data-testid="stAppViewContainer"] {
  background-color: var(--apple-bg) !important;
  background-image: radial-gradient(var(--apple-hairline) 1px, transparent 1px) !important;
  background-size: 24px 24px !important;
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
}
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
.photo-note {
  background: var(--apple-surface);
  border: 1px solid var(--apple-hairline);
  border-radius: 12px;
  padding: 16px 20px;
  font-family: var(--apple-font);
  font-size: 13px;
  color: var(--apple-ink-secondary);
  letter-spacing: -0.12px;
  line-height: 1.5;
  margin: 8px 0;
}
.det-pill {
  display: inline-block;
  margin: 2px 4px 2px 0;
  padding: 3px 10px;
  border-radius: 9999px;
  font-family: var(--apple-font);
  font-size: 12px;
  font-weight: 400;
  background: rgba(0,0,0,0.03);
  border: 1px solid var(--apple-hairline);
  color: var(--apple-ink);
  letter-spacing: -0.12px;
}

/* ── Streamlit Widget Overrides ── */
.stTabs [data-baseweb="tab-list"] {
  gap: 0;
  background: var(--apple-surface);
  border-radius: 12px;
  border: 1px solid var(--apple-hairline);
  padding: 4px;
}
.stTabs [data-baseweb="tab"] {
  font-family: var(--apple-font);
  font-size: 14px;
  font-weight: 400;
  letter-spacing: -0.224px;
  color: var(--apple-ink-secondary);
  border-radius: 8px;
  padding: 8px 20px;
  background: transparent;
}
.stTabs [aria-selected="true"] {
  background: var(--apple-bg) !important;
  color: var(--apple-ink) !important;
  font-weight: 600;
}
.stTabs [data-baseweb="tab-highlight"],
.stTabs [data-baseweb="tab-border"] { display: none; }

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

[data-testid="stMetric"] {
  background: var(--apple-surface);
  border: 1px solid var(--apple-hairline);
  border-radius: 18px;
  padding: 20px;
}
[data-testid="stMetricValue"] {
  font-family: var(--apple-font) !important;
  color: var(--apple-blue) !important;
}
[data-testid="stMetricLabel"] {
  font-family: var(--apple-font) !important;
  color: var(--apple-ink-secondary) !important;
}

.stProgress > div > div { background-color: var(--apple-blue) !important; }
.stAlert { border-radius: 12px; font-family: var(--apple-font); }

/* ── Moving Train Background Animation ── */

/* Faint rail track at viewport bottom */
.stApp::before {
  content: "";
  position: fixed;
  bottom: 12px;
  left: 0;
  width: 100vw;
  height: 2px;
  background: repeating-linear-gradient(
    90deg,
    var(--apple-hairline) 0px,
    var(--apple-hairline) 20px,
    transparent 20px,
    transparent 28px
  );
  opacity: 0.5;
  pointer-events: none;
  z-index: 0;
}

/* Animated train silhouette — locomotive + 5 coaches */
.stApp::after {
  content: "";
  position: fixed;
  bottom: 4px;
  left: 0;
  width: 420px;
  height: 28px;
  background-image: url("data:image/svg+xml,%3Csvg xmlns='http://www.w3.org/2000/svg' viewBox='0 0 420 28' fill='none'%3E%3Cline x1='0' y1='24' x2='420' y2='24' stroke='%23d2d2d7' stroke-width='1'/%3E%3Crect x='2' y='6' width='60' height='16' rx='4' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='1.2'/%3E%3Crect x='6' y='9' width='14' height='9' rx='2' fill='%230066cc' fill-opacity='0.08' stroke='%230066cc' stroke-width='0.8'/%3E%3Crect x='24' y='9' width='14' height='9' rx='2' fill='%230066cc' fill-opacity='0.08' stroke='%230066cc' stroke-width='0.8'/%3E%3Crect x='48' y='2' width='6' height='6' rx='1' fill='%230066cc' fill-opacity='0.12' stroke='%230066cc' stroke-width='0.8'/%3E%3Ccircle cx='51' cy='1' r='2' fill='%230066cc' fill-opacity='0.06'/%3E%3Ccircle cx='55' cy='-1' r='2.5' fill='%230066cc' fill-opacity='0.04'/%3E%3Ccircle cx='14' cy='24' r='3' fill='%230066cc' fill-opacity='0.2' stroke='%230066cc' stroke-width='1'/%3E%3Ccircle cx='30' cy='24' r='3' fill='%230066cc' fill-opacity='0.2' stroke='%230066cc' stroke-width='1'/%3E%3Ccircle cx='50' cy='24' r='3' fill='%230066cc' fill-opacity='0.2' stroke='%230066cc' stroke-width='1'/%3E%3Cline x1='62' y1='16' x2='72' y2='16' stroke='%230066cc' stroke-width='1' stroke-dasharray='2 2'/%3E%3Crect x='72' y='8' width='58' height='14' rx='3' fill='%230066cc' fill-opacity='0.08' stroke='%230066cc' stroke-width='1'/%3E%3Cline x1='84' y1='10' x2='84' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='96' y1='10' x2='96' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='108' y1='10' x2='108' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='118' y1='10' x2='118' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Ccircle cx='84' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Ccircle cx='118' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Cline x1='130' y1='16' x2='140' y2='16' stroke='%230066cc' stroke-width='1' stroke-dasharray='2 2'/%3E%3Crect x='140' y='8' width='58' height='14' rx='3' fill='%230066cc' fill-opacity='0.08' stroke='%230066cc' stroke-width='1'/%3E%3Cline x1='152' y1='10' x2='152' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='164' y1='10' x2='164' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='176' y1='10' x2='176' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='186' y1='10' x2='186' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Ccircle cx='152' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Ccircle cx='186' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Cline x1='198' y1='16' x2='208' y2='16' stroke='%230066cc' stroke-width='1' stroke-dasharray='2 2'/%3E%3Crect x='208' y='8' width='58' height='14' rx='3' fill='%230066cc' fill-opacity='0.08' stroke='%230066cc' stroke-width='1'/%3E%3Cline x1='220' y1='10' x2='220' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='232' y1='10' x2='232' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='244' y1='10' x2='244' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='254' y1='10' x2='254' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Ccircle cx='220' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Ccircle cx='254' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Cline x1='266' y1='16' x2='276' y2='16' stroke='%230066cc' stroke-width='1' stroke-dasharray='2 2'/%3E%3Crect x='276' y='8' width='58' height='14' rx='3' fill='%230066cc' fill-opacity='0.08' stroke='%230066cc' stroke-width='1'/%3E%3Cline x1='288' y1='10' x2='288' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='300' y1='10' x2='300' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='312' y1='10' x2='312' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='322' y1='10' x2='322' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Ccircle cx='288' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Ccircle cx='322' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Cline x1='334' y1='16' x2='344' y2='16' stroke='%230066cc' stroke-width='1' stroke-dasharray='2 2'/%3E%3Crect x='344' y='8' width='58' height='14' rx='3' fill='%230066cc' fill-opacity='0.1' stroke='%230066cc' stroke-width='1'/%3E%3Cline x1='356' y1='10' x2='356' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='368' y1='10' x2='368' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='380' y1='10' x2='380' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Cline x1='390' y1='10' x2='390' y2='20' stroke='%230066cc' stroke-width='0.5' stroke-opacity='0.3'/%3E%3Ccircle cx='356' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Ccircle cx='390' cy='24' r='2.5' fill='%230066cc' fill-opacity='0.15' stroke='%230066cc' stroke-width='0.8'/%3E%3Ccircle cx='404' cy='14' r='2' fill='%23ff3b30' fill-opacity='0.3'/%3E%3C/svg%3E");
  background-repeat: no-repeat;
  background-size: contain;
  opacity: 0.18;
  pointer-events: none;
  z-index: 0;
  animation: train-roll 25s linear infinite;
}

@keyframes train-roll {
  0%   { transform: translateX(-440px); }
  100% { transform: translateX(100vw); }
}
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
    c = "#34c759" if conf >= 0.65 else "#ff9f0a" if conf >= 0.45 else "#ff3b30"
    return f'<span style="color:{c};font-weight:600">{conf:.3f}</span>'


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
    <p class="btcas-title">BTCAS Test — Video + Photo Inspection</p>
    <p class="btcas-subtitle">Central Railway · LHB Coach · Test Mode · YOLOv8s v2</p>
</div>
""", unsafe_allow_html=True)

try:
    _load_model()
    st.markdown(
        f'<p style="font-size:12px;color:#34c759;letter-spacing:-0.12px">'
        f'● Model loaded — {MODEL_PATH} &nbsp;|&nbsp; conf={CONF_THRESHOLD} &nbsp;'
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
        with st.expander("⚙️ Performance & Frame Sampling Settings", expanded=False):
            target_fps_val = st.selectbox(
                "Target Processing FPS (Frames analyzed per second of video)",
                options=[0.5, 1.0, 1.5, 2.0, 5.0],
                index=1,
                key="test_target_fps",
                help="Controls how many frames per second of video duration are sampled for YOLO detection.\n"
                     "• 0.5 FPS: ~60 frames for 2-min video (Fastest)\n"
                     "• 1.0 FPS: ~120 frames for 2-min video (Recommended — 83% fewer frames)\n"
                     "• 1.5 FPS: ~180 frames for 2-min video\n"
                     "• 2.0 FPS: ~240 frames for 2-min video\n"
                     "• 5.0 FPS: ~600 frames for 2-min video (Legacy)",
                format_func=lambda x: {
                    0.5: "0.5 FPS (Fastest — ~60 frames for 2-min video)",
                    1.0: "1.0 FPS (Recommended — ~120 frames for 2-min video)",
                    1.5: "1.5 FPS (Balanced — ~180 frames for 2-min video)",
                    2.0: "2.0 FPS (High Precision — ~240 frames for 2-min video)",
                    5.0: "5.0 FPS (Legacy Heavy — ~600 frames for 2-min video)",
                }.get(x, f"{x} FPS")
            )

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
                            f'<span style="'
                            f'font-size:13px;color:#0066cc;letter-spacing:-0.12px">'
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
                        '<span style="'
                        'font-size:13px;color:#0066cc;letter-spacing:-0.12px">'
                        '⏳ Processing LEFT camera...</span>',
                        unsafe_allow_html=True,
                    )
                    lr = process_video(lp, camera_side="LEFT",
                                       progress_callback=make_progress_cb("LEFT", 0.0, 0.5),
                                       target_fps=target_fps_val)

                    status_text.markdown(
                        '<span style="'
                        'font-size:13px;color:#0066cc;letter-spacing:-0.12px">'
                        '⏳ Processing RIGHT camera...</span>',
                        unsafe_allow_html=True,
                    )
                    rr = process_video(rp, camera_side="RIGHT",
                                       progress_callback=make_progress_cb("RIGHT", 0.5, 0.5),
                                       target_fps=target_fps_val)

                progress_bar.progress(1.0)
                status_text.markdown(
                    '<span style="'
                    'font-size:13px;color:#34c759;letter-spacing:-0.12px">'
                    '✓ Processing complete</span>',
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
            <div><span class="summary-num" style="color:#0066cc">{res['total_coaches_counted']}</span>
                 <span class="summary-label">Coaches (L:{lcc}·R:{rcc})</span></div>
            <div><span class="summary-num" style="color:#6e6e73">{res['left_tanks']}</span>
                 <span class="summary-label">Left Tanks</span></div>
            <div><span class="summary-num" style="color:#6e6e73">{res['right_tanks']}</span>
                 <span class="summary-label">Right Tanks</span></div>
            <div><span class="summary-num" style="color:#34c759">{total-maint}</span>
                 <span class="summary-label">Normal</span></div>
            <div><span class="summary-num" style="color:#ff3b30">{maint}</span>
                 <span class="summary-label">Maintenance Req.</span></div>
            <div><span class="summary-num" style="font-size:14px;color:#86868b">{res['job_id']}</span>
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
                    st.markdown('<p style="color:#86868b;font-size:14px">No tanks detected.</p>',
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
                                '<p style="color:#86868b;font-size:12px;margin-bottom:8px">'
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
