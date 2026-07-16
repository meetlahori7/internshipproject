"""
BTCAS Pipeline — 7-class YOLOv8 inference, tank tracking, coach counting,
and maintenance decision algorithm.

Classes (from btcas_yolov8s_v2_best.pt):
    {0: 'Stairs',
     1: 'bio tank front',
     2: 'cbc_coupler',
     3: 'bio tank top surface',
     4: 'connected pipe',
     5: 'connecting_wire',
     6: 'pipesupport'}

Maintenance decision (v3 rule, per info.md):
    Normal  = bio tank top surface detected overlapping the tank.
    Maintenance Required = otherwise.

Coach counting algorithm (couplers OR stairs):
    A coach boundary is confirmed when EITHER a cbc_coupler OR a Stairs
    detection appears in the temporal gap between two tank-group clusters.
    Fallback 1: if no coupler/stairs detected, use timestamp-gap heuristic
    (>5 sec gap between tanks = new coach, tuned for LHB local trains).
    Fallback 2: if still 1 coach but multiple tanks, each tank = 1 coach.
"""

from __future__ import annotations

import json
import os
from dataclasses import dataclass, field
from datetime import datetime
from typing import Optional

import cv2
import numpy as np

from ultralytics import YOLO

# ─────────────────────────────────────────
# CONFIG
# ─────────────────────────────────────────
MODEL_PATH = "btcas_yolov8s_v2_best.pt"

CONF_THRESHOLD = 0.35
IOU_THRESHOLD = 0.35

CONFIRMATION_GAP_FRAMES = 15
TANK_OVERLAP_IOU = 0.3
CHILD_OVERLAP_THRESHOLD = 0.25

INFERENCE_EVERY_N = 5
RESIZE_WIDTH = 640
RESIZE_HEIGHT = 480

# Timestamp-gap fallback: if the time gap between two consecutive tank
# detections exceeds this threshold (seconds), treat it as a coach boundary.
# Tuned for LHB local trains at typical inspection speeds (5–15 km/h).
TIMESTAMP_GAP_THRESHOLD_SEC = 5.0

CROPPED_TANKS_DIR = "cropped_tanks"

CLASS_NAMES = {
    0: "Stairs",
    1: "bio tank front",
    2: "cbc_coupler",
    3: "bio tank top surface",
    4: "connected pipe",
    5: "connecting_wire",
    6: "pipesupport",
}

CLASS_INDEX = {v: k for k, v in CLASS_NAMES.items()}

CLS_TANK = CLASS_INDEX["bio tank front"]
CLS_SURFACE = CLASS_INDEX["bio tank top surface"]
CLS_PIPE = CLASS_INDEX["connected pipe"]
CLS_SUPPORT = CLASS_INDEX["pipesupport"]
CLS_COUPLER = CLASS_INDEX["cbc_coupler"]
CLS_WIRE = CLASS_INDEX["connecting_wire"]
CLS_STAIRS = CLASS_INDEX["Stairs"]


# ─────────────────────────────────────────
# DATA STRUCTURES
# ─────────────────────────────────────────
@dataclass
class Box:
    x1: float
    y1: float
    x2: float
    y2: float
    conf: float
    cls: int

    @property
    def area(self) -> float:
        return max(0.0, self.x2 - self.x1) * max(0.0, self.y2 - self.y1)

    def iou(self, other: "Box") -> float:
        xa = max(self.x1, other.x1)
        ya = max(self.y1, other.y1)
        xb = min(self.x2, other.x2)
        yb = min(self.y2, other.y2)
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        union = self.area + other.area - inter
        return inter / union if union > 0 else 0.0

    def overlap_fraction(self, other: "Box") -> float:
        """Fraction of `other`'s area that lies inside `self`."""
        xa = max(self.x1, other.x1)
        ya = max(self.y1, other.y1)
        xb = min(self.x2, other.x2)
        yb = min(self.y2, other.y2)
        inter = max(0.0, xb - xa) * max(0.0, yb - ya)
        return inter / other.area if other.area > 0 else 0.0


@dataclass
class TankTrack:
    """Tracks one bio tank across consecutive frames."""
    tank_id: int
    camera_side: str
    last_box: Box
    first_seen_frame: int
    last_seen_frame: int
    frame_count: int = 1
    conf_sum: float = 0.0
    pipe_seen: bool = False
    support_seen: bool = False
    surface_seen: bool = False
    coupler_seen: bool = False
    wire_seen: bool = False
    stairs_seen: bool = False
    timestamp_sec: float = 0.0
    best_frame_index: int = 0
    best_frame_conf: float = 0.0
    best_frame_boxes: list[Box] = field(default_factory=list)
    raw_image: Optional[np.ndarray] = None


# ─────────────────────────────────────────
# INFERENCE
# ─────────────────────────────────────────
_model_cache: dict = {}


def _load_model() -> YOLO:
    if MODEL_PATH not in _model_cache:
        _model_cache[MODEL_PATH] = YOLO(MODEL_PATH)
    return _model_cache[MODEL_PATH]


def _resize_for_inference(frame: np.ndarray) -> np.ndarray:
    """Resize frame to fixed dimensions for consistent, fast inference."""
    h, w = frame.shape[:2]
    if w == RESIZE_WIDTH and h == RESIZE_HEIGHT:
        return frame
    return cv2.resize(frame, (RESIZE_WIDTH, RESIZE_HEIGHT), interpolation=cv2.INTER_LINEAR)


def _run_inference(frame: np.ndarray) -> list[Box]:
    """Run YOLO on a single frame and return list of Box objects.
    Frames are resized before inference for speed."""
    model = _load_model()
    resized = _resize_for_inference(frame)
    results = model(
        resized,
        conf=CONF_THRESHOLD,
        iou=IOU_THRESHOLD,
        verbose=False,
    )
    # Scale boxes back to original frame coordinates
    orig_h, orig_w = frame.shape[:2]
    sx = orig_w / RESIZE_WIDTH
    sy = orig_h / RESIZE_HEIGHT
    boxes: list[Box] = []
    for r in results:
        if r.boxes is None:
            continue
        for b in r.boxes:
            xyxy = b.xyxy[0].cpu().numpy()
            boxes.append(
                Box(
                    x1=float(xyxy[0] * sx),
                    y1=float(xyxy[1] * sy),
                    x2=float(xyxy[2] * sx),
                    y2=float(xyxy[3] * sy),
                    conf=float(b.conf[0].cpu().numpy()),
                    cls=int(b.cls[0].cpu().numpy()),
                )
            )
    return boxes


# ─────────────────────────────────────────
# TRACKING & DETECTION FLOW
# ─────────────────────────────────────────
def _associate_tank(
    box: Box,
    tracks: list[TankTrack],
    current_max_id: list[int],
    current_frame: int,
) -> TankTrack:
    """Match a tank detection to an existing track or create a new one.

    A track is considered stale (no longer matchable) if it hasn't been
    seen for more than CONFIRMATION_GAP_FRAMES frames.
    """
    best_track: Optional[TankTrack] = None
    best_iou = TANK_OVERLAP_IOU
    for t in tracks:
        # Skip stale tracks that haven't been seen recently
        if (current_frame - t.last_seen_frame) > CONFIRMATION_GAP_FRAMES:
            continue
        iou = box.iou(t.last_box)
        if iou > best_iou:
            best_iou = iou
            best_track = t

    if best_track is None:
        current_max_id[0] += 1
        best_track = TankTrack(
            tank_id=current_max_id[0],
            camera_side="",
            last_box=box,
            first_seen_frame=current_frame,
            last_seen_frame=current_frame,
        )
        tracks.append(best_track)

    best_track.last_box = box
    best_track.last_seen_frame = current_frame
    best_track.frame_count += 1
    best_track.conf_sum += box.conf
    if box.conf > best_track.best_frame_conf:
        best_track.best_frame_conf = box.conf
        best_track.best_frame_index = current_frame
    return best_track


def _check_child_in_tank(
    child: Box, tank_box: Box, track: TankTrack, class_id: int
) -> None:
    """If `child` lies inside `tank_box`, mark the corresponding flag on track."""
    if tank_box.overlap_fraction(child) < CHILD_OVERLAP_THRESHOLD:
        return
    if class_id == CLS_PIPE:
        track.pipe_seen = True
    elif class_id == CLS_SUPPORT:
        track.support_seen = True
    elif class_id == CLS_SURFACE:
        track.surface_seen = True
    elif class_id == CLS_COUPLER:
        track.coupler_seen = True
    elif class_id == CLS_WIRE:
        track.wire_seen = True
    elif class_id == CLS_STAIRS:
        track.stairs_seen = True


# ─────────────────────────────────────────
# COACH COUNTING ALGORITHM
# ─────────────────────────────────────────
def _compute_coach_count(
    inter_tank_events: list[dict],
    tank_track_count: int,
) -> int:
    """
    Coach counting algorithm — couplers OR stairs.

    A boundary between two coaches is confirmed when EITHER a cbc_coupler
    OR a Stairs detection appears in the temporal gap between two tank-group
    clusters (per camera side).  Using OR instead of AND because the model's
    recall for these classes is low (~38–45%), making co-occurrence rare.

    Returns the number of confirmed coach boundaries + 1 (= number of
    coaches).  Fallback: if no boundaries detected at all, returns
    tank_track_count so we never report 0 coaches when tanks were found.
    """
    if tank_track_count == 0:
        return 0

    boundaries = 0
    saw_coupler_in_gap = False
    saw_stairs_in_gap = False

    for event in inter_tank_events:
        if event.get("kind") == "gap_start":
            saw_coupler_in_gap = False
            saw_stairs_in_gap = False
            continue

        cls_id = event.get("cls")
        if cls_id == CLS_COUPLER:
            saw_coupler_in_gap = True
        elif cls_id == CLS_STAIRS:
            saw_stairs_in_gap = True

        if event.get("kind") == "gap_end" and (saw_coupler_in_gap or saw_stairs_in_gap):
            boundaries += 1
            saw_coupler_in_gap = False
            saw_stairs_in_gap = False

    if boundaries == 0:
        return tank_track_count
    return boundaries + 1


def _get_confirmed_gap_ends(events: list[dict]) -> list[int]:
    """Return frame indices of gap ends where coupler OR stairs was detected."""
    ends: list[int] = []
    saw_coupler = False
    saw_stairs = False
    for event in events:
        if event["kind"] == "gap_start":
            saw_coupler = False
            saw_stairs = False
        elif event["kind"] == "detect":
            if event["cls"] == CLS_COUPLER:
                saw_coupler = True
            elif event["cls"] == CLS_STAIRS:
                saw_stairs = True
        elif event["kind"] == "gap_end" and (saw_coupler or saw_stairs):
            ends.append(event["frame"])
    return ends


def _assign_coach_numbers(
    finalized: list[TankTrack],
    confirmed_gap_ends: list[int],
) -> dict[int, int]:
    """
    Assign 1-based coach_number to each tank track.

    Tanks before the 1st confirmed gap boundary → coach 1.
    Tanks after 1st confirmed gap → coach 2, after 2nd → coach 3, etc.
    If no confirmed boundaries, all tanks → coach 1.
    """
    if not confirmed_gap_ends or not finalized:
        return {t.tank_id: 1 for t in finalized}

    result: dict[int, int] = {}
    for t in finalized:
        coach = 1
        for end in sorted(confirmed_gap_ends):
            if t.first_seen_frame > end:
                coach += 1
            else:
                break
        result[t.tank_id] = coach
    return result


def _build_inter_tank_events(
    all_detections: list[dict],
    gaps: list[tuple[int, int]],
) -> list[dict]:
    """
    Build an ordered event stream:
      - 'gap_start' marks the start of a temporal gap between tank clusters
      - detections of cbc_coupler / Stairs during the gap
      - 'gap_end' marks the end of the gap

    `gaps` is a list of (start_frame_idx, end_frame_idx) tuples marking where
    no tank was detected for a sustained period (= coach boundary region).
    """
    events: list[dict] = []
    for start_idx, end_idx in gaps:
        events.append({"kind": "gap_start", "frame": start_idx})
        for d in all_detections:
            if start_idx <= d["frame"] <= end_idx:
                if d["cls"] in (CLS_COUPLER, CLS_STAIRS):
                    events.append({"kind": "detect", "frame": d["frame"], "cls": d["cls"]})
        events.append({"kind": "gap_end", "frame": end_idx})
    return events


# ─────────────────────────────────────────
# MAINTENANCE DECISION
# ─────────────────────────────────────────
def _maintenance_decision(track: TankTrack) -> str:
    """Normal if bio tank top surface detected — surface clean = no maintenance needed."""
    return "Normal" if track.surface_seen else "Maintenance Required"


def _fmt_status(seen: bool, present_label: str, absent_label: str) -> str:
    return present_label if seen else absent_label


# ─────────────────────────────────────────
# CROPPED IMAGE SAVE & ANNOTATION
# ─────────────────────────────────────────

CLASS_COLORS_BGR = {
    "Stairs":               (180, 255, 0  ),
    "bio tank front":       (0,   140, 255),
    "cbc_coupler":          (255, 180, 0  ),
    "bio tank top surface": (80,  255, 0  ),
    "connected pipe":       (60,  60,  255),
    "connecting_wire":      (0,   220, 255),
    "pipesupport":          (255, 0,   180),
}


def _annotate_frame(frame: np.ndarray, boxes: list[Box]) -> np.ndarray:
    """Draw bounding boxes + labels for all detected classes."""
    out = frame.copy()
    font = cv2.FONT_HERSHEY_SIMPLEX
    for b in boxes:
        cls_name = CLASS_NAMES.get(b.cls, str(b.cls))
        color = CLASS_COLORS_BGR.get(cls_name, (200, 200, 200))
        x1, y1, x2, y2 = int(b.x1), int(b.y1), int(b.x2), int(b.y2)
        cv2.rectangle(out, (x1, y1), (x2, y2), color, 2)
        label = f"{cls_name} {b.conf:.2f}"
        (lw, lh), _ = cv2.getTextSize(label, font, 0.45, 1)
        cv2.rectangle(out, (x1, y1 - lh - 6), (x1 + lw + 4, y1), color, -1)
        cv2.putText(out, label, (x1 + 2, y1 - 3), font, 0.45, (0, 0, 0), 1)
    return out


def _save_crop_and_annotated(
    frame: np.ndarray,
    tank_box: Box,
    boxes: list[Box],
    side_prefix: str,
    tank_label: str,
) -> tuple[str, str]:
    """Save the crop and the full annotated frame. Returns (crop_path, annotated_path)."""
    side_dir = os.path.join(CROPPED_TANKS_DIR, side_prefix.lower())
    os.makedirs(side_dir, exist_ok=True)

    h, w = frame.shape[:2]
    x1 = max(0, int(tank_box.x1))
    y1 = max(0, int(tank_box.y1))
    x2 = min(w, int(tank_box.x2))
    y2 = min(h, int(tank_box.y2))

    crop_path = ""
    if x2 > x1 and y2 > y1:
        crop = frame[y1:y2, x1:x2]
        crop_path = os.path.join(side_dir, f"{tank_label}.jpg")
        cv2.imwrite(crop_path, crop)

    ann = _annotate_frame(frame, boxes)
    ann_path = os.path.join(side_dir, f"{tank_label}_full.jpg")
    cv2.imwrite(ann_path, ann)

    return crop_path, ann_path


# ─────────────────────────────────────────
# MAIN PROCESSING ENTRY POINT
# ─────────────────────────────────────────
def process_video(video_path: str, camera_side: str) -> list[dict]:
    """
    Process a video file frame-by-frame, producing one tank report dict per
    detected bio tank. Also performs coach counting via the coupler+stairs
    co-occurrence algorithm; the coach count is injected onto each tank record
    as `coach_count_side` so the UI can pick it up from any record.

    Uses frame skipping (YOLO every INFERENCE_EVERY_N frames) and pre-inference
    resize for speed on CPU.

    Returns a list of tank-report dicts (see module docstring for schema).
    """
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS) or 25.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)

    tracks: list[TankTrack] = []
    current_max_id = [0]
    side_prefix = "L" if camera_side.lower().startswith("l") else "R"

    all_detections: list[dict] = []
    frames_with_tank: list[int] = []
    frames_with_tank_set: set[int] = set()

    cached_boxes: list[Box] = []
    last_infer_frame = -INFERENCE_EVERY_N - 1

    frame_idx = -1

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        frame_idx += 1

        if frame_idx - last_infer_frame >= INFERENCE_EVERY_N:
            cached_boxes = _run_inference(frame)
            last_infer_frame = frame_idx

        boxes = cached_boxes

        tank_boxes = [b for b in boxes if b.cls == CLS_TANK]

        for tb in tank_boxes:
            track = _associate_tank(tb, tracks, current_max_id, frame_idx)
            track.camera_side = camera_side
            track.timestamp_sec = frame_idx / fps if fps > 0 else 0.0

            if tb.conf >= track.best_frame_conf:
                track.best_frame_conf = tb.conf
                track.best_frame_index = frame_idx
                track.raw_image = frame.copy()
                track.best_frame_boxes = list(boxes)
                track.last_box = tb

            for cb in boxes:
                if cb is tb:
                    continue
                _check_child_in_tank(cb, tb, track, cb.cls)

            all_detections.append({"frame": frame_idx, "cls": tb.cls})
            if frame_idx not in frames_with_tank_set:
                frames_with_tank.append(frame_idx)
                frames_with_tank_set.add(frame_idx)

        for cb in boxes:
            if cb.cls != CLS_TANK:
                all_detections.append({"frame": frame_idx, "cls": cb.cls})

    cap.release()

    finalized: list[TankTrack] = [
        t for t in tracks if t.frame_count >= 1 and t.camera_side == camera_side
    ]

    tank_track_count = len(finalized)

    cluster_starts = []
    cluster_ends = []
    sorted_tank_frames = sorted(set(frames_with_tank))
    if sorted_tank_frames:
        cluster_starts.append(sorted_tank_frames[0])
        for i in range(1, len(sorted_tank_frames)):
            if sorted_tank_frames[i] - sorted_tank_frames[i - 1] > 8:
                cluster_ends.append(sorted_tank_frames[i - 1])
                cluster_starts.append(sorted_tank_frames[i])
        cluster_ends.append(sorted_tank_frames[-1])

    gaps: list[tuple[int, int]] = []
    for i in range(1, len(cluster_starts)):
        gap_start = cluster_ends[i - 1] + 1
        gap_end = cluster_starts[i] - 1
        if gap_end >= gap_start:
            gaps.append((gap_start, gap_end))

    events = _build_inter_tank_events(all_detections, gaps)
    coach_count = _compute_coach_count(events, tank_track_count)
    confirmed_gap_ends = _get_confirmed_gap_ends(events)
    coach_map = _assign_coach_numbers(finalized, confirmed_gap_ends)

    # ── Fallback: timestamp-gap heuristic ──────────────────────────
    # If the coupler/stairs algorithm still yields 1 coach but we have
    # multiple tanks, use the temporal gap between consecutive tank
    # first-seen times. A gap > TIMESTAMP_GAP_THRESHOLD_SEC seconds
    # is treated as a coach boundary (tuned for LHB local trains).
    if coach_count <= 1 and tank_track_count > 1:
        finalized.sort(key=lambda t: t.first_seen_frame)
        current_coach = 1
        for i, t in enumerate(finalized):
            if i == 0:
                coach_map[t.tank_id] = 1
            else:
                time_gap = (
                    (t.first_seen_frame - finalized[i - 1].first_seen_frame) / fps
                    if fps > 0 else 0.0
                )
                if time_gap > TIMESTAMP_GAP_THRESHOLD_SEC:
                    current_coach += 1
                coach_map[t.tank_id] = current_coach
        coach_count = current_coach

    finalized.sort(key=lambda t: (t.first_seen_frame, t.tank_id))

    reports: list[dict] = []
    for i, t in enumerate(finalized, start=1):
        tank_label = f"{side_prefix}{i}"
        avg_conf = (t.conf_sum / max(1, t.frame_count)) if t.frame_count else 0.0

        crop_path = ""
        ann_path = ""
        if t.raw_image is not None and t.last_box is not None:
            crop_path, ann_path = _save_crop_and_annotated(
                t.raw_image, t.last_box, t.best_frame_boxes, side_prefix, tank_label
            )

        report = {
            "tank_id": tank_label,
            "bio_tank_number": tank_label,
            "coach_number": coach_map.get(t.tank_id, 1),
            "camera_side": camera_side,
            "timestamp_sec": round(t.timestamp_sec, 2),
            "pipe_status": _fmt_status(t.pipe_seen, "Connected", "Not Connected"),
            "pipe_support_status": _fmt_status(t.support_seen, "Present", "Absent"),
            "surface_status": _fmt_status(t.surface_seen, "Clean", "Not Clean"),
            "coupler_status": _fmt_status(t.coupler_seen, "Detected", "Not Detected"),
            "connecting_wire_status": _fmt_status(t.wire_seen, "Detected", "Not Detected"),
            "stairs_status": _fmt_status(t.stairs_seen, "Detected", "Not Detected"),
            "detection_confidence": round(avg_conf, 3),
            "maintenance_status": _maintenance_decision(t),
            "tank_image_path": crop_path,
            "annotated_frame_path": ann_path,
        }
        reports.append(report)

    return reports


# ─────────────────────────────────────────
# SUMMARY HELPER (used by app for top-level result)
# ─────────────────────────────────────────
def summarize(reports: list[dict], side: str, side_prefix: str) -> dict:
    total = len(reports)
    maint = sum(1 for r in reports if r["maintenance_status"] == "Maintenance Required")
    normal = total - maint
    # Use max coach_number across all reports — not just the first one
    coach_count = max((r["coach_number"] for r in reports), default=0)
    return {
        "side": side,
        "side_prefix": side_prefix,
        "total_tanks": total,
        "coach_count": coach_count,
        "normal": normal,
        "maintenance_required": maint,
    }


# ─────────────────────────────────────────
# API CONTRACT — transforms internal reports → backend JSON file
# ─────────────────────────────────────────

API_CONTRACT_FIELDS = {
    "coach_number",
    "tank_id",
    "camera_side",
    "timestamp_sec",
    "detection_confidence",
    "pipe_status",
    "pipe_support_status",
    "surface_status",
    "maintenance_status",
    "tank_image_path",
}


def build_api_result(
    inspection_run_id: str,
    processing_timestamp: str,
    left_reports: list[dict],
    right_reports: list[dict],
) -> dict:
    """
    Build the API-contract JSON dict from internal per-tank reports.
    Strips internal-only fields (coupler_status, connecting_wire_status,
    stairs_status, bio_tank_number).
    """
    tanks: list[dict] = []
    for r in left_reports + right_reports:
        tank = {k: r[k] for k in API_CONTRACT_FIELDS}
        tanks.append(tank)
    tanks.sort(key=lambda x: (0 if x["camera_side"] == "LEFT" else 1, x["tank_id"]))

    return {
        "inspection_run_id": inspection_run_id,
        "processing_timestamp": processing_timestamp,
        "status": "COMPLETE",
        "tanks": tanks,
    }


def save_api_report(
    result: dict,
    output_dir: str,
) -> str:
    """Write API-contract JSON to disk. Returns the file path."""
    os.makedirs(output_dir, exist_ok=True)
    run_id = result["inspection_run_id"]
    fname = f"btcas_api_report_{run_id}.json"
    fpath = os.path.join(output_dir, fname)
    with open(fpath, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=2)
    return fpath
