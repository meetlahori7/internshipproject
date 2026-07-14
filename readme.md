# AI Railway Bio-Toilet Inspection System (BTCAS v3)

Streamlit-based inspection system for railway bio-toilets using a custom YOLOv8
model trained on **7 classes**. Supports video mode (left + right camera) and
photo mode (single-frame debug).

## What the app does

- Runs detection on uploaded left + right camera videos
- Tracks each bio tank across frames and detects: pipe, support, surface, coupler, wire, stairs
- Counts coaches via the **coupler + stairs co-occurrence algorithm**
- Decides maintenance per tank (surface-driven: clean surface → Normal)
- Saves a cropped tank image + full annotated frame (all 7 class boxes drawn)
- Auto-saves an API-contract JSON file to disk after inspection
- Two download buttons: Full Report (all fields) + API Contract (stripped for backend)

## Model — 7 classes

| Index | Class name                  |
|-------|-----------------------------|
| 0     | Stairs                      |
| 1     | bio tank front              |
| 2     | cbc_coupler                 |
| 3     | bio tank top surface        |
| 4     | connected pipe              |
| 5     | connecting_wire             |
| 6     | pipesupport                 |

Inference: `conf=0.35`, `iou=0.35`. Frame skip every 5 frames, resize to 640×480.

## Per-tank output fields

| Field                   | Description                          |
|-------------------------|--------------------------------------|
| `tank_id`               | `L1`, `L2`, `R1`, ... (per side)     |
| `coach_number`          | Sequential coach position (1,2,3...) |
| `camera_side`           | `LEFT` / `RIGHT`                     |
| `timestamp_sec`         | Seconds into the video               |
| `pipe_status`           | Connected / Not Connected            |
| `pipe_support_status`   | Present / Absent                     |
| `surface_status`        | Clean / Not Clean                    |
| `coupler_status`        | Detected / Not Detected              |
| `connecting_wire_status`| Detected / Not Detected              |
| `stairs_status`         | Detected / Not Detected              |
| `detection_confidence`  | Mean confidence across tracked frames|
| `maintenance_status`    | Normal / Maintenance Required        |
| `tank_image_path`       | Path to cropped tank image           |
| `annotated_frame_path`  | Path to full frame with all boxes    |

## Coach counting algorithm (couplers + stairs)

1. Tanks tracked across frames via IoU (≥0.3).
2. Temporal gaps between tank clusters are examined.
3. A **coach boundary** is confirmed only when BOTH a `cbc_coupler` AND a
   `Stairs` detection appear in the same gap.
4. **Fallback**: if no coupler+stairs co-occurrence is found, coach_count falls
   back to number of distinct tank tracks.

## Maintenance decision

If `bio tank top surface` is detected overlapping the tank across any tracked
frame → **Normal**. Otherwise → **Maintenance Required**.

Surface cleanliness is the sole maintenance criterion. Pipe, support, coupler,
wire, and stairs status are displayed for situational awareness but do not
drive the decision.

## Files

| File                   | Purpose                                    |
|------------------------|--------------------------------------------|
| `btcas_app.py`         | Main Streamlit UI (video mode)             |
| `test_app.py`          | Streamlit UI with video + photo tabs       |
| `btcas_pipeline.py`    | YOLO inference, tracking, algorithms       |
| `btcas_yolov8s_v2_best.pt` | Trained YOLOv8 model weights          |
| `info.md`              | API contract + backend integration notes   |
| `requirements.txt`     | Python dependencies                        |

## Outputs

After each inspection run:
- `cropped_tanks/l/L1.jpg` — crop of each tank
- `cropped_tanks/l/L1_full.jpg` — full annotated frame with detection boxes
- `btcas_api_report_INSP_*.json` — API-contract JSON (auto-saved)

## Run

```bash
streamlit run btcas_app.py      # video only
streamlit run test_app.py       # video + photo tabs
```

## Notes

- `surface_status` uses `"Not Clean"` (not `"Inspection Required"`) per API contract.
- Extra fields (`coupler_status`, `connecting_wire_status`, `stairs_status`) are
  visible in the UI but stripped from the API-contract JSON download.
- The app is a local prototype. See `info.md` for backend integration details.
