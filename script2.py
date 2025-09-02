"""
Pipeline: Webcam → YOLO (person detect) → DeepSORT (track) → target center vs frame center → error (dx, dy)

Controls
- Click inside a person's box to select that person as the target.
- Press 'n' to cycle through visible track IDs as target.
- Press 'u' to unselect the target (free mode).
- Press 'q' to quit.

Requirements
pip install ultralytics opencv-python deep-sort-realtime numpy

Notes
- Uses YOLOv8 from Ultralytics with class filter = person (id=0)
- Uses deep-sort-realtime for tracking
- Works with any webcam (default index=0). Change CAM_INDEX for your Rapoo camera if needed.
- If you have a dGPU with CUDA, Ultralytics will auto-use it when available (or set DEVICE='cpu'/'cuda').
"""

import cv2
import math
import time
import numpy as np
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import platform

# -------------------- Config --------------------
CAM_INDEX = 0              # Rapoo camera index (changed from 0 to 2)
CONF_THRES = 0.5           # YOLO confidence threshold
IOU_THRES = 0.45           # YOLO NMS IoU
TARGET_CLASS = 0           # YOLO "person" class id
DEVICE = 'cpu'             # Force CPU for stability; change to 'cuda' if you have NVIDIA drivers set up
MAX_AGE = 30               # DeepSORT track persistence (frames)
N_INIT = 3                 # Frames to confirm a track
NMS_MAX_OVERLAP = 1.0

# Visualization
FONT = cv2.FONT_HERSHEY_SIMPLEX
TEXT_SCALE = 0.6
TEXT_THICK = 2
BOX_THICK = 2
CROSS_SIZE = 10

# -------------------- Helpers --------------------

def draw_crosshair(img, center, size=10, thickness=1):
    x, y = int(center[0]), int(center[1])
    cv2.line(img, (x - size, y), (x + size, y), (0, 255, 255), thickness)
    cv2.line(img, (x, y - size), (x, y + size), (0, 255, 255), thickness)


def put_label(img, text, org, bg=True):
    (tw, th), _ = cv2.getTextSize(text, FONT, TEXT_SCALE, TEXT_THICK)
    x, y = org
    if bg:
        cv2.rectangle(img, (x, y - th - 6), (x + tw + 6, y + 4), (0, 0, 0), -1)
    cv2.putText(img, text, (x + 3, y - 3), FONT, TEXT_SCALE, (255, 255, 255), TEXT_THICK, cv2.LINE_AA)


# -------------------- Init models --------------------
print("Loading YOLOv8n (person detection)...")
model = YOLO("yolov8n.pt")
if DEVICE:
    model.to(DEVICE)

print("Init DeepSORT tracker...")
tracker = DeepSort(
    max_age=MAX_AGE,
    n_init=N_INIT,
    nms_max_overlap=NMS_MAX_OVERLAP,
    max_cosine_distance=0.2,
    nn_budget=None,
    embedder="mobilenet",
    half=True,
)

# -------------------- Mouse selection --------------------
selected_track_id = None
last_click = None


def point_in_box(pt, box):
    x1, y1, x2, y2 = box
    return (x1 <= pt[0] <= x2) and (y1 <= pt[1] <= y2)


def on_mouse(event, x, y, flags, param):
    global last_click
    if event == cv2.EVENT_LBUTTONDOWN:
        last_click = (x, y)


# -------------------- Capture --------------------
# Set camera backend based on OS (V4L2 for Linux)
os_name = platform.system()
if os_name == 'Linux':
    backend = cv2.CAP_V4L2
elif os_name == 'Windows':
    backend = cv2.CAP_DSHOW
else:
    backend = cv2.CAP_ANY

cap = cv2.VideoCapture(CAM_INDEX, backend)
if not cap.isOpened():
    # Fallback to any backend
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_ANY)

# Set camera resolution and other settings
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)   # Reduced for stability
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)  # Reduced for stability
cap.set(cv2.CAP_PROP_FPS, 30)            # Standard FPS
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))  # YUYV for color
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)      # Reduce buffering for lower latency

# Force color mode
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)     # Ensure color output

print(f"Camera opened with resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"Camera FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"Camera format: {cap.get(cv2.CAP_PROP_FORMAT)}")

cv2.namedWindow("YOLO + DeepSORT")
cv2.setMouseCallback("YOLO + DeepSORT", on_mouse)

fps_time = time.time()
cycle_ids = []  # for 'n' key cycling

try:
    while True:
        ok, frame = cap.read()
        if not ok:
            print("Camera read failed. Check CAM_INDEX / permissions.")
            break

        # Ensure frame is in color (BGR format)
        if len(frame.shape) == 2:  # Grayscale frame
            print("Warning: Camera is providing grayscale frames, converting to color")
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 1:  # Single channel
            print("Warning: Camera is providing single-channel frames, converting to color")
            frame = cv2.cvtColor(frame, cv2.COLOR_GRAY2BGR)
        elif len(frame.shape) == 3 and frame.shape[2] == 3:
            # Frame is already in color (BGR), which is what we want
            pass
        else:
            print(f"Warning: Unexpected frame format: {frame.shape}")

        H, W = frame.shape[:2]
        fx, fy = W / 2.0, H / 2.0

        # 1) YOLO inference (only "person")
        results = model.predict(source=frame, verbose=False, conf=CONF_THRES, iou=IOU_THRES, classes=[TARGET_CLASS])
        dets = []  # DeepSORT format: [ [x1,y1,x2,y2], conf, class ]
        yolo_boxes = []  # keep to map click → track
        if results and len(results) > 0:
            r = results[0]
            if r.boxes is not None and len(r.boxes) > 0:
                for b in r.boxes:
                    xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                    conf = float(b.conf.cpu().numpy().item())  # Fix NumPy deprecation warning
                    cls = int(b.cls.cpu().numpy().item()) if b.cls is not None else TARGET_CLASS  # Fix NumPy deprecation warning
                    x1, y1, x2, y2 = xyxy.tolist()
                    dets.append(([x1, y1, x2, y2], conf, cls))
                    yolo_boxes.append((x1, y1, x2, y2))

        # 2) DeepSORT update
        tracks = tracker.update_tracks(dets, frame=frame)

        # collect current track IDs (visible)
        visible_tracks = []
        boxes_by_id = {}
        centers_by_id = {}

        for t in tracks:
            if not t.is_confirmed() or t.time_since_update > 0:
                continue
            tid = t.track_id
            ltrb = t.to_ltrb()  # left, top, right, bottom
            x1, y1, x2, y2 = map(int, ltrb)
            cx, cy = (x1 + x2) / 2.0, (y1 + y2) / 2.0
            visible_tracks.append(tid)
            boxes_by_id[tid] = (x1, y1, x2, y2)
            centers_by_id[tid] = (cx, cy)

        # Handle click → choose the track whose box contains the click (or nearest center)
        if last_click is not None and visible_tracks:
            clicked = last_click
            candidates = []
            for tid in visible_tracks:
                box = boxes_by_id[tid]
                if point_in_box(clicked, box):
                    candidates.append((tid, 0.0))
                else:
                    cx, cy = centers_by_id[tid]
                    d = math.hypot(clicked[0] - cx, clicked[1] - cy)
                    candidates.append((tid, d))
            candidates.sort(key=lambda x: x[1])
            selected_track_id = candidates[0][0]
            last_click = None

        # Cycle key 'n'
        key = cv2.waitKey(1) & 0xFF
        if key == ord('n'):
            if not cycle_ids:
                cycle_ids = list(visible_tracks)
            if cycle_ids:
                selected_track_id = cycle_ids[0]
                cycle_ids = cycle_ids[1:] + cycle_ids[:1]
        elif key == ord('u'):
            selected_track_id = None
        elif key == ord('q'):
            break

        # Draw frame center
        draw_crosshair(frame, (fx, fy), size=CROSS_SIZE, thickness=1)
        put_label(frame, f"Frame center (fx, fy)=({int(fx)}, {int(fy)})", (10, 24))

        # Draw tracks
        for tid in visible_tracks:
            x1, y1, x2, y2 = boxes_by_id[tid]
            cx, cy = centers_by_id[tid]
            color = (0, 255, 0) if tid == selected_track_id else (255, 0, 0)
            cv2.rectangle(frame, (x1, y1), (x2, y2), color, BOX_THICK)
            draw_crosshair(frame, (cx, cy), size=6, thickness=1)
            put_label(frame, f"ID {tid}", (x1, y1 - 8))

        # Compute and display error for selected target
        if selected_track_id is not None and selected_track_id in centers_by_id:
            cx, cy = centers_by_id[selected_track_id]
            dx, dy = cx - fx, cy - fy
            # Visualize error vector
            cv2.arrowedLine(frame, (int(fx), int(fy)), (int(cx), int(cy)), (0, 255, 255), 2, tipLength=0.2)
            put_label(frame, f"Target center (cx, cy)=({int(cx)}, {int(cy)})", (10, 50))
            put_label(frame, f"Error (dx, dy)=({int(dx)}, {int(dy)})", (10, 76))
        else:
            put_label(frame, "No target selected (click a box, or press 'n')", (10, 50))

        # FPS
        now = time.time()
        fps = 1.0 / max(1e-6, (now - fps_time))
        fps_time = now
        put_label(frame, f"FPS: {fps:.1f}", (10, H - 10))

        cv2.imshow("YOLO + DeepSORT", frame)

finally:
    cap.release()
    cv2.destroyAllWindows()
