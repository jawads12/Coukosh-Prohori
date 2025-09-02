"""
Clean YOLO + DeepSORT Person Tracking Script
Optimized for Rapoo Camera with V4L2 backend for stable video

Pipeline: Clean Camera → YOLO (person detect) → DeepSORT (track) → target center vs frame center → error (dx, dy)

Controls
- Click inside a person's box to select that person as the target.
- Press 'n' to cycle through visible track IDs as target.
- Press 'u' to unselect the target (free mode).
- Press 'q' to quit.
"""

import cv2
import math
import time
import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort
import platform

# Fix Qt platform plugin warning
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# -------------------- Config --------------------
CAM_INDEX = 2              # Rapoo camera index
CONF_THRES = 0.5           # YOLO confidence threshold
IOU_THRES = 0.45           # YOLO NMS IoU
TARGET_CLASS = 0           # YOLO "person" class id
DEVICE = 'cpu'             # Force CPU for stability
MAX_AGE = 30               # DeepSORT track persistence (frames)
N_INIT = 3                 # Frames to confirm a track
NMS_MAX_OVERLAP = 1.0

# Video Quality Settings - Optimized for clean feed
SKIP_FRAMES = 1            # Process every 2nd frame for performance

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

def put_label(img, text, pos):
    cv2.putText(img, text, pos, FONT, TEXT_SCALE, (0, 0, 0), TEXT_THICK + 1)
    cv2.putText(img, text, pos, FONT, TEXT_SCALE, (255, 255, 255), TEXT_THICK)

def point_in_box(point, box):
    x, y = point
    x1, y1, x2, y2 = box
    return x1 <= x <= x2 and y1 <= y <= y2

# -------------------- Globals --------------------
selected_track_id = None
last_click = None

def on_mouse(event, x, y, flags, param):
    global last_click
    if event == cv2.EVENT_LBUTTONDOWN:
        last_click = (x, y)

# -------------------- Main --------------------

print("Loading YOLOv8n (person detection)...")
model = YOLO('yolov8n.pt')

print("Init DeepSORT tracker...")
tracker = DeepSort(max_age=MAX_AGE, n_init=N_INIT, nms_max_overlap=NMS_MAX_OVERLAP)

print("Setting up clean camera...")

# Use V4L2 backend with clean settings (from advanced test)
cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)

if not cap.isOpened():
    print(f"Error: Could not open Rapoo camera with index {CAM_INDEX}")
    print("Available camera indices to try: 0, 1, 2, 3...")
    exit(1)

# Apply clean settings that worked in the test
cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))  # YUYV uncompressed
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)        # Stable resolution
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)       # Stable resolution
cap.set(cv2.CAP_PROP_FPS, 30)                 # Good FPS
cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)           # Real-time processing

# Disable auto settings for stability
cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)     # Manual exposure
cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)            # Disable autofocus
cap.set(cv2.CAP_PROP_AUTO_WB, 0)              # Disable auto white balance

# Camera quality settings
cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
cap.set(cv2.CAP_PROP_SATURATION, 0.5)

# Print final settings
print("=== Clean Camera Settings ===")
print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
print(f"FOURCC: {cap.get(cv2.CAP_PROP_FOURCC)}")
print("Backend: V4L2 with YUYV format")
print("=============================")

# Skip initial frames to let camera stabilize
print("Stabilizing camera...")
for _ in range(30):
    cap.read()

cv2.namedWindow("Clean Rapoo Camera - Person Tracking", cv2.WINDOW_NORMAL)
cv2.resizeWindow("Clean Rapoo Camera - Person Tracking", 800, 600)
cv2.setMouseCallback("Clean Rapoo Camera - Person Tracking", on_mouse)

fps_time = time.time()
cycle_ids = []  # for 'n' key cycling
frame_count = 0  # For frame skipping
clean_frames = 0
corrupted_frames = 0

print("=== Clean Person Tracking Started ===")
print("Camera is stable and ready!")

try:
    while True:
        ret, frame = cap.read()
        frame_count += 1
        
        if not ret:
            print("Camera read failed.")
            break
        
        # Validate frame quality
        if frame is None or frame.size == 0:
            corrupted_frames += 1
            continue
            
        # Check for corruption
        mean_brightness = frame.mean()
        if mean_brightness < 5 or mean_brightness > 250:
            corrupted_frames += 1
            continue
        
        clean_frames += 1
        
        # Frame skipping for performance
        should_process_yolo = (frame_count % (SKIP_FRAMES + 1) == 0)

        H, W = frame.shape[:2]
        fx, fy = W / 2.0, H / 2.0

        # Draw frame center
        draw_crosshair(frame, (fx, fy), size=CROSS_SIZE, thickness=2)

        # 1) YOLO inference (only "person")
        if should_process_yolo:
            try:
                results = model.predict(source=frame, verbose=False, conf=CONF_THRES, iou=IOU_THRES, classes=[TARGET_CLASS])
                dets = []  # DeepSORT format: [ [x1,y1,x2,y2], conf, class ]
                yolo_boxes = []  # keep to map click → track
                if results and len(results) > 0:
                    r = results[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        for b in r.boxes:
                            xyxy = b.xyxy.cpu().numpy().astype(int).flatten()
                            conf = float(b.conf.cpu().numpy().item())
                            cls = int(b.cls.cpu().numpy().item()) if b.cls is not None else TARGET_CLASS
                            x1, y1, x2, y2 = xyxy.tolist()
                            dets.append(([x1, y1, x2, y2], conf, cls))
                            yolo_boxes.append((x1, y1, x2, y2))
            except Exception as e:
                print(f"YOLO inference error: {e}")
                dets = []
                yolo_boxes = []
        else:
            dets = []
            yolo_boxes = []

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

        # Handle click → choose the track whose box contains the click
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
                try:
                    idx = cycle_ids.index(selected_track_id)
                    selected_track_id = cycle_ids[(idx + 1) % len(cycle_ids)]
                except (ValueError, TypeError):
                    selected_track_id = cycle_ids[0] if cycle_ids else None
        elif key == ord('u'):
            selected_track_id = None
            cycle_ids = []
        elif key == ord('q'):
            break

        # Update cycle_ids with current visible tracks
        cycle_ids = list(visible_tracks)

        # 3) Draw all tracks
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

        # FPS and quality info
        now = time.time()
        fps = 1.0 / max(1e-6, (now - fps_time))
        fps_time = now
        success_rate = (clean_frames / frame_count) * 100 if frame_count > 0 else 0
        
        put_label(frame, f"FPS: {fps:.1f} | Clean: {success_rate:.1f}%", (10, H - 30))
        put_label(frame, f"Frame: {frame_count} | Processing: {'YOLO' if should_process_yolo else 'Skip'}", (10, H - 10))

        cv2.imshow("Clean Rapoo Camera - Person Tracking", frame)

except KeyboardInterrupt:
    print("\nInterrupted by user")

finally:
    print(f"\nFinal Statistics:")
    print(f"Total frames: {frame_count}")
    print(f"Clean frames: {clean_frames}")
    print(f"Corrupted frames: {corrupted_frames}")
    if frame_count > 0:
        print(f"Success rate: {(clean_frames/frame_count)*100:.1f}%")
    
    cap.release()
    cv2.destroyAllWindows()
    print("Clean person tracking completed")
