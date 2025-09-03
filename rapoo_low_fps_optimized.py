"""
Rapoo Script3 - Optimized for Low FPS Camera
This version is specifically optimized for cameras with 7-9 FPS limitations
Uses frame skipping and processing optimization for smooth tracking
"""

import cv2
import numpy as np
import os
import time
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Fix Qt warnings
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Optimized Configuration for Low FPS Camera
CAM_INDEX = 2              # Your camera index
CONF_THRES = 0.3          # Lower threshold for better detection at low FPS
TARGET_CLASS = 0          # Person class ID
PROCESS_EVERY_N_FRAMES = 2  # Process every 2nd frame to improve responsiveness

# Global variables
selected_track_id = None
frame_count = 0
last_process_time = time.time()

# Human keypoint indices for YOLOv8 pose model (17 keypoints)
KEYPOINT_NAMES = [
    'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
    'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
    'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
    'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
]

# Important body parts for human shape validation
CORE_KEYPOINTS = {
    'shoulders': [5, 6],  # left_shoulder, right_shoulder
    'hips': [11, 12],     # left_hip, right_hip
    'head': [0, 1, 2, 3, 4],  # nose, eyes, ears
    'arms': [7, 8, 9, 10],    # elbows, wrists
    'legs': [13, 14, 15, 16]  # knees, ankles
}

def validate_human_keypoints(keypoints, confidence_threshold=0.25):
    """Validate if detected keypoints represent a real human shape - optimized for low FPS"""
    if keypoints is None or len(keypoints) == 0:
        return False, "No keypoints detected"
    
    # Count visible keypoints for each body part
    visible_count = {'head': 0, 'shoulders': 0, 'hips': 0, 'arms': 0, 'legs': 0}
    
    for part, indices in CORE_KEYPOINTS.items():
        for idx in indices:
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                x, y, conf = keypoints[idx][:3]
                if conf > confidence_threshold and x > 0 and y > 0:
                    visible_count[part] += 1
    
    # Relaxed validation for low FPS cameras
    has_head = visible_count['head'] >= 1
    has_shoulders = visible_count['shoulders'] >= 1
    has_torso = visible_count['hips'] >= 1
    has_limbs = visible_count['arms'] >= 1 or visible_count['legs'] >= 1
    
    # More lenient criteria for low FPS
    is_human = (has_shoulders or has_torso) and (has_head or has_limbs)
    
    reason = f"H:{visible_count['head']}, S:{visible_count['shoulders']}, Hip:{visible_count['hips']}, A:{visible_count['arms']}, L:{visible_count['legs']}"
    
    return is_human, reason

def get_human_chest_center_from_keypoints(keypoints):
    """Calculate chest center using shoulder and hip keypoints - optimized for low confidence"""
    if keypoints is None or len(keypoints) == 0:
        return None, None
    
    # Extract key body landmarks with lower confidence threshold
    left_shoulder = keypoints[5] if len(keypoints) > 5 else [0, 0, 0]
    right_shoulder = keypoints[6] if len(keypoints) > 6 else [0, 0, 0]
    left_hip = keypoints[11] if len(keypoints) > 11 else [0, 0, 0]
    right_hip = keypoints[12] if len(keypoints) > 12 else [0, 0, 0]
    
    # Collect valid shoulder points (lower threshold for low FPS)
    shoulder_points = []
    if len(left_shoulder) >= 3 and left_shoulder[2] > 0.2:
        shoulder_points.append((left_shoulder[0], left_shoulder[1]))
    if len(right_shoulder) >= 3 and right_shoulder[2] > 0.2:
        shoulder_points.append((right_shoulder[0], right_shoulder[1]))
    
    # Collect valid hip points
    hip_points = []
    if len(left_hip) >= 3 and left_hip[2] > 0.2:
        hip_points.append((left_hip[0], left_hip[1]))
    if len(right_hip) >= 3 and right_hip[2] > 0.2:
        hip_points.append((right_hip[0], right_hip[1]))
    
    # Calculate chest center with fallback options
    if shoulder_points and hip_points:
        shoulder_center_x = sum([p[0] for p in shoulder_points]) / len(shoulder_points)
        shoulder_center_y = sum([p[1] for p in shoulder_points]) / len(shoulder_points)
        
        hip_center_x = sum([p[0] for p in hip_points]) / len(hip_points)
        hip_center_y = sum([p[1] for p in hip_points]) / len(hip_points)
        
        chest_x = shoulder_center_x + 0.3 * (hip_center_x - shoulder_center_x)
        chest_y = shoulder_center_y + 0.3 * (hip_center_y - shoulder_center_y)
        
    elif shoulder_points:
        chest_x = sum([p[0] for p in shoulder_points]) / len(shoulder_points)
        chest_y = sum([p[1] for p in shoulder_points]) / len(shoulder_points) + 40
        
    elif hip_points:
        chest_x = sum([p[0] for p in hip_points]) / len(hip_points)
        chest_y = sum([p[1] for p in hip_points]) / len(hip_points) - 60
        
    else:
        # More lenient fallback for low FPS
        valid_points = []
        for kp in keypoints:
            if len(kp) >= 3 and kp[2] > 0.2:  # Lower threshold
                valid_points.append((kp[0], kp[1]))
        
        if valid_points:
            chest_x = sum([p[0] for p in valid_points]) / len(valid_points)
            chest_y = sum([p[1] for p in valid_points]) / len(valid_points)
        else:
            return None, None
    
    return chest_x, chest_y

def get_largest_person(active_tracks):
    """Find the largest person among active tracks"""
    if not active_tracks:
        return None
    
    largest_id = None
    largest_area = 0
    
    for tid, (bbox, keypoints) in active_tracks.items():
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        is_valid, _ = validate_human_keypoints(keypoints)
        
        if is_valid and area > largest_area:
            largest_area = area
            largest_id = tid
    
    return largest_id

def main():
    global selected_track_id, frame_count, last_process_time
    
    print("=== Rapoo Camera - Optimized for Low FPS ===")
    print("1. Loading YOLOv8 Pose model...")
    model = YOLO('yolov8n-pose.pt')
    
    print("2. Initializing DeepSORT tracker...")
    tracker = DeepSort(max_age=50, n_init=2)  # Increased max_age for low FPS
    
    print("3. Setting up camera with optimal settings...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera at index {CAM_INDEX}")
        return
    
    # Optimal settings for your Rapoo camera based on test results
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)     # Best compromise: 720p @ 9.3 FPS
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)    
    cap.set(cv2.CAP_PROP_FPS, 30)              # Request 30, get ~9
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # Minimize latency
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    
    # Quality optimizations for low FPS
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.8)        # Higher contrast for better detection
    cap.set(cv2.CAP_PROP_SATURATION, 0.7)
    cap.set(cv2.CAP_PROP_SHARPNESS, 0.9)       # Maximum sharpness for keypoint detection
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Fixed exposure for consistent FPS
    
    # Get actual settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"=== Camera Settings (Optimized for Low FPS) ===")
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"Target FPS: {actual_fps} (Expected: ~9 FPS)")
    print(f"Processing: Every {PROCESS_EVERY_N_FRAMES} frames")
    print("============================================")
    
    cv2.namedWindow("Rapoo - Low FPS Optimized Tracking", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rapoo - Low FPS Optimized Tracking", 960, 540)
    
    print("4. Starting optimized tracking pipeline...")
    print("   - Frame skipping enabled for responsiveness")
    print("   - Relaxed validation for low FPS cameras")
    print("   - Enhanced tracking persistence")
    print("   - Press 'q' to quit")
    
    fps_start_time = time.time()
    fps_frame_count = 0
    current_fps = 0
    
    last_detections = []
    last_keypoints = {}
    
    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            frame_count += 1
            fps_frame_count += 1
            
            H, W = frame.shape[:2]
            fx, fy = W / 2.0, H / 2.0
            
            # Calculate actual FPS
            if fps_frame_count % 30 == 0:
                elapsed = time.time() - fps_start_time
                current_fps = 30 / elapsed if elapsed > 0 else 0
                fps_start_time = time.time()
                print(f"Actual camera FPS: {current_fps:.1f}")
            
            # Draw frame center
            cv2.drawMarker(frame, (int(fx), int(fy)), (255, 255, 255), 
                          cv2.MARKER_CROSS, 15, 2)
            
            # Process every N frames for better performance
            if frame_count % PROCESS_EVERY_N_FRAMES == 0:
                process_start = time.time()
                
                # YOLO pose detection
                results = model.predict(source=frame, verbose=False, conf=CONF_THRES)
                
                detections = []
                pose_data = {}
                
                if results and len(results) > 0:
                    r = results[0]
                    if r.boxes is not None and len(r.boxes) > 0:
                        for i, box in enumerate(r.boxes):
                            xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
                            conf = float(box.conf.cpu().numpy().item())
                            x1, y1, x2, y2 = xyxy.tolist()
                            
                            keypoints = None
                            if hasattr(r, 'keypoints') and r.keypoints is not None and i < len(r.keypoints.data):
                                kp_data = r.keypoints.data[i].cpu().numpy()
                                keypoints = kp_data.tolist()
                            
                            if keypoints is not None:
                                is_human, reason = validate_human_keypoints(keypoints)
                                
                                if is_human:
                                    detections.append(([x1, y1, x2, y2], conf, 0))
                                    pose_data[len(detections)-1] = keypoints
                
                last_detections = detections
                last_keypoints = pose_data
                
                process_time = time.time() - process_start
                print(f"Processing time: {process_time*1000:.1f}ms | Detections: {len(detections)}")
            
            else:
                # Use last frame's detections for smooth display
                detections = last_detections
                pose_data = last_keypoints
            
            # DeepSORT tracking
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Collect active tracks
            active_tracks = {}
            for track in tracks:
                if track.is_confirmed() and track.time_since_update <= 1:  # More lenient for low FPS
                    tid = track.track_id
                    bbox = track.to_ltrb()
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    keypoints = None
                    if pose_data:
                        keypoints = list(pose_data.values())[0] if pose_data else None
                    
                    active_tracks[tid] = ([x1, y1, x2, y2], keypoints)
            
            # Auto-target selection
            if active_tracks:
                if selected_track_id is None or selected_track_id not in active_tracks:
                    selected_track_id = get_largest_person(active_tracks)
                    if selected_track_id:
                        print(f"Auto-selected target: ID {selected_track_id}")
            else:
                if selected_track_id is not None:
                    selected_track_id = None
                    print("Target lost - waiting for new detection...")
            
            # Calculate target center
            target_center = None
            if selected_track_id is not None and selected_track_id in active_tracks:
                bbox, keypoints = active_tracks[selected_track_id]
                cx, cy = get_human_chest_center_from_keypoints(keypoints)
                if cx is not None and cy is not None:
                    target_center = (cx, cy)
            
            # Visualization
            for tid, (bbox, keypoints) in active_tracks.items():
                x1, y1, x2, y2 = bbox
                color = (0, 255, 0) if tid == selected_track_id else (255, 0, 0)
                
                is_valid, reason = validate_human_keypoints(keypoints)
                
                if is_valid:
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw keypoints
                    if keypoints is not None:
                        for i, kp in enumerate(keypoints):
                            if len(kp) >= 3:
                                x, y, conf = kp[:3]
                                if conf > 0.2 and x > 0 and y > 0:
                                    cv2.circle(frame, (int(x), int(y)), 2, color, -1)
                    
                    # Draw chest center
                    chest_cx, chest_cy = get_human_chest_center_from_keypoints(keypoints)
                    if chest_cx is not None and chest_cy is not None:
                        cv2.circle(frame, (int(chest_cx), int(chest_cy)), 6, color, -1)
                        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Display target information
            if target_center is not None:
                cx, cy = target_center
                bbox, keypoints = active_tracks[selected_track_id]
                x1, y1, x2, y2 = bbox
                box_area = (x2 - x1) * (y2 - y1)
                
                dx = cx - fx
                dy = cy - fy
                
                # Count keypoints
                visible_keypoints = 0
                if keypoints:
                    for kp in keypoints:
                        if len(kp) >= 3 and kp[2] > 0.2:
                            visible_keypoints += 1
                
                distance_category = "Very Close" if box_area > 15000 else \
                                  "Close" if box_area > 8000 else \
                                  "Medium" if box_area > 4000 else "Far"
                
                # Draw targeting line
                cv2.arrowedLine(frame, (int(fx), int(fy)), (int(cx), int(cy)), 
                               (0, 255, 255), 2, tipLength=0.1)
                
                # Display info
                cv2.putText(frame, f"Target: ID {selected_track_id} | Error: ({int(dx)}, {int(dy)})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Chest: ({int(cx)}, {int(cy)}) | {distance_category} | KP: {visible_keypoints}/17", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                cv2.putText(frame, f"Camera FPS: {current_fps:.1f} | Process: 1/{PROCESS_EVERY_N_FRAMES} frames", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
                
                print(f"ID: {selected_track_id} | Chest: ({int(cx)}, {int(cy)}) | Error: ({int(dx)}, {int(dy)}) | {distance_category} | KP: {visible_keypoints}/17")
            
            else:
                cv2.putText(frame, "No target - waiting for human detection...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Camera FPS: {current_fps:.1f} | Detected humans: {len(active_tracks)}", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
            
            cv2.imshow("Rapoo - Low FPS Optimized Tracking", frame)
            
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Optimized tracking completed")

if __name__ == "__main__":
    main()