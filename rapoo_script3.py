"""
Rapoo Script3: Advanced Human Pose-Based Tracking with Rapoo Camera
Optimized for Rapoo Camera at Full HD (1920x1080 @ 30 FPS)

Workflow:
1. Capture Frame from Rapoo USB camera (index 2)
2. Detect Humans using YOLOv8 Pose Detection (17 keypoints)
3. Validate human shape using keypoint analysis
4. Track humans with DeepSORT
5. Automatically target the largest/closest validated human
6. Compute Human Chest Center using shoulder/hip keypoints
7. Compare with Frame Center and calculate error values

Features:
- Full HD 1920x1080 @ 30 FPS for maximum quality
- Human-only detection using pose keypoints
- Advanced keypoint validation (shoulders, head, hips, limbs)
- Chest center calculation using anatomical landmarks
- Distance compensation for accurate tracking
- Auto-selection of largest valid human

Controls:
- Automatically selects the largest detected human
- Press 'q' to quit
- Press 'n' to cycle through detected persons
"""

import cv2
import numpy as np
import os
from ultralytics import YOLO
from deep_sort_realtime.deepsort_tracker import DeepSort

# Fix Qt warnings
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Configuration - Optimized for Rapoo Camera
CAM_INDEX = 2              # Rapoo camera index
CONF_THRES = 0.4          # YOLO confidence threshold (lowered for better detection)
TARGET_CLASS = 0          # Person class ID

# Global variables
selected_track_id = None
person_cycle_index = 0

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

def validate_human_keypoints(keypoints, confidence_threshold=0.3):
    """Validate if detected keypoints represent a real human shape"""
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
    
    # Human validation criteria
    has_head = visible_count['head'] >= 1  # At least nose or one eye
    has_shoulders = visible_count['shoulders'] >= 1  # At least one shoulder
    has_torso = visible_count['hips'] >= 1  # At least one hip
    has_limbs = visible_count['arms'] >= 1 or visible_count['legs'] >= 1
    
    # Must have core body structure
    is_human = has_shoulders and (has_head or has_torso) and has_limbs
    
    reason = f"Head:{visible_count['head']}, Shoulders:{visible_count['shoulders']}, Hips:{visible_count['hips']}, Arms:{visible_count['arms']}, Legs:{visible_count['legs']}"
    
    return is_human, reason

def create_human_shape_bbox(keypoints, padding=10):
    """Create a tight bounding box around detected human keypoints"""
    if keypoints is None or len(keypoints) == 0:
        return None
    
    valid_points = []
    for kp in keypoints:
        if len(kp) >= 3:
            x, y, conf = kp[:3]
            if conf > 0.3 and x > 0 and y > 0:
                valid_points.append((x, y))
    
    if len(valid_points) < 3:  # Need at least 3 points for a meaningful bbox
        return None
    
    # Find bounding box of all valid keypoints
    xs = [p[0] for p in valid_points]
    ys = [p[1] for p in valid_points]
    
    x1 = max(0, min(xs) - padding)
    y1 = max(0, min(ys) - padding)
    x2 = max(xs) + padding
    y2 = max(ys) + padding
    
    return [int(x1), int(y1), int(x2), int(y2)]

def get_human_chest_center_from_keypoints(keypoints):
    """Calculate chest center using shoulder and hip keypoints for anatomical accuracy"""
    if keypoints is None or len(keypoints) == 0:
        return None, None
    
    # Extract key body landmarks
    left_shoulder = keypoints[5] if len(keypoints) > 5 else [0, 0, 0]
    right_shoulder = keypoints[6] if len(keypoints) > 6 else [0, 0, 0]
    left_hip = keypoints[11] if len(keypoints) > 11 else [0, 0, 0]
    right_hip = keypoints[12] if len(keypoints) > 12 else [0, 0, 0]
    
    # Collect valid shoulder points
    shoulder_points = []
    if len(left_shoulder) >= 3 and left_shoulder[2] > 0.3:
        shoulder_points.append((left_shoulder[0], left_shoulder[1]))
    if len(right_shoulder) >= 3 and right_shoulder[2] > 0.3:
        shoulder_points.append((right_shoulder[0], right_shoulder[1]))
    
    # Collect valid hip points
    hip_points = []
    if len(left_hip) >= 3 and left_hip[2] > 0.3:
        hip_points.append((left_hip[0], left_hip[1]))
    if len(right_hip) >= 3 and right_hip[2] > 0.3:
        hip_points.append((right_hip[0], right_hip[1]))
    
    # Calculate chest center based on available landmarks
    if shoulder_points and hip_points:
        # Ideal case: use midpoint between shoulder center and hip center
        shoulder_center_x = sum([p[0] for p in shoulder_points]) / len(shoulder_points)
        shoulder_center_y = sum([p[1] for p in shoulder_points]) / len(shoulder_points)
        
        hip_center_x = sum([p[0] for p in hip_points]) / len(hip_points)
        hip_center_y = sum([p[1] for p in hip_points]) / len(hip_points)
        
        # Chest is approximately 1/3 of the way from shoulders to hips
        chest_x = shoulder_center_x + 0.3 * (hip_center_x - shoulder_center_x)
        chest_y = shoulder_center_y + 0.3 * (hip_center_y - shoulder_center_y)
        
    elif shoulder_points:
        # Use shoulder center and estimate chest position
        chest_x = sum([p[0] for p in shoulder_points]) / len(shoulder_points)
        chest_y = sum([p[1] for p in shoulder_points]) / len(shoulder_points) + 40  # Offset down for chest
        
    elif hip_points:
        # Use hip center and estimate chest position
        chest_x = sum([p[0] for p in hip_points]) / len(hip_points)
        chest_y = sum([p[1] for p in hip_points]) / len(hip_points) - 60  # Offset up for chest
        
    else:
        # Fallback: use any available high-confidence keypoints
        valid_points = []
        for kp in keypoints:
            if len(kp) >= 3 and kp[2] > 0.4:
                valid_points.append((kp[0], kp[1]))
        
        if valid_points:
            chest_x = sum([p[0] for p in valid_points]) / len(valid_points)
            chest_y = sum([p[1] for p in valid_points]) / len(valid_points)
        else:
            return None, None
    
    return chest_x, chest_y

def get_largest_person(active_tracks):
    """Find the largest person among active tracks (closest/most prominent)"""
    if not active_tracks:
        return None
    
    largest_id = None
    largest_area = 0
    
    for tid, (bbox, keypoints) in active_tracks.items():
        x1, y1, x2, y2 = bbox
        area = (x2 - x1) * (y2 - y1)
        
        # Validate this is a real human
        is_valid, _ = validate_human_keypoints(keypoints)
        
        if is_valid and area > largest_area:
            largest_area = area
            largest_id = tid
    
    return largest_id

def main():
    global selected_track_id, person_cycle_index
    
    print("=== Rapoo Camera - Human Pose-Based Tracking Pipeline ===")
    print("1. Loading YOLOv8 Pose model...")
    model = YOLO('yolov8n-pose.pt')  # Use pose detection model
    
    print("2. Initializing DeepSORT tracker...")
    tracker = DeepSort(max_age=30, n_init=3)
    
    print("3. Setting up Rapoo camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open Rapoo camera at index {CAM_INDEX}")
        print("Make sure your Rapoo camera is connected and try different indices (0, 1, 2, 3)")
        return
    
    # Rapoo Camera settings for BEST QUALITY (Based on your specs)
    print("4. Configuring camera for maximum quality...")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)    # Full HD width
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)   # Full HD height  
    cap.set(cv2.CAP_PROP_FPS, 30)              # 30 FPS as per specs
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)        # Reduce latency
    
    # Enhanced quality settings for Rapoo camera
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)      # Optimal brightness
    cap.set(cv2.CAP_PROP_CONTRAST, 0.7)        # Good contrast for detection
    cap.set(cv2.CAP_PROP_SATURATION, 0.6)      # Natural colors
    cap.set(cv2.CAP_PROP_SHARPNESS, 0.8)       # High sharpness for keypoint detection
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)  # Auto exposure
    cap.set(cv2.CAP_PROP_GAIN, 0.3)           # Low gain for clean image
    
    # Get actual camera settings
    actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    actual_fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"=== Rapoo Camera Configuration ===")
    print(f"Resolution: {actual_width}x{actual_height}")
    print(f"FPS: {actual_fps}")
    print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
    print(f"Saturation: {cap.get(cv2.CAP_PROP_SATURATION)}")
    print(f"Sharpness: {cap.get(cv2.CAP_PROP_SHARPNESS)}")
    print("================================")
    
    # Setup display window (scaled down for viewing)
    cv2.namedWindow("Rapoo - Human Pose Tracking (Full HD)", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rapoo - Human Pose Tracking (Full HD)", 1280, 720)  # Display size
    
    print("5. Starting human pose tracking pipeline...")
    print("   - Full HD capture: 1920x1080 @ 30 FPS")
    print("   - Uses keypoint detection for accurate human identification") 
    print("   - Focuses on chest center using shoulder/hip joints")
    print("   - Press 'q' to quit")
    
    try:
        while True:
            # Step 1: Capture Frame from Rapoo camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame from Rapoo camera")
                break
            
            H, W = frame.shape[:2]
            fx, fy = W / 2.0, H / 2.0  # Frame center
            
            # Draw frame center crosshair
            cv2.drawMarker(frame, (int(fx), int(fy)), (255, 255, 255), 
                          cv2.MARKER_CROSS, 20, 2)
            
            # Step 2: Detect Humans with Pose (YOLOv8-pose)
            results = model.predict(source=frame, verbose=False, conf=CONF_THRES)
            
            # Prepare detections for DeepSORT
            detections = []
            pose_data = {}  # Store keypoints for each detection
            
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for i, box in enumerate(r.boxes):
                        # Extract box coordinates and confidence
                        xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
                        conf = float(box.conf.cpu().numpy().item())
                        x1, y1, x2, y2 = xyxy.tolist()
                        
                        # Get keypoints if available
                        keypoints = None
                        if hasattr(r, 'keypoints') and r.keypoints is not None and i < len(r.keypoints.data):
                            kp_data = r.keypoints.data[i].cpu().numpy()  # Shape: (17, 3) for YOLOv8-pose
                            keypoints = kp_data.tolist()
                        
                        # Validate human using keypoints
                        if keypoints is not None:
                            is_human, reason = validate_human_keypoints(keypoints)
                            
                            if is_human:
                                # Create tight human-shape bounding box
                                human_bbox = create_human_shape_bbox(keypoints)
                                if human_bbox is not None:
                                    x1, y1, x2, y2 = human_bbox
                                
                                # Format for DeepSORT: [x1, y1, x2, y2], confidence, class
                                detections.append(([x1, y1, x2, y2], conf, 0))
                                pose_data[len(detections)-1] = keypoints
                            else:
                                print(f"Rejected detection: {reason}")
            
            # Step 3: Track Humans (DeepSORT)
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Collect active tracks with pose data
            active_tracks = {}
            for track in tracks:
                if track.is_confirmed() and track.time_since_update == 0:
                    tid = track.track_id
                    bbox = track.to_ltrb()  # left, top, right, bottom
                    x1, y1, x2, y2 = map(int, bbox)
                    
                    # Find corresponding keypoints
                    keypoints = None
                    if pose_data:
                        # Use the first available keypoints (simplified matching)
                        keypoints = list(pose_data.values())[0] if pose_data else None
                    
                    active_tracks[tid] = ([x1, y1, x2, y2], keypoints)
            
            # Automatic target selection - choose largest valid human
            if active_tracks:
                # Auto-select largest person if no target or target lost
                if selected_track_id is None or selected_track_id not in active_tracks:
                    selected_track_id = get_largest_person(active_tracks)
                    if selected_track_id:
                        print(f"Auto-selected largest person: ID {selected_track_id}")
            else:
                if selected_track_id is not None:
                    selected_track_id = None
                    print("No valid humans detected, target unselected")
            
            # Step 4: Compute Human Chest Center using keypoints
            target_center = None
            if selected_track_id is not None and selected_track_id in active_tracks:
                bbox, keypoints = active_tracks[selected_track_id]
                cx, cy = get_human_chest_center_from_keypoints(keypoints)
                if cx is not None and cy is not None:
                    target_center = (cx, cy)
            
            # Step 5: Draw all tracked humans
            for tid, (bbox, keypoints) in active_tracks.items():
                x1, y1, x2, y2 = bbox
                
                # Color: Green for selected target, Blue for others
                color = (0, 255, 0) if tid == selected_track_id else (255, 0, 0)
                
                # Validate using keypoints
                is_valid, reason = validate_human_keypoints(keypoints)
                
                if is_valid:
                    # Draw human-shape bounding box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                    
                    # Draw keypoints
                    if keypoints is not None:
                        for i, kp in enumerate(keypoints):
                            if len(kp) >= 3:
                                x, y, conf = kp[:3]
                                if conf > 0.3 and x > 0 and y > 0:
                                    cv2.circle(frame, (int(x), int(y)), 3, color, -1)
                    
                    # Draw chest center using keypoints
                    chest_cx, chest_cy = get_human_chest_center_from_keypoints(keypoints)
                    if chest_cx is not None and chest_cy is not None:
                        cv2.circle(frame, (int(chest_cx), int(chest_cy)), 8, color, -1)
                        cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), 
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, color, 2)
                else:
                    # This shouldn't happen with pose detection, but just in case
                    cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 1)
                    cv2.putText(frame, f"ID: {tid} (INVALID: {reason[:20]})", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 1)
            
            # Draw target information and errors
            if target_center is not None:
                cx, cy = target_center
                
                # Get human info for display
                bbox, keypoints = active_tracks[selected_track_id]
                x1, y1, x2, y2 = bbox
                box_area = (x2 - x1) * (y2 - y1)
                is_valid, validation_reason = validate_human_keypoints(keypoints)
                
                # Calculate frame coverage percentage
                frame_area = W * H
                coverage = (box_area / frame_area) * 100
                
                # Calculate error values (target center vs frame center)
                dx = cx - fx  # horizontal error
                dy = cy - fy  # vertical error
                
                # Count visible keypoints
                visible_keypoints = 0
                if keypoints:
                    for kp in keypoints:
                        if len(kp) >= 3 and kp[2] > 0.3:
                            visible_keypoints += 1
                
                distance_category = "Very Close" if box_area > 20000 else \
                                  "Close" if box_area > 12000 else \
                                  "Medium" if box_area > 8000 else \
                                  "Far" if box_area > 4000 else "Very Far"
                
                # Draw line from frame center to target center
                cv2.arrowedLine(frame, (int(fx), int(fy)), (int(cx), int(cy)), 
                               (0, 255, 255), 3, tipLength=0.1)
                
                # Display target coordinates on Full HD frame
                cv2.putText(frame, f"Chest Center: ({int(cx)}, {int(cy)})", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (255, 255, 255), 3)
                
                # Display error values
                cv2.putText(frame, f"Error (dx, dy): ({int(dx)}, {int(dy)})", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 255), 3)
                
                # Display human validation information
                cv2.putText(frame, f"Distance: {distance_category} | Keypoints: {visible_keypoints}/17 | HUMAN VALIDATED", 
                           (20, 180), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                # Display target info
                cv2.putText(frame, f"Target: Human ID {selected_track_id} (Full HD Pose-Based Tracking)", 
                           (20, 240), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 3)
                
                # Print error values to console with enhanced info
                print(f"ID: {selected_track_id} | Chest: ({int(cx)}, {int(cy)}) | Error: ({int(dx)}, {int(dy)}) | {distance_category} | Keypoints: {visible_keypoints}/17 | HUMAN | Coverage: {coverage:.1f}%")
            else:
                # Display "no target" message
                cv2.putText(frame, "No valid humans detected", 
                           (20, 60), cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 0, 255), 3)
                cv2.putText(frame, "Waiting for human with visible keypoints...", 
                           (20, 120), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 3)
            
            # Display frame center coordinates
            cv2.putText(frame, f"Frame Center: ({int(fx)}, {int(fy)})", 
                       (20, H - 60), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Display number of detected humans
            cv2.putText(frame, f"Valid humans detected: {len(active_tracks)}", 
                       (20, H - 20), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 3)
            
            # Display camera info
            cv2.putText(frame, f"Rapoo Full HD: {W}x{H} @ {actual_fps:.1f} FPS", 
                       (W - 600, 40), cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 0), 3)
            
            # Show frame
            cv2.imshow("Rapoo - Human Pose Tracking (Full HD)", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Rapoo camera pipeline completed")

if __name__ == "__main__":
    main()
