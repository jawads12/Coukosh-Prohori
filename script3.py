"""
Script3: Clean Person Tracking Pipeline
Workflow:
1. Capture Frame from camera
2. Detect Humans (YOLO)
3. Automatically target the largest/closest person
4. Compu            # Step 4: Compute Target Center (cx, cy) - Using chest center with distance compensation
            target_center = None
            if selected_track_id is not None and selected_track_id in active_tracks:
                bbox = active_tracks[selected_track_id]
                cx, cy = get_human_chest_center(bbox)  # Get chest center instead of bbox center
                target_center = (cx, cy)
                
                # Calculate distance info for debugging
                x1, y1, x2, y2 = bbox
                box_area = (x2 - x1) * (y2 - y1)
                distance_category = "Very Close" if box_area > 25000 else 
                                  "Close" if box_area > 15000 else 
                                  "Medium" if box_area > 8000 else 
                                  "Far" if box_area > 4000 else "Very Far"an Chest Center (cx, cy) - Upper body focus
5. Compare with Frame Center (fx, fy)
6. Generate Error Values (dx, dy)

Controls:
- Automatically selects the largest detected person
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

# Configuration
CAM_INDEX = 0              # Default webcam
CONF_THRES = 0.4          # YOLO confidence threshold (lowered for better detection)
TARGET_CLASS = 0          # Person class ID

# Global variables
selected_track_id = None
person_cycle_index = 0

# Human keypoint indices for YOLOv8 pose model
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

def create_human_shape_bbox(keypoints, padding_ratio=0.15):
    """Create tight bounding box around actual human keypoints only"""
    if keypoints is None or len(keypoints) == 0:
        return None
    
    # Extract valid keypoints (confidence > 0.3)
    valid_points = []
    for kp in keypoints:
        if len(kp) >= 3:
            x, y, conf = kp[:3]
            if conf > 0.3 and x > 0 and y > 0:
                valid_points.append([x, y])
    
    if len(valid_points) < 3:  # Need at least 3 points for meaningful bbox
        return None
    
    valid_points = np.array(valid_points)
    
    # Find tight bounding box around keypoints
    x_min, y_min = np.min(valid_points, axis=0)
    x_max, y_max = np.max(valid_points, axis=0)
    
    # Add padding to ensure we don't crop the human
    width = x_max - x_min
    height = y_max - y_min
    
    x_padding = width * padding_ratio
    y_padding = height * padding_ratio
    
    x1 = max(0, x_min - x_padding)
    y1 = max(0, y_min - y_padding)
    x2 = x_max + x_padding
    y2 = y_max + y_padding
    
    return [int(x1), int(y1), int(x2), int(y2)]

def get_human_chest_center_from_keypoints(keypoints):
    """Calculate chest center using shoulder and hip keypoints"""
    if keypoints is None or len(keypoints) < 17:
        return None, None
    
    # Get shoulder and hip coordinates
    left_shoulder = keypoints[5][:3] if len(keypoints[5]) >= 3 else [0, 0, 0]
    right_shoulder = keypoints[6][:3] if len(keypoints[6]) >= 3 else [0, 0, 0]
    left_hip = keypoints[11][:3] if len(keypoints[11]) >= 3 else [0, 0, 0]
    right_hip = keypoints[12][:3] if len(keypoints[12]) >= 3 else [0, 0, 0]
    
    # Find valid points for chest calculation
    valid_shoulders = []
    valid_hips = []
    
    if left_shoulder[2] > 0.3:  # confidence > 0.3
        valid_shoulders.append(left_shoulder[:2])
    if right_shoulder[2] > 0.3:
        valid_shoulders.append(right_shoulder[:2])
    if left_hip[2] > 0.3:
        valid_hips.append(left_hip[:2])
    if right_hip[2] > 0.3:
        valid_hips.append(right_hip[:2])
    
    # Calculate chest center
    if len(valid_shoulders) >= 1 and len(valid_hips) >= 1:
        # Use average of shoulders and hips
        shoulder_center = np.mean(valid_shoulders, axis=0)
        hip_center = np.mean(valid_hips, axis=0)
        
        # Chest is between shoulders and hips (closer to shoulders)
        chest_x = shoulder_center[0]
        chest_y = shoulder_center[1] + (hip_center[1] - shoulder_center[1]) * 0.3
        
        return chest_x, chest_y
    
    elif len(valid_shoulders) >= 1:
        # Use shoulders only, estimate chest below
        shoulder_center = np.mean(valid_shoulders, axis=0)
        chest_x = shoulder_center[0]
        chest_y = shoulder_center[1] + 50  # Estimate chest 50 pixels below shoulders
        
        return chest_x, chest_y
    
    return None, None

def get_largest_person(active_tracks):
    """Get the track ID of the largest valid detected person (closest/most prominent)"""
    if not active_tracks:
        return None
    
    largest_area = 0
    largest_tid = None
    
    for tid, track_data in active_tracks.items():
        bbox, keypoints = track_data
        x1, y1, x2, y2 = bbox
        
        # Validate using keypoints instead of bounding box
        is_valid, _ = validate_human_keypoints(keypoints)
        
        if is_valid:
            area = (x2 - x1) * (y2 - y1)
            if area > largest_area:
                largest_area = area
                largest_tid = tid
    
    return largest_tid

def main():
    global selected_track_id, person_cycle_index
    
    print("=== Human Pose-Based Tracking Pipeline ===")
    print("1. Loading YOLOv8 Pose model...")
    model = YOLO('yolov8n-pose.pt')  # Use pose detection model
    
    print("2. Initializing DeepSORT tracker...")
    tracker = DeepSort(max_age=30, n_init=3)
    
    print("3. Setting up camera...")
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAM_INDEX}")
        return
    
    # Camera settings for best quality
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    print(f"Camera resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    
    # Setup window
    cv2.namedWindow("Human Pose Tracking - Chest Center", cv2.WINDOW_NORMAL)
    
    print("4. Starting human pose tracking pipeline...")
    print("   - Uses keypoint detection for accurate human identification")
    print("   - Focuses on chest center using shoulder/hip joints")
    print("   - Press 'q' to quit")
    
    try:
        while True:
            # Step 1: Capture Frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            H, W = frame.shape[:2]
            
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
                    
                    # Find corresponding keypoints (this is simplified - in practice you'd need better matching)
                    keypoints = None
                    if pose_data:
                        # Use the first available keypoints (this could be improved)
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
            
            # Step 5: Compare with Frame Center (fx, fy)
            fx, fy = W / 2.0, H / 2.0
            
            # Step 6: Generate Error Values (dx, dy)
            dx, dy = 0, 0
            if target_center is not None:
                cx, cy = target_center
                dx = cx - fx
                dy = cy - fy
            
            # === VISUALIZATION ===
            
            # Draw frame center
            cv2.line(frame, (int(fx - 10), int(fy)), (int(fx + 10), int(fy)), (0, 255, 255), 2)
            cv2.line(frame, (int(fx), int(fy - 10)), (int(fx), int(fy + 10)), (0, 255, 255), 2)
            
            # Draw all detected humans with keypoints
            for tid, track_data in active_tracks.items():
                bbox, keypoints = track_data
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
                        
                        # Draw cross-hair on chest center for selected target
                        if tid == selected_track_id:
                            cv2.line(frame, (int(chest_cx - 12), int(chest_cy)), (int(chest_cx + 12), int(chest_cy)), (0, 255, 255), 3)
                            cv2.line(frame, (int(chest_cx), int(chest_cy - 12)), (int(chest_cx), int(chest_cy + 12)), (0, 255, 255), 3)
                    
                    # Draw track ID
                    cv2.putText(frame, f"ID: {tid} (HUMAN)", (x1, y1 - 10), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
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
                               (0, 255, 255), 2, tipLength=0.1)
                
                # Display target coordinates
                cv2.putText(frame, f"Chest Center: ({int(cx)}, {int(cy)})", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Display error values
                cv2.putText(frame, f"Error (dx, dy): ({int(dx)}, {int(dy)})", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
                
                # Display human validation information
                cv2.putText(frame, f"Distance: {distance_category} | Keypoints: {visible_keypoints}/17 | HUMAN VALIDATED", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Display target info
                cv2.putText(frame, f"Target: Human ID {selected_track_id} (Pose-Based Tracking)", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Print error values to console with enhanced info
                print(f"ID: {selected_track_id} | Chest: ({int(cx)}, {int(cy)}) | Error: ({int(dx)}, {int(dy)}) | {distance_category} | Keypoints: {visible_keypoints}/17 | HUMAN")
            else:
                # Display "no target" message
                cv2.putText(frame, "No valid humans detected", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, "Waiting for human with visible keypoints...", 
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 0, 255), 2)
            
            # Display frame center coordinates
            cv2.putText(frame, f"Frame Center: ({int(fx)}, {int(fy)})", 
                       (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display number of detected humans
            cv2.putText(frame, f"Valid humans detected: {len(active_tracks)}", 
                       (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Human Pose Tracking - Chest Center", frame)
            
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
        print("Pipeline completed")

if __name__ == "__main__":
    main()
