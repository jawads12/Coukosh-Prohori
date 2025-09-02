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
CONF_THRES = 0.5          # YOLO confidence threshold
TARGET_CLASS = 0          # Person class ID

# Global variables
selected_track_id = None
person_cycle_index = 0

def get_human_chest_center(bbox):
    """Calculate the chest/torso center of a human bounding box with distance compensation"""
    x1, y1, x2, y2 = bbox
    
    # Calculate bounding box dimensions
    width = x2 - x1
    height = y2 - y1
    box_area = width * height
    
    # Horizontal center (always middle)
    cx = x1 + width / 2.0
    
    # Distance-aware chest center calculation
    # Larger box = closer person, smaller box = farther person
    if box_area > 25000:      # Very close (large in frame)
        chest_ratio = 0.25    # Chest higher in frame when very close
    elif box_area > 15000:    # Close
        chest_ratio = 0.27    # Standard close distance
    elif box_area > 8000:     # Medium distance
        chest_ratio = 0.30    # Slightly lower
    elif box_area > 4000:     # Far
        chest_ratio = 0.33    # Compensate for perspective
    else:                     # Very far (small in frame)
        chest_ratio = 0.36    # More compensation needed
    
    # Calculate distance-compensated chest Y coordinate
    cy = y1 + height * chest_ratio
    
    return cx, cy

def get_largest_person(active_tracks):
    """Get the track ID of the largest detected person (closest/most prominent)"""
    if not active_tracks:
        return None
    
    largest_area = 0
    largest_tid = None
    
    for tid, (x1, y1, x2, y2) in active_tracks.items():
        area = (x2 - x1) * (y2 - y1)
        if area > largest_area:
            largest_area = area
            largest_tid = tid
    
    return largest_tid

def main():
    global selected_track_id, person_cycle_index
    
    print("=== Automatic Person Tracking Pipeline ===")
    print("1. Loading YOLO model...")
    model = YOLO('yolov8n.pt')
    
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
    cv2.namedWindow("Auto Person Tracking - Chest Center", cv2.WINDOW_NORMAL)
    
    print("4. Starting automatic tracking pipeline...")
    print("   - Automatically targets largest detected person")
    print("   - Focuses on chest/torso center")
    print("   - Press 'n' to cycle through persons, 'q' to quit")
    
    try:
        while True:
            # Step 1: Capture Frame from camera
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            H, W = frame.shape[:2]
            
            # Step 2: Detect Humans (YOLO)
            results = model.predict(source=frame, verbose=False, conf=CONF_THRES, classes=[TARGET_CLASS])
            
            # Prepare detections for DeepSORT
            detections = []
            yolo_boxes = []
            
            if results and len(results) > 0:
                r = results[0]
                if r.boxes is not None and len(r.boxes) > 0:
                    for box in r.boxes:
                        # Extract box coordinates and confidence
                        xyxy = box.xyxy.cpu().numpy().astype(int).flatten()
                        conf = float(box.conf.cpu().numpy().item())
                        x1, y1, x2, y2 = xyxy.tolist()
                        
                        # Format for DeepSORT: [x1, y1, x2, y2], confidence, class
                        detections.append(([x1, y1, x2, y2], conf, 0))
                        yolo_boxes.append((x1, y1, x2, y2))
            
            # Step 3: Track Selected Person (DeepSORT)
            tracks = tracker.update_tracks(detections, frame=frame)
            
            # Collect active tracks
            active_tracks = {}
            for track in tracks:
                if track.is_confirmed() and track.time_since_update == 0:
                    tid = track.track_id
                    bbox = track.to_ltrb()  # left, top, right, bottom
                    x1, y1, x2, y2 = map(int, bbox)
                    active_tracks[tid] = (x1, y1, x2, y2)
            
            # Automatic target selection - choose largest person
            if active_tracks:
                # Auto-select largest person if no target or target lost
                if selected_track_id is None or selected_track_id not in active_tracks:
                    selected_track_id = get_largest_person(active_tracks)
                    if selected_track_id:
                        print(f"Auto-selected largest person: ID {selected_track_id}")
            else:
                if selected_track_id is not None:
                    selected_track_id = None
                    print("No persons detected, target unselected")
            
            # Step 4: Compute Human Chest Center (cx, cy) 
            target_center = None
            if selected_track_id is not None and selected_track_id in active_tracks:
                bbox = active_tracks[selected_track_id]
                cx, cy = get_human_chest_center(bbox)  # Get chest center instead of bbox center
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
            
            # Draw all detected persons
            for tid, (x1, y1, x2, y2) in active_tracks.items():
                # Color: Green for selected target, Blue for others
                color = (0, 255, 0) if tid == selected_track_id else (255, 0, 0)
                
                # Draw bounding box
                cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
                
                # Draw chest center (not bbox center)
                chest_cx, chest_cy = get_human_chest_center((x1, y1, x2, y2))
                cv2.circle(frame, (int(chest_cx), int(chest_cy)), 6, color, -1)
                
                # Draw cross-hair on chest center for selected target
                if tid == selected_track_id:
                    cv2.line(frame, (int(chest_cx - 8), int(chest_cy)), (int(chest_cx + 8), int(chest_cy)), (0, 255, 255), 2)
                    cv2.line(frame, (int(chest_cx), int(chest_cy - 8)), (int(chest_cx), int(chest_cy + 8)), (0, 255, 255), 2)
                
                # Draw track ID
                cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
            
            # Draw target information and errors
            if target_center is not None:
                cx, cy = target_center
                
                # Get distance info for display
                bbox = active_tracks[selected_track_id]
                x1, y1, x2, y2 = bbox
                box_area = (x2 - x1) * (y2 - y1)
                distance_category = "Very Close" if box_area > 25000 else \
                                  "Close" if box_area > 15000 else \
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
                
                # Display distance information
                cv2.putText(frame, f"Distance: {distance_category} (Area: {int(box_area)})", 
                           (10, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 0), 2)
                
                # Display target info
                cv2.putText(frame, f"Target: Person ID {selected_track_id} (Auto-selected)", 
                           (10, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                
                # Print error values to console with distance info
                print(f"ID: {selected_track_id} | Chest: ({int(cx)}, {int(cy)}) | Error: ({int(dx)}, {int(dy)}) | {distance_category} (Area: {int(box_area)})")
            else:
                cv2.putText(frame, "Waiting for person detection...", 
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame center coordinates
            cv2.putText(frame, f"Frame Center: ({int(fx)}, {int(fy)})", 
                       (10, H - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Display number of detected persons
            cv2.putText(frame, f"Persons detected: {len(active_tracks)}", 
                       (10, H - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
            
            # Show frame
            cv2.imshow("Auto Person Tracking - Chest Center", frame)
            
            # Handle keyboard input
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('n') and active_tracks:
                # Cycle through detected persons
                track_ids = list(active_tracks.keys())
                if selected_track_id in track_ids:
                    current_index = track_ids.index(selected_track_id)
                    next_index = (current_index + 1) % len(track_ids)
                    selected_track_id = track_ids[next_index]
                else:
                    selected_track_id = track_ids[0]
                print(f"Cycled to target: Person ID {selected_track_id}")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Pipeline completed")

if __name__ == "__main__":
    main()
