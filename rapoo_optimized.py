#!/usr/bin/env python3
"""
Optimized USB Camera Video Script for Rapoo Camera
This script captures and displays video from USB camera with maximum FPS.
"""

import cv2
import sys
import time

def test_camera_capabilities(camera_index):
    """Test different camera settings to find optimal configuration."""
    print(f"\n=== Testing Camera {camera_index} Capabilities ===")
    
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Camera {camera_index} not available")
        return None
    
    # Test different backends
    backends = [
        (cv2.CAP_V4L2, "V4L2"),
        (cv2.CAP_FFMPEG, "FFMPEG"), 
        (cv2.CAP_GSTREAMER, "GSTREAMER")
    ]
    
    best_config = None
    best_fps = 0
    
    for backend_id, backend_name in backends:
        cap.release()
        cap = cv2.VideoCapture(camera_index, backend_id)
        
        if not cap.isOpened():
            continue
            
        print(f"\nTesting {backend_name} backend:")
        
        # Test different resolutions and FPS combinations
        test_configs = [
            (1920, 1080, 30),  # Full HD
            (1920, 1080, 24),  # Full HD reduced FPS
            (1280, 720, 30),   # HD
            (1280, 720, 60),   # HD high FPS
            (640, 480, 30),    # Standard
            (640, 480, 60),    # Standard high FPS
        ]
        
        for width, height, fps in test_configs:
            # Set camera properties
            cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
            cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
            cap.set(cv2.CAP_PROP_FPS, fps)
            cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
            cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Reduce buffer for lower latency
            
            # Get actual values
            actual_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = cap.get(cv2.CAP_PROP_FPS)
            
            # Test actual FPS by measuring frame capture time
            start_time = time.time()
            frame_count = 0
            
            for _ in range(10):  # Test 10 frames
                ret, frame = cap.read()
                if ret:
                    frame_count += 1
            
            elapsed_time = time.time() - start_time
            measured_fps = frame_count / elapsed_time if elapsed_time > 0 else 0
            
            print(f"  {width}x{height}@{fps}: Actual={actual_width}x{actual_height}@{actual_fps:.1f}, Measured={measured_fps:.1f} FPS")
            
            # Track best configuration
            if measured_fps > best_fps and frame_count >= 8:  # At least 80% success rate
                best_fps = measured_fps
                best_config = {
                    'backend': backend_id,
                    'backend_name': backend_name,
                    'width': actual_width,
                    'height': actual_height,
                    'fps': actual_fps,
                    'measured_fps': measured_fps
                }
    
    cap.release()
    return best_config

def main():
    """Main function with optimized camera settings."""
    
    print("Rapoo Camera FPS Optimization")
    print("=" * 40)
    
    # Test all available cameras
    best_camera = None
    best_config = None
    
    for camera_index in range(5):  # Test cameras 0-4
        config = test_camera_capabilities(camera_index)
        if config and (best_config is None or config['measured_fps'] > best_config['measured_fps']):
            best_camera = camera_index
            best_config = config
    
    if not best_config:
        print("No suitable camera found!")
        return False
    
    print(f"\n🎯 BEST CONFIGURATION FOUND:")
    print(f"Camera Index: {best_camera}")
    print(f"Backend: {best_config['backend_name']}")
    print(f"Resolution: {best_config['width']}x{best_config['height']}")
    print(f"Target FPS: {best_config['fps']}")
    print(f"Measured FPS: {best_config['measured_fps']:.1f}")
    print("\nStarting optimized camera...")
    
    # Initialize camera with best settings
    cap = cv2.VideoCapture(best_camera, best_config['backend'])
    
    if not cap.isOpened():
        print("Error: Could not open camera with best settings")
        return False
    
    # Apply optimal settings
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, best_config['width'])
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, best_config['height'])
    cap.set(cv2.CAP_PROP_FPS, best_config['fps'])
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Additional optimizations
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Disable auto exposure for consistent FPS
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Disable autofocus for consistent FPS
    
    print("Press 'q' to quit, 's' for screenshot, 'f' to show FPS")
    
    screenshot_count = 0
    show_fps = True
    fps_counter = 0
    start_time = time.time()
    
    try:
        while True:
            frame_start = time.time()
            
            # Capture frame
            ret, frame = cap.read()
            
            if not ret:
                print("Warning: Failed to capture frame")
                continue
            
            # Calculate FPS
            fps_counter += 1
            if fps_counter % 30 == 0:  # Update every 30 frames
                elapsed = time.time() - start_time
                current_fps = 30 / elapsed
                start_time = time.time()
                
                if show_fps:
                    print(f"Current FPS: {current_fps:.1f}")
            
            # Add FPS overlay to frame
            if show_fps:
                elapsed = time.time() - start_time if fps_counter % 30 != 0 else time.time() - frame_start
                instant_fps = 1.0 / (time.time() - frame_start) if time.time() - frame_start > 0 else 0
                cv2.putText(frame, f"FPS: {instant_fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Display frame
            cv2.imshow('Optimized Rapoo Camera', frame)
            
            # Handle keys
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('f'):
                show_fps = not show_fps
                print(f"FPS display: {'ON' if show_fps else 'OFF'}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released")
    
    return True

if __name__ == "__main__":
    success = main()
    
    if not success:
        print("\nTroubleshooting:")
        print("1. Check USB connection and try different USB ports")
        print("2. Close other applications using the camera")
        print("3. Try running with sudo for camera permissions")
        print("4. Update camera drivers")
        sys.exit(1)