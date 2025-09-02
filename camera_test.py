"""
Simple Camera Test Script
Just displays raw camera feed to check video quality
No AI processing, no tracking - pure camera output

Controls:
- Press 'q' to quit
- Press 's' to save a screenshot
- Press 'f' to toggle fullscreen
"""

import cv2
import time
import os

# Fix Qt warnings
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Camera settings
CAM_INDEX = 2  # Your Rapoo camera

def test_camera_settings(cap):
    """Test different camera settings to find the best quality"""
    
    print("=== Testing Camera Settings ===")
    
    # Test different resolutions
    resolutions = [
        (640, 480, "480p"),
        (1280, 720, "720p"), 
        (1920, 1080, "1080p")
    ]
    
    for width, height, name in resolutions:
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
        
        actual_width = cap.get(cv2.CAP_PROP_FRAME_WIDTH)
        actual_height = cap.get(cv2.CAP_PROP_FRAME_HEIGHT)
        
        print(f"{name}: Requested {width}x{height}, Got {int(actual_width)}x{int(actual_height)}")
    
    # Test different codecs
    codecs = [
        ('MJPG', cv2.VideoWriter_fourcc(*'MJPG')),
        ('YUYV', cv2.VideoWriter_fourcc(*'YUYV')),
        ('H264', cv2.VideoWriter_fourcc(*'H264'))
    ]
    
    print("\n=== Testing Codecs ===")
    for name, fourcc in codecs:
        cap.set(cv2.CAP_PROP_FOURCC, fourcc)
        actual_fourcc = cap.get(cv2.CAP_PROP_FOURCC)
        print(f"{name}: Set {fourcc}, Got {int(actual_fourcc)}")

def main():
    print("Starting Rapoo Camera Test...")
    
    # Initialize camera
    cap = cv2.VideoCapture(CAM_INDEX)
    if not cap.isOpened():
        print(f"Error: Cannot open camera {CAM_INDEX}")
        print("Try different camera indices: 0, 1, 2, 3...")
        return
    
    # Test camera capabilities
    test_camera_settings(cap)
    
    # Set optimal settings for quality testing
    print("\n=== Setting Optimal Configuration ===")
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc(*'MJPG'))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Camera quality settings
    cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
    cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
    cap.set(cv2.CAP_PROP_SATURATION, 0.5)
    cap.set(cv2.CAP_PROP_SHARPNESS, 0.7)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.75)
    
    # Print final settings
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"Brightness: {cap.get(cv2.CAP_PROP_BRIGHTNESS)}")
    print(f"Contrast: {cap.get(cv2.CAP_PROP_CONTRAST)}")
    
    # Create window
    cv2.namedWindow("Rapoo Camera Test", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Rapoo Camera Test", 1280, 720)
    
    # Variables for FPS calculation
    fps_start_time = time.time()
    fps_frame_count = 0
    screenshot_count = 0
    fullscreen = False
    
    print("\n=== Camera Feed Started ===")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save screenshot")
    print("- Press 'f' to toggle fullscreen")
    print("- Press 'r' to reset settings")
    
    try:
        while True:
            # Read frame
            ret, frame = cap.read()
            if not ret:
                print("Failed to read frame")
                break
            
            # Calculate FPS
            fps_frame_count += 1
            if fps_frame_count % 30 == 0:  # Update every 30 frames
                fps_end_time = time.time()
                fps = 30 / (fps_end_time - fps_start_time)
                fps_start_time = fps_end_time
                
                # Add FPS text to frame
                cv2.putText(frame, f"FPS: {fps:.1f}", (10, 30), 
                           cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            
            # Add info overlay
            height, width = frame.shape[:2]
            cv2.putText(frame, f"Resolution: {width}x{height}", (10, height - 60), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' for screenshot", (10, height - 30), 
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Rapoo Camera Test", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                screenshot_count += 1
                filename = f"rapoo_camera_screenshot_{screenshot_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
            elif key == ord('f'):
                if fullscreen:
                    cv2.setWindowProperty("Rapoo Camera Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_NORMAL)
                    fullscreen = False
                else:
                    cv2.setWindowProperty("Rapoo Camera Test", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
                    fullscreen = True
            elif key == ord('r'):
                print("Resetting camera settings...")
                cap.set(cv2.CAP_PROP_BRIGHTNESS, 0.5)
                cap.set(cv2.CAP_PROP_CONTRAST, 0.6)
                cap.set(cv2.CAP_PROP_SATURATION, 0.5)
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        cap.release()
        cv2.destroyAllWindows()
        print("Camera test completed")

if __name__ == "__main__":
    main()
