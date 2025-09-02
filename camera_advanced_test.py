"""
Advanced Camera Test Script for Rapoo Camera
Uses multiple backends and methods to get clean video feed
"""

import cv2
import time
import os
import platform

# Fix Qt warnings
os.environ['QT_QPA_PLATFORM'] = 'xcb'

# Camera settings
CAM_INDEX = 2  # Your Rapoo camera

def test_all_backends():
    """Test all available camera backends"""
    backends = [
        ('DirectShow (Windows)', cv2.CAP_DSHOW),
        ('V4L2 (Linux)', cv2.CAP_V4L2),
        ('GStreamer', cv2.CAP_GSTREAMER),
        ('FFMPEG', cv2.CAP_FFMPEG),
        ('Default', cv2.CAP_ANY)
    ]
    
    print("=== Testing Camera Backends ===")
    working_backends = []
    
    for name, backend in backends:
        print(f"\nTesting {name}...")
        cap = cv2.VideoCapture(CAM_INDEX, backend)
        
        if cap.isOpened():
            ret, frame = cap.read()
            if ret and frame is not None:
                print(f"✅ {name} - Working")
                working_backends.append((name, backend))
                
                # Test frame quality
                height, width = frame.shape[:2]
                print(f"   Resolution: {width}x{height}")
                print(f"   Frame size: {frame.size} bytes")
                
                # Check for corruption (very dark or very bright frames indicate issues)
                mean_brightness = frame.mean()
                print(f"   Brightness: {mean_brightness:.1f}")
                
                if mean_brightness < 10 or mean_brightness > 245:
                    print(f"   ⚠️  Possible corruption detected")
                else:
                    print(f"   ✅ Frame looks healthy")
            else:
                print(f"❌ {name} - Failed to read frame")
        else:
            print(f"❌ {name} - Failed to open")
        
        cap.release()
    
    return working_backends

def test_with_gstreamer():
    """Try GStreamer pipeline for better compatibility"""
    print("\n=== Testing GStreamer Pipeline ===")
    
    # GStreamer pipeline for USB camera
    gst_pipeline = (
        f"v4l2src device=/dev/video{CAM_INDEX} ! "
        "video/x-raw,format=YUY2,width=1280,height=720,framerate=30/1 ! "
        "videoconvert ! "
        "appsink"
    )
    
    print(f"Pipeline: {gst_pipeline}")
    
    cap = cv2.VideoCapture(gst_pipeline, cv2.CAP_GSTREAMER)
    
    if cap.isOpened():
        print("✅ GStreamer pipeline opened successfully")
        return cap
    else:
        print("❌ GStreamer pipeline failed")
        return None

def clean_camera_with_v4l2():
    """Use V4L2 backend with optimal settings"""
    print("\n=== Testing V4L2 with Clean Settings ===")
    
    cap = cv2.VideoCapture(CAM_INDEX, cv2.CAP_V4L2)
    
    if not cap.isOpened():
        print("❌ V4L2 backend failed")
        return None
    
    # Set pixel format to uncompressed YUV
    cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
    
    # Set lower resolution first for stability
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    cap.set(cv2.CAP_PROP_FPS, 15)
    
    # Disable auto settings that might cause issues
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)  # Manual exposure
    cap.set(cv2.CAP_PROP_AUTOFOCUS, 0)         # Disable autofocus
    cap.set(cv2.CAP_PROP_AUTO_WB, 0)           # Disable auto white balance
    
    # Set buffer size to 1 for real-time
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    # Test frame read
    for i in range(10):  # Skip first few frames
        ret, frame = cap.read()
        if not ret:
            print(f"❌ Failed to read frame {i}")
            cap.release()
            return None
    
    print("✅ V4L2 clean setup successful")
    return cap

def main():
    print("Advanced Rapoo Camera Diagnostics")
    print("=================================")
    
    # Test all backends first
    working_backends = test_all_backends()
    
    if not working_backends:
        print("\n❌ No working backends found!")
        return
    
    print(f"\n✅ Found {len(working_backends)} working backends")
    
    # Try GStreamer first (often best for Linux)
    cap = test_with_gstreamer()
    
    # If GStreamer fails, try clean V4L2
    if cap is None:
        cap = clean_camera_with_v4l2()
    
    # If still fails, use best working backend
    if cap is None:
        print(f"\nFalling back to best backend: {working_backends[0][0]}")
        cap = cv2.VideoCapture(CAM_INDEX, working_backends[0][1])
        
        # Apply clean settings
        cap.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('Y', 'U', 'Y', 'V'))
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 15)
        cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if cap is None or not cap.isOpened():
        print("❌ All methods failed!")
        return
    
    # Print final settings
    print(f"\n=== Final Camera Settings ===")
    print(f"Resolution: {int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))}")
    print(f"FPS: {cap.get(cv2.CAP_PROP_FPS)}")
    print(f"FOURCC: {cap.get(cv2.CAP_PROP_FOURCC)}")
    print(f"Buffer Size: {cap.get(cv2.CAP_PROP_BUFFERSIZE)}")
    
    # Create window
    cv2.namedWindow("Clean Rapoo Camera", cv2.WINDOW_NORMAL)
    cv2.resizeWindow("Clean Rapoo Camera", 800, 600)
    
    print(f"\n=== Clean Camera Feed Started ===")
    print("Controls:")
    print("- Press 'q' to quit")
    print("- Press 's' to save screenshot")
    print("- Press SPACE to skip corrupted frames")
    
    frame_count = 0
    clean_frame_count = 0
    corrupted_frame_count = 0
    
    try:
        while True:
            ret, frame = cap.read()
            frame_count += 1
            
            if not ret:
                print(f"Frame {frame_count}: Failed to read")
                continue
            
            # Check frame health
            if frame is None or frame.size == 0:
                corrupted_frame_count += 1
                continue
            
            # Check for extreme brightness (corruption indicator)
            mean_brightness = frame.mean()
            if mean_brightness < 5 or mean_brightness > 250:
                corrupted_frame_count += 1
                print(f"Frame {frame_count}: Corrupted (brightness: {mean_brightness:.1f})")
                continue
            
            clean_frame_count += 1
            
            # Add info overlay
            height, width = frame.shape[:2]
            cv2.putText(frame, f"Frame: {frame_count} | Clean: {clean_frame_count} | Corrupted: {corrupted_frame_count}", 
                       (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}x{height} | Brightness: {mean_brightness:.1f}", 
                       (10, height - 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            cv2.putText(frame, "Press 'q' to quit, 's' for screenshot", 
                       (10, height - 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
            
            # Display frame
            cv2.imshow("Clean Rapoo Camera", frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            if key == ord('q'):
                break
            elif key == ord('s'):
                filename = f"rapoo_clean_screenshot_{clean_frame_count}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Clean screenshot saved: {filename}")
            elif key == ord(' '):  # Space to skip frames
                for _ in range(5):
                    cap.read()
                print("Skipped 5 frames")
                
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        print(f"\nStatistics:")
        print(f"Total frames: {frame_count}")
        print(f"Clean frames: {clean_frame_count}")
        print(f"Corrupted frames: {corrupted_frame_count}")
        if frame_count > 0:
            print(f"Success rate: {(clean_frame_count/frame_count)*100:.1f}%")
        
        cap.release()
        cv2.destroyAllWindows()
        print("Camera test completed")

if __name__ == "__main__":
    main()
