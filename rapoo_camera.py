#!/usr/bin/env python3
"""
Simple USB Camera Video Script for Rapoo Camera
This script captures and displays video from a USB connected camera.
"""

import cv2
import sys

def main():
    """Main function to run the USB camera video capture."""
    
    # Try different camera indices (0, 1, 2, etc.) to find your Rapoo camera
    camera_index = 2
    
    print("Starting Rapoo USB Camera...")
    print("Press 'q' to quit, 's' to take screenshot")
    
    # Initialize the camera
    cap = cv2.VideoCapture(camera_index)
    
    if not cap.isOpened():
        print(f"Error: Could not open camera at index {camera_index}")
        print("Try changing the camera_index value (0, 1, 2, etc.)")
        return False
    
    # Set camera properties (optional - adjust as needed)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
    cap.set(cv2.CAP_PROP_FPS, 30)
    
    # Get actual camera properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    print(f"Camera initialized: {width}x{height} @ {fps} FPS")
    
    screenshot_count = 0
    
    try:
        while True:
            # Capture frame-by-frame
            ret, frame = cap.read()
            
            if not ret:
                print("Error: Failed to capture frame")
                break
            
            # Display the frame
            cv2.imshow('Rapoo USB Camera', frame)
            
            # Handle key presses
            key = cv2.waitKey(1) & 0xFF
            
            if key == ord('q'):
                print("Quitting...")
                break
            elif key == ord('s'):
                # Take screenshot
                screenshot_count += 1
                filename = f"screenshot_{screenshot_count:03d}.jpg"
                cv2.imwrite(filename, frame)
                print(f"Screenshot saved: {filename}")
    
    except KeyboardInterrupt:
        print("\nInterrupted by user")
    
    finally:
        # Clean up
        cap.release()
        cv2.destroyAllWindows()
        print("Camera released and windows closed")
    
    return True

def list_cameras():
    """List available cameras to help find the correct index."""
    print("Available cameras:")
    for i in range(10):  # Check first 10 indices
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            ret, _ = cap.read()
            if ret:
                print(f"  Camera {i}: Available")
            else:
                print(f"  Camera {i}: Connected but no video")
            cap.release()
        else:
            break
    print()

if __name__ == "__main__":
    print("Rapoo USB Camera Script")
    print("=" * 30)
    
    # First, list available cameras
    list_cameras()
    
    # Run the main camera function
    success = main()
    
    if not success:
        print("\nTroubleshooting tips:")
        print("1. Make sure your Rapoo camera is connected")
        print("2. Try different camera_index values (0, 1, 2, etc.)")
        print("3. Check if the camera is being used by another application")
        print("4. Make sure you have proper permissions to access the camera")
        sys.exit(1)
