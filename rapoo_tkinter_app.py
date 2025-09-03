#!/usr/bin/env python3
"""
Rapoo Tkinter GUI Application
A comprehensive Tkinter interface implementing all features from rapoo_low_fps_optimized.py
Professional GUI for Rapoo camera human tracking system with optimized low FPS performance
"""

import tkinter as tk
from tkinter import ttk, messagebox, filedialog, scrolledtext
import cv2
import numpy as np
from PIL import Image, ImageTk
import threading
import time
import json
import subprocess
import os
import sys
from datetime import datetime
import queue
import logging

# Import tracking modules
try:
    from ultralytics import YOLO
    from deep_sort_realtime.deepsort_tracker import DeepSort
    TRACKING_AVAILABLE = True
except ImportError:
    TRACKING_AVAILABLE = False

class RapooTrackingApp:
    def __init__(self, root):
        self.root = root
        self.setup_ui()
        self.initialize_variables()
        self.setup_tracking_constants()
        self.setup_logging()
        
    def setup_logging(self):
        """Setup logging system"""
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s - %(levelname)s - %(message)s'
        )
        self.logger = logging.getLogger(__name__)

    def setup_ui(self):
        """Create the main user interface"""
        self.root.title("Rapoo Camera - Advanced Human Tracking System")
        self.root.geometry("1600x1000")
        self.root.configure(bg='#1e1e1e')
        
        # Configure styles
        self.style = ttk.Style()
        self.style.theme_use('clam')
        self.configure_styles()
        
        # Create main layout
        self.create_main_layout()
        
        # Create menu bar
        self.create_menu_bar()
        
        # Create status bar
        self.create_status_bar()
        
        # Bind closing event
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def configure_styles(self):
        """Configure custom styles for dark theme"""
        self.style.configure('Dark.TFrame', background='#1e1e1e')
        self.style.configure('Dark.TLabel', background='#1e1e1e', foreground='white')
        self.style.configure('Dark.TButton', background='#404040', foreground='white')
        self.style.configure('Success.TButton', background='#4CAF50', foreground='white')
        self.style.configure('Danger.TButton', background='#f44336', foreground='white')
        self.style.configure('Info.TButton', background='#2196F3', foreground='white')
        self.style.configure('Warning.TButton', background='#FF9800', foreground='white')

    def create_main_layout(self):
        """Create the main application layout"""
        # Main container
        self.main_frame = ttk.Frame(self.root, style='Dark.TFrame')
        self.main_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=5)

        # Create paned window for resizable layout
        self.paned_window = ttk.PanedWindow(self.main_frame, orient=tk.HORIZONTAL)
        self.paned_window.pack(fill=tk.BOTH, expand=True)

        # Left panel - Video and controls
        self.left_panel = ttk.Frame(self.paned_window, style='Dark.TFrame')
        self.paned_window.add(self.left_panel, weight=3)

        # Right panel - Configuration and information
        self.right_panel = ttk.Frame(self.paned_window, style='Dark.TFrame')
        self.paned_window.add(self.right_panel, weight=1)

        # Create left panel contents
        self.create_video_section()
        self.create_control_section()

        # Create right panel contents
        self.create_info_section()
        self.create_config_section()
        self.create_logs_section()

    def create_video_section(self):
        """Create video display section"""
        video_frame = ttk.LabelFrame(self.left_panel, text="Rapoo Camera Feed", style='Dark.TFrame')
        video_frame.pack(fill=tk.BOTH, expand=True, pady=(0, 10))

        # Video canvas
        self.video_canvas = tk.Canvas(
            video_frame,
            bg='black',
            width=960,
            height=540,
            highlightthickness=0
        )
        self.video_canvas.pack(expand=True, fill=tk.BOTH, padx=5, pady=5)

        # Placeholder text
        self.video_canvas.create_text(
            480, 270,
            text="Rapoo Camera Feed\nClick 'Start Tracking' to begin\nOptimized for 7-9 FPS cameras",
            fill='white',
            font=('Arial', 16),
            justify=tk.CENTER,
            tags="placeholder"
        )

    def create_control_section(self):
        """Create control buttons section"""
        control_frame = ttk.LabelFrame(self.left_panel, text="Controls", style='Dark.TFrame')
        control_frame.pack(fill=tk.X, pady=5)

        # Main controls
        main_controls = ttk.Frame(control_frame, style='Dark.TFrame')
        main_controls.pack(side=tk.LEFT, padx=10, pady=10)

        self.start_button = ttk.Button(
            main_controls,
            text="🎯 Start Tracking",
            style='Success.TButton',
            command=self.start_tracking,
            width=15
        )
        self.start_button.pack(side=tk.LEFT, padx=(0, 10))

        self.stop_button = ttk.Button(
            main_controls,
            text="⏹ Stop Tracking",
            style='Danger.TButton',
            command=self.stop_tracking,
            state='disabled',
            width=15
        )
        self.stop_button.pack(side=tk.LEFT, padx=(0, 10))

        self.test_button = ttk.Button(
            main_controls,
            text="📷 Test Camera",
            style='Info.TButton',
            command=self.test_camera,
            width=15
        )
        self.test_button.pack(side=tk.LEFT, padx=(0, 10))

        # Quick actions
        quick_actions = ttk.Frame(control_frame, style='Dark.TFrame')
        quick_actions.pack(side=tk.RIGHT, padx=10, pady=10)

        ttk.Button(
            quick_actions,
            text="📷 Screenshot",
            command=self.take_screenshot,
            width=12
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            quick_actions,
            text="🎥 Record",
            command=self.toggle_recording,
            width=12
        ).pack(side=tk.LEFT, padx=(0, 5))

        ttk.Button(
            quick_actions,
            text="🎯 Reset Target",
            command=self.reset_target,
            width=12
        ).pack(side=tk.LEFT)

    def create_info_section(self):
        """Create information display section"""
        info_frame = ttk.LabelFrame(self.right_panel, text="Tracking Information", style='Dark.TFrame')
        info_frame.pack(fill=tk.X, pady=(0, 10))

        # Status indicator
        status_frame = ttk.Frame(info_frame, style='Dark.TFrame')
        status_frame.pack(fill=tk.X, padx=5, pady=5)

        tk.Label(
            status_frame,
            text="Status:",
            font=('Arial', 10, 'bold'),
            fg='white',
            bg='#1e1e1e'
        ).pack(side=tk.LEFT)

        self.status_indicator = tk.Label(
            status_frame,
            text="● Ready",
            font=('Arial', 12, 'bold'),
            fg='#4CAF50',
            bg='#1e1e1e'
        )
        self.status_indicator.pack(side=tk.LEFT, padx=(10, 0))

        # Target information
        target_frame = ttk.LabelFrame(info_frame, text="Target Data", style='Dark.TFrame')
        target_frame.pack(fill=tk.X, padx=5, pady=5)

        self.target_info = {}
        target_fields = [
            ("Target ID:", "target_id"),
            ("Head Position:", "head_position"),
            ("Chest Position:", "chest_position"),
            ("Legs Position:", "legs_position"),
            ("Error (dx, dy):", "error"),
            ("Distance:", "distance"),
            ("Keypoints:", "keypoints"),
            ("Box Area:", "box_area")
        ]

        for i, (label_text, key) in enumerate(target_fields):
            row_frame = ttk.Frame(target_frame, style='Dark.TFrame')
            row_frame.pack(fill=tk.X, padx=5, pady=2)

            tk.Label(
                row_frame,
                text=label_text,
                font=('Arial', 9, 'bold'),
                fg='white',
                bg='#1e1e1e',
                width=15,
                anchor='w'
            ).pack(side=tk.LEFT)

            self.target_info[key] = tk.Label(
                row_frame,
                text="--",
                font=('Arial', 9),
                fg='#00ff00',
                bg='#1e1e1e',
                anchor='w'
            )
            self.target_info[key].pack(side=tk.LEFT, padx=(5, 0))

        # Performance information
        perf_frame = ttk.LabelFrame(info_frame, text="Performance", style='Dark.TFrame')
        perf_frame.pack(fill=tk.X, padx=5, pady=5)

        self.perf_info = {}
        perf_fields = [
            ("Camera FPS:", "camera_fps"),
            ("Processing Time:", "process_time"),
            ("Frames Processed:", "frames_processed"),
            ("Active Tracks:", "active_tracks"),
            ("Detection Rate:", "detection_rate")
        ]

        for i, (label_text, key) in enumerate(perf_fields):
            row_frame = ttk.Frame(perf_frame, style='Dark.TFrame')
            row_frame.pack(fill=tk.X, padx=5, pady=2)

            tk.Label(
                row_frame,
                text=label_text,
                font=('Arial', 9, 'bold'),
                fg='white',
                bg='#1e1e1e',
                width=15,
                anchor='w'
            ).pack(side=tk.LEFT)

            self.perf_info[key] = tk.Label(
                row_frame,
                text="0",
                font=('Arial', 9),
                fg='#ffff00',
                bg='#1e1e1e',
                anchor='w'
            )
            self.perf_info[key].pack(side=tk.LEFT, padx=(5, 0))

    def create_config_section(self):
        """Create configuration section"""
        config_frame = ttk.LabelFrame(self.right_panel, text="Configuration", style='Dark.TFrame')
        config_frame.pack(fill=tk.X, pady=(0, 10))

        # Camera settings
        cam_frame = ttk.LabelFrame(config_frame, text="Camera Settings", style='Dark.TFrame')
        cam_frame.pack(fill=tk.X, padx=5, pady=5)

        # Camera index
        tk.Label(cam_frame, text="Camera Index:", fg='white', bg='#1e1e1e').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.camera_index_var = tk.StringVar(value="2")
        camera_combo = ttk.Combobox(cam_frame, textvariable=self.camera_index_var, values=["0", "1", "2", "3", "4"], width=8)
        camera_combo.grid(row=0, column=1, padx=5, pady=2)

        # Resolution
        tk.Label(cam_frame, text="Resolution:", fg='white', bg='#1e1e1e').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.resolution_var = tk.StringVar(value="1280x720")
        resolution_combo = ttk.Combobox(cam_frame, textvariable=self.resolution_var, 
                                       values=["640x480", "1280x720", "1920x1080"], width=12)
        resolution_combo.grid(row=1, column=1, padx=5, pady=2)

        # Detection settings
        detect_frame = ttk.LabelFrame(config_frame, text="Detection Settings", style='Dark.TFrame')
        detect_frame.pack(fill=tk.X, padx=5, pady=5)

        # Confidence threshold
        tk.Label(detect_frame, text="Confidence:", fg='white', bg='#1e1e1e').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.confidence_var = tk.DoubleVar(value=0.3)
        confidence_scale = tk.Scale(detect_frame, from_=0.1, to=0.9, resolution=0.05,
                                   orient=tk.HORIZONTAL, variable=self.confidence_var,
                                   bg='#404040', fg='white', length=150)
        confidence_scale.grid(row=0, column=1, padx=5, pady=2)

        # Process every N frames
        tk.Label(detect_frame, text="Process Every:", fg='white', bg='#1e1e1e').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.process_frames_var = tk.StringVar(value="2")
        process_spinbox = tk.Spinbox(detect_frame, from_=1, to=5, textvariable=self.process_frames_var, width=8)
        process_spinbox.grid(row=1, column=1, padx=5, pady=2)

        # Tracking settings
        track_frame = ttk.LabelFrame(config_frame, text="Tracking Settings", style='Dark.TFrame')
        track_frame.pack(fill=tk.X, padx=5, pady=5)

        # Max age
        tk.Label(track_frame, text="Max Age:", fg='white', bg='#1e1e1e').grid(row=0, column=0, sticky='w', padx=5, pady=2)
        self.max_age_var = tk.StringVar(value="50")
        max_age_spinbox = tk.Spinbox(track_frame, from_=10, to=100, textvariable=self.max_age_var, width=8)
        max_age_spinbox.grid(row=0, column=1, padx=5, pady=2)

        # N init
        tk.Label(track_frame, text="N Init:", fg='white', bg='#1e1e1e').grid(row=1, column=0, sticky='w', padx=5, pady=2)
        self.n_init_var = tk.StringVar(value="2")
        n_init_spinbox = tk.Spinbox(track_frame, from_=1, to=10, textvariable=self.n_init_var, width=8)
        n_init_spinbox.grid(row=1, column=1, padx=5, pady=2)

        # Display options
        display_frame = ttk.LabelFrame(config_frame, text="Display Options", style='Dark.TFrame')
        display_frame.pack(fill=tk.X, padx=5, pady=5)

        self.show_keypoints_var = tk.BooleanVar(value=True)
        tk.Checkbutton(display_frame, text="Show Keypoints", variable=self.show_keypoints_var,
                      fg='white', bg='#1e1e1e', selectcolor='#404040').pack(anchor='w', padx=5)

        self.show_bbox_var = tk.BooleanVar(value=True)
        tk.Checkbutton(display_frame, text="Show Bounding Boxes", variable=self.show_bbox_var,
                      fg='white', bg='#1e1e1e', selectcolor='#404040').pack(anchor='w', padx=5)

        self.show_fps_var = tk.BooleanVar(value=True)
        tk.Checkbutton(display_frame, text="Show FPS Info", variable=self.show_fps_var,
                      fg='white', bg='#1e1e1e', selectcolor='#404040').pack(anchor='w', padx=5)

        # Body part tracking options with threat levels
        body_parts_frame = ttk.LabelFrame(config_frame, text="Threat Level Detection", style='Dark.TFrame')
        body_parts_frame.pack(fill=tk.X, padx=5, pady=5)

        # Threat level selection dropdown
        threat_selection_frame = tk.Frame(body_parts_frame, bg='#1e1e1e')
        threat_selection_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(threat_selection_frame, text="Select Threat Level to Detect:", fg='white', bg='#1e1e1e', font=('Arial', 10, 'bold')).pack(anchor='w')
        
        self.threat_level_var = tk.StringVar(value="All")
        threat_combo = ttk.Combobox(threat_selection_frame, textvariable=self.threat_level_var,
                                   values=["All", "High (Head)", "Medium (Chest)", "Low (Legs)"], width=20)
        threat_combo.pack(anchor='w', padx=5, pady=5)
        threat_combo.bind('<<ComboboxSelected>>', self.on_threat_level_change)

        # Threat level information display
        threat_info_frame = tk.Frame(body_parts_frame, bg='#1e1e1e')
        threat_info_frame.pack(fill=tk.X, padx=5, pady=5)
        
        tk.Label(threat_info_frame, text="Threat Level Guide:", fg='white', bg='#1e1e1e', font=('Arial', 9, 'bold')).pack(anchor='w')
        tk.Label(threat_info_frame, text="🔴 HIGH - Head Detection (Critical Zone)", fg='#ff6b6b', bg='#1e1e1e', font=('Arial', 8)).pack(anchor='w', padx=10)
        tk.Label(threat_info_frame, text="🟡 MEDIUM - Chest Detection (Important Zone)", fg='#ffd93d', bg='#1e1e1e', font=('Arial', 8)).pack(anchor='w', padx=10)
        tk.Label(threat_info_frame, text="🟢 LOW - Legs Detection (Secondary Zone)", fg='#6bcf7f', bg='#1e1e1e', font=('Arial', 8)).pack(anchor='w', padx=10)

        # Set default values for body parts (controlled by threat level selection)
        self.show_head_var = tk.BooleanVar(value=True)
        self.show_chest_var = tk.BooleanVar(value=True)
        self.show_legs_var = tk.BooleanVar(value=True)

        # Targeting priority
        tk.Label(body_parts_frame, text="Primary Target:", fg='white', bg='#1e1e1e').pack(anchor='w', padx=5, pady=(10,2))
        self.target_priority_var = tk.StringVar(value="chest")
        priority_combo = ttk.Combobox(body_parts_frame, textvariable=self.target_priority_var,
                                     values=["head", "chest", "legs"], width=15)
        priority_combo.pack(anchor='w', padx=5, pady=2)

    def create_logs_section(self):
        """Create logs section"""
        logs_frame = ttk.LabelFrame(self.right_panel, text="Activity Logs", style='Dark.TFrame')
        logs_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        # Log display
        self.log_text = scrolledtext.ScrolledText(
            logs_frame,
            bg='#0d1117',
            fg='#58a6ff',
            font=('Courier New', 9),
            wrap=tk.WORD,
            height=15
        )
        self.log_text.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

        # Log controls
        log_controls = ttk.Frame(logs_frame, style='Dark.TFrame')
        log_controls.pack(fill=tk.X, padx=5, pady=(0, 5))

        ttk.Button(log_controls, text="Clear", command=self.clear_logs, width=8).pack(side=tk.LEFT, padx=(0, 5))
        ttk.Button(log_controls, text="Save", command=self.save_logs, width=8).pack(side=tk.LEFT, padx=(0, 5))

        self.auto_scroll_var = tk.BooleanVar(value=True)
        tk.Checkbutton(log_controls, text="Auto Scroll", variable=self.auto_scroll_var,
                      fg='white', bg='#1e1e1e', selectcolor='#404040').pack(side=tk.LEFT, padx=(10, 0))

    def create_menu_bar(self):
        """Create application menu bar"""
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        # File menu
        file_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="File", menu=file_menu)
        file_menu.add_command(label="Save Settings", command=self.save_settings)
        file_menu.add_command(label="Load Settings", command=self.load_settings)
        file_menu.add_separator()
        file_menu.add_command(label="Exit", command=self.on_closing)

        # Camera menu
        camera_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Camera", menu=camera_menu)
        camera_menu.add_command(label="Test Camera", command=self.test_camera)
        camera_menu.add_command(label="Run Advanced Test", command=self.run_advanced_test)
        camera_menu.add_command(label="Run Original Script", command=self.run_original_script)

        # Help menu
        help_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Help", menu=help_menu)
        help_menu.add_command(label="About", command=self.show_about)

    def create_status_bar(self):
        """Create status bar"""
        self.status_bar = ttk.Frame(self.root, style='Dark.TFrame')
        self.status_bar.pack(side=tk.BOTTOM, fill=tk.X)

        self.status_text = tk.Label(
            self.status_bar,
            text="Ready - Rapoo Camera Optimized for Low FPS",
            font=('Arial', 10),
            fg='white',
            bg='#1e1e1e',
            anchor='w'
        )
        self.status_text.pack(side=tk.LEFT, padx=5, pady=2)

        # Progress bar
        self.progress_bar = ttk.Progressbar(
            self.status_bar,
            mode='indeterminate',
            length=200
        )
        self.progress_bar.pack(side=tk.RIGHT, padx=5, pady=2)

    def initialize_variables(self):
        """Initialize application variables"""
        self.is_tracking = False
        self.tracking_thread = None
        self.camera = None
        self.model = None
        self.tracker = None

        # Tracking state from original script
        self.selected_track_id = None
        self.current_frame = None
        self.frame_count = 0
        self.fps_start_time = None
        self.fps_frame_count = 0
        self.current_fps = 0
        self.start_time = None

        # Processing optimization
        self.last_detections = []
        self.last_keypoints = {}
        self.last_process_time = 0

        # Recording state
        self.is_recording = False
        self.video_writer = None

        # UI update queue
        self.ui_queue = queue.Queue()

    def setup_tracking_constants(self):
        """Setup tracking constants from original script"""
        # Human keypoint indices for YOLOv8 pose model (17 keypoints)
        self.KEYPOINT_NAMES = [
            'nose', 'left_eye', 'right_eye', 'left_ear', 'right_ear',
            'left_shoulder', 'right_shoulder', 'left_elbow', 'right_elbow',
            'left_wrist', 'right_wrist', 'left_hip', 'right_hip',
            'left_knee', 'right_knee', 'left_ankle', 'right_ankle'
        ]

        # Important body parts for human shape validation
        self.CORE_KEYPOINTS = {
            'shoulders': [5, 6],  # left_shoulder, right_shoulder
            'hips': [11, 12],     # left_hip, right_hip
            'head': [0, 1, 2, 3, 4],  # nose, eyes, ears
            'arms': [7, 8, 9, 10],    # elbows, wrists
            'legs': [13, 14, 15, 16]  # knees, ankles
        }

    def on_threat_level_change(self, event=None):
        """Handle threat level selection change"""
        threat_level = self.threat_level_var.get()
        
        # Reset all detection flags
        self.show_head_var.set(False)
        self.show_chest_var.set(False) 
        self.show_legs_var.set(False)
        
        # Enable detection based on selected threat level
        if threat_level == "All":
            self.show_head_var.set(True)
            self.show_chest_var.set(True)
            self.show_legs_var.set(True)
            self.log_message("🎯 Threat Level: ALL - Detecting all body parts")
        elif threat_level == "High (Head)":
            self.show_head_var.set(True)
            self.target_priority_var.set("head")
            self.log_message("🔴 Threat Level: HIGH - Head detection only")
        elif threat_level == "Medium (Chest)":
            self.show_chest_var.set(True)
            self.target_priority_var.set("chest")
            self.log_message("🟡 Threat Level: MEDIUM - Chest detection only")
        elif threat_level == "Low (Legs)":
            self.show_legs_var.set(True)
            self.target_priority_var.set("legs")
            self.log_message("🟢 Threat Level: LOW - Legs detection only")

    def validate_human_keypoints(self, keypoints, confidence_threshold=0.25):
        """Validate if detected keypoints represent a real human shape - optimized for low FPS"""
        if keypoints is None or len(keypoints) == 0:
            return False, "No keypoints detected"

        # Count visible keypoints for each body part
        visible_count = {'head': 0, 'shoulders': 0, 'hips': 0, 'arms': 0, 'legs': 0}

        for part, indices in self.CORE_KEYPOINTS.items():
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

    def get_human_chest_center_from_keypoints(self, keypoints):
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

    def get_human_head_center_from_keypoints(self, keypoints):
        """Calculate head center using nose, eyes, and ears keypoints"""
        if keypoints is None or len(keypoints) == 0:
            return None, None

        # Head keypoints: nose, left_eye, right_eye, left_ear, right_ear
        head_keypoints = [0, 1, 2, 3, 4]  # nose, eyes, ears
        
        valid_head_points = []
        
        for idx in head_keypoints:
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                x, y, conf = keypoints[idx][:3]
                if conf > 0.2 and x > 0 and y > 0:  # Lower threshold for low FPS
                    valid_head_points.append((x, y, conf))
        
        if not valid_head_points:
            return None, None
        
        # Weighted average based on confidence
        total_weight = sum([point[2] for point in valid_head_points])
        if total_weight == 0:
            return None, None
            
        head_x = sum([point[0] * point[2] for point in valid_head_points]) / total_weight
        head_y = sum([point[1] * point[2] for point in valid_head_points]) / total_weight
        
        return head_x, head_y

    def get_human_legs_center_from_keypoints(self, keypoints):
        """Calculate legs center using primarily knee and ankle keypoints (avoiding hip area)"""
        if keypoints is None or len(keypoints) == 0:
            return None, None

        # Focus on lower leg keypoints: left_knee, right_knee, left_ankle, right_ankle
        # Avoid hips (11, 12) to prevent pointing to inappropriate areas
        lower_leg_keypoints = [13, 14, 15, 16]  # knees and ankles only
        
        valid_leg_points = []
        
        for idx in lower_leg_keypoints:
            if idx < len(keypoints) and len(keypoints[idx]) >= 3:
                x, y, conf = keypoints[idx][:3]
                if conf > 0.2 and x > 0 and y > 0:  # Lower threshold for low FPS
                    # Give higher weight to ankle points (15, 16) to focus on lower legs
                    weight_multiplier = 2.0 if idx in [15, 16] else 1.0  # Prioritize ankles
                    valid_leg_points.append((x, y, conf * weight_multiplier))
        
        if not valid_leg_points:
            return None, None
        
        # Weighted average based on confidence, prioritizing ankle points
        total_weight = sum([point[2] for point in valid_leg_points])
        if total_weight == 0:
            return None, None
            
        legs_x = sum([point[0] * point[2] for point in valid_leg_points]) / total_weight
        legs_y = sum([point[1] * point[2] for point in valid_leg_points]) / total_weight
        
        # Adjust to get center point of actual legs (much lower, focusing on shin/ankle area)
        legs_y = legs_y + 40  # Move significantly down to target actual leg area
        
        return legs_x, legs_y
        
        return legs_x, legs_y

    def get_all_body_centers(self, keypoints):
        """Get all body part centers (head, chest, legs) in one call"""
        head_x, head_y = self.get_human_head_center_from_keypoints(keypoints)
        chest_x, chest_y = self.get_human_chest_center_from_keypoints(keypoints)
        legs_x, legs_y = self.get_human_legs_center_from_keypoints(keypoints)
        
        return {
            'head': (head_x, head_y) if head_x is not None else None,
            'chest': (chest_x, chest_y) if chest_x is not None else None,
            'legs': (legs_x, legs_y) if legs_x is not None else None
        }

    def get_largest_person(self, active_tracks):
        """Find the largest person among active tracks"""
        if not active_tracks:
            return None

        largest_id = None
        largest_area = 0

        for tid, (bbox, keypoints) in active_tracks.items():
            x1, y1, x2, y2 = bbox
            area = (x2 - x1) * (y2 - y1)

            is_valid, _ = self.validate_human_keypoints(keypoints)

            if is_valid and area > largest_area:
                largest_area = area
                largest_id = tid

        return largest_id

    def log_message(self, message, level="INFO"):
        """Log message to UI and console"""
        timestamp = datetime.now().strftime("%H:%M:%S")
        log_entry = f"[{timestamp}] {level}: {message}\n"

        # Update status bar
        self.status_text.config(text=message)

        # Add to log display
        if hasattr(self, 'log_text'):
            self.log_text.insert(tk.END, log_entry)
            if self.auto_scroll_var.get():
                self.log_text.see(tk.END)

        # Console log
        if level == "ERROR":
            self.logger.error(message)
        elif level == "WARNING":
            self.logger.warning(message)
        else:
            self.logger.info(message)

    def start_tracking(self):
        """Start the tracking system"""
        if not TRACKING_AVAILABLE:
            messagebox.showerror("Error", "Tracking modules not available.\nPlease install ultralytics and deep-sort-realtime.")
            return

        if self.is_tracking:
            return

        try:
            self.log_message("🎯 Starting Rapoo tracking system...")
            self.progress_bar.start()

            # Initialize tracking components
            self.log_message("Loading YOLOv8 Pose model...")
            self.model = YOLO('yolov8n-pose.pt')

            self.log_message("Initializing DeepSORT tracker...")
            self.tracker = DeepSort(
                max_age=int(self.max_age_var.get()),
                n_init=int(self.n_init_var.get())
            )

            # Setup camera with Rapoo optimized settings
            camera_index = int(self.camera_index_var.get())
            self.log_message(f"Setting up Rapoo camera at index {camera_index}...")
            self.camera = cv2.VideoCapture(camera_index)

            if not self.camera.isOpened():
                raise Exception(f"Cannot open camera {camera_index}")

            # Apply Rapoo camera optimization settings
            resolution = self.resolution_var.get().split('x')
            self.camera.set(cv2.CAP_PROP_FRAME_WIDTH, int(resolution[0]))
            self.camera.set(cv2.CAP_PROP_FRAME_HEIGHT, int(resolution[1]))
            self.camera.set(cv2.CAP_PROP_FPS, 30)  # Request 30, get ~9
            self.camera.set(cv2.CAP_PROP_BUFFERSIZE, 1)  # Minimize latency
            self.camera.set(cv2.CAP_PROP_FOURCC, cv2.VideoWriter_fourcc('M', 'J', 'P', 'G'))

            # Quality optimizations for low FPS
            self.camera.set(cv2.CAP_PROP_BRIGHTNESS, 0.6)
            self.camera.set(cv2.CAP_PROP_CONTRAST, 0.8)
            self.camera.set(cv2.CAP_PROP_SATURATION, 0.7)
            self.camera.set(cv2.CAP_PROP_SHARPNESS, 0.9)
            self.camera.set(cv2.CAP_PROP_AUTO_EXPOSURE, 0.25)

            # Get actual settings
            actual_width = int(self.camera.get(cv2.CAP_PROP_FRAME_WIDTH))
            actual_height = int(self.camera.get(cv2.CAP_PROP_FRAME_HEIGHT))
            actual_fps = self.camera.get(cv2.CAP_PROP_FPS)

            self.log_message(f"Camera configured: {actual_width}x{actual_height} @ {actual_fps} FPS")
            self.log_message("Optimized for Rapoo camera (7-9 FPS expected)")

            # Initialize tracking variables
            self.is_tracking = True
            self.start_time = time.time()
            self.frame_count = 0
            self.fps_start_time = time.time()
            self.fps_frame_count = 0
            self.selected_track_id = None
            self.last_detections = []
            self.last_keypoints = {}

            # Update UI
            self.start_button.config(state='disabled')
            self.stop_button.config(state='normal')
            self.status_indicator.config(text="● Tracking", fg='#ff9800')

            # Start tracking thread
            self.tracking_thread = threading.Thread(target=self.tracking_loop, daemon=True)
            self.tracking_thread.start()

            # Start UI update timer
            self.update_ui()

            self.log_message("✅ Rapoo tracking started successfully")

        except Exception as e:
            self.log_message(f"❌ Failed to start tracking: {e}", "ERROR")
            self.progress_bar.stop()
            messagebox.showerror("Error", f"Failed to start tracking:\n{e}")

    def stop_tracking(self):
        """Stop the tracking system"""
        if not self.is_tracking:
            return

        self.log_message("⏹ Stopping tracking system...")

        # Stop tracking
        self.is_tracking = False

        # Wait for thread to finish
        if self.tracking_thread and self.tracking_thread.is_alive():
            self.tracking_thread.join(timeout=2)

        # Release camera
        if self.camera:
            self.camera.release()
            self.camera = None

        # Stop recording if active
        if self.is_recording:
            self.toggle_recording()

        # Reset tracking components
        self.model = None
        self.tracker = None
        self.selected_track_id = None
        self.current_frame = None

        # Update UI
        self.start_button.config(state='normal')
        self.stop_button.config(state='disabled')
        self.status_indicator.config(text="● Ready", fg='#4CAF50')
        self.progress_bar.stop()

        # Clear video display
        self.video_canvas.delete("all")
        self.video_canvas.create_text(
            480, 270,
            text="Tracking Stopped\nClick 'Start Tracking' to begin",
            fill='white',
            font=('Arial', 16),
            justify=tk.CENTER,
            tags="placeholder"
        )

        # Reset info displays
        for label in self.target_info.values():
            label.config(text="--")

        for label in self.perf_info.values():
            label.config(text="0")

        self.log_message("✅ Tracking stopped")

    def tracking_loop(self):
        """Main tracking loop implementing the optimized algorithm"""
        try:
            while self.is_tracking:
                ret, frame = self.camera.read()
                if not ret:
                    continue

                self.current_frame = frame.copy()
                self.frame_count += 1
                self.fps_frame_count += 1

                H, W = frame.shape[:2]
                fx, fy = W / 2.0, H / 2.0

                # Calculate actual FPS
                if self.fps_frame_count % 30 == 0:
                    elapsed = time.time() - self.fps_start_time
                    self.current_fps = 30 / elapsed if elapsed > 0 else 0
                    self.fps_start_time = time.time()

                # Process every N frames for better performance
                process_interval = int(self.process_frames_var.get())
                if self.frame_count % process_interval == 0:
                    process_start = time.time()

                    # YOLO pose detection
                    results = self.model.predict(
                        source=frame,
                        verbose=False,
                        conf=self.confidence_var.get()
                    )

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
                                    is_human, reason = self.validate_human_keypoints(keypoints)

                                    if is_human:
                                        detections.append(([x1, y1, x2, y2], conf, 0))
                                        pose_data[len(detections)-1] = keypoints

                    self.last_detections = detections
                    self.last_keypoints = pose_data
                    self.last_process_time = time.time() - process_start

                else:
                    # Use last frame's detections for smooth display
                    detections = self.last_detections
                    pose_data = self.last_keypoints

                # DeepSORT tracking
                tracks = self.tracker.update_tracks(detections, frame=frame)

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
                    if self.selected_track_id is None or self.selected_track_id not in active_tracks:
                        self.selected_track_id = self.get_largest_person(active_tracks)

                # Draw visualization
                self.draw_visualization(frame, active_tracks, fx, fy)

                # Queue frame for UI update
                self.ui_queue.put(('frame', frame))

                # Queue tracking data for UI update
                tracking_data = {
                    'active_tracks': active_tracks,
                    'selected_track_id': self.selected_track_id,
                    'fps': self.current_fps,
                    'process_time': self.last_process_time,
                    'frame_center': (fx, fy)
                }
                self.ui_queue.put(('data', tracking_data))

                # Record frame if recording
                if self.is_recording and self.video_writer:
                    self.video_writer.write(frame)

                time.sleep(0.033)  # ~30 FPS limit

        except Exception as e:
            self.log_message(f"❌ Tracking loop error: {e}", "ERROR")
        finally:
            self.is_tracking = False

    def draw_visualization(self, frame, active_tracks, fx, fy):
        """Draw tracking visualization on current frame"""
        H, W = frame.shape[:2]

        # Draw frame center
        cv2.drawMarker(frame, (int(fx), int(fy)), (255, 255, 255),
                      cv2.MARKER_CROSS, 15, 2)

        # Draw active tracks
        for tid, (bbox, keypoints) in active_tracks.items():
            x1, y1, x2, y2 = bbox
            color = (0, 255, 0) if tid == self.selected_track_id else (255, 0, 0)

            is_valid, reason = self.validate_human_keypoints(keypoints)

            if is_valid:
                # Draw bounding box
                if self.show_bbox_var.get():
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)

                # Draw keypoints
                if self.show_keypoints_var.get() and keypoints is not None:
                    for i, kp in enumerate(keypoints):
                        if len(kp) >= 3:
                            x, y, conf = kp[:3]
                            if conf > 0.2 and x > 0 and y > 0:
                                cv2.circle(frame, (int(x), int(y)), 2, color, -1)

                # Draw body part centers
                body_centers = self.get_all_body_centers(keypoints)
                
                # Draw head center with threat level indication
                if body_centers['head'] is not None:
                    head_x, head_y = body_centers['head']
                    cv2.circle(frame, (int(head_x), int(head_y)), 8, (0, 0, 255), -1)  # Red for HIGH threat
                    cv2.putText(frame, "HIGH", (int(head_x) + 10, int(head_y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 0, 255), 2)

                # Draw chest center with threat level indication
                if body_centers['chest'] is not None:
                    chest_x, chest_y = body_centers['chest']
                    cv2.circle(frame, (int(chest_x), int(chest_y)), 8, (0, 165, 255), -1)  # Orange for MEDIUM threat
                    cv2.putText(frame, "MED", (int(chest_x) + 10, int(chest_y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 165, 255), 2)

                # Draw legs center with threat level indication
                if body_centers['legs'] is not None:
                    legs_x, legs_y = body_centers['legs']
                    cv2.circle(frame, (int(legs_x), int(legs_y)), 8, (0, 255, 0), -1)  # Green for LOW threat
                    cv2.putText(frame, "LOW", (int(legs_x) + 10, int(legs_y) - 10),
                               cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 2)

                cv2.putText(frame, f"ID: {tid}", (x1, y1 - 10),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)

        # Display target information
        if self.selected_track_id is not None and self.selected_track_id in active_tracks:
            bbox, keypoints = active_tracks[self.selected_track_id]
            body_centers = self.get_all_body_centers(keypoints)
            
            # Get the primary target center based on priority setting
            target_priority = self.target_priority_var.get()
            target_center = body_centers.get(target_priority)
            
            # Fallback to other body parts if primary is not available
            if target_center is None:
                for fallback in ['chest', 'head', 'legs']:
                    if body_centers.get(fallback) is not None:
                        target_center = body_centers[fallback]
                        break

            if target_center is not None:
                target_x, target_y = target_center
                x1, y1, x2, y2 = bbox
                box_area = (x2 - x1) * (y2 - y1)

                dx = target_x - fx
                dy = target_y - fy

                # Count keypoints
                visible_keypoints = 0
                if keypoints:
                    for kp in keypoints:
                        if len(kp) >= 3 and kp[2] > 0.2:
                            visible_keypoints += 1

                distance_category = "Very Close" if box_area > 15000 else \
                                  "Close" if box_area > 8000 else \
                                  "Medium" if box_area > 4000 else "Far"

                # Draw targeting line to primary target
                cv2.arrowedLine(frame, (int(fx), int(fy)), (int(target_x), int(target_y)),
                               (0, 255, 255), 3, tipLength=0.1)
                
                # Draw target priority indicator
                cv2.putText(frame, f"Tracking: {target_priority.upper()}", (int(target_x) - 30, int(target_y) - 20),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

                # Display info on frame
                if self.show_fps_var.get():
                    cv2.putText(frame, f"Target: ID {self.selected_track_id} | {target_priority.upper()} Error: ({int(dx)}, {int(dy)})",
                               (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    
                    # Show all body part positions if available
                    info_line = f"Head: "
                    if body_centers['head']:
                        hx, hy = body_centers['head']
                        info_line += f"({int(hx)},{int(hy)}) "
                    else:
                        info_line += "N/A "
                    
                    info_line += f"| Chest: "
                    if body_centers['chest']:
                        cx, cy = body_centers['chest']
                        info_line += f"({int(cx)},{int(cy)}) "
                    else:
                        info_line += "N/A "
                    
                    info_line += f"| Legs: "
                    if body_centers['legs']:
                        lx, ly = body_centers['legs']
                        info_line += f"({int(lx)},{int(ly)})"
                    else:
                        info_line += "N/A"
                    
                    cv2.putText(frame, info_line, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)
                    cv2.putText(frame, f"{distance_category} | KP: {visible_keypoints}/17 | Primary: {target_priority}",
                               (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(frame, f"Camera FPS: {self.current_fps:.1f} | Process: 1/{self.process_frames_var.get()} frames",
                               (10, 100), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

        else:
            if self.show_fps_var.get():
                cv2.putText(frame, "No target - waiting for human detection...",
                           (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)
                cv2.putText(frame, f"Camera FPS: {self.current_fps:.1f} | Detected humans: {len(active_tracks)}",
                           (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)

    def update_ui(self):
        """Update UI elements"""
        try:
            # Process UI queue
            while not self.ui_queue.empty():
                try:
                    item_type, data = self.ui_queue.get_nowait()

                    if item_type == 'frame' and data is not None:
                        self.update_video_display(data)

                    elif item_type == 'data':
                        self.update_tracking_info(data)

                except queue.Empty:
                    break

        except Exception as e:
            self.log_message(f"UI update error: {e}", "ERROR")

        # Schedule next update
        if self.is_tracking:
            self.root.after(100, self.update_ui)  # Update every 100ms

    def update_video_display(self, frame):
        """Update video display with current frame"""
        try:
            # Resize frame to fit canvas
            canvas_width = self.video_canvas.winfo_width()
            canvas_height = self.video_canvas.winfo_height()

            if canvas_width > 1 and canvas_height > 1:
                # Calculate scaling
                h, w = frame.shape[:2]
                scale = min(canvas_width / w, canvas_height / h)

                new_w = int(w * scale)
                new_h = int(h * scale)

                # Resize frame
                resized_frame = cv2.resize(frame, (new_w, new_h))

                # Convert to RGB and create PhotoImage
                rgb_frame = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2RGB)
                pil_image = Image.fromarray(rgb_frame)
                photo = ImageTk.PhotoImage(pil_image)

                # Update canvas
                self.video_canvas.delete("all")
                self.video_canvas.create_image(
                    canvas_width // 2,
                    canvas_height // 2,
                    image=photo,
                    anchor=tk.CENTER
                )

                # Keep reference to prevent garbage collection
                self.video_canvas.photo = photo

        except Exception as e:
            self.log_message(f"Video display error: {e}", "ERROR")

    def update_tracking_info(self, data):
        """Update tracking information displays"""
        try:
            active_tracks = data.get('active_tracks', {})
            selected_track_id = data.get('selected_track_id')
            fps = data.get('fps', 0)
            process_time = data.get('process_time', 0)
            fx, fy = data.get('frame_center', (0, 0))

            # Update target information
            if selected_track_id is not None and selected_track_id in active_tracks:
                bbox, keypoints = active_tracks[selected_track_id]
                body_centers = self.get_all_body_centers(keypoints)
                
                # Get the primary target center based on priority setting
                target_priority = self.target_priority_var.get()
                target_center = body_centers.get(target_priority)
                
                # Fallback to other body parts if primary is not available
                if target_center is None:
                    for fallback in ['chest', 'head', 'legs']:
                        if body_centers.get(fallback) is not None:
                            target_center = body_centers[fallback]
                            break

                if target_center is not None:
                    target_x, target_y = target_center
                    x1, y1, x2, y2 = bbox
                    box_area = (x2 - x1) * (y2 - y1)
                    dx = target_x - fx
                    dy = target_y - fy

                    # Count keypoints
                    visible_keypoints = 0
                    if keypoints:
                        for kp in keypoints:
                            if len(kp) >= 3 and kp[2] > 0.2:
                                visible_keypoints += 1

                    distance_category = "Very Close" if box_area > 15000 else \
                                      "Close" if box_area > 8000 else \
                                      "Medium" if box_area > 4000 else "Far"

                    # Update UI labels
                    self.target_info['target_id'].config(text=str(selected_track_id))
                    
                    # Update body part positions
                    if body_centers['head']:
                        hx, hy = body_centers['head']
                        self.target_info['head_position'].config(text=f"({int(hx)}, {int(hy)})")
                    else:
                        self.target_info['head_position'].config(text="N/A")
                    
                    if body_centers['chest']:
                        cx, cy = body_centers['chest']
                        self.target_info['chest_position'].config(text=f"({int(cx)}, {int(cy)})")
                    else:
                        self.target_info['chest_position'].config(text="N/A")
                    
                    if body_centers['legs']:
                        lx, ly = body_centers['legs']
                        self.target_info['legs_position'].config(text=f"({int(lx)}, {int(ly)})")
                    else:
                        self.target_info['legs_position'].config(text="N/A")
                    
                    self.target_info['error'].config(text=f"({int(dx)}, {int(dy)}) [{target_priority}]")
                    self.target_info['distance'].config(text=distance_category)
                    self.target_info['keypoints'].config(text=f"{visible_keypoints}/17")
                    self.target_info['box_area'].config(text=str(int(box_area)))

            else:
                # Clear target info
                for label in self.target_info.values():
                    label.config(text="--")

            # Update performance info
            self.perf_info['camera_fps'].config(text=f"{fps:.1f}")
            self.perf_info['process_time'].config(text=f"{process_time*1000:.1f}ms")
            self.perf_info['frames_processed'].config(text=str(self.frame_count))
            self.perf_info['active_tracks'].config(text=str(len(active_tracks)))

            # Calculate detection rate
            if self.frame_count > 0:
                detection_rate = (len(active_tracks) / self.frame_count) * 100
                self.perf_info['detection_rate'].config(text=f"{detection_rate:.1f}%")

        except Exception as e:
            self.log_message(f"Info update error: {e}", "ERROR")

    def test_camera(self):
        """Test camera functionality"""
        try:
            camera_index = int(self.camera_index_var.get())
            self.log_message(f"🔍 Testing Rapoo camera {camera_index}...")

            cap = cv2.VideoCapture(camera_index)

            if cap.isOpened():
                ret, frame = cap.read()
                cap.release()

                if ret and frame is not None:
                    h, w = frame.shape[:2]
                    self.log_message(f"✅ Rapoo camera {camera_index} working - Resolution: {w}x{h}")
                    messagebox.showinfo("Camera Test", f"Rapoo camera {camera_index} is working properly!\nResolution: {w}x{h}")
                else:
                    self.log_message(f"❌ Rapoo camera {camera_index} failed to capture frame")
                    messagebox.showerror("Camera Test", f"Rapoo camera {camera_index} failed to capture frame")
            else:
                self.log_message(f"❌ Rapoo camera {camera_index} not accessible")
                messagebox.showerror("Camera Test", f"Rapoo camera {camera_index} is not accessible")

        except Exception as e:
            self.log_message(f"❌ Camera test error: {e}", "ERROR")
            messagebox.showerror("Camera Test", f"Camera test failed:\n{e}")

    def run_advanced_test(self):
        """Run the advanced camera test script"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "camera_advanced_test.py")
            if os.path.exists(script_path):
                subprocess.Popen([sys.executable, script_path])
                self.log_message("🔍 Advanced camera test started in separate window")
            else:
                self.log_message("❌ Advanced camera test script not found", "ERROR")
                messagebox.showerror("Error", "Advanced camera test script not found")
        except Exception as e:
            self.log_message(f"❌ Failed to run advanced camera test: {e}", "ERROR")

    def run_original_script(self):
        """Run the original rapoo_low_fps_optimized script"""
        try:
            script_path = os.path.join(os.path.dirname(__file__), "rapoo_low_fps_optimized.py")
            if os.path.exists(script_path):
                subprocess.Popen([sys.executable, script_path])
                self.log_message("🎯 Original Rapoo script started in separate window")
            else:
                self.log_message("❌ Original script not found", "ERROR")
                messagebox.showerror("Error", "rapoo_low_fps_optimized.py not found")
        except Exception as e:
            self.log_message(f"❌ Failed to run original script: {e}", "ERROR")

    def take_screenshot(self):
        """Take a screenshot of current frame"""
        if self.current_frame is not None:
            try:
                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rapoo_screenshot_{timestamp}.jpg"
                cv2.imwrite(filename, self.current_frame)
                self.log_message(f"📷 Screenshot saved: {filename}")
                messagebox.showinfo("Screenshot", f"Screenshot saved as:\n{filename}")
            except Exception as e:
                self.log_message(f"❌ Screenshot error: {e}", "ERROR")
        else:
            messagebox.showwarning("Screenshot", "No frame available for screenshot")

    def toggle_recording(self):
        """Toggle video recording"""
        if not self.is_recording:
            # Start recording
            try:
                if self.current_frame is None:
                    messagebox.showwarning("Recording", "No video feed available")
                    return

                timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
                filename = f"rapoo_recording_{timestamp}.mp4"

                h, w = self.current_frame.shape[:2]
                fourcc = cv2.VideoWriter_fourcc(*'mp4v')
                self.video_writer = cv2.VideoWriter(filename, fourcc, 9.0, (w, h))  # 9 FPS for Rapoo

                self.is_recording = True
                self.log_message(f"🎥 Recording started: {filename}")
                messagebox.showinfo("Recording", f"Recording started:\n{filename}")

            except Exception as e:
                self.log_message(f"❌ Recording error: {e}", "ERROR")
        else:
            # Stop recording
            if self.video_writer:
                self.video_writer.release()
                self.video_writer = None

            self.is_recording = False
            self.log_message("⏹ Recording stopped")
            messagebox.showinfo("Recording", "Recording stopped")

    def reset_target(self):
        """Reset selected target"""
        self.selected_track_id = None
        self.log_message("🎯 Target reset")

        # Clear target info
        for label in self.target_info.values():
            label.config(text="--")

    def save_settings(self):
        """Save current settings to file"""
        try:
            settings = {
                'camera_index': self.camera_index_var.get(),
                'resolution': self.resolution_var.get(),
                'confidence': self.confidence_var.get(),
                'process_frames': self.process_frames_var.get(),
                'max_age': self.max_age_var.get(),
                'n_init': self.n_init_var.get(),
                'show_keypoints': self.show_keypoints_var.get(),
                'show_bbox': self.show_bbox_var.get(),
                'show_fps': self.show_fps_var.get(),
                'target_priority': self.target_priority_var.get()
            }

            filename = filedialog.asksaveasfilename(
                title="Save Settings",
                defaultextension=".json",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filename:
                with open(filename, 'w') as f:
                    json.dump(settings, f, indent=2)
                self.log_message(f"💾 Settings saved: {filename}")
                messagebox.showinfo("Save Settings", f"Settings saved to:\n{filename}")

        except Exception as e:
            self.log_message(f"❌ Save settings error: {e}", "ERROR")

    def load_settings(self):
        """Load settings from file"""
        try:
            filename = filedialog.askopenfilename(
                title="Load Settings",
                filetypes=[("JSON files", "*.json"), ("All files", "*.*")]
            )

            if filename:
                with open(filename, 'r') as f:
                    settings = json.load(f)

                # Apply loaded settings
                self.camera_index_var.set(settings.get('camera_index', '2'))
                self.resolution_var.set(settings.get('resolution', '1280x720'))
                self.confidence_var.set(settings.get('confidence', 0.3))
                self.process_frames_var.set(settings.get('process_frames', '2'))
                self.max_age_var.set(settings.get('max_age', '50'))
                self.n_init_var.set(settings.get('n_init', '2'))
                self.show_keypoints_var.set(settings.get('show_keypoints', True))
                self.show_bbox_var.set(settings.get('show_bbox', True))
                self.show_fps_var.set(settings.get('show_fps', True))
                self.target_priority_var.set(settings.get('target_priority', 'chest'))

                self.log_message(f"📁 Settings loaded: {filename}")
                messagebox.showinfo("Load Settings", f"Settings loaded from:\n{filename}")

        except Exception as e:
            self.log_message(f"❌ Load settings error: {e}", "ERROR")

    def clear_logs(self):
        """Clear log display"""
        if hasattr(self, 'log_text'):
            self.log_text.delete(1.0, tk.END)
            self.log_message("Logs cleared")

    def save_logs(self):
        """Save logs to file"""
        try:
            filename = filedialog.asksaveasfilename(
                title="Save Logs",
                defaultextension=".log",
                filetypes=[("Log files", "*.log"), ("Text files", "*.txt"), ("All files", "*.*")]
            )

            if filename and hasattr(self, 'log_text'):
                with open(filename, 'w') as f:
                    f.write(self.log_text.get(1.0, tk.END))
                self.log_message(f"📝 Logs saved: {filename}")
                messagebox.showinfo("Save Logs", f"Logs saved to:\n{filename}")

        except Exception as e:
            self.log_message(f"❌ Save logs error: {e}", "ERROR")

    def show_about(self):
        """Show about dialog"""
        about_text = """
Rapoo Tkinter GUI v1.0.0
Advanced Human Tracking System

Developed by: Jawad
Repository: github.com/jawads12/Coukosh-Prohori

Features:
• Optimized for Rapoo USB cameras (7-9 FPS)
• Real-time human detection using YOLOv8
• Pose estimation with 17 keypoints
• Multi-object tracking with DeepSORT
• Frame skipping optimization
• Professional Tkinter interface
• Recording and screenshot capabilities

Based on: rapoo_low_fps_optimized.py
© 2024 - Open Source License
        """

        messagebox.showinfo("About Rapoo Tracking System", about_text.strip())

    def on_closing(self):
        """Handle application closing"""
        if self.is_tracking:
            if messagebox.askokcancel("Quit", "Tracking is active. Stop tracking and quit?"):
                self.stop_tracking()
                self.root.after(1000, self.root.destroy)
        else:
            self.root.destroy()

def main():
    """Main application entry point"""
    # Set environment variables
    os.environ['QT_QPA_PLATFORM'] = 'xcb'

    root = tk.Tk()
    app = RapooTrackingApp(root)

    # Center window on screen
    root.update_idletasks()
    x = (root.winfo_screenwidth() // 2) - (root.winfo_width() // 2)
    y = (root.winfo_screenheight() // 2) - (root.winfo_height() // 2)
    root.geometry(f"+{x}+{y}")

    # Add icon if available
    try:
        if os.path.exists("assets/app_icon.png"):
            icon = tk.PhotoImage(file="assets/app_icon.png")
            root.iconphoto(False, icon)
    except:
        pass

    root.mainloop()

if __name__ == "__main__":
    main()
