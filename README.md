# Coukosh Prohori - Advanced Human Tracking System

A comprehensive computer vision project implementing real-time human detection, pose estimation, and tracking using YOLOv8 and DeepSORT algorithms, specifically optimized for Rapoo USB cameras.

## 🚀 Features

- **Real-time Human Detection**: YOLOv8-based person detection with high accuracy
- **Pose Estimation**: 17-keypoint human pose analysis for detailed body tracking
- **Multi-object Tracking**: DeepSORT algorithm for persistent human ID tracking across frames
- **Camera Optimization**: Specialized support for Rapoo USB cameras with FPS optimization
- **Anatomical Analysis**: Chest center calculation and human shape validation
- **Performance Monitoring**: Real-time FPS counter and processing statistics

## 📁 Project Structure

```
Coukosh Prohori/
├── rapoo_camera.py              # Basic Rapoo camera test and capture
├── rapoo_script3.py             # Full HD pose-based human tracking
├── rapoo_low_fps_optimized.py   # Optimized version for 7-9 FPS cameras
├── script1.py                   # Basic human detection script
├── script2.py                   # Enhanced tracking implementation
├── script3.py                   # Advanced multi-person tracking
├── human_detec_trac_perfect.py  # Perfect tracking implementation
├── clean_tracking.py            # Clean tracking algorithm
├── camera_test.py               # General camera testing
├── camera_advanced_test.py      # Advanced camera diagnostics
├── requirements.txt             # Project dependencies
├── .gitignore                   # Git ignore rules
└── README.md                    # Project documentation
```

## 🛠️ Installation

### Prerequisites

- Python 3.12 or higher
- CUDA-compatible GPU (recommended for optimal performance)
- USB camera (Rapoo camera specifically tested and optimized)

### Setup

1. **Clone the repository:**
   ```bash
   git clone https://github.com/jawads12/Coukosh-Prohori.git
   cd Coukosh-Prohori
   ```

2. **Create and activate virtual environment:**
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Linux/Mac
   # or
   venv\Scripts\activate     # On Windows
   ```

3. **Install dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLOv8 models:**
   The scripts will automatically download required models (yolov8n-pose.pt) on first run.

## 🎯 Usage

### Basic Camera Testing

Test your Rapoo camera functionality:
```bash
python rapoo_camera.py
```

### Human Tracking Scripts

#### Full HD Tracking (Best Quality)
```bash
python rapoo_script3.py
```
- Resolution: 1920x1080
- Target FPS: 30 (limited by hardware to ~7-9 FPS)
- Full pose estimation with 17 keypoints

#### Optimized Low FPS Tracking (Recommended)
```bash
python rapoo_low_fps_optimized.py
```
- Resolution: 1280x720
- Optimized for 7-9 FPS cameras
- Frame skipping and relaxed validation
- Enhanced tracking persistence

#### Legacy Scripts
```bash
python script1.py    # Basic detection
python script2.py    # Enhanced tracking
python script3.py    # Advanced multi-person
```

## ⚙️ Configuration

### Camera Settings

**Rapoo Camera Specifications:**
- Model: 2.07 MP USB Camera
- Maximum Resolution: 1920x1080
- Actual FPS: 7-9 (hardware limitation)
- Optimal Backend: V4L2 (Linux)
- Recommended Codec: MJPG

### Performance Optimization

**For Rapoo cameras with FPS limitations:**
- Use `rapoo_low_fps_optimized.py`
- Frame processing every 2nd frame (PROCESS_EVERY_N_FRAMES=2)
- Relaxed confidence thresholds (0.2 vs 0.3)
- Increased tracking persistence (max_age=50)

## 🔧 Technical Details

### YOLOv8 Pose Keypoints

The system tracks 17 human keypoints:
1. Nose (0)
2. Eyes (1, 2)
3. Ears (3, 4)
4. Shoulders (5, 6)
5. Elbows (7, 8)
6. Wrists (9, 10)
7. Hips (11, 12)
8. Knees (13, 14)
9. Ankles (15, 16)

### Tracking Algorithm

**DeepSORT Configuration:**
- Maximum age: 50 frames
- Minimum initialization: 3 detections
- Maximum IOU distance: 0.7
- Maximum cosine distance: 0.4

### Human Validation

The system validates human shapes using:
- Shoulder width analysis
- Hip alignment verification
- Head-to-body proportion checks
- Limb connectivity validation

## 📊 Performance Metrics

| Script | Resolution | Target FPS | Actual FPS | Features |
|--------|------------|------------|------------|----------|
| rapoo_camera.py | 640x480 | 30 | 7-9 | Basic capture |
| rapoo_script3.py | 1920x1080 | 30 | 7-9 | Full HD pose |
| rapoo_low_fps_optimized.py | 1280x720 | 15 | 7-9 | Optimized tracking |

## 🐛 Troubleshooting

### Common Issues

1. **Low FPS Performance:**
   - Use `rapoo_low_fps_optimized.py`
   - Reduce resolution to 1280x720
   - Enable frame skipping

2. **Camera Not Detected:**
   ```bash
   python rapoo_camera.py  # Test camera detection
   ```

3. **CUDA Out of Memory:**
   - Reduce batch size in YOLOv8 settings
   - Use CPU inference: `device='cpu'`

4. **Missing Dependencies:**
   ```bash
   pip install -r requirements.txt
   ```

## 🤝 Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit your changes (`git commit -m 'Add some AmazingFeature'`)
4. Push to the branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

## 📝 License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## 🙏 Acknowledgments

- **YOLOv8**: Ultralytics for the pose estimation model
- **DeepSORT**: Real-time multi-object tracking algorithm
- **OpenCV**: Computer vision library for camera handling
- **PyTorch**: Deep learning framework

## 📞 Contact

**Developer:** Jawad  
**Repository:** [Coukosh-Prohori](https://github.com/jawads12/Coukosh-Prohori)  
**Issues:** [GitHub Issues](https://github.com/jawads12/Coukosh-Prohori/issues)

---

**Note:** This project is specifically optimized for Rapoo USB cameras but can work with other USB cameras with appropriate configuration adjustments.
