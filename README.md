# 🚀 YOLO Detection Suite

[![Python 3.9+](https://img.shields.io/badge/python-3.9+-blue.svg)](https://www.python.org/downloads/)
[![YOLOv8](https://img.shields.io/badge/YOLOv8-Ultralytics-00FFFF)](https://github.com/ultralytics/ultralytics)
[![Streamlit](https://img.shields.io/badge/Streamlit-1.28+-FF4B4B)](https://streamlit.io)
[![OpenCV](https://img.shields.io/badge/OpenCV-4.8+-green)](https://opencv.org/)
[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A comprehensive collection of real-time object detection applications powered by **YOLOv8**. From robot safety systems to household item recognition and vehicle tracking - all optimized for performance and ease of use.

---

## 🎯 Overview

This repository contains four production-ready object detection applications built with state-of-the-art YOLOv8 models:

| Application | Description | Type | Key Features |
|------------|-------------|------|--------------|
| 🤖 **Robot Safety-Bot** | Real-time human proximity detection | Streamlit Web App | Safety zones, debounce logic, proximity alerts |
| 🏠 **Household Scanner** | Live household item detection | Streamlit Web App | 15+ item categories, real-time counting |
| 📦 **Item Detector** | Batch household item detection | Video Processing | Video annotation, frame-by-frame analysis |
| 🚗 **Vehicle Tracker** | Multi-vehicle detection & counting | Video Processing | Cars, trucks, buses, motorcycles |

---

## ✨ Features

### 🤖 Robot Safety-Bot (`human.py`)
The flagship application - a safety monitoring system designed for robotic environments:

- **Real-time Human Detection**: YOLOv8n-powered person detection from webcam
- **Proximity-Based Safety Zones**:
  - 🟢 **SAFE**: No humans or distant (bbox < 15% of frame)
  - 🔴 **STOP**: Human too close (bbox > threshold)
- **Intelligent Debounce System**: 3-frame consensus prevents flickering
- **Configurable Thresholds**: Adjustable safety distance and debounce settings
- **Live Statistics Dashboard**: FPS, detection counts, bbox area percentages
- **Production-Ready**: Designed for integration with robotic control systems

**Perfect for:** Collaborative robots, warehouse automation, safety-critical applications

### 🏠 Household Scanner (`Streaming.py`)
Real-time object recognition for everyday items:

- Detects **15+ household items**: bottles, cups, utensils, fruits, books, etc.
- Model selection: Choose from YOLOv8n (fast) to YOLOv8x (accurate)
- Live item counting with visual feedback
- Confidence threshold adjustment
- FPS monitoring and performance stats

### 📦 Item Detector (`detection_2.py`)
Batch processing for video file analysis:

- Process pre-recorded videos with household item detection
- Frame-by-frame annotation with counts
- Progress tracking and video export
- Optimized for longer videos

### 🚗 Vehicle Tracker (`vehical_detection.py`)
Multi-class vehicle detection system:

- Detects: Cars, Motorcycles, Buses, Trucks
- Real-time vehicle counting per category
- Video annotation and export
- Traffic analysis capabilities

---

## 🎬 Demo

### Robot Safety-Bot in Action

```
🟢 SAFE                           📊 Detection Stats
                                  Persons Detected: 0
┌─────────────────────────┐      Max Bbox Area: 0.00%
│                         │      Threshold: 15.0%
│   [Video Feed]          │      FPS: 28.5
│                         │
└─────────────────────────┘

[Person enters frame - far away]

🟢 SAFE                           📊 Detection Stats
                                  Persons Detected: 1
┌─────────────────────────┐      Max Bbox Area: 8.32%
│   ┌───┐                 │      Threshold: 15.0%
│   │ P │  [bounding box] │      FPS: 27.8
│   └───┘                 │
└─────────────────────────┘

[Person approaches camera]

🔴 STOP ⚠️                        📊 Detection Stats
                                  Persons Detected: 1
┌─────────────────────────┐      Max Bbox Area: 23.47%
│   ┌────────┐            │      Threshold: 15.0%
│   │Person  │            │      FPS: 26.9
│   │0.94|23%│            │
│   └────────┘            │
└─────────────────────────┘
```

---

## 🛠️ Installation

### Prerequisites

- Python 3.9 or higher
- Webcam (for real-time applications)
- 4GB RAM minimum (8GB recommended)

### Quick Start

1. **Clone the repository**
   ```bash
   git clone https://github.com/yourusername/yolo-detection-suite.git
   cd yolo-detection-suite
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

4. **Download YOLO model** (automatic on first run)
   ```bash
   # The YOLOv8n model (~6MB) downloads automatically
   # Or manually download:
   python -c "from ultralytics import YOLO; YOLO('yolov8n.pt')"
   ```

---

## 🚀 Usage

### 1️⃣ Robot Safety-Bot (Streamlit Web App)

Launch the interactive safety monitoring system:

```bash
streamlit run human.py
```

**Controls:**
- Adjust **Safety Threshold** (5-50%) in sidebar
- Set **Debounce Frames** (1-10) for stability
- Click **Start Camera** to begin monitoring
- Watch for 🟢 SAFE / 🔴 STOP indicators

**Configuration Options:**
- `Safety Threshold`: Bounding box area percentage that triggers UNSAFE state (default: 15%)
- `Debounce Frames`: Number of consecutive frames required for state change (default: 3)
- `Show FPS`: Toggle FPS counter display

### 2️⃣ Household Scanner (Streamlit Web App)

Real-time household item detection:

```bash
streamlit run Streaming.py
```

**Features:**
- Select YOLO model (n/s/m/l/x) based on speed vs accuracy needs
- Adjust confidence threshold (0.25 recommended)
- Choose camera index if multiple cameras available
- View live detection stats in sidebar

### 3️⃣ Item Detector (Video Processing)

Process video files with household item detection:

```bash
python detection_2.py
```

**Configuration** (edit in script):
```python
VIDEO_PATH = "./C2.mp4"           # Input video
OUTPUT_PATH = "output.mp4"         # Output video
CONFIDENCE = 0.25                  # Detection confidence
```

### 4️⃣ Vehicle Tracker (Video Processing)

Detect and count vehicles in video files:

```bash
python vehical_detection.py
```

**Configuration** (edit in script):
```python
VIDEO_PATH = "./video.mp4"         # Input video
OUTPUT_PATH = "output.mp4"         # Output video
MODEL = "yolov8n.pt"               # Model selection
```

---

## 📁 Project Structure

```
yolo-detection-suite/
│
├── human.py                    # 🤖 Robot Safety-Bot (Streamlit)
├── Streaming.py                # 🏠 Household Scanner (Streamlit)
├── detection_2.py              # 📦 Item Detector (Video)
├── vehical_detection.py        # 🚗 Vehicle Tracker (Video)
│
├── requirements.txt            # Python dependencies
├── human.md                    # Safety-Bot roadmap & documentation
├── humanfaq.md                 # FAQ and troubleshooting
│
├── yolov8n.pt                  # YOLOv8 nano model (auto-downloaded)
│
├── video.mp4                   # Sample input videos
├── C2.mp4
│
└── output_detected.mp4         # Sample output videos
    output_household_detected.mp4
```

---

## 🧰 Technologies

| Technology | Purpose | Version |
|-----------|---------|---------|
| [Ultralytics YOLOv8](https://github.com/ultralytics/ultralytics) | Real-time object detection | 8.0+ |
| [Streamlit](https://streamlit.io) | Interactive web interfaces | 1.28+ |
| [OpenCV](https://opencv.org/) | Video/image processing | 4.8+ |
| [NumPy](https://numpy.org/) | Numerical computations | 1.24+ |
| [Python](https://python.org) | Core programming language | 3.9+ |

---

## 🎯 Detectable Objects

### Robot Safety-Bot
- **Person** (COCO class 0)

### Household Scanner & Item Detector
- **Kitchenware**: Bottle, Wine glass, Cup, Fork, Knife, Spoon, Bowl
- **Food**: Banana, Apple, Orange, Sandwich, Broccoli, Carrot
- **Objects**: Book, Clock, Vase

### Vehicle Tracker
- **Vehicles**: Car, Motorcycle, Bus, Truck

---

## 🚦 How Robot Safety-Bot Works

### Safety Logic

The safety system uses **bounding box area as a proximity metric**:

1. **Detection**: YOLOv8n detects persons in frame
2. **Area Calculation**: Bbox area computed as % of total frame
3. **Safety Evaluation**:
   - If `max_bbox_area > threshold` → Potentially UNSAFE
   - If `max_bbox_area ≤ threshold` → Potentially SAFE
4. **Debounce**: State only changes after N consecutive frames agree
5. **Status Update**: 🟢 SAFE or 🔴 STOP indicator displayed

### Key Insight
**Larger bounding boxes = closer proximity = higher risk**

A person occupying 5% of the frame is far away (safe), while 25% indicates dangerous proximity.

### Debounce Mechanism

Prevents false alarms from detection noise:

```
Frame:  1    2    3    4    5    6    7
Unsafe: Y    Y    Y    N    N    N    N
State:  ⏳   ⏳   🔴   🔴   ⏳   ⏳   🟢
        (gathering votes) (change!)
```

---

## 🗺️ Roadmap

This project follows a phased development approach:

### ✅ Phase 1 - MVP (Complete)
- [x] Webcam integration
- [x] YOLOv8n human detection
- [x] Safety threshold logic
- [x] Red/green status indicators
- [x] Debounce system
- [x] Streamlit deployment ready

### 🚧 Phase 2 - Enhanced Safety (Planned)
- [ ] Region of Interest (ROI) zones
- [ ] Distance estimation
- [ ] Multi-zone safety levels
- [ ] Audio alerts

### 🔮 Phase 3 - Tracking (Future)
- [ ] DeepSORT integration
- [ ] Per-person ID tracking
- [ ] Temporal smoothing
- [ ] Multi-human scenarios

### 🧠 Phase 4 - AI Verification (Future)
- [ ] Human vs Non-Human classifier
- [ ] False positive reduction
- [ ] Pose estimation integration

### 🏭 Phase 5 - Production (Future)
- [ ] ONNX/TensorRT export
- [ ] ROS integration
- [ ] Hardware E-Stop trigger
- [ ] Multi-camera support
- [ ] Cloud telemetry

---

## 📊 Performance

### Model Comparison

| Model | Speed (FPS) | Accuracy | Size | Best For |
|-------|-------------|----------|------|----------|
| YOLOv8n | 25-35 | Good | 6MB | Real-time, edge devices |
| YOLOv8s | 20-28 | Better | 22MB | Balanced use cases |
| YOLOv8m | 15-22 | Great | 52MB | Accuracy priority |
| YOLOv8l | 10-15 | Excellent | 87MB | High-accuracy needs |
| YOLOv8x | 5-10 | Best | 136MB | Maximum accuracy |

**Recommended:** YOLOv8n for real-time applications, YOLOv8m+ for offline processing

---

## 🔧 Configuration

### Camera Issues

If camera doesn't open, try different indices:
```python
# In Streamlit apps, use sidebar "Camera Index" slider
# In Python scripts:
cap = cv2.VideoCapture(0)  # Try 0, 1, 2, etc.
```

### Performance Optimization

**For faster FPS:**
```python
# Reduce input resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

# Lower confidence threshold
conf_threshold = 0.25  # or lower

# Use YOLOv8n instead of larger models
```

**For better accuracy:**
```python
# Use larger model
model = YOLO('yolov8m.pt')

# Increase confidence threshold
conf_threshold = 0.5

# Higher input resolution
cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
```

---

## 🤝 Contributing

Contributions are welcome! Here's how:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/AmazingFeature`)
3. Commit changes (`git commit -m 'Add AmazingFeature'`)
4. Push to branch (`git push origin feature/AmazingFeature`)
5. Open a Pull Request

### Ideas for Contributions
- Add support for additional object classes
- Implement tracking algorithms (SORT, DeepSORT)
- Create Docker containerization
- Add unit tests
- Improve documentation
- Optimize for edge devices (Raspberry Pi, Jetson)

---

## 📝 License

This project is licensed under the MIT License - see the LICENSE file for details.

---

## 🙏 Acknowledgments

- [Ultralytics](https://github.com/ultralytics/ultralytics) for the amazing YOLOv8 implementation
- [Streamlit](https://streamlit.io) for the intuitive web framework
- [OpenCV](https://opencv.org/) for robust computer vision tools
- COCO dataset for pre-trained model weights

---

## 📧 Contact

Have questions or suggestions? Feel free to:
- Open an issue on GitHub
- Submit a pull request
- Star ⭐ this repository if you find it useful!

---

## 🎓 Use Cases

This project is ideal for:

- 🏭 **Industrial Robotics**: Safety monitoring for collaborative robots (cobots)
- 🏪 **Retail Analytics**: Customer counting and item inventory
- 🚗 **Traffic Management**: Vehicle detection and flow analysis
- 🏠 **Smart Home**: Object recognition and monitoring
- 🎓 **Education**: Learning computer vision and object detection
- 🔬 **Research**: Baseline for custom detection systems

---

<div align="center">

### ⭐ Star this repository if you find it helpful!

**Made with ❤️ and YOLOv8**

</div>
