Here’s a clear, structured roadmap for your MVP safety-bot Streamlit app — written as a **`plan.md`** you can use directly in your repo 👇

---

## 🧠 Project: Robot Safety-Bot (Streamlit MVP)

### 🎯 Goal

Develop a Streamlit web app that uses the **device webcam** to detect humans in real-time using **YOLOv8-n** and shows **red/green safety status lights** based on proximity (bounding-box area).
This simulates a robot’s parallel “safety bot” running alongside manipulation models.

---

## 🚀 Phase 1 — MVP (Webcam + Safety Light)

### ✅ Core Features

1. **Webcam Stream**

   * Use `streamlit-webrtc` or `cv2.VideoCapture` for real-time webcam feed.
   * Display live video inside Streamlit app.

2. **YOLOv8n Integration**

   * Use `ultralytics` library or load `yolov8n.pt` from Hugging Face.
   * Run inference on each frame (downsample for speed if needed).
   * Filter detections for class = `person`.

3. **Safety Logic**

   * Compute bounding box area as a percentage of total frame area.
   * Define thresholds:

     * Safe → Green light (no human detected or small bbox area)
     * Unsafe → Red light (bbox area > threshold)
   * Maintain small debounce (2–3 consecutive frames before state change).

4. **Safety Indicator UI**

   * Top of Streamlit app:

     * 🟢 **Green** = SAFE
     * 🔴 **Red** = STOP (Human too close)
   * Use Streamlit container (`st.empty()`) to dynamically update color.

5. **Deployment**

   * Host on [Streamlit Cloud](https://streamlit.io/cloud) using GitHub repo.
   * Test on laptop webcam and mobile (browser permission required).

### 🧩 Tech Stack

* Python 3.9+
* Streamlit
* Ultralytics / YOLOv8n
* OpenCV
* (Optional) `streamlit-webrtc` for lower latency

### 📁 Folder Structure

```
robot-safety-bot/
│
├── app.py                  # Streamlit main app
├── plan.md                 # This roadmap
├── requirements.txt        # Dependencies
└── assets/
    └── icon.png            # Optional app logo
```

### 📦 requirements.txt

```
streamlit
ultralytics
opencv-python
numpy
```

*(Add `streamlit-webrtc` later if needed for smoother video.)*

---

## 🧰 Phase 2 — Functional Enhancements (Post-MVP)

### 2.1 Distance & Zone Safety

* Add configurable **Region of Interest (ROI)** (lower 1/3rd of frame).
* Calculate bbox overlap with ROI to improve safety logic.
* Optionally integrate **depth estimation** using monocular depth model.

---

## 🧭 Phase 3 — Tracking & Stability

### 3.1 Tracking with DeepSORT

* Integrate DeepSORT or SORT for ID persistence.
* Reduces flicker from single-frame false positives.
* Enables per-person tracking for multi-human scenarios.

### 3.2 Temporal Smoothing

* Maintain moving average of bbox size or proximity score.
* Red/green transitions smoother and more stable.

---

## 🧍 Phase 4 — Semantic Human Verification

### 4.1 Add Human/Non-Human Filter

* Integrate [**prithivMLmods/Human-vs-NonHuman-Detection**](https://huggingface.co/prithivMLmods/Human-vs-NonHuman-Detection).
* Run after YOLOv8 detection to verify if detected object is truly a human.

---

## 🧩 Phase 5 — Policy & Contextual Safety

### 5.1 SAFE or ShieldAgent-style Safety Policies

* Implement a **policy model** that checks if robot actions violate safety constraints.
* Uses image + text context to evaluate semantic risk.

### 5.2 VLA Safety Integration

* Run policy model in parallel to your robot’s Vision-Language-Action (VLA) model.
* Safety bot sends binary signal to control node → hardware stop if unsafe.

---

## 🧠 Phase 6 — Production Deployment

### 6.1 Containerization & Edge Deployment

* Export YOLOv8n to **ONNX/TensorRT** for Jetson / Orin deployment.
* Dockerize Streamlit + YOLO service.
* Add MQTT or ROS topic publisher for `/safety_stop`.

### 6.2 Telemetry & Logging

* Log safety events (frames + timestamp).
* Dashboard for reviewing safety triggers and false alarms.

---

## 🧩 Optional Future Work

* Add audio alert (“Stop!”) when unsafe.
* Integrate multi-camera support.
* Add physical E-Stop relay trigger via GPIO (on Jetson).

---

## 🗓️ Development Timeline (Recommended)

| Phase | Feature                         | Est. Time |
| ----- | ------------------------------- | --------- |
| 1     | MVP webcam + red/green lights   | 2 days    |
| 2     | ROI / Distance logic            | 1 day     |
| 3     | Tracking (DeepSORT)             | 2–3 days  |
| 4     | Human-vs-NonHuman model         | 1 day     |
| 5     | SAFE / policy-based integration | 3–5 days  |
| 6     | Edge + ROS integration          | 3–5 days  |

---

## ✅ Deliverables for MVP

* [ ] Working Streamlit app on streamlit.io
* [ ] Live webcam video feed
* [ ] YOLOv8n inference detecting persons
* [ ] Safety threshold & red/green indicator
* [ ] Clean UI + clear “STOP” / “SAFE” status
* [ ] Public GitHub repo (for deployment + future extensions)

---

Would you like me to now generate the **`app.py`** Streamlit file that fulfills **Phase 1 (MVP)** exactly as above — i.e., webcam feed, YOLOv8n detection, red/green light signal, and ready for deployment to Streamlit Cloud?
