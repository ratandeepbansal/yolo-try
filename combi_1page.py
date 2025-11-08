"""
Robot Vision System - Unified View
Real-time Safety Monitoring + Depth Estimation in One Page
"""

import streamlit as st
import cv2
import numpy as np
import torch
from ultralytics import YOLO
from collections import deque
import time

# Page configuration
st.set_page_config(
    page_title="Robot Vision - Unified",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for safety indicators and layout
st.markdown("""
<style>
    .safety-indicator {
        padding: 40px;
        border-radius: 15px;
        text-align: center;
        font-size: 48px;
        font-weight: bold;
        margin: 20px 0;
        box-shadow: 0 4px 6px rgba(0, 0, 0, 0.1);
    }
    .safe {
        background-color: #4CAF50;
        color: white;
    }
    .unsafe {
        background-color: #f44336;
        color: white;
        animation: pulse 1s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .stats-box {
        background-color: #B8BABE;
        padding: 15px;
        border-radius: 10px;
        margin: 10px 0;
    }
    .video-container {
        border: 3px solid #ddd;
        border-radius: 10px;
        padding: 5px;
        background-color: #f9f9f9;
    }
</style>
""", unsafe_allow_html=True)


# ============================================================================
# MODEL LOADING
# ============================================================================

@st.cache_resource
def load_yolo_model():
    """Load YOLOv8n model (cached for performance)"""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Failed to load YOLO model: {e}")
        st.info("The model will be downloaded automatically on first run.")
        return None


@st.cache_resource
def load_midas_model():
    """Load MiDaS depth estimation model"""
    try:
        model_type = "MiDaS_small"
        midas = torch.hub.load("intel-isl/MiDaS", model_type)
        device = "mps" if torch.backends.mps.is_available() else "cpu"
        midas.to(device).eval()

        transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
        transform = transforms.small_transform if model_type == "MiDaS_small" else transforms.dpt_transform

        return midas, transform, device
    except Exception as e:
        st.error(f"Failed to load MiDaS model: {e}")
        return None, None, None


# ============================================================================
# SAFETY MONITOR CLASS
# ============================================================================

def calculate_bbox_area_percentage(bbox, frame_shape):
    """Calculate bounding box area as percentage of frame area"""
    x1, y1, x2, y2 = bbox
    bbox_area = (x2 - x1) * (y2 - y1)
    frame_area = frame_shape[0] * frame_shape[1]
    return (bbox_area / frame_area) * 100


class SafetyMonitor:
    """Manages safety state with debounce logic"""
    def __init__(self, debounce_frames=3, area_threshold=15.0):
        self.debounce_frames = debounce_frames
        self.area_threshold = area_threshold
        self.state_history = deque(maxlen=debounce_frames)
        self.current_state = "SAFE"

    def update(self, is_unsafe):
        """Update safety state with debounce"""
        self.state_history.append(is_unsafe)

        # Only change state if all recent frames agree
        if len(self.state_history) == self.debounce_frames:
            if all(self.state_history):
                self.current_state = "UNSAFE"
            elif not any(self.state_history):
                self.current_state = "SAFE"

    def get_state(self):
        """Return current safety state"""
        return self.current_state


# ============================================================================
# UNIFIED PROCESSING FUNCTION
# ============================================================================

def process_unified_frame(frame, yolo_model, midas_model, transform, device, 
                         safety_monitor, area_threshold, show_fps=True, fps_value=0):
    """
    Process frame through both YOLO and MiDaS simultaneously
    
    Args:
        frame: BGR video frame from camera
        yolo_model: YOLOv8 model
        midas_model: MiDaS model
        transform: MiDaS transform
        device: torch device
        safety_monitor: SafetyMonitor instance
        area_threshold: safety threshold percentage
        show_fps: whether to display FPS
        fps_value: current FPS value
        
    Returns:
        yolo_frame: annotated RGB frame with YOLO detections
        depth_frame: RGB depth map visualization
        max_area: maximum bbox area percentage
        num_persons: number of persons detected
        safety_state: current safety state
    """
    # ========== YOLO PROCESSING ==========
    results = yolo_model(frame, verbose=False)
    
    yolo_frame = frame.copy()
    max_area_percentage = 0.0
    num_persons = 0

    # Process YOLO detections
    for result in results:
        boxes = result.boxes
        for box in boxes:
            cls = int(box.cls[0])
            conf = float(box.conf[0])

            # Only process 'person' detections with confidence > 0.5
            if cls == 0 and conf > 0.5:
                num_persons += 1

                # Get bounding box coordinates
                x1, y1, x2, y2 = box.xyxy[0].cpu().numpy()

                # Calculate area percentage
                area_pct = calculate_bbox_area_percentage(
                    [x1, y1, x2, y2],
                    frame.shape
                )
                max_area_percentage = max(max_area_percentage, area_pct)

                # Draw bounding box
                color = (0, 0, 255) if area_pct > area_threshold else (0, 255, 0)
                cv2.rectangle(
                    yolo_frame,
                    (int(x1), int(y1)),
                    (int(x2), int(y2)),
                    color,
                    3
                )

                # Add label
                label = f"Person {conf:.2f} | {area_pct:.1f}%"
                cv2.putText(
                    yolo_frame,
                    label,
                    (int(x1), int(y1) - 10),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.6,
                    color,
                    2
                )

    # Update safety monitor
    is_unsafe = max_area_percentage > area_threshold
    safety_monitor.update(is_unsafe)
    safety_state = safety_monitor.get_state()

    # Add FPS to YOLO frame if enabled
    if show_fps:
        cv2.putText(
            yolo_frame,
            f"FPS: {fps_value:.1f}",
            (10, 30),
            cv2.FONT_HERSHEY_SIMPLEX,
            1,
            (255, 255, 255),
            2
        )

    # Convert YOLO frame to RGB
    yolo_frame_rgb = cv2.cvtColor(yolo_frame, cv2.COLOR_BGR2RGB)

    # ========== MIDAS DEPTH PROCESSING ==========
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Apply MiDaS transforms
    input_batch = transform(frame_rgb).to(device)

    # Predict depth
    with torch.no_grad():
        prediction = midas_model(input_batch)
        prediction = torch.nn.functional.interpolate(
            prediction.unsqueeze(1),
            size=frame_rgb.shape[:2],
            mode="bicubic",
            align_corners=False,
        ).squeeze()

    # Convert to numpy and normalize
    depth_map = prediction.cpu().numpy()
    depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
    depth_map = np.uint8(depth_map)
    depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
    depth_frame_rgb = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

    return yolo_frame_rgb, depth_frame_rgb, max_area_percentage, num_persons, safety_state


# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main unified application"""
    
    # Title
    st.title("🤖 Robot Vision System - Unified View")
    st.markdown("**Real-time Safety Monitoring + Depth Estimation**")
    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")
    
    st.sidebar.subheader("Safety Monitor")
    area_threshold = st.sidebar.slider(
        "Safety Threshold (%)",
        min_value=5.0,
        max_value=50.0,
        value=15.0,
        step=1.0,
        help="Bounding box area percentage that triggers UNSAFE state"
    )

    debounce_frames = st.sidebar.slider(
        "Debounce Frames",
        min_value=1,
        max_value=10,
        value=3,
        help="Number of consecutive frames needed to change safety state"
    )

    show_fps = st.sidebar.checkbox("Show FPS", value=True)

    st.sidebar.markdown("---")
    st.sidebar.subheader("📊 Legend")
    st.sidebar.info(
        "**Safety Monitor:**\n"
        "🟢 Green box = Safe distance\n"
        "🔴 Red box = Too close!\n\n"
        "**Depth Map:**\n"
        "🟡 Bright/Yellow = Close\n"
        "🟣 Dark/Purple = Far"
    )

    # Load models
    st.sidebar.markdown("---")
    with st.sidebar:
        with st.spinner("Loading models..."):
            yolo_model = load_yolo_model()
            midas_model, transform, device = load_midas_model()

    if yolo_model is None or midas_model is None:
        st.error("❌ Failed to load models. Please check your installation.")
        return

    st.sidebar.success("✅ Models loaded successfully!")

    # Initialize safety monitor
    if 'safety_monitor' not in st.session_state:
        st.session_state.safety_monitor = SafetyMonitor(
            debounce_frames=debounce_frames,
            area_threshold=area_threshold
        )
    else:
        st.session_state.safety_monitor.debounce_frames = debounce_frames
        st.session_state.safety_monitor.area_threshold = area_threshold

    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 3])
    with col_btn1:
        start_button = st.button("🎥 Start Camera", type="primary", use_container_width=True)
    with col_btn2:
        stop_button = st.button("⏹️ Stop Camera", use_container_width=True)

    # Initialize camera state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    if start_button:
        st.session_state.camera_running = True

    if stop_button:
        st.session_state.camera_running = False

    # ========== MAIN DISPLAY LAYOUT ==========
    
    # Safety status (top, centered)
    status_placeholder = st.empty()
    
    st.markdown("---")
    
    # Video feeds side by side
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("### 🚦 Safety Monitor (YOLO)")
        yolo_placeholder = st.empty()
    
    with col2:
        st.markdown("### 📏 Depth Estimation (MiDaS)")
        depth_placeholder = st.empty()
    
    # Statistics below
    st.markdown("---")
    stats_placeholder = st.empty()

    # ========== CAMERA LOOP ==========
    
    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam. Please check your camera permissions.")
            return

        # FPS calculation
        fps_history = deque(maxlen=30)

        try:
            while st.session_state.camera_running:
                start_time = time.time()

                ret, frame = cap.read()
                if not ret:
                    st.error("Failed to read from webcam")
                    break

                # Calculate FPS
                fps = 1.0 / max(time.time() - start_time, 0.001)
                fps_history.append(fps)
                avg_fps = np.mean(fps_history) if fps_history else 0

                # Process frame through both models
                yolo_frame, depth_frame, max_area, num_persons, safety_state = process_unified_frame(
                    frame,
                    yolo_model,
                    midas_model,
                    transform,
                    device,
                    st.session_state.safety_monitor,
                    area_threshold,
                    show_fps,
                    avg_fps
                )

                # Display safety status
                if safety_state == "SAFE":
                    status_placeholder.markdown(
                        '<div class="safety-indicator safe">🟢 SAFE TO OPERATE</div>',
                        unsafe_allow_html=True
                    )
                else:
                    status_placeholder.markdown(
                        '<div class="safety-indicator unsafe">🔴 STOP - HUMAN TOO CLOSE</div>',
                        unsafe_allow_html=True
                    )

                # Display video feeds
                yolo_placeholder.image(yolo_frame, channels="RGB", use_container_width=True)
                depth_placeholder.image(depth_frame, channels="RGB", use_container_width=True)

                # Display statistics
                stats_col1, stats_col2, stats_col3, stats_col4 = st.columns(4)
                
                with stats_col1:
                    st.metric("👥 Persons Detected", num_persons)
                with stats_col2:
                    st.metric("📊 Max Bbox Area", f"{max_area:.2f}%")
                with stats_col3:
                    st.metric("⚠️ Safety Threshold", f"{area_threshold:.1f}%")
                with stats_col4:
                    st.metric("⚡ FPS", f"{avg_fps:.1f}")

                # Small delay
                time.sleep(0.01)

        finally:
            cap.release()

    else:
        st.info("👆 Click 'Start Camera' to begin real-time monitoring")
        
        # Show placeholder info
        col_info1, col_info2 = st.columns(2)
        with col_info1:
            st.info("🚦 **Safety Monitor** will detect humans and show safety status based on proximity")
        with col_info2:
            st.info("📏 **Depth Estimation** will show distance information using monocular vision")


if __name__ == "__main__":
    main()