import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
import time

# Configure Streamlit page
st.set_page_config(
    page_title="YOLO Household Items Detector",
    page_icon=":video_camera:",
    layout="wide"
)

# Household item classes in COCO dataset (YOLOv8 default)
HOUSEHOLD_CLASSES = {
    39: 'bottle',
    40: 'wine glass',
    41: 'cup',
    42: 'fork',
    43: 'knife',
    44: 'spoon',
    45: 'bowl',
    46: 'banana',
    47: 'apple',
    48: 'sandwich',
    49: 'orange',
    50: 'broccoli',
    51: 'carrot',
    73: 'book',
    74: 'clock',
    75: 'vase',
    84: 'book',
}

@st.cache_resource
def load_model(model_name):
    """Load YOLOv8 model (cached to avoid reloading)"""
    return YOLO(model_name)

def process_frame(frame, model, conf_threshold):
    """Process a single frame with YOLO detection"""
    # Run YOLOv8 inference
    results = model(frame, conf=conf_threshold, verbose=False)

    # Annotate frame with detections
    annotated_frame = results[0].plot()

    # Count detections by class
    detections = results[0].boxes
    item_counts = {}

    if detections is not None:
        for box in detections:
            cls_id = int(box.cls[0])
            if cls_id in HOUSEHOLD_CLASSES:
                item_name = HOUSEHOLD_CLASSES[cls_id]
                item_counts[item_name] = item_counts.get(item_name, 0) + 1

    # Display counts on frame
    y_offset = 30
    if item_counts:
        for item_type, count in sorted(item_counts.items()):
            text = f"{item_type.capitalize()}: {count}"
            cv2.putText(annotated_frame, text, (10, y_offset),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            y_offset += 30
    else:
        cv2.putText(annotated_frame, "No items detected", (10, 30),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

    return annotated_frame, item_counts

def main():
    st.title("Real-Time Household Items Detection")
    st.markdown("Using YOLOv8 to detect household items from webcam stream")

    # Sidebar controls
    st.sidebar.header("Settings")

    # Model selection
    model_option = st.sidebar.selectbox(
        "Select YOLO Model",
        ["yolov8n.pt", "yolov8s.pt", "yolov8m.pt", "yolov8l.pt", "yolov8x.pt"],
        help="Nano (n) is fastest, X is most accurate but slower"
    )

    # Confidence threshold
    conf_threshold = st.sidebar.slider(
        "Confidence Threshold",
        min_value=0.0,
        max_value=1.0,
        value=0.25,
        step=0.05,
        help="Higher values = fewer but more confident detections"
    )

    # Camera selection
    camera_index = st.sidebar.number_input(
        "Camera Index",
        min_value=0,
        max_value=5,
        value=0,
        help="Usually 0 for default webcam"
    )

    # Display detected classes
    st.sidebar.markdown("### Detectable Items:")
    st.sidebar.markdown("""
    - Bottle, Wine glass, Cup
    - Fork, Knife, Spoon, Bowl
    - Banana, Apple, Orange
    - Sandwich, Broccoli, Carrot
    - Book, Clock, Vase
    """)

    # Main content area
    col1, col2 = st.columns([3, 1])

    with col1:
        stframe = st.empty()

    with col2:
        st.subheader("Detection Stats")
        stats_placeholder = st.empty()
        fps_placeholder = st.empty()

    # Control buttons
    start_button = st.button("Start Camera", type="primary")
    stop_button = st.button("Stop Camera", type="secondary")

    # Initialize session state
    if 'streaming' not in st.session_state:
        st.session_state.streaming = False

    if start_button:
        st.session_state.streaming = True

    if stop_button:
        st.session_state.streaming = False

    # Streaming logic
    if st.session_state.streaming:
        st.sidebar.success("Camera is running")

        # Load model
        with st.spinner(f"Loading {model_option}..."):
            model = load_model(model_option)

        # Open webcam
        cap = cv2.VideoCapture(camera_index)

        if not cap.isOpened():
            st.error(f"Error: Cannot access camera {camera_index}")
            st.session_state.streaming = False
            return

        # Set camera properties for better performance
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

        frame_count = 0
        start_time = time.time()

        try:
            while st.session_state.streaming:
                ret, frame = cap.read()

                if not ret:
                    st.error("Failed to read from camera")
                    break

                frame_count += 1

                # Process frame
                annotated_frame, item_counts = process_frame(frame, model, conf_threshold)

                # Convert BGR to RGB for Streamlit
                annotated_frame_rgb = cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB)

                # Display frame
                stframe.image(annotated_frame_rgb, channels="RGB", use_container_width=True)

                # Calculate FPS
                elapsed_time = time.time() - start_time
                fps = frame_count / elapsed_time if elapsed_time > 0 else 0

                # Update stats
                with stats_placeholder.container():
                    if item_counts:
                        for item_type, count in sorted(item_counts.items()):
                            st.metric(item_type.capitalize(), count)
                    else:
                        st.info("No items detected")

                fps_placeholder.metric("FPS", f"{fps:.2f}")

                # Small delay to prevent overwhelming the system
                time.sleep(0.01)

        except Exception as e:
            st.error(f"Error during streaming: {str(e)}")

        finally:
            cap.release()
            st.session_state.streaming = False

    else:
        st.sidebar.info("Camera is stopped")
        st.info("Click 'Start Camera' to begin detection")

if __name__ == "__main__":
    main()
