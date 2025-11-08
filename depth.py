import streamlit as st
import torch
import cv2
import numpy as np
from PIL import Image

st.set_page_config(page_title="Monocular Depth Map", layout="wide")

@st.cache_resource
def load_midas_model():
    model_type = "MiDaS_small"  # use "DPT_Large" for higher quality (slower)
    midas = torch.hub.load("intel-isl/MiDaS", model_type)
    device = "mps" if torch.backends.mps.is_available() else "cpu"
    midas.to(device).eval()

    transforms = torch.hub.load("intel-isl/MiDaS", "transforms")
    transform = transforms.small_transform if model_type == "MiDaS_small" else transforms.dpt_transform

    return midas, transform, device

midas, transform, device = load_midas_model()

st.title("📷 Real-Time Monocular Depth Estimation (MiDaS)")
st.markdown("Move your laptop webcam to see depth perception from a single camera. "
            "Brighter = closer, darker = farther.")

run_button = st.toggle("Start Camera", value=False)
frame_window = st.image([])

if run_button:
    cap = cv2.VideoCapture(0)
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)

    while True:
        ret, frame = cap.read()
        if not ret:
            st.error("Camera not found or not accessible.")
            break

        # Convert OpenCV (BGR) -> RGB
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        # Apply MiDaS transforms (directly on numpy array, not PIL)
        input_batch = transform(frame_rgb).to(device)

        # Predict depth
        with torch.no_grad():
            prediction = midas(input_batch)
            prediction = torch.nn.functional.interpolate(
                prediction.unsqueeze(1),
                size=frame_rgb.shape[:2],
                mode="bicubic",
                align_corners=False,
            ).squeeze()

        # Convert to numpy
        depth_map = prediction.cpu().numpy()
        depth_map = cv2.normalize(depth_map, None, 0, 255, cv2.NORM_MINMAX)
        depth_map = np.uint8(depth_map)
        depth_color = cv2.applyColorMap(depth_map, cv2.COLORMAP_MAGMA)
        depth_color = cv2.cvtColor(depth_color, cv2.COLOR_BGR2RGB)

        # Combine RGB + Depth side by side
        combined = np.hstack((frame_rgb, depth_color))
        frame_window.image(combined)

    cap.release()
else:
    st.info("▶️ Click the toggle to start the webcam.")