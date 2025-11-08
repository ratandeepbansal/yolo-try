import streamlit as st
import cv2
import numpy as np
import time
import matplotlib.pyplot as plt
from io import BytesIO

st.set_page_config(page_title="Pure Visual SLAM", layout="wide")

st.title("📸 Pure Visual SLAM (Monocular, No IMU)")
st.write("""
This app demonstrates a minimal **Visual SLAM** pipeline using a single camera (no IMU).
It detects and tracks **ORB features** between frames, estimates **camera motion (pose)**,
and builds a simple **3D trajectory** in real time.
""")

# Sidebar Controls
st.sidebar.header("Controls")
frame_skip = st.sidebar.slider("Frame skip (processing frequency)", 1, 5, 2)
feature_count = st.sidebar.slider("Number of ORB features", 500, 3000, 1000)
show_features = st.sidebar.checkbox("Show keypoints on video", True)

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=feature_count)

# Streamlit camera input
camera_input = st.camera_input("Use your webcam for Visual SLAM")

# Initialize session state
if "prev_kp" not in st.session_state:
    st.session_state.prev_kp = None
    st.session_state.prev_desc = None
    st.session_state.prev_frame = None
    st.session_state.R = np.eye(3)
    st.session_state.t = np.zeros((3, 1))
    st.session_state.trajectory = []

if camera_input is not None:
    # Read frame
    bytes_data = camera_input.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    # Detect and compute ORB features
    kp, desc = orb.detectAndCompute(frame_gray, None)

    if st.session_state.prev_kp is not None and st.session_state.prev_desc is not None:
        # Match features between frames
        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        matches = bf.match(st.session_state.prev_desc, desc)
        matches = sorted(matches, key=lambda x: x.distance)

        # Extract matched points
        src_pts = np.float32([st.session_state.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
        dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

        # Estimate Essential matrix
        E, mask = cv2.findEssentialMat(dst_pts, src_pts, focal=700, pp=(frame.shape[1]/2, frame.shape[0]/2), method=cv2.RANSAC, prob=0.999, threshold=1.0)

        if E is not None:
            _, R, t, mask_pose = cv2.recoverPose(E, dst_pts, src_pts)

            # Update global pose
            st.session_state.t += st.session_state.R.dot(t)
            st.session_state.R = R.dot(st.session_state.R)

            # Record trajectory
            st.session_state.trajectory.append(st.session_state.t.flatten())

        # Draw features
        if show_features:
            frame_matches = cv2.drawMatches(st.session_state.prev_frame, st.session_state.prev_kp,
                                            frame, kp, matches[:50], None, flags=2)
            st.image(frame_matches, channels="BGR", caption="Feature Matching")
        else:
            st.image(frame, channels="BGR", caption="Current Frame")

        # Plot trajectory
        if len(st.session_state.trajectory) > 1:
            traj = np.array(st.session_state.trajectory)
            fig, ax = plt.subplots()
            ax.plot(traj[:, 0], traj[:, 2], "-b")
            ax.set_title("Estimated Camera Trajectory (Top View)")
            ax.set_xlabel("X")
            ax.set_ylabel("Z")
            buf = BytesIO()
            fig.savefig(buf, format="png")
            st.image(buf)

    # Update previous frame and features
    st.session_state.prev_kp = kp
    st.session_state.prev_desc = desc
    st.session_state.prev_frame = frame_gray

else:
    st.info("👆 Turn on your webcam and click 'Capture' to start tracking.")
