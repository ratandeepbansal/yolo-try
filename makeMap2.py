import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO

# --------------------------------------------------
# PAGE CONFIG
# --------------------------------------------------
st.set_page_config(page_title="Visual SLAM - 3D Room Mapping", layout="wide")
st.title("🗺️ Visual SLAM - 3D Room Mapping")

st.info("""
📸 **How to use:** 
1. Take a photo with your camera  
2. Click **'Add Frame to Map'**  
3. Move your camera slightly  
4. Take another photo  
5. Repeat to build your 3D map!
""")

# --------------------------------------------------
# SIDEBAR CONTROLS
# --------------------------------------------------
st.sidebar.header("Controls")
frame_skip = st.sidebar.slider("Frame skip (processing frequency)", 1, 5, 2)
feature_count = st.sidebar.slider("Number of ORB features", 500, 3000, 1500)
show_features = st.sidebar.checkbox("Show keypoints on video", True)
point_size = st.sidebar.slider("Point size in 3D view", 1, 20, 5)
max_points = st.sidebar.slider("Max points to display", 100, 5000, 2000)

# Reset button
if st.sidebar.button("🔄 Reset Map"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# --------------------------------------------------
# ORB INITIALIZATION
# --------------------------------------------------
orb = cv2.ORB_create(nfeatures=feature_count)
focal_length = 700  # Approximate camera intrinsics

# --------------------------------------------------
# INITIALIZE STATE
# --------------------------------------------------
if "initialized" not in st.session_state:
    st.session_state.prev_kp = None
    st.session_state.prev_desc = None
    st.session_state.prev_frame = None
    st.session_state.prev_frame_color = None
    st.session_state.R = np.eye(3)
    st.session_state.t = np.zeros((3, 1))
    st.session_state.trajectory = []
    st.session_state.map_points = []
    st.session_state.map_colors = []
    st.session_state.frame_count = 0
    st.session_state.initialized = True


# --------------------------------------------------
# HELPER FUNCTIONS
# --------------------------------------------------
def triangulate_points(pts1, pts2, P1, P2):
    """Triangulate 3D points from 2D correspondences"""
    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    return points_3d.T


def is_valid_point(point, max_depth=10.0):
    """Filter out invalid 3D points"""
    if point[2] < 0 or point[2] > max_depth:
        return False
    if np.linalg.norm(point) > max_depth * 2:
        return False
    return True


# --------------------------------------------------
# CAMERA INPUT HANDLING
# --------------------------------------------------
if "camera_input_cleared" in st.session_state:
    del st.session_state["camera_input_cleared"]
    camera_input = None
else:
    camera_input = st.camera_input("📷 Take a photo, then click 'Add Frame to Map' below")

# --------------------------------------------------
# ADD FRAME BUTTON
# --------------------------------------------------
add_frame_button = st.button("➕ Add Frame to Map", type="primary", use_container_width=True)

# --------------------------------------------------
# FRAME PROCESSING
# --------------------------------------------------
if camera_input is not None and add_frame_button:
    bytes_data = camera_input.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

    kp, desc = orb.detectAndCompute(frame_gray, None)

    if desc is not None and len(kp) > 10:
        st.session_state.frame_count += 1

        # If not first frame, compute motion
        if st.session_state.prev_kp is not None and st.session_state.prev_desc is not None:
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(st.session_state.prev_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)

            if len(matches) > 20:
                src_pts = np.float32([st.session_state.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

                E, mask = cv2.findEssentialMat(dst_pts, src_pts, focal=focal_length, pp=(cx, cy),
                                               method=cv2.RANSAC, prob=0.999, threshold=1.0)

                if E is not None and mask is not None:
                    _, R, t, mask_pose = cv2.recoverPose(E, dst_pts, src_pts, focal=focal_length, pp=(cx, cy))

                    inlier_mask = (mask.ravel() == 1) & (mask_pose.ravel() > 0)
                    src_pts_inliers = src_pts[inlier_mask]
                    dst_pts_inliers = dst_pts[inlier_mask]

                    if len(src_pts_inliers) > 10:
                        P1 = np.hstack([st.session_state.R, st.session_state.t])
                        K = np.array([[focal_length, 0, cx],
                                      [0, focal_length, cy],
                                      [0, 0, 1]])
                        P1 = K @ P1

                        t_scaled = t * 0.1
                        st.session_state.t += st.session_state.R @ t_scaled
                        st.session_state.R = R @ st.session_state.R

                        P2 = np.hstack([st.session_state.R, st.session_state.t])
                        P2 = K @ P2

                        st.session_state.trajectory.append(st.session_state.t.flatten().copy())

                        if len(src_pts_inliers) > 10 and st.session_state.frame_count % frame_skip == 0:
                            points_3d = triangulate_points(src_pts_inliers, dst_pts_inliers, P1, P2)
                            points_added = 0
                            for i, pt in enumerate(dst_pts_inliers):
                                x, y = int(pt[0, 0]), int(pt[0, 1])
                                if 0 <= x < w and 0 <= y < h:
                                    color = frame[y, x] / 255.0
                                    point_3d = points_3d[i]
                                    if is_valid_point(point_3d):
                                        st.session_state.map_points.append(point_3d)
                                        st.session_state.map_colors.append(color[::-1])
                                        points_added += 1

                            st.sidebar.success(f"➕ Added {points_added} new 3D points")

                        if len(st.session_state.map_points) > max_points:
                            st.session_state.map_points = st.session_state.map_points[-max_points:]
                            st.session_state.map_colors = st.session_state.map_colors[-max_points:]

                        st.success(f"✅ Frame {st.session_state.frame_count} processed! Take another photo.")

                    else:
                        st.warning("⚠️ Not enough inlier matches.")
                else:
                    st.warning("⚠️ Could not estimate camera motion.")
            else:
                st.warning(f"⚠️ Only {len(matches)} matches. Need 20+.")
        else:
            st.success("✅ First frame added! Now take another photo.")

        # Save current frame as previous
        st.session_state.prev_kp = kp
        st.session_state.prev_desc = desc
        st.session_state.prev_frame = frame_gray
        st.session_state.prev_frame_color = frame.copy()

        # Reset camera for next photo
        st.session_state["camera_input_cleared"] = True
        st.experimental_rerun()

    else:
        st.error("❌ Not enough features detected. Try a textured surface.")

# --------------------------------------------------
# DISPLAY SECTION
# --------------------------------------------------
col1, col2 = st.columns(2)

# Left Column - Camera and Status
with col1:
    st.subheader("📸 Current Status")
    if camera_input is not None:
        bytes_data = camera_input.getvalue()
        frame_display = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        st.image(frame_display, channels="BGR", caption="Current camera view")

    if st.session_state.frame_count > 0:
        st.info(f"Frames processed: {st.session_state.frame_count}")
    else:
        st.info("Take a photo and click 'Add Frame to Map' to start!")

    if show_features and st.session_state.prev_frame_color is not None:
        st.subheader("🔍 Last Frame")
        st.image(st.session_state.prev_frame_color, channels="BGR", caption="Previous frame")

# Right Column - 3D Map
with col2:
    st.subheader("🗺️ Room Map")
    if len(st.session_state.map_points) > 0:
        fig = plt.figure(figsize=(10, 8))
        map_points_array = np.array(st.session_state.map_points)
        colors_array = np.array(st.session_state.map_colors)

        # Top View
        ax1 = fig.add_subplot(211)
        ax1.scatter(map_points_array[:, 0], map_points_array[:, 2], c=colors_array, s=point_size, alpha=0.6)
        if len(st.session_state.trajectory) > 1:
            traj = np.array(st.session_state.trajectory)
            ax1.plot(traj[:, 0], traj[:, 2], "-r", linewidth=2)
        ax1.set_title("Top View - Room Map")
        ax1.set_xlabel("X (m)")
        ax1.set_ylabel("Z (m)")
        ax1.axis('equal')

        # 3D View
        ax2 = fig.add_subplot(212, projection='3d')
        ax2.scatter(map_points_array[:, 0], map_points_array[:, 1], map_points_array[:, 2],
                    c=colors_array, s=point_size, alpha=0.6)
        if len(st.session_state.trajectory) > 1:
            traj = np.array(st.session_state.trajectory)
            ax2.plot(traj[:, 0], traj[:, 1], traj[:, 2], "-r", linewidth=2)
        ax2.set_title("3D View - Room Map")
        ax2.set_xlabel("X")
        ax2.set_ylabel("Y")
        ax2.set_zlabel("Z")

        plt.tight_layout()
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100)
        plt.close(fig)
        st.image(buf, caption=f"Mapped {len(st.session_state.map_points)} 3D points")

    elif st.session_state.frame_count == 0:
        st.info("👆 Take a photo to start building the map!")
    elif st.session_state.frame_count == 1:
        st.info("🎯 Good! Move camera and add another frame.")
    else:
        st.info("Keep adding frames to build the map...")

# --------------------------------------------------
# SIDEBAR STATS
# --------------------------------------------------
st.sidebar.divider()
st.sidebar.subheader("📊 Statistics")
st.sidebar.metric("Frames Processed", st.session_state.frame_count)
st.sidebar.metric("Map Points", len(st.session_state.map_points))
st.sidebar.metric("Trajectory Points", len(st.session_state.trajectory))

# --------------------------------------------------
# INSTRUCTIONS EXPANDER
# --------------------------------------------------
with st.expander("📖 Instructions", expanded=(st.session_state.frame_count == 0)):
    st.write("""
    ### Step-by-Step:
    1. **Take first photo** of a textured surface  
    2. **Click 'Add Frame to Map'**  
    3. **Move camera slightly** (10–15° or 10–20 cm)  
    4. **Take next photo**  
    5. Repeat steps 2–4  
    6. Watch your 3D map grow in real-time ✨
    """)
