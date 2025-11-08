import streamlit as st
import cv2
import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from io import BytesIO
import pandas as pd

st.set_page_config(page_title="Advanced Visual SLAM", layout="wide")
st.title("🗺️ Advanced Visual SLAM - Room Mapping System")

st.info("""
📸 **Workflow:** Take photo → Click 'Add Frame to Map' → Move camera → Take photo → Click 'Add Frame to Map' → Repeat!
""")

# Sidebar Controls
st.sidebar.header("⚙️ Settings")

# Camera settings
st.sidebar.subheader("Camera")
focal_length = st.sidebar.slider("Focal length", 400, 1200, 700)
feature_count = st.sidebar.slider("ORB features", 500, 3000, 1500)

# Processing settings
st.sidebar.subheader("Processing")
frame_skip = st.sidebar.slider("Frame skip", 1, 5, 2)
translation_scale = st.sidebar.slider("Translation scale", 0.01, 0.5, 0.1, 0.01)

# Visualization settings
st.sidebar.subheader("Visualization")
point_size = st.sidebar.slider("Point size", 1, 30, 8)
max_points = st.sidebar.slider("Max points", 500, 10000, 3000, 100)
show_features = st.sidebar.checkbox("Show feature matching", True)
show_grid = st.sidebar.checkbox("Show grid", True)

# Filtering settings
st.sidebar.subheader("Filtering")
max_depth = st.sidebar.slider("Max depth (m)", 1.0, 20.0, 8.0, 0.5)

# View settings
st.sidebar.subheader("View")
view_mode = st.sidebar.selectbox("View mode", ["Both", "Top View Only", "3D View Only"])
z_invert = st.sidebar.checkbox("Invert Z axis", False)

# Reset
st.sidebar.subheader("Actions")
if st.sidebar.button("🔄 Reset Map", type="primary"):
    for key in list(st.session_state.keys()):
        del st.session_state[key]
    st.rerun()

# Initialize ORB detector
orb = cv2.ORB_create(nfeatures=feature_count)

# Initialize session state
if "prev_kp" not in st.session_state:
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

def triangulate_points(pts1, pts2, P1, P2):
    """Triangulate 3D points from 2D correspondences"""
    pts1 = pts1.reshape(-1, 2)
    pts2 = pts2.reshape(-1, 2)
    
    points_4d = cv2.triangulatePoints(P1, P2, pts1.T, pts2.T)
    points_3d = points_4d[:3, :] / points_4d[3, :]
    return points_3d.T

def is_valid_point(point, max_depth_val):
    """Filter out invalid 3D points"""
    if point[2] < 0 or point[2] > max_depth_val:
        return False
    if np.linalg.norm(point) > max_depth_val * 2:
        return False
    if np.any(np.isnan(point)) or np.any(np.isinf(point)):
        return False
    return True

def create_visualization(map_points_array, colors_array, trajectory, view_mode, point_size, show_grid, z_invert):
    """Create visualization based on selected view mode"""
    if z_invert:
        map_points_array = map_points_array.copy()
        map_points_array[:, 2] = -map_points_array[:, 2]
        trajectory = trajectory.copy()
        trajectory[:, 2] = -trajectory[:, 2]
    
    if view_mode == "Top View Only":
        fig, ax = plt.subplots(figsize=(12, 10))
        
        ax.scatter(map_points_array[:, 0], map_points_array[:, 2], 
                  c=colors_array, s=point_size, alpha=0.7, edgecolors='none')
        
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 2], "-r", linewidth=3, 
                   label="Camera Path", alpha=0.8)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 2], c='red', s=150, 
                      marker='o', edgecolors='darkred', linewidths=2, label="Current Position")
            ax.scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=150, 
                      marker='s', edgecolors='darkgreen', linewidths=2, label="Start Position")
        
        ax.set_title("Top-Down View of Room Map", fontsize=14, fontweight='bold')
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Z (meters)", fontsize=12)
        ax.legend(fontsize=10)
        if show_grid:
            ax.grid(True, alpha=0.3, linestyle='--')
        ax.axis('equal')
        ax.set_facecolor('#f0f0f0')
        
    elif view_mode == "3D View Only":
        fig = plt.figure(figsize=(12, 10))
        ax = fig.add_subplot(111, projection='3d')
        
        ax.scatter(map_points_array[:, 0], map_points_array[:, 1], map_points_array[:, 2],
                  c=colors_array, s=point_size, alpha=0.6, edgecolors='none')
        
        if len(trajectory) > 1:
            ax.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                   "-r", linewidth=3, label="Camera Path", alpha=0.8)
            ax.scatter(trajectory[-1, 0], trajectory[-1, 1], trajectory[-1, 2], 
                      c='red', s=150, marker='o', edgecolors='darkred', linewidths=2)
        
        ax.set_title("3D Room Map", fontsize=14, fontweight='bold')
        ax.set_xlabel("X (meters)", fontsize=12)
        ax.set_ylabel("Y (meters)", fontsize=12)
        ax.set_zlabel("Z (meters)", fontsize=12)
        ax.legend(fontsize=10)
        if show_grid:
            ax.grid(True, alpha=0.3)
        
    else:  # Both
        fig = plt.figure(figsize=(14, 12))
        
        # Top view
        ax1 = fig.add_subplot(211)
        ax1.scatter(map_points_array[:, 0], map_points_array[:, 2], 
                   c=colors_array, s=point_size, alpha=0.7, edgecolors='none')
        
        if len(trajectory) > 1:
            ax1.plot(trajectory[:, 0], trajectory[:, 2], "-r", linewidth=2.5, 
                    label="Camera Path", alpha=0.8)
            ax1.scatter(trajectory[-1, 0], trajectory[-1, 2], c='red', s=120, 
                       marker='o', edgecolors='darkred', linewidths=2, label="Current")
            ax1.scatter(trajectory[0, 0], trajectory[0, 2], c='green', s=120, 
                       marker='s', edgecolors='darkgreen', linewidths=2, label="Start")
        
        ax1.set_title("Top-Down View", fontsize=13, fontweight='bold')
        ax1.set_xlabel("X (meters)", fontsize=11)
        ax1.set_ylabel("Z (meters)", fontsize=11)
        ax1.legend(fontsize=9)
        if show_grid:
            ax1.grid(True, alpha=0.3, linestyle='--')
        ax1.axis('equal')
        ax1.set_facecolor('#f0f0f0')
        
        # 3D view
        ax2 = fig.add_subplot(212, projection='3d')
        ax2.scatter(map_points_array[:, 0], map_points_array[:, 1], map_points_array[:, 2],
                   c=colors_array, s=point_size, alpha=0.6, edgecolors='none')
        
        if len(trajectory) > 1:
            ax2.plot(trajectory[:, 0], trajectory[:, 1], trajectory[:, 2], 
                    "-r", linewidth=2.5, label="Camera Path", alpha=0.8)
        
        ax2.set_title("3D Perspective View", fontsize=13, fontweight='bold')
        ax2.set_xlabel("X (meters)", fontsize=11)
        ax2.set_ylabel("Y (meters)", fontsize=11)
        ax2.set_zlabel("Z (meters)", fontsize=11)
        ax2.legend(fontsize=9)
        if show_grid:
            ax2.grid(True, alpha=0.3)
    
    plt.tight_layout()
    return fig

# Main camera input
camera_input = st.camera_input("📷 Take a photo")

# Add frame button - this is the key!
add_frame_col1, add_frame_col2 = st.columns([3, 1])
with add_frame_col1:
    add_frame_button = st.button("➕ Add Frame to Map", type="primary", use_container_width=True)
with add_frame_col2:
    if camera_input is not None:
        st.success("📸 Ready")
    else:
        st.info("Take photo")

# Process frame when button is clicked
if camera_input is not None and add_frame_button:
    # Read frame
    bytes_data = camera_input.getvalue()
    frame = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
    h, w = frame.shape[:2]
    cx, cy = w / 2, h / 2
    
    frame_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    
    # Detect ORB features
    kp, desc = orb.detectAndCompute(frame_gray, None)
    
    if desc is not None and len(kp) > 10:
        st.session_state.frame_count += 1
        
        # Check if we can process with previous frame
        if st.session_state.prev_kp is not None and st.session_state.prev_desc is not None:
            # Match features
            bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
            matches = bf.match(st.session_state.prev_desc, desc)
            matches = sorted(matches, key=lambda x: x.distance)
            
            st.sidebar.success(f"✅ {len(matches)} matches found")
            
            if len(matches) > 30:
                # Extract matched points
                src_pts = np.float32([st.session_state.prev_kp[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
                dst_pts = np.float32([kp[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)
                
                # Estimate Essential matrix
                E, mask = cv2.findEssentialMat(dst_pts, src_pts, focal=focal_length, 
                                              pp=(cx, cy), method=cv2.RANSAC, 
                                              prob=0.999, threshold=1.0)
                
                if E is not None and mask is not None:
                    # Recover pose
                    _, R, t, mask_pose = cv2.recoverPose(E, dst_pts, src_pts, focal=focal_length, pp=(cx, cy))
                    
                    inlier_mask = (mask.ravel() == 1) & (mask_pose.ravel() > 0)
                    src_pts_inliers = src_pts[inlier_mask]
                    dst_pts_inliers = dst_pts[inlier_mask]
                    
                    st.sidebar.info(f"🎯 {len(src_pts_inliers)} inliers")
                    
                    if len(src_pts_inliers) > 10:
                        # Camera projection matrices
                        K = np.array([[focal_length, 0, cx],
                                     [0, focal_length, cy],
                                     [0, 0, 1]])
                        
                        P1 = K @ np.hstack([st.session_state.R, st.session_state.t])
                        
                        # Update pose
                        t_scaled = t * translation_scale
                        st.session_state.t += st.session_state.R @ t_scaled
                        st.session_state.R = R @ st.session_state.R
                        
                        P2 = K @ np.hstack([st.session_state.R, st.session_state.t])
                        
                        # Record trajectory
                        st.session_state.trajectory.append(st.session_state.t.flatten().copy())
                        
                        # Triangulate points
                        if st.session_state.frame_count % frame_skip == 0:
                            points_3d = triangulate_points(src_pts_inliers, dst_pts_inliers, P1, P2)
                            
                            points_added = 0
                            for i, pt in enumerate(dst_pts_inliers):
                                x, y = int(pt[0, 0]), int(pt[0, 1])
                                if 0 <= x < w and 0 <= y < h:
                                    color = frame[y, x] / 255.0
                                    point_3d = points_3d[i]
                                    
                                    if is_valid_point(point_3d, max_depth):
                                        st.session_state.map_points.append(point_3d)
                                        st.session_state.map_colors.append(color[::-1])
                                        points_added += 1
                            
                            st.sidebar.success(f"➕ Added {points_added} points")
                        
                        # Limit points
                        if len(st.session_state.map_points) > max_points:
                            st.session_state.map_points = st.session_state.map_points[-max_points:]
                            st.session_state.map_colors = st.session_state.map_colors[-max_points:]
                        
                        st.success(f"✅ Frame {st.session_state.frame_count} added! Move camera and add another frame.")
                    else:
                        st.warning("⚠️ Not enough inliers. Move camera more between frames.")
                else:
                    st.warning("⚠️ Could not estimate motion. Try different movement.")
            else:
                st.warning(f"⚠️ Only {len(matches)} matches (need 30+). Move camera more.")
        else:
            st.success(f"✅ First frame added! Move camera and add another frame.")
        
        # Store current frame
        st.session_state.prev_kp = kp
        st.session_state.prev_desc = desc
        st.session_state.prev_frame = frame_gray
        st.session_state.prev_frame_color = frame.copy()
    else:
        st.error("❌ Not enough features! Point at textured surfaces.")

# Display
col1, col2 = st.columns([1, 1])

with col1:
    st.subheader("📸 Camera View")
    if camera_input is not None:
        bytes_data = camera_input.getvalue()
        frame_display = cv2.imdecode(np.frombuffer(bytes_data, np.uint8), cv2.IMREAD_COLOR)
        st.image(frame_display, channels="BGR", caption="Current view")
    else:
        st.info("Take a photo to start")
    
    if show_features and st.session_state.prev_frame_color is not None:
        st.subheader("🔍 Previous Frame")
        st.image(st.session_state.prev_frame_color, channels="BGR", caption=f"Frame #{st.session_state.frame_count}")

with col2:
    st.subheader("🗺️ Room Map")
    if len(st.session_state.map_points) > 10:
        map_points_array = np.array(st.session_state.map_points)
        colors_array = np.array(st.session_state.map_colors)
        trajectory = np.array(st.session_state.trajectory)
        
        fig = create_visualization(
            map_points_array, colors_array, trajectory, 
            view_mode, point_size, show_grid, z_invert
        )
        
        buf = BytesIO()
        fig.savefig(buf, format="png", dpi=100, bbox_inches='tight')
        plt.close(fig)
        st.image(buf, caption=f"{len(st.session_state.map_points)} points mapped")
    elif st.session_state.frame_count == 0:
        st.info("Take photo → Click 'Add Frame to Map' → Start mapping!")
    elif st.session_state.frame_count == 1:
        st.info("Move camera → Take another photo → Click 'Add Frame to Map'")
    else:
        st.info("Building map... keep adding frames!")

# Sidebar stats
st.sidebar.divider()
st.sidebar.subheader("📊 Statistics")
st.sidebar.metric("Frames Processed", st.session_state.frame_count)
st.sidebar.metric("Map Points", len(st.session_state.map_points))
st.sidebar.metric("Trajectory Points", len(st.session_state.trajectory))

# Export
if len(st.session_state.map_points) > 0:
    st.sidebar.divider()
    if st.sidebar.button("💾 Export Point Cloud"):
        df = pd.DataFrame(
            st.session_state.map_points,
            columns=['X', 'Y', 'Z']
        )
        colors_df = pd.DataFrame(
            st.session_state.map_colors,
            columns=['R', 'G', 'B']
        )
        df = pd.concat([df, colors_df], axis=1)
        
        csv = df.to_csv(index=False)
        st.sidebar.download_button(
            label="⬇️ Download CSV",
            data=csv,
            file_name="room_map.csv",
            mime="text/csv"
        )

# Instructions
with st.expander("📖 Instructions"):
    st.write("""
    ### The Right Way to Use This App:
    
    **Step 1:** Take a photo  
    **Step 2:** Click 'Add Frame to Map' (this saves the frame)  
    **Step 3:** Move camera slightly (10-15° rotation or 10-20cm translation)  
    **Step 4:** Take another photo  
    **Step 5:** Click 'Add Frame to Map' again  
    **Step 6:** Repeat steps 3-5 to build your map!
    
    ### Why the button?
    Streamlit's camera only holds ONE photo at a time. The button "locks in" each frame before you take the next one, so previous frames aren't lost.
    
    ### Tips:
    - Add 10-20 frames for a good map
    - Move slowly between frames
    - Point at textured surfaces
    - Overlap views by 50%
    - Good lighting is essential
    """)