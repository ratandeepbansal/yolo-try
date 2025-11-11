"""
Object Tracking & Dwell Time Analytics System
Real-time tracking with zone-based analytics, heatmaps, and queue detection
"""

import streamlit as st
import cv2
import numpy as np
from ultralytics import YOLO
from collections import defaultdict, deque
import time
from datetime import datetime, timedelta
import json
import pandas as pd
from pathlib import Path

# Page configuration
st.set_page_config(
    page_title="Tracking Analytics",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS
st.markdown("""
<style>
    .metric-card {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 20px;
        border-radius: 10px;
        color: white;
        text-align: center;
        margin: 10px 0;
    }
    .zone-card {
        background-color: #f0f2f6;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        border-left: 4px solid #667eea;
    }
    .alert-box {
        background-color: #ff4444;
        color: white;
        padding: 15px;
        border-radius: 8px;
        margin: 10px 0;
        animation: pulse 1.5s infinite;
    }
    @keyframes pulse {
        0%, 100% { opacity: 1; }
        50% { opacity: 0.7; }
    }
    .stat-number {
        font-size: 36px;
        font-weight: bold;
    }
</style>
""", unsafe_allow_html=True)


class Zone:
    """Represents a Region of Interest (ROI) zone"""
    def __init__(self, name, polygon, zone_type="general"):
        self.name = name
        self.polygon = np.array(polygon, dtype=np.int32)
        self.zone_type = zone_type  # general, queue, entrance, exit
        self.color = self._get_color_by_type()

    def _get_color_by_type(self):
        colors = {
            "general": (100, 200, 100),
            "queue": (255, 200, 100),
            "entrance": (100, 255, 100),
            "exit": (255, 100, 100),
            "restricted": (200, 100, 255)
        }
        return colors.get(self.zone_type, (150, 150, 150))

    def contains_point(self, point):
        """Check if a point is inside the zone polygon"""
        result = cv2.pointPolygonTest(self.polygon, point, False)
        return result >= 0

    def draw(self, frame, alpha=0.3):
        """Draw zone on frame with transparency"""
        overlay = frame.copy()
        cv2.polylines(overlay, [self.polygon], True, self.color, 2)
        cv2.fillPoly(overlay, [self.polygon], self.color)
        cv2.addWeighted(overlay, alpha, frame, 1 - alpha, 0, frame)

        # Add label
        M = cv2.moments(self.polygon)
        if M["m00"] != 0:
            cx = int(M["m10"] / M["m00"])
            cy = int(M["m01"] / M["m00"])
            cv2.putText(frame, self.name, (cx - 30, cy),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)


class TrackedObject:
    """Represents a tracked object with history"""
    def __init__(self, track_id, class_name, position):
        self.track_id = track_id
        self.class_name = class_name
        self.positions = deque(maxlen=100)  # Last 100 positions
        self.positions.append(position)
        self.first_seen = time.time()
        self.last_seen = time.time()
        self.current_zones = set()
        self.zone_entry_times = {}  # zone_name: entry_timestamp
        self.zone_dwell_times = defaultdict(float)  # zone_name: total_seconds
        self.total_distance = 0.0
        self.is_stationary = False
        self.stationary_start = None

    def update(self, position, current_time):
        """Update position and calculate movement"""
        if len(self.positions) > 0:
            prev_pos = self.positions[-1]
            distance = np.linalg.norm(np.array(position) - np.array(prev_pos))
            self.total_distance += distance

            # Check if stationary (moved less than 5 pixels)
            if distance < 5:
                if self.stationary_start is None:
                    self.stationary_start = current_time
                elif current_time - self.stationary_start > 30:  # 30 seconds
                    self.is_stationary = True
            else:
                self.stationary_start = None
                self.is_stationary = False

        self.positions.append(position)
        self.last_seen = current_time

    def enter_zone(self, zone_name, entry_time):
        """Mark entry into a zone"""
        if zone_name not in self.current_zones:
            self.current_zones.add(zone_name)
            self.zone_entry_times[zone_name] = entry_time

    def exit_zone(self, zone_name, exit_time):
        """Mark exit from a zone and calculate dwell time"""
        if zone_name in self.current_zones:
            self.current_zones.remove(zone_name)
            if zone_name in self.zone_entry_times:
                dwell_time = exit_time - self.zone_entry_times[zone_name]
                self.zone_dwell_times[zone_name] += dwell_time
                del self.zone_entry_times[zone_name]

    def get_total_dwell_time(self):
        """Get total time tracked (seconds)"""
        return self.last_seen - self.first_seen

    def get_average_speed(self):
        """Get average speed (pixels per second)"""
        total_time = self.get_total_dwell_time()
        if total_time > 0:
            return self.total_distance / total_time
        return 0


class TrackingAnalytics:
    """Main tracking analytics system"""
    def __init__(self):
        self.tracked_objects = {}  # track_id: TrackedObject
        self.zones = []
        self.heatmap = None
        self.heatmap_decay = 0.95  # Decay factor for heatmap
        self.total_tracks = 0
        self.abandoned_objects = []

    def add_zone(self, zone):
        """Add a zone for tracking"""
        self.zones.append(zone)

    def update_tracks(self, detections, current_time, frame_shape):
        """Update all tracks with new detections"""
        if self.heatmap is None:
            self.heatmap = np.zeros(frame_shape[:2], dtype=np.float32)

        # Decay heatmap
        self.heatmap *= self.heatmap_decay

        active_ids = set()

        for detection in detections:
            track_id = detection['id']
            class_name = detection['class']
            position = detection['center']

            active_ids.add(track_id)

            # Create or update tracked object
            if track_id not in self.tracked_objects:
                self.tracked_objects[track_id] = TrackedObject(track_id, class_name, position)
                self.total_tracks += 1
            else:
                self.tracked_objects[track_id].update(position, current_time)

            obj = self.tracked_objects[track_id]

            # Update heatmap
            x, y = int(position[0]), int(position[1])
            if 0 <= y < self.heatmap.shape[0] and 0 <= x < self.heatmap.shape[1]:
                cv2.circle(self.heatmap, (x, y), 20, 1, -1)

            # Check zone membership
            current_zones = set()
            for zone in self.zones:
                if zone.contains_point(position):
                    current_zones.add(zone.name)
                    if zone.name not in obj.current_zones:
                        obj.enter_zone(zone.name, current_time)

            # Check for zone exits
            exited_zones = obj.current_zones - current_zones
            for zone_name in exited_zones:
                obj.exit_zone(zone_name, current_time)

        # Remove stale tracks (not seen for 5 seconds)
        stale_ids = []
        for track_id, obj in self.tracked_objects.items():
            if track_id not in active_ids:
                if current_time - obj.last_seen > 5:
                    stale_ids.append(track_id)

        for track_id in stale_ids:
            del self.tracked_objects[track_id]

        # Update abandoned objects list
        self.abandoned_objects = [
            obj for obj in self.tracked_objects.values()
            if obj.is_stationary and obj.class_name != 'person'
        ]

    def get_zone_statistics(self):
        """Get statistics for each zone"""
        zone_stats = {}
        for zone in self.zones:
            count = sum(1 for obj in self.tracked_objects.values()
                       if zone.name in obj.current_zones)
            avg_dwell = []
            for obj in self.tracked_objects.values():
                if zone.name in obj.zone_dwell_times:
                    avg_dwell.append(obj.zone_dwell_times[zone.name])

            zone_stats[zone.name] = {
                'current_count': count,
                'type': zone.zone_type,
                'avg_dwell_time': np.mean(avg_dwell) if avg_dwell else 0
            }
        return zone_stats

    def draw_trajectories(self, frame):
        """Draw movement trajectories on frame"""
        for obj in self.tracked_objects.values():
            if len(obj.positions) > 1:
                points = np.array(obj.positions, dtype=np.int32)
                cv2.polylines(frame, [points], False, (255, 255, 0), 2)

                # Draw current position
                if len(obj.positions) > 0:
                    current_pos = obj.positions[-1]
                    cv2.circle(frame, (int(current_pos[0]), int(current_pos[1])),
                             5, (0, 255, 255), -1)

                    # Add ID label
                    label = f"ID:{obj.track_id} {obj.class_name}"
                    cv2.putText(frame, label,
                              (int(current_pos[0]) + 10, int(current_pos[1]) - 10),
                              cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 255), 2)

    def get_heatmap_overlay(self, frame):
        """Generate heatmap overlay"""
        if self.heatmap is None:
            return frame

        # Normalize and colorize heatmap
        heatmap_normalized = cv2.normalize(self.heatmap, None, 0, 255, cv2.NORM_MINMAX)
        heatmap_colored = cv2.applyColorMap(heatmap_normalized.astype(np.uint8),
                                           cv2.COLORMAP_JET)

        # Blend with original frame
        overlay = cv2.addWeighted(frame, 0.7, heatmap_colored, 0.3, 0)
        return overlay

    def export_data(self):
        """Export tracking data to JSON and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

        # Prepare data
        tracking_data = []
        for obj in self.tracked_objects.values():
            tracking_data.append({
                'track_id': obj.track_id,
                'class': obj.class_name,
                'total_time': obj.get_total_dwell_time(),
                'distance_traveled': obj.total_distance,
                'avg_speed': obj.get_average_speed(),
                'zone_dwell_times': dict(obj.zone_dwell_times),
                'is_stationary': obj.is_stationary
            })

        # Save JSON
        json_path = f"tracking_data_{timestamp}.json"
        with open(json_path, 'w') as f:
            json.dump(tracking_data, f, indent=2)

        # Save CSV
        csv_path = f"tracking_data_{timestamp}.csv"
        df = pd.DataFrame(tracking_data)
        df.to_csv(csv_path, index=False)

        return json_path, csv_path


@st.cache_resource
def load_model():
    """Load YOLOv8 model with tracking"""
    try:
        model = YOLO('yolov8n.pt')
        return model
    except Exception as e:
        st.error(f"Failed to load model: {e}")
        return None


def create_default_zones(frame_shape):
    """Create default zones based on frame dimensions"""
    h, w = frame_shape[:2]
    zones = []

    # Entrance zone (left side)
    entrance = Zone("Entrance", [
        [0, 0], [w//4, 0], [w//4, h], [0, h]
    ], "entrance")
    zones.append(entrance)

    # Center zone (browsing area)
    center = Zone("Browsing", [
        [w//4, h//4], [3*w//4, h//4], [3*w//4, 3*h//4], [w//4, 3*h//4]
    ], "general")
    zones.append(center)

    # Queue zone (bottom right)
    queue = Zone("Queue", [
        [3*w//4, 2*h//3], [w, 2*h//3], [w, h], [3*w//4, h]
    ], "queue")
    zones.append(queue)

    return zones


def main():
    """Main Streamlit application"""

    st.title("📊 Object Tracking & Dwell Time Analytics")
    st.markdown("**Real-time tracking with zone-based analytics and heatmaps**")
    st.markdown("---")

    # Sidebar controls
    st.sidebar.header("⚙️ Settings")

    track_classes = st.sidebar.multiselect(
        "Track Objects",
        ["person", "car", "truck", "bus", "bicycle", "motorcycle"],
        default=["person"]
    )

    confidence_threshold = st.sidebar.slider(
        "Confidence Threshold",
        0.1, 0.9, 0.5, 0.05
    )

    show_trajectories = st.sidebar.checkbox("Show Trajectories", value=True)
    show_heatmap = st.sidebar.checkbox("Show Heatmap", value=False)
    show_zones = st.sidebar.checkbox("Show Zones", value=True)

    abandoned_threshold = st.sidebar.slider(
        "Abandoned Object Alert (seconds)",
        10, 120, 30, 10
    )

    st.sidebar.markdown("---")
    st.sidebar.markdown("### 📖 Features")
    st.sidebar.info(
        "🎯 **Persistent Tracking**: Track objects with unique IDs\n\n"
        "⏱️ **Dwell Time**: Calculate time spent in zones\n\n"
        "🗺️ **Heatmaps**: Visualize popular areas\n\n"
        "📊 **Queue Detection**: Monitor queue lengths\n\n"
        "🚨 **Abandoned Objects**: Alert for stationary items"
    )

    # Load model
    model = load_model()
    if model is None:
        st.error("Failed to load model")
        return

    # Initialize analytics system
    if 'analytics' not in st.session_state:
        st.session_state.analytics = TrackingAnalytics()
        st.session_state.zones_initialized = False

    # Layout
    col1, col2 = st.columns([2, 1])

    with col1:
        st.subheader("📹 Live Tracking View")
        video_placeholder = st.empty()

    with col2:
        st.subheader("📊 Real-Time Analytics")
        metrics_placeholder = st.empty()
        zone_stats_placeholder = st.empty()
        alerts_placeholder = st.empty()

    # Control buttons
    col_btn1, col_btn2, col_btn3 = st.columns(3)
    with col_btn1:
        start_button = st.button("🎥 Start Tracking", type="primary")
    with col_btn2:
        stop_button = st.button("⏹️ Stop")
    with col_btn3:
        export_button = st.button("💾 Export Data")

    # Initialize camera state
    if 'camera_running' not in st.session_state:
        st.session_state.camera_running = False

    if start_button:
        st.session_state.camera_running = True

    if stop_button:
        st.session_state.camera_running = False

    if export_button and len(st.session_state.analytics.tracked_objects) > 0:
        json_path, csv_path = st.session_state.analytics.export_data()
        st.success(f"✅ Data exported to {json_path} and {csv_path}")

    # Main tracking loop
    if st.session_state.camera_running:
        cap = cv2.VideoCapture(0)

        if not cap.isOpened():
            st.error("❌ Cannot access webcam")
            return

        try:
            # Initialize zones on first frame
            if not st.session_state.zones_initialized:
                ret, frame = cap.read()
                if ret:
                    zones = create_default_zones(frame.shape)
                    for zone in zones:
                        st.session_state.analytics.add_zone(zone)
                    st.session_state.zones_initialized = True
                cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

            while st.session_state.camera_running:
                ret, frame = cap.read()
                if not ret:
                    break

                current_time = time.time()

                # Run tracking
                results = model.track(frame, persist=True, verbose=False,
                                     conf=confidence_threshold)

                # Extract detections
                detections = []
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes
                    for i in range(len(boxes)):
                        track_id = int(boxes.id[i])
                        cls = int(boxes.cls[i])
                        class_name = model.names[cls]

                        if class_name in track_classes:
                            x1, y1, x2, y2 = boxes.xyxy[i].cpu().numpy()
                            center = ((x1 + x2) / 2, (y1 + y2) / 2)

                            detections.append({
                                'id': track_id,
                                'class': class_name,
                                'bbox': [x1, y1, x2, y2],
                                'center': center
                            })

                # Update analytics
                st.session_state.analytics.update_tracks(detections, current_time,
                                                        frame.shape)

                # Visualize
                annotated_frame = frame.copy()

                # Draw zones
                if show_zones:
                    for zone in st.session_state.analytics.zones:
                        zone.draw(annotated_frame)

                # Draw trajectories
                if show_trajectories:
                    st.session_state.analytics.draw_trajectories(annotated_frame)

                # Apply heatmap
                if show_heatmap:
                    annotated_frame = st.session_state.analytics.get_heatmap_overlay(
                        annotated_frame)

                # Draw bounding boxes
                for det in detections:
                    x1, y1, x2, y2 = det['bbox']
                    cv2.rectangle(annotated_frame, (int(x1), int(y1)),
                                (int(x2), int(y2)), (0, 255, 0), 2)

                # Display video
                video_placeholder.image(cv2.cvtColor(annotated_frame, cv2.COLOR_BGR2RGB),
                                       channels="RGB", use_container_width=True)

                # Display metrics
                analytics = st.session_state.analytics
                active_tracks = len(analytics.tracked_objects)

                metrics_html = f"""
                <div class="metric-card">
                    <div class="stat-number">{active_tracks}</div>
                    <div>Active Tracks</div>
                </div>
                <div class="metric-card">
                    <div class="stat-number">{analytics.total_tracks}</div>
                    <div>Total Objects Seen</div>
                </div>
                """
                metrics_placeholder.markdown(metrics_html, unsafe_allow_html=True)

                # Display zone statistics
                zone_stats = analytics.get_zone_statistics()
                zone_html = "<h4>Zone Statistics</h4>"
                for zone_name, stats in zone_stats.items():
                    zone_html += f"""
                    <div class="zone-card">
                        <strong>{zone_name}</strong> ({stats['type']})<br>
                        Current Count: <strong>{stats['current_count']}</strong><br>
                        Avg Dwell Time: <strong>{stats['avg_dwell_time']:.1f}s</strong>
                    </div>
                    """
                zone_stats_placeholder.markdown(zone_html, unsafe_allow_html=True)

                # Display alerts
                if analytics.abandoned_objects:
                    alerts_html = "<h4>🚨 Alerts</h4>"
                    for obj in analytics.abandoned_objects:
                        stationary_time = current_time - obj.stationary_start
                        alerts_html += f"""
                        <div class="alert-box">
                            Abandoned {obj.class_name} detected!<br>
                            ID: {obj.track_id} | Duration: {stationary_time:.0f}s
                        </div>
                        """
                    alerts_placeholder.markdown(alerts_html, unsafe_allow_html=True)
                else:
                    alerts_placeholder.empty()

                time.sleep(0.01)

        finally:
            cap.release()

    else:
        st.info("👆 Click 'Start Tracking' to begin analytics")


if __name__ == "__main__":
    main()
