# 📊 Object Tracking & Dwell Time Analytics

## Overview

The **Object Tracking & Dwell Time Analytics** system provides comprehensive real-time tracking capabilities with zone-based analytics, heatmaps, queue detection, and abandoned object alerts. Built on YOLOv8 with ByteTrack integration.

---

## 🎯 Key Features

### 1. **Persistent Object Tracking**
- Unique ID assignment for each tracked object
- Maintains track continuity across frames
- Handles occlusions and re-identification
- Supports multiple object classes (person, vehicles, etc.)

### 2. **Zone-Based Analytics**
- Define custom Regions of Interest (ROIs)
- Multiple zone types:
  - **General**: Regular monitoring areas
  - **Queue**: Queue length detection
  - **Entrance**: Entry point monitoring
  - **Exit**: Exit point monitoring
  - **Restricted**: Alert zones

### 3. **Dwell Time Tracking**
- Calculate time spent in each zone per object
- Average dwell time across all visitors
- Per-zone occupancy statistics
- Historical dwell time data

### 4. **Path Trajectory Visualization**
- Track complete movement paths
- Visualize trajectories with polylines
- Store last 100 positions per object
- Distance traveled calculation

### 5. **Heatmap Generation**
- Real-time heatmap overlay
- Visualize popular/hot zones
- Decay-based accumulation
- Jet colormap for intensity

### 6. **Queue Detection**
- Real-time queue length monitoring
- Zone-specific people counting
- Queue zone alerts when overcrowded
- Average wait time estimation

### 7. **Abandoned Object Detection**
- Detect stationary objects (non-person)
- Configurable stationary threshold (default: 30s)
- Visual and dashboard alerts
- Track ID and duration reporting

### 8. **Analytics Dashboard**
- Real-time metrics display
- Active tracks counter
- Total objects seen
- Zone-specific statistics
- Visual alerts for abandoned objects

### 9. **Data Export**
- JSON export with full tracking data
- CSV export for spreadsheet analysis
- Includes:
  - Track IDs and classes
  - Total tracking time
  - Distance traveled
  - Average speed
  - Zone dwell times
  - Stationary status

---

## 🚀 Quick Start

### Installation

```bash
# Install dependencies
pip install -r requirements.txt

# Run the application
streamlit run tracking_analytics.py
```

### First Run

1. Click **"Start Tracking"** to begin
2. Default zones are automatically created:
   - **Entrance** (left side)
   - **Browsing** (center)
   - **Queue** (bottom right)
3. Adjust settings in sidebar
4. Objects will be tracked with unique IDs

---

## ⚙️ Configuration

### Sidebar Settings

| Setting | Description | Default |
|---------|-------------|---------|
| **Track Objects** | Select object classes to track | person |
| **Confidence Threshold** | Minimum detection confidence | 0.5 |
| **Show Trajectories** | Display movement paths | ✅ Enabled |
| **Show Heatmap** | Overlay heatmap visualization | ❌ Disabled |
| **Show Zones** | Display zone boundaries | ✅ Enabled |
| **Abandoned Object Alert** | Stationary time threshold | 30 seconds |

---

## 📊 Understanding the Dashboard

### Active Tracks
- Number of currently visible and tracked objects
- Updates in real-time as objects enter/exit frame

### Total Objects Seen
- Cumulative count of all unique objects tracked
- Persists throughout session

### Zone Statistics Cards
Each zone displays:
- **Zone Name** and type
- **Current Count**: Objects currently in zone
- **Avg Dwell Time**: Average time spent in zone (seconds)

### Alerts Section
- Shows abandoned object warnings
- Includes object ID and duration
- Red pulsing animation for attention

---

## 🏗️ System Architecture

### Core Classes

#### `Zone`
- Represents a Region of Interest
- Polygon-based geometry
- Contains point-in-polygon detection
- Draws zone overlays with transparency

#### `TrackedObject`
- Tracks individual object history
- Maintains position deque (last 100 positions)
- Calculates movement metrics:
  - Distance traveled
  - Average speed
  - Dwell times per zone
- Detects stationary behavior

#### `TrackingAnalytics`
- Main analytics engine
- Manages all tracked objects
- Updates heatmap
- Calculates zone statistics
- Handles data export

---

## 🔧 Technical Details

### Tracking Algorithm: ByteTrack
- Built into Ultralytics YOLOv8 (`model.track()`)
- No external dependencies required
- Maintains ID consistency across occlusions
- Fast and efficient for real-time use

### Heatmap Algorithm
- Accumulation-based with decay factor (0.95)
- Gaussian blur applied around each position
- Normalized and colorized with COLORMAP_JET
- Blended with video at 30% opacity

### Zone Detection
- Uses OpenCV `pointPolygonTest()`
- Checks if object center is inside polygon
- O(1) complexity for convex polygons
- Handles entry/exit events

### Dwell Time Calculation
```python
dwell_time = exit_timestamp - entry_timestamp
total_dwell = sum(all_zone_dwell_times)
```

### Abandoned Object Logic
```python
if movement_distance < 5_pixels AND time > 30_seconds:
    mark_as_abandoned()
```

---

## 📈 Use Cases

### 1. **Retail Analytics**
- Track customer paths through store
- Identify hot zones and cold zones
- Optimize store layout based on heatmaps
- Monitor checkout queue lengths
- Calculate average shopping time

### 2. **Security & Surveillance**
- Detect abandoned bags/packages
- Monitor restricted area access
- Track suspicious loitering
- Analyze entry/exit patterns
- Generate security reports

### 3. **Queue Management**
- Real-time queue length monitoring
- Average wait time estimation
- Alert staff when queues exceed threshold
- Optimize staff allocation

### 4. **Event Analytics**
- Track attendee movement at events
- Popular booth/area identification
- Crowd density heatmaps
- Entry/exit flow analysis

### 5. **Workspace Optimization**
- Monitor workspace utilization
- Identify underused areas
- Track meeting room occupancy
- Optimize office layout

---

## 💾 Data Export Format

### JSON Structure
```json
[
  {
    "track_id": 1,
    "class": "person",
    "total_time": 45.3,
    "distance_traveled": 1234.5,
    "avg_speed": 27.2,
    "zone_dwell_times": {
      "Entrance": 5.2,
      "Browsing": 35.1,
      "Queue": 5.0
    },
    "is_stationary": false
  }
]
```

### CSV Columns
- `track_id`: Unique identifier
- `class`: Object class (person, car, etc.)
- `total_time`: Total tracking duration (seconds)
- `distance_traveled`: Total distance (pixels)
- `avg_speed`: Average speed (pixels/second)
- `zone_dwell_times`: JSON string of zone times
- `is_stationary`: Boolean flag

---

## 🎨 Customization

### Creating Custom Zones

Edit the `create_default_zones()` function in `tracking_analytics.py`:

```python
def create_default_zones(frame_shape):
    h, w = frame_shape[:2]
    zones = []

    # Custom zone example
    my_zone = Zone(
        name="Custom Area",
        polygon=[
            [100, 100],   # Top-left
            [400, 100],   # Top-right
            [400, 400],   # Bottom-right
            [100, 400]    # Bottom-left
        ],
        zone_type="general"
    )
    zones.append(my_zone)

    return zones
```

### Zone Types
- `"general"` - Green color
- `"queue"` - Orange color
- `"entrance"` - Bright green
- `"exit"` - Red color
- `"restricted"` - Purple color

---

## 🔍 Troubleshooting

### Issue: Tracks keep losing ID
**Solution**:
- Increase confidence threshold
- Reduce frame rate
- Improve lighting conditions
- Use larger YOLOv8 model (s/m/l instead of n)

### Issue: Heatmap too intense/faint
**Solution**:
- Adjust `heatmap_decay` value (0.90-0.99)
- Lower value = faster decay = less intense
- Higher value = slower decay = more persistent

### Issue: False abandoned object alerts
**Solution**:
- Increase `abandoned_threshold` in sidebar
- Adjust stationary movement threshold (currently 5 pixels)
- Filter out specific object classes

### Issue: Zones not detecting objects
**Solution**:
- Verify polygon coordinates are within frame bounds
- Ensure object center (not bbox) is used for detection
- Check zone polygon orientation (clockwise/counter-clockwise)

---

## 📚 API Reference

### `Zone` Class

```python
Zone(name: str, polygon: List[List[int]], zone_type: str)
```
- **name**: Display name for the zone
- **polygon**: List of [x, y] coordinates
- **zone_type**: Type of zone (general, queue, entrance, exit, restricted)

**Methods:**
- `contains_point(point: Tuple[float, float]) -> bool`
- `draw(frame: np.ndarray, alpha: float = 0.3)`

### `TrackedObject` Class

```python
TrackedObject(track_id: int, class_name: str, position: Tuple[float, float])
```

**Key Attributes:**
- `track_id`: Unique identifier
- `positions`: Deque of last 100 positions
- `zone_dwell_times`: Dict of zone_name -> seconds
- `is_stationary`: Boolean flag
- `total_distance`: Cumulative distance traveled

**Methods:**
- `update(position, current_time)`
- `enter_zone(zone_name, entry_time)`
- `exit_zone(zone_name, exit_time)`
- `get_total_dwell_time() -> float`
- `get_average_speed() -> float`

### `TrackingAnalytics` Class

```python
TrackingAnalytics()
```

**Methods:**
- `add_zone(zone: Zone)`
- `update_tracks(detections, current_time, frame_shape)`
- `get_zone_statistics() -> Dict`
- `draw_trajectories(frame: np.ndarray)`
- `get_heatmap_overlay(frame: np.ndarray) -> np.ndarray`
- `export_data() -> Tuple[str, str]`

---

## 🚀 Future Enhancements

### Phase 2 (Planned)
- [ ] Interactive zone drawing with mouse
- [ ] Multi-camera support
- [ ] Video file input (batch processing)
- [ ] SQLite database for historical data
- [ ] Advanced analytics charts (matplotlib/plotly)

### Phase 3 (Planned)
- [ ] Real-time alerts via webhook/email
- [ ] Cloud export (AWS S3, Google Cloud Storage)
- [ ] REST API for remote monitoring
- [ ] Mobile dashboard view
- [ ] Custom alert rules engine

### Phase 4 (Planned)
- [ ] Machine learning for behavior prediction
- [ ] Anomaly detection
- [ ] Integration with point-of-sale systems
- [ ] A/B testing zone configurations
- [ ] Multi-site dashboard aggregation

---

## 📄 License & Credits

Built on top of:
- **Ultralytics YOLOv8**: Object detection
- **ByteTrack**: Object tracking algorithm
- **OpenCV**: Computer vision operations
- **Streamlit**: Web interface
- **NumPy/Pandas**: Data processing

---

## 🤝 Contributing

To extend this system:
1. Fork the repository
2. Add your feature
3. Test thoroughly
4. Submit pull request with documentation

---

## 📞 Support

For issues or questions:
1. Check troubleshooting section
2. Review code comments in `tracking_analytics.py`
3. Open GitHub issue with:
   - System info
   - Error messages
   - Steps to reproduce

---

**Version**: 1.0.0
**Last Updated**: 2025-01-11
**Status**: Production Ready ✅
