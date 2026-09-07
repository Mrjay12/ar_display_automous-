# GPS-Denied Visual Localization & AR Mapping System - Architecture Document

## Executive Summary

This document defines the complete system architecture for a GPS-denied visual localization and augmented-reality mapping system using an OAK-D Pro camera. The system will prove that visual + geometric + map-based localization can replace GPS for determining camera pose and projecting geographic information into real-time video.

**Key principle:** This is NOT an AR visualization project. The primary research problem is GPS-denied global localization using visual + geometric reasoning. The geographic map is part of the perception system, not merely a rendering asset.

---

## 1. System Architecture Overview

### High-Level Data Flow

```
     OAK-D PRO (Hardware)
           │
    ┌──────┼──────┐
    ▼      ▼      ▼
   RGB   DEPTH   IMU
    │      │      │
    └──────┼──────┘
           ▼
    SENSOR PIPELINE
           │
    ┌──────┴──────┐
    ▼             ▼
  LIVE         RECORDED
  CAMERA       DATASET
    │             │
    └──────┬──────┘
           ▼
    SCENE PERCEPTION
    ├─ Object Detection
    ├─ Feature Extraction
    └─ Visual Understanding
           │
           ▼
    VISUAL PLACE RECOGNITION
           │
           ▼
    CANDIDATE LOCATIONS
    (with confidence scores)
           │
           ▼
    GEOGRAPHIC MAP LOADING
           │
           ▼
    MULTI-BUILDING GEOMETRIC
    VERIFICATION
           │
           ▼
    GLOBAL POSE ESTIMATION
    (Latitude, Longitude, Altitude, Roll, Pitch, Yaw)
           │
           ▼
    LOCAL VISUAL/DEPTH/IMU TRACKING
           │
           ▼
    CONFIDENCE MONITORING
           │
    ┌──────┴──────┐
    ▼             ▼
  STABLE      TRACKING LOST
   TRACK      ─→ Relocalization
    │
    ▼
    AR PROJECTION ENGINE
    │
    ├─ Building Wireframes
    ├─ Road Boundaries
    ├─ Building/Street Labels
    ├─ Navigation Paths
    ├─ Obstacle Bounding Boxes
    └─ Debug Overlays
    │
    ▼
    VISUALIZATION OUTPUT
```

### Module Decomposition

```
ar_display_autonomous/
├── camera/                          # Hardware abstraction
│   ├── oak_d_interface.py          # OAK-D Pro interface
│   ├── camera_calibration.py       # Calibration management
│   ├── sensor_recording.py         # Dataset recording
│   └── sensor_replay.py            # Dataset replay
│
├── perception/                      # Scene understanding
│   ├── object_detector.py          # Real-time object detection
│   ├── depth_processor.py          # Depth map processing
│   ├── feature_extractor.py        # Visual feature extraction
│   └── scene_analyzer.py           # Scene understanding
│
├── localization/                    # GPS-denied positioning
│   ├── visual_place_recognition.py # Coarse localization (VPR)
│   ├── geometric_verifier.py       # Map-to-scene matching
│   ├── pose_estimator.py           # 6-DoF pose calculation
│   ├── confidence_estimator.py     # Localization confidence
│   └── relocalization.py           # Recovery on tracking loss
│
├── mapping/                         # Geographic data
│   ├── map_provider.py             # Abstraction layer
│   ├── osm_loader.py               # OpenStreetMap source
│   ├── building_geometry.py        # Building mesh generation
│   ├── road_geometry.py            # Road polygon generation
│   ├── coordinate_system.py        # Coordinate transformations
│   └── map_database.py             # Map caching/indexing
│
├── tracking/                        # Continuous pose tracking
│   ├── visual_tracker.py           # Feature-based tracking
│   ├── depth_tracker.py            # Depth-based tracking
│   ├── imu_integrator.py           # IMU pre-integration
│   ├── pose_predictor.py           # Pose prediction
│   └── tracking_monitor.py         # Confidence/failure detection
│
├── ar/                              # Real-time visualization
│   ├── camera_projection.py        # Camera model + projection
│   ├── wireframe_renderer.py       # Building wireframes
│   ├── road_renderer.py            # Road boundaries
│   ├── label_renderer.py           # Text labels
│   ├── path_renderer.py            # Navigation paths
│   ├── obstacle_renderer.py        # Detection bounding boxes
│   └── ar_visualizer.py            # Composite AR display
│
├── config/                          # Configuration
│   ├── camera_config.yaml          # Camera settings
│   ├── localization_config.yaml    # Algorithm parameters
│   ├── map_config.yaml             # Map source settings
│   └── calibration.yaml            # Camera calibration (auto-loaded)
│
├── data/                            # Data storage
│   ├── calibrations/               # Camera calibration files
│   ├── recordings/                 # Recorded datasets
│   ├── maps/                        # Map cache
│   └── evaluation/                 # Ground truth + metrics
│
├── evaluation/                      # Testing & validation
│   ├── acceptance_tests.py         # Milestone acceptance tests
│   ├── metrics.py                  # Performance metrics
│   ├── ground_truth.py             # GPS ground truth loader
│   └── benchmark.py                # Benchmark suite
│
├── utils/                           # Utilities
│   ├── logging_config.py           # Logging setup
│   ├── performance_monitor.py      # CPU/memory/FPS tracking
│   ├── timestamp_sync.py           # Multi-sensor sync
│   └── visualization_utils.py      # Common visualization helpers
│
└── main.py                          # Entry point
```

---

## 2. Coordinate Frame Design

### Frame Hierarchy

```
GLOBAL (WGS84)
    ├─ Latitude
    ├─ Longitude  
    └─ Altitude (MSL)
         │
         └─→ LOCAL_METRIC (ENU)
             ├─ East (X)
             ├─ North (Y)
             └─ Up (Z)
                  │
                  └─→ MAP_FRAME (Local tangent plane)
                      ├─ X_map
                      ├─ Y_map
                      └─ Z_map (often = Up)
                           │
                           └─→ CAMERA_FRAME
                               ├─ X_cam (right)
                               ├─ Y_cam (down)
                               └─ Z_cam (forward)
                                    │
                                    └─→ IMAGE_FRAME
                                        ├─ u (column, pixels)
                                        └─ v (row, pixels)
```

### Coordinate Transformations

**Geographic to Local:**
```
pyproj.Transformer or GeographicLib
(lat, lon, alt) ──→ (east, north, up)
```

**Map to Camera:**
```
SE3/Pose representation
T_camera_map = [R_camera_map | t_camera_map]
```

**Camera to Image:**
```
Camera intrinsic matrix K
p_image = K @ p_camera
```

### Reference Locations

Each operation must define:
- Origin point
- Coordinate system
- Units (meters, pixels, degrees)
- Orientation/handedness
- Uncertainty/error estimates

---

## 3. Camera Calibration Management

### Calibration Data Retrieved from OAK-D Pro

```yaml
rgb_camera:
  intrinsics:
    fx: float        # Focal length x
    fy: float        # Focal length y
    cx: float        # Principal point x
    cy: float        # Principal point y
  distortion:
    model: string    # OpenCV distortion model
    coefficients: [k1, k2, p1, p2, k3, ...]
  resolution: [width, height]
  
stereo_cameras:
  left:
    intrinsics: {...}
    distortion: {...}
    resolution: [width, height]
  right:
    intrinsics: {...}
    distortion: {...}
    resolution: [width, height]
  extrinsics:
    baseline: float           # Stereo baseline (meters)
    rectification_matrix: 3x3
    projection_matrix: 3x4
    
camera_to_imu:
  rotation: 3x3 rotation matrix
  translation: [x, y, z] in camera frame
```

### Storage

- Automatically retrieved at startup
- Stored in `data/calibrations/device_{serial}.yaml`
- Verified on every initialization
- No hard-coded values anywhere in code

---

## 4. Sensor Pipeline Architecture (Milestone 1)

### OAK-D Pro Acquisition

```python
OAKDInterface
├── initialize()           # Detect device, start streams
├── get_rgb_frame()       # Retrieve RGB with timestamp
├── get_stereo_frames()   # Retrieve left/right with timestamps
├── get_depth_frame()     # Retrieve depth with timestamp
├── get_imu_data()        # Retrieve IMU samples with timestamps
├── get_calibration()     # Retrieve camera calibration
└── shutdown()            # Clean release of resources
```

### Synchronization Strategy

**Timestamp Integration:**
- Every measurement has microsecond-precision timestamp
- RGB, depth, IMU are time-tagged at sensor level
- Software reconstructs temporal alignment
- No frame dropping; lag measurement is explicit

**Recording Format:**

```
recording/
├── metadata.yaml         # Recording params, duration, sensor list
├── calibration.yaml      # Camera calibration from this device
├── rgb/
│   ├── 000000.png (with timestamp in metadata)
│   ├── 000001.png
│   └── index.yaml (timestamps for all frames)
├── stereo_left/
│   ├── 000000.png
│   └── index.yaml
├── stereo_right/
│   ├── 000000.png
│   └── index.yaml
├── depth/
│   ├── 000000.npz (compressed)
│   └── index.yaml
└── imu/
    └── imu_log.csv (timestamp, ax, ay, az, gx, gy, gz)
```

### Replay Engine

Capable of re-feeding recorded data as if live, with correct timestamps and synchronization.

---

## 5. Perception Pipeline (Phases 2+)

### Object Detection

**Detector:** YOLOv8-based detector optimized for edge deployment

**Inputs:**
- RGB frame (current frame)
- Optional: Temporal context (2-3 prior frames)

**Outputs:**
- Class label (person, car, bike, etc.)
- 2D bounding box (u, v, width, height)
- Confidence score (0-1)

**Performance Target:**
- 20-30 FPS on laptop GPU

### Depth Processing

**Inputs:**
- Raw stereo depth from OAK-D
- Camera calibration

**Outputs:**
- Dense depth map (meters)
- Confidence per pixel
- 3D point cloud for regions of interest

### Feature Extraction

**Detector:** SuperPoint or equivalent (trained on SfM datasets)

**Outputs:**
- Keypoint locations (u, v in image)
- Descriptors (128-D float vectors)
- Per-keypoint confidence

---

## 6. Localization Pipeline (Core Algorithm)

### Stage A: Coarse Visual Place Recognition

**Problem:** Given RGB frame, identify ~10 candidate geographic locations.

**Approach:** DINOv2-based embedding + retrieval
- Extract full-image embedding
- Candidate locations retrieved from geographic map index
- Return top-K candidates with confidence scores

**Inputs:**
- RGB frame
- Geographic map database

**Outputs:**
- Candidate locations: [(lat, lon, confidence), ...]
- Top candidate map region

**Computational Cost:** ~100-300 ms (GPU dependent)

### Stage B: Geometric Verification

**Problem:** Given observed buildings + depths + map data, which candidate explains the scene best?

**Approach:**
1. For each candidate location:
   - Extract buildings from map within search radius
   - Project buildings into camera frame using camera pose hypothesis
   - Compare observed depth with projected building geometry
   - Score based on alignment quality

2. Building Multi-Geometry Matching:
   - Observed: { Building_A at 35m bearing +20°, Building_B at 60m bearing -45°, ... }
   - Map candidates: Search for locations containing similar configurations
   - Evaluate spatial constraints: distances, relative bearings, heights

**Inputs:**
- Candidate locations
- Observed depth + building positions
- Camera calibration
- Map building database

**Outputs:**
- Selected location: (lat, lon)
- Confidence score
- Estimated camera height

**Computational Cost:** ~500 ms - 2 s (depends on map density)

### Stage C: Precise Pose Estimation

**Problem:** Given matched location + features, estimate full 6-DoF camera pose.

**Approach:**
1. Feature matching: visible buildings ↔ detected features
2. PnP (Perspective-n-Point) with RANSAC
3. Optional: depth-assisted PnP
4. Refinement using tracked features

**Inputs:**
- Matched geographic location
- Detected visual features (from building edges, textures)
- Building 3D geometry
- Camera calibration

**Outputs:**
- Camera position: (lat, lon, alt)
- Camera orientation: (roll, pitch, yaw)
- Pose covariance

**Computational Cost:** ~100-300 ms

### Stage D: Continuous Tracking

**Problem:** Maintain pose estimate across frames without expensive global re-localization.

**Approach:**
1. Visual tracking: KLT or SuperGlue-based feature tracking
2. Depth tracking: Register current depth against reference depth map
3. IMU integration: Predict motion between frames
4. Fusion: Combine tracking results with weighted confidence

**Inputs:**
- Reference frame + pose estimate
- Current frame + depth
- IMU measurements

**Outputs:**
- Updated camera pose
- Tracking confidence (0-1)
- Motion estimate (velocity, angular velocity)

**Failure Detection:**
- Tracking confidence < threshold → trigger relocalization
- Feature desert (low trackable features) → alert
- Large pose jump → validate against map geometry

**Computational Cost:** ~20-50 ms per frame (target 20+ FPS)

---

## 7. Mapping System

### Geographic Data Source

**Primary:** OpenStreetMap
- Building footprints (via Overpass API or planet file)
- Roads, streets, intersections
- Building names, POIs
- Street names

**Secondary:** Local height data
- SRTM (Shuttle Radar Topography Mission)
- Building height estimation (OSM has partial height data)

### Map Abstraction Layer

```python
class MapProvider(ABC):
    @abstractmethod
    def load_buildings(self, lat, lon, radius_m) -> List[Building]:
        pass
    
    @abstractmethod
    def load_roads(self, lat, lon, radius_m) -> List[Road]:
        pass
    
    @abstractmethod
    def get_elevation(self, lat, lon) -> float:
        pass
    
    @abstractmethod
    def search_poi(self, lat, lon, radius_m, tags) -> List[POI]:
        pass

class OSMProvider(MapProvider):
    # Implementation for OpenStreetMap
    pass

class Provider2GIS(MapProvider):
    # Future: 2GIS implementation
    pass
```

### Building Representation

```python
@dataclass
class Building:
    osm_id: int
    name: Optional[str]
    footprint: Polygon            # lat/lon vertices
    height: float                 # estimated or tagged
    center: Point                 # (lat, lon)
    
    def to_3d_geometry(self, local_origin) -> Mesh:
        """Convert to 3D extruded mesh in local coordinates."""
        # Returns edges only (wireframe)
        pass
```

### Road Representation

```python
@dataclass
class Road:
    osm_id: int
    name: str
    way_type: str                 # primary, secondary, residential
    path: LineString              # lat/lon vertices
    width: float                  # estimated
    
    def to_3d_geometry(self, local_origin) -> Geometry:
        """Convert to 3D line/polygon in local coordinates."""
        pass
```

---

## 8. Coordinate Transformation Implementation

### Geographic (WGS84) → Local ENU

```python
import pyproj

transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978")  # WGS84 to ECEF
# Then apply local ENU computation from ECEF reference

# Or use:
from pyproj.aoi import AreaOfInterest
from pyproj.database import CRS

crs = CRS.from_proj4("+proj=utm +zone=N ...")  # If in UTM zone
```

### Local ENU → Camera

```python
import numpy as np
from scipy.spatial.transform import Rotation

# Pose representation
class Pose:
    def __init__(self, position_enu, orientation_quat):
        self.position = position_enu      # [east, north, up]
        self.quat = orientation_quat      # [x, y, z, w]
    
    @property
    def R(self):
        return Rotation.from_quat(self.quat).as_matrix()
    
    def transform_point(self, p_global):
        """Transform point from world to camera frame."""
        p_local = p_global - self.position
        p_camera = self.R.T @ p_local
        return p_camera
```

### Camera → Image

```python
def project_to_image(p_camera, K, distortion_coeffs):
    """Project 3D camera point to 2D image."""
    x, y, z = p_camera
    
    # Undistorted
    u_norm = x / z
    v_norm = y / z
    
    # Apply distortion (if necessary)
    # u_dist, v_dist = apply_distortion(u_norm, v_norm, distortion_coeffs)
    
    # Project to pixel coordinates
    u = K[0, 0] * u_norm + K[0, 2]
    v = K[1, 1] * v_norm + K[1, 2]
    
    return np.array([u, v])
```

---

## 9. Confidence & Error Metrics

### Localization Confidence

```python
confidence = (
    vpr_confidence * w_vpr +
    geometric_match_score * w_geometric +
    tracking_consistency * w_tracking +
    map_coverage * w_map
)

# where w_* are learned or tuned weights
```

### Position Uncertainty Estimation

- Covariance from PnP solver
- Tracking drift estimation
- Confidence thresholds for relocalization trigger

### Logging

```yaml
localization_event:
  timestamp: 2026-09-07T12:34:56.789Z
  global_position:
    latitude: 54.xxxxx
    longitude: 25.xxxxx
    altitude: 125.5
  orientation:
    roll_deg: 0.5
    pitch_deg: -2.3
    yaw_deg: 175.2
  confidence:
    global_localization: 0.89
    tracking: 0.94
  estimated_error_m: 2.4
  stage: "tracking"  # or "global_localization" or "relocalization"
  num_tracked_features: 42
  num_matched_buildings: 3
```

---

## 10. GPS Integration (Development Mode)

### Not a Dependency

GPS is used ONLY for:
- Reference ground truth
- Initial search region hint
- Algorithm evaluation
- Debugging

```python
class GPSSource:
    def get_position(self) -> Optional[Tuple[float, float, float]]:
        """Return (lat, lon, altitude) or None if unavailable."""
        pass

class LocalizationEngine:
    def initialize_with_gps(self, gps_source):
        """Optional initialization from GPS."""
        pos = gps_source.get_position()
        if pos:
            self.search_region = self._expand_region(pos, radius_m=500)
        else:
            # GPS-denied: search entire map region
            self.search_region = self.map.full_coverage_region()
    
    def get_position(self) -> Pose:
        """Returns camera pose. NO GPS read here."""
        # Returns result from visual + geometric localization only
        return self.current_pose
```

---

## 11. AR Rendering Pipeline

### Projection Sequence

```
GEOGRAPHIC BUILDING
    └─→ WORLD COORDINATES (local ENU)
        └─→ CAMERA COORDINATES
            └─→ IMAGE PIXEL COORDINATES
                └─→ 2D SCREEN OVERLAY
```

### Building Wireframe Rendering

```python
def render_building_wireframe(building, pose, K, rgb_frame):
    """Project building edges into camera view."""
    
    # Convert building footprint to local coordinates
    building_3d = building.to_3d_geometry(self.origin)
    
    # Transform to camera frame
    edges_camera = [pose.transform_point(v) for v in building_3d.vertices]
    
    # Project to image
    pixels = [project_to_image(v, K) for v in edges_camera]
    
    # Draw wireframe edges
    for edge in building_3d.edges:
        p1_pix = pixels[edge[0]]
        p2_pix = pixels[edge[1]]
        cv2.line(rgb_frame, p1_pix, p2_pix, color=(0, 255, 0), thickness=2)
```

### Label Rendering

```python
def render_label(name, world_pos, pose, K, rgb_frame):
    """Render geographically-anchored text label."""
    
    # Project label position to image
    cam_pos = pose.transform_point(world_pos)
    img_pos = project_to_image(cam_pos, K)
    
    # Only render if in front of camera and within image bounds
    if cam_pos[2] > 0.1 and is_within_image(img_pos, rgb_frame.shape):
        cv2.putText(rgb_frame, name, tuple(img_pos.astype(int)), ...)
```

---

## 12. Testing & Validation Methodology

### Milestone 1 Tests

**Acceptance Test 1: Device Detection**
- Connect OAK-D Pro
- Verify it's detected
- Retrieve serial number
- Initialize all streams
- Shutdown cleanly

**Acceptance Test 2: RGB Acquisition**
- Continuous RGB stream for 5 minutes
- Measure: FPS, dropped frames, resolution consistency

**Acceptance Test 3: Stereo Acquisition**
- Left/right stream synchronization
- Retrieve stereo calibration
- Verify rectification

**Acceptance Test 4: Depth Generation**
- Depth at known distances (0.5m, 1m, 2m, 5m)
- Measure depth error for each distance

**Acceptance Test 5: RGB/Depth Alignment**
- Object in RGB should align with depth values
- Visual verification via composite display

**Acceptance Test 6: IMU Acquisition**
- 5-minute IMU stream
- Static tests (values near zero or gravity)
- Rotation tests (values change appropriately)

**Acceptance Test 7: Timestamp Integrity**
- Every measurement has valid timestamp
- Timestamps are monotonically increasing
- No impossible future timestamps

**Acceptance Test 8: Synchronization**
- Move camera through known sequence
- Verify all sensors record motion in correct order

**Acceptance Test 9: Calibration Retrieval**
- Automatically retrieve intrinsics, distortion, extrinsics
- Store in configuration file
- No hard-coded values

**Acceptance Test 10: Coordinate Sanity**
- Move camera in known directions
- Verify measured coordinates have correct signs/units

**Acceptance Test 11: Continuous Recording**
- Record 5 minutes of synchronized data
- Store RGB, depth, IMU, calibration, timestamps

**Acceptance Test 12: Replay**
- Disconnect camera
- Replay recorded dataset
- Verify temporal alignment

**Acceptance Test 13: 5-Minute Stability**
- Run pipeline 5 minutes without crash
- Measure: CPU, RAM, FPS stability, dropped frames

**Acceptance Test 14: Repeatability**
- Disconnect/reconnect OAK-D 5 times
- Verify successful initialization each time

### Success Metrics (Later Milestones)

```
Global Localization Error:
  Target: < 50 m (proof of concept)
  Good:   < 10 m
  Advanced: < 2-5 m

Orientation Error:
  Target: < 5°

Tracking:
  Initial time: < 10 seconds
  FPS: 15-30
  Relocalization time: < 5 seconds
```

---

## 13. Performance Targets

### Real-Time Constraints

- **RGB Acquisition:** 30 FPS (33 ms per frame)
- **Depth Processing:** 15-30 FPS (33-67 ms)
- **IMU:** 200+ Hz (5 ms between samples)
- **Visual Tracking:** 15-30 FPS
- **Global Relocalization:** < 3 seconds (not every frame)
- **AR Rendering:** 15-30 FPS
- **Overall visualization:** 15-30 FPS

### Resource Constraints

- CPU: Laptop GPU available
- RAM: < 2 GB for active dataset
- Disk: Dataset storage ~1 GB per 5 minutes

### Optimization Strategy

- Global localization: 1-2 Hz (occasional)
- Local tracking: 15-30 Hz (continuous)
- Features extracted selectively (not every pixel)
- Map loaded only for region of interest
- Depth processed at lower resolution where possible

---

## 14. Failure Modes & Recovery

### Common Failure Modes

1. **Tracking Lost**
   - Few trackable features
   - Fast camera motion
   - Sudden illumination change
   - Textureless environment

   **Recovery:** Trigger global relocalization

2. **Map/Reality Mismatch**
   - Building demolished/newly constructed
   - Temporary obstacles blocking view
   - Map inaccuracies

   **Recovery:** Detect discrepancy, lower confidence, maintain tracking with uncertainty

3. **Ambiguous Candidate Locations**
   - Similar buildings in multiple areas
   - Insufficient depth variation

   **Recovery:** Require additional motion to disambiguate

4. **IMU Drift**
   - Accumulated error over time
   - Temperature changes

   **Recovery:** Fuse with visual tracking, periodic visual alignment

### Relocalization Trigger Conditions

- Tracking confidence < 40%
- Feature tracking lost for > 1 second
- Pose uncertainty exceeds threshold
- Periodic re-verification (e.g., every 30 seconds)

---

## 15. Development Phases

### Phase 1: Milestone 1 (OAK-D Sensor Pipeline)
**Duration:** 2-4 weeks
- Sensor acquisition
- Calibration handling
- Recording/replay
- All 14 acceptance tests passing

### Phase 2: Milestones 2-3 (Visual Tracking)
**Duration:** 2-3 weeks
- Feature extraction
- Feature tracking
- IMU integration
- Local trajectory estimation

### Phase 3: Milestones 4-6 (GPS-Assisted Map Projection)
**Duration:** 3-4 weeks
- Geographic map loading
- Building mesh generation
- Camera projection
- AR rendering of buildings + roads + labels

### Phase 4: Milestones 7-8 (Perception + Navigation)
**Duration:** 2-3 weeks
- Object detection
- Depth-based distance estimation
- Navigation path rendering

### Phase 5: Milestones 9-10 (Visual Localization)
**Duration:** 4-6 weeks
- Visual place recognition (VPR)
- Geometric verification
- Multi-building matching algorithm
- Pose estimation

### Phase 6: Milestones 11-12 (GPS-Denied Autonomy)
**Duration:** 2-3 weeks
- Continuous tracking
- Relocalization
- Confidence monitoring
- End-to-end GPS-denied operation

---

## 16. Technology Stack

| Component | Technology | Rationale |
|-----------|-----------|-----------|
| OAK-D Interface | DepthAI SDK (Python) | Official, well-documented |
| Computer Vision | OpenCV 4.8+ | Standard, well-optimized |
| Feature Extraction | SuperPoint or ORB | SuperPoint more robust but slower; ORB faster |
| Feature Matching | SuperGlue or FLANN | SuperGlue more accurate; FLANN faster |
| VPR | DINOv2 embeddings + retrieval | Foundation model, good generalization |
| Pose Estimation | OpenCV solvePnP + RANSAC | Robust, well-tested |
| Geometric Verification | OpenCV/PCL point cloud processing | Standard point cloud ops |
| Map Data | OpenStreetMap (Overpass API) | Free, global coverage |
| Coordinate Transform | pyproj | Standard geospatial library |
| 3D Processing | Open3D or trimesh | 3D geometry operations |
| Visualization | OpenCV + NumPy | Direct, no heavy dependencies |
| Object Detection | YOLOv8 | Real-time, good accuracy |
| IMU Processing | NumPy + SciPy | Basic signal processing |

### Avoided Technologies

- **ROS:** Not required, adds complexity
- **Heavy 3D Engines (Unity/Unreal):** Overkill for MVP, use simple wireframe rendering first
- **TensorFlow/PyTorch for everything:** Use only where necessary; OpenCV for basics
- **Excessive cloud dependencies:** All processing on local device

---

## 17. Known Constraints & Limitations

### First Version

1. **Offline map requirement:** System needs pre-loaded geographic data
2. **Limited to mapped areas:** Performance degrades in unmapped regions
3. **Weather dependent:** Heavy rain/snow affects vision
4. **Lighting sensitive:** Low-light performance limited by sensor
5. **Map accuracy:** Dependent on source (OSM can have errors)
6. **Building height estimation:** May be inaccurate, affects geometry matching

### Future Improvements

- Real-time map updates from online sources
- Weather-robust feature detectors
- Low-light adapted algorithms
- Crowdsourced map corrections
- Learning-based height estimation from images

---

## 18. Success Criteria (MVP)

The system is considered MVP success when:

1. ✓ Milestone 1: Reliable sensor pipeline with all 14 acceptance tests passing
2. ✓ Milestone 2-3: Can estimate local camera trajectory without global location
3. ✓ Milestone 4: Can project known geographic buildings into camera view with manual GPS initialization
4. ✓ Milestone 5-6: Complete AR visualization with buildings, roads, labels
5. ✓ Milestone 7-8: Real-time obstacle detection with distances
6. ✓ Milestone 9: Visual place recognition produces candidate locations
7. ✓ Milestone 10: Geometric verification disambiguates candidates
8. ✓ Milestone 11-12: Fully autonomous global localization and tracking (GPS-denied)
9. ✓ Evaluation: Position error < 50 m, orientation error < 5°, runs at 15+ FPS

---

## 19. Documentation Standards

Every module must include:

```python
"""
Module description.

Responsibilities:
- Responsibility 1
- Responsibility 2

Input:
  - Format description
  - Expected range/constraints

Output:
  - Format description
  - Guaranteed properties

Performance:
  - Computational cost (ms, FPS)
  - Memory usage
  - Typical latency

Failure modes:
  - Mode 1: description, recovery
  - Mode 2: description, recovery

Example:
    >>> obj = MyClass(param=value)
    >>> result = obj.process(input_data)
"""
```

Coordinate frames must be explicitly documented:

```python
def transform(self, point_in_frame_A):
    """
    Transform point from Frame A to Frame B.
    
    Frame A: Local tangent plane (ENU), origin at (54.xxx, 25.xxx), units: meters
    Frame B: Camera frame, Z forward, X right, Y down, origin at camera center
    
    Returns: Point in Frame B with same units.
    """
```

---

## 20. Repository Structure Example

```
ar_display_autonomous/
│
├── ARCHITECTURE.md              # This document
├── README.md                    # Quick start
├── requirements.txt             # Python dependencies
├── setup.py                     # Installation script
│
├── src/
│   ├── camera/
│   │   ├── __init__.py
│   │   ├── oak_d_interface.py
│   │   ├── camera_calibration.py
│   │   ├── sensor_recording.py
│   │   └── sensor_replay.py
│   │
│   ├── config/
│   │   ├── __init__.py
│   │   ├── config_loader.py
│   │   ├── camera_config.yaml
│   │   └── localization_config.yaml
│   │
│   ├── perception/
│   │   ├── __init__.py
│   │   ├── object_detector.py
│   │   ├── feature_extractor.py
│   │   └── depth_processor.py
│   │
│   ├── localization/
│   │   ├── __init__.py
│   │   ├── visual_place_recognition.py
│   │   ├── geometric_verifier.py
│   │   ├── pose_estimator.py
│   │   └── confidence_estimator.py
│   │
│   ├── mapping/
│   │   ├── __init__.py
│   │   ├── map_provider.py
│   │   ├── osm_loader.py
│   │   ├── building_geometry.py
│   │   └── coordinate_system.py
│   │
│   ├── ar/
│   │   ├── __init__.py
│   │   ├── ar_visualizer.py
│   │   ├── wireframe_renderer.py
│   │   └── camera_projection.py
│   │
│   ├── tracking/
│   │   ├── __init__.py
│   │   ├── visual_tracker.py
│   │   ├── imu_integrator.py
│   │   └── pose_predictor.py
│   │
│   ├── evaluation/
│   │   ├── __init__.py
│   │   ├── acceptance_tests.py
│   │   ├── metrics.py
│   │   └── ground_truth.py
│   │
│   └── utils/
│       ├── __init__.py
│       ├── logging_config.py
│       ├── performance_monitor.py
│       └── timestamp_sync.py
│
├── data/
│   ├── calibrations/         # Stored calibration files
│   ├── recordings/           # Recorded datasets
│   └── maps/                 # Map cache
│
├── scripts/
│   ├── record_dataset.py     # Record sensor data
│   ├── replay_dataset.py     # Replay recorded data
│   ├── run_acceptance_tests.py
│   └── benchmark.py
│
├── tests/
│   ├── test_camera.py
│   ├── test_calibration.py
│   ├── test_recording.py
│   └── test_coordinate_transforms.py
│
└── main.py                   # Application entry point
```

---

## 21. Configuration Example

**camera_config.yaml:**
```yaml
camera:
  device_id: null          # null = auto-detect first device
  rgb:
    fps: 30
    resolution: [1280, 720]
  
  stereo:
    fps: 30
    resolution: [640, 400]
  
  depth:
    enabled: true
    median_filter: true
  
  imu:
    enabled: true
    report_rate: 200       # Hz
    
recording:
  enabled: false           # Set true to record
  output_dir: ./data/recordings
  duration_sec: 300
  
gps:
  enabled: false
  port: /dev/ttyUSB0
  baud_rate: 115200
  use_for_localization: false  # Never!
  use_for_ground_truth: true   # Evaluation only
```

---

## 22. Milestone 1 Specification (Detailed)

See separate **MILESTONE_1.md** for detailed acceptance tests and deliverables.

---

This architecture document will be updated as implementation proceeds and design decisions are validated.
