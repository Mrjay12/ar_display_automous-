# Coordinate Frame Definitions

## Overview

This document defines all coordinate frames used in the GPS-Denied Visual Localization System and their transformations.

---

## Global Frame (WGS84)

**Name:** `GLOBAL` or `WGS84`

**Definition:**
- Latitude (φ): -90° to +90° (South to North)
- Longitude (λ): -180° to +180° (West to East)
- Altitude (h): Height above Mean Sea Level (meters)

**Properties:**
- Used for geographic ground truth
- Stored in configuration and map data
- Subject to GPS measurements (development mode only)
- **NOT used for live localization** (GPS-denied operation)

**Example:**
```
Latitude:  54.687381°N
Longitude: 25.279652°E
Altitude:  125.5 m
```

---

## Local Metric Frame (ENU)

**Name:** `LOCAL` or `ENU`

**Definition:**
- **E (East):** X-axis pointing eastward (tangent to local meridian)
- **N (North):** Y-axis pointing northward (perpendicular to meridian)
- **U (Up):** Z-axis pointing upward (normal to ellipsoid)

**Properties:**
- Right-handed coordinate system
- Origin at arbitrary local reference point
- Metric units (meters)
- Suitable for navigation and localization

**Transformation from WGS84:**
```python
import pyproj
transformer = pyproj.Transformer.from_crs("EPSG:4326", "EPSG:4978")  # WGS84 to ECEF
# Convert WGS84 (lat, lon, alt) → ECEF
# Convert ECEF → ENU (with local origin)
```

**Reference Implementation:**
```python
from pyproj import CRS, Transformer
# Define local origin
origin_lat, origin_lon, origin_alt = 54.687381, 25.279652, 125.5

# Create transformer
crs_wgs84 = CRS.from_epsg(4326)
crs_local_enu = CRS.from_proj4("+proj=enu +lat_0={} +lon_0={} +ellps=WGS84".format(
    origin_lat, origin_lon))

transformer = Transformer.from_crs(crs_wgs84, crs_local_enu)

# Transform point
point_wgs84 = (54.688, 25.280, 126)  # (lat, lon, alt)
point_enu = transformer.transform(*point_wgs84)
print(f"ENU: {point_enu}")  # (east_m, north_m, up_m)
```

**Example:**
```
Origin: 54.687381°N, 25.279652°E, 125.5 m

Point:  54.688°N, 25.280°E, 126 m
ENU:    (+850.2 m East, +690.5 m North, +0.5 m Up)
```

---

## Camera Frame

**Name:** `CAMERA` or `CAM`

**Definition:**
- **X-axis:** Points rightward in camera image plane
- **Y-axis:** Points downward in camera image plane
- **Z-axis:** Points forward (outward from camera lens)
- Origin: RGB camera center

**Properties:**
- Right-handed coordinate system
- Units: meters (depth) or pixels (image plane)
- Defined by OAK-D Pro hardware

**Transformation from Local ENU:**
- Requires camera pose estimate:
  - **Position:** (east, north, up) in ENU
  - **Orientation:** (roll, pitch, yaw) Euler angles

```python
import numpy as np
from scipy.spatial.transform import Rotation

# Camera pose in ENU
pos_enu = np.array([850.2, 690.5, 1.5])      # [east, north, up]
rpy = np.array([0.0, 0.0, 45.0]) * np.pi/180  # [roll, pitch, yaw] in radians

# Rotation matrix from ENU to Camera
R_enu_to_cam = Rotation.from_euler('xyz', rpy).as_matrix()

# Transform: p_camera = R^T @ (p_enu - t_enu)
def transform_enu_to_camera(p_enu, pos_enu, R_enu_to_cam):
    p_rel = p_enu - pos_enu  # Relative position in ENU
    p_cam = R_enu_to_cam.T @ p_rel  # Transform to camera frame
    return p_cam
```

**Axis Verification Procedure:**
```
1. Place camera on flat horizontal surface, pointing north
2. Move camera 1 meter to the east
   Expected: +1.0 in X coordinate
   
3. Move camera 1 meter forward (northward)
   Expected: +1.0 in Z coordinate (forward depth)
   
4. Move camera 1 meter upward
   Expected: -1.0 in Y coordinate (down is positive)
   
5. Rotate camera 90° clockwise (viewed from above)
   Expected: Rotation matrix reflects this transformation
```

---

## Image Frame (Pixel)

**Name:** `IMAGE` or `PIXEL`

**Definition:**
- **u-axis:** Column index (0 at left, max at right)
- **v-axis:** Row index (0 at top, max at bottom)
- Origin: Top-left pixel (0, 0)

**Properties:**
- Integer coordinates (pixel indices)
- 2D projection of 3D camera frame
- Units: pixels

**Transformation from Camera Frame:**
Using camera intrinsic matrix K:

```python
def project_to_image(p_camera, K):
    """Project 3D camera point to 2D image pixel."""
    x, y, z = p_camera
    
    # Perspective projection
    u_norm = x / z
    v_norm = y / z
    
    # Apply distortion (if needed)
    # ... distortion coefficients applied here ...
    
    # Project to pixel coordinates
    u = K[0, 0] * u_norm + K[0, 2]
    v = K[1, 1] * v_norm + K[1, 2]
    
    return np.array([u, v])
```

**Camera Intrinsic Matrix (K):**
```
K = [ fx   0  cx ]
    [  0  fy  cy ]
    [  0   0   1 ]

where:
  fx, fy = focal length (pixels)
  cx, cy = principal point (pixels)
```

**OAK-D Pro Example:**
```yaml
rgb_camera:
  intrinsic_matrix:
    - [1395.6, 0.0, 640.0]
    - [0.0, 1395.8, 360.0]
    - [0.0, 0.0, 1.0]
  resolution: [1280, 720]
  distortion_coefficients: [k1, k2, p1, p2, k3, ...]
```

---

## Stereo Frame

**Name:** `STEREO` or `STEREO_LEFT`

**Definition:**
- Separate coordinate systems for left and right cameras
- Origin at left camera center
- Right camera displaced by baseline (typically 75 mm for OAK-D Pro)

**Stereo Baseline:**
```
Left Camera ──────75 mm────── Right Camera
(origin)                          (75, 0, 0)
```

**Disparity to Depth:**
```python
def disparity_to_depth(disparity, focal_length, baseline):
    """Convert stereo disparity to depth."""
    if disparity == 0:
        return 0
    depth = (focal_length * baseline) / disparity
    return depth
```

---

## IMU Frame

**Name:** `IMU`

**Definition:**
- **X-axis:** Rightward (same as camera X)
- **Y-axis:** Downward (same as camera Y)
- **Z-axis:** Forward (same as camera Z)

**Sensor Outputs:**
- **Accelerometer:** m/s² (gravitational acceleration ≈ 9.81 m/s² when stationary)
- **Gyroscope:** rad/s or °/s (depending on configuration)

**Static Test (Verification):**
```
Camera at rest on horizontal surface, facing up:
  ax ≈ 0 m/s²
  ay ≈ 0 m/s²
  az ≈ 9.81 m/s² (gravity)
  gx, gy, gz ≈ 0 °/s (no rotation)
```

---

## Coordinate Frame Transformation Chain

```
┌─────────────────┐
│  GLOBAL (WGS84) │
│ lat, lon, alt   │
└────────┬────────┘
         │
    pyproj transform
         │
         ▼
┌─────────────────┐
│  LOCAL (ENU)    │
│ east, north, up │
└────────┬────────┘
         │
    camera pose estimate
    (R, t)
         │
         ▼
┌─────────────────┐
│  CAMERA         │
│ x (right)       │
│ y (down)        │
│ z (forward)     │
└────────┬────────┘
         │
   intrinsic matrix K
   distortion coefficients
         │
         ▼
┌─────────────────┐
│  IMAGE (PIXEL)  │
│ u (column)      │
│ v (row)         │
└─────────────────┘
```

---

## Implementation in Code

### Frame Class

```python
from dataclasses import dataclass
from typing import Literal

@dataclass
class Frame:
    """Georeferenced coordinate frame."""
    name: Literal["global", "local", "camera", "image", "imu"]
    point: np.ndarray  # Coordinate values
    timestamp_us: int

    def to_frame(self, target_frame, transformer):
        """Transform to another frame."""
        if self.name == target_frame:
            return self
        return Frame(
            name=target_frame,
            point=transformer.transform(self.name, target_frame, self.point),
            timestamp_us=self.timestamp_us
        )
```

### Transformation Functions

```python
class CoordinateSystem:
    """Manage coordinate transformations."""
    
    def __init__(self, origin_wgs84, camera_pose):
        self.origin_wgs84 = origin_wgs84  # (lat, lon, alt)
        self.camera_pose = camera_pose    # (pos_enu, orientation)
        self._setup_transformers()
    
    def wgs84_to_enu(self, lat, lon, alt):
        """Transform WGS84 to ENU."""
        # Implementation using pyproj
        pass
    
    def enu_to_camera(self, p_enu):
        """Transform ENU to camera frame."""
        # Implementation using pose
        pass
    
    def camera_to_image(self, p_camera):
        """Project camera 3D to image 2D."""
        # Implementation using K and distortion
        pass
```

---

## Coordinate Frame Documentation Requirements

Every function that transforms coordinates MUST include:

```python
def some_transform(input_point):
    """
    Transform input from Frame A to Frame B.
    
    INPUT FRAME:
    - Name: Frame A
    - Origin: [description]
    - Units: meters/pixels
    - Axes: x/y/z definitions
    - Assumptions: list of frame properties assumed
    
    OUTPUT FRAME:
    - Name: Frame B
    - Origin: [description]
    - Units: meters/pixels
    - Axes: x/y/z definitions
    
    TRANSFORMATION:
    - Method: [rotation, translation, projection, etc.]
    - Intermediate steps: [if multiple]
    
    FAILURE MODES:
    - Condition: z ≤ 0 in camera frame (behind camera)
    - Condition: point outside image bounds
    
    EXAMPLE:
    Input: ENU point (100, 200, 10)
    Output: Image pixel (640, 360)
    """
    # Implementation
    pass
```

---

## Testing Coordinate Frames

### Test 1: Axis Directions

```python
def test_camera_axes():
    """Verify camera axes point in correct directions."""
    # Move camera 1m east → X should increase
    # Move camera 1m north → Z should increase
    # Move camera 1m up → Y should decrease
```

### Test 2: Transformation Consistency

```python
def test_round_trip():
    """Verify transformations are invertible."""
    p_enu = np.array([100, 200, 50])
    p_cam = enu_to_camera(p_enu)
    p_enu_back = camera_to_enu(p_cam)
    assert np.allclose(p_enu, p_enu_back)  # Should match
```

### Test 3: Known Positions

```python
def test_known_geometry():
    """Verify transformations against known building positions."""
    # Building footprint from map: [lat, lon]
    # Expected camera view: [pixels]
    # Actual camera view: [measured pixels]
    # Should match within tolerance
```

---

This document is the authoritative reference for all coordinate frames in the system. All code must follow these definitions explicitly.
