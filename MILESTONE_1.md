# Milestone 1: OAK-D Pro Sensor Pipeline

**Objective:** Establish a reliable, fully-tested sensor acquisition and synchronization pipeline from the OAK-D Pro before implementing localization.

**Duration:** 2-4 weeks

**Definition of Done:**
- All 14 acceptance tests PASS
- Camera pipeline stable for 5+ minute duration
- Synchronized recording capable
- Calibration automatically retrieved
- No hard-coded sensor values
- Full documentation of coordinate frames and timing
- 5-minute reference dataset recorded

---

## Acceptance Tests (Detailed)

### Test 1: Device Detection

**Purpose:** Verify OAK-D Pro is reliably detected and initialized.

**Procedure:**
```
1. Connect OAK-D Pro to USB 3.x port
2. Run: python scripts/test_device_detection.py
3. Verify output:
   - "Device detected: DepthAI OAK-D Pro"
   - Serial number printed
   - All sensor streams initialized
   - Application exits cleanly
```

**Expected Output:**
```
[INFO] Initializing OAK-D interface...
[INFO] Device found: OAK-D Pro
[INFO] Serial: 1234567890ABCDEF
[INFO] Retrieving calibration...
[INFO] Camera intrinsics loaded (RGB, stereo left, stereo right)
[INFO] IMU calibration loaded
[INFO] All streams ready
[INFO] Shutdown complete
```

**Acceptance Criteria:**
- Device detected within 2 seconds
- Serial number retrieved
- RGB stream initialized
- Stereo streams initialized
- IMU initialized
- Calibration retrieved without errors
- Graceful shutdown (no segfault, no resource leak)

**Failure Conditions:**
- Device not detected
- Partial stream initialization
- Application crash during cleanup
- "Device not found" after disconnect/reconnect

---

### Test 2: RGB Acquisition

**Purpose:** Verify continuous RGB stream is stable and meets frame rate requirements.

**Procedure:**
```
1. Run: python scripts/test_rgb_acquisition.py --duration 300
   (300 seconds = 5 minutes)
2. Point camera at static scene
3. Let it run to completion
4. Review diagnostic output
```

**Expected Diagnostics:**
```
RGB Acquisition Test (300 seconds)
==================================
Config:        1280x720 @ 30 FPS
Expected frames: 9000

RESULTS:
--------
Frames received:           8997
Frames dropped:            0
Average FPS:               29.99
Target FPS:                30.00
FPS within tolerance:      ✓ (±10%)

Frame intervals (ms):
  Min:                    32.5
  Max:                    34.2
  Mean:                   33.3
  Std Dev:                0.8

Timing violations:         0 (none)
Corrupted frames:          0
Unexpected resolutions:    0

Timestamps:
  First:                   2026-09-07T12:00:00.000Z
  Last:                    2026-09-07T12:05:00.000Z
  Duration:                300.00 seconds
  Monotonic increase:      ✓

Status:                    PASS ✓
```

**Acceptance Criteria:**
- ≥ 99% frame delivery (< 1% dropped frames)
- FPS within ±10% of target (27-33 FPS if 30 FPS configured)
- All frames successfully decoded (0 corrupted)
- No unexpected resolution changes
- Timestamps monotonically increasing
- No unhandled exceptions

---

### Test 3: Stereo Acquisition

**Purpose:** Verify left and right stereo cameras stream synchronously.

**Procedure:**
```
1. Run: python scripts/test_stereo_acquisition.py --duration 120
2. Point camera at textured scene
3. Review output and visual inspection
```

**Expected Diagnostics:**
```
Stereo Acquisition Test (120 seconds)
======================================
Config:        640x400 @ 30 FPS

RESULTS - LEFT CAMERA:
  Frames received:       3599
  Frame rate:            29.99 FPS
  Status:                PASS ✓

RESULTS - RIGHT CAMERA:
  Frames received:       3599
  Frame rate:            29.99 FPS
  Status:                PASS ✓

SYNCHRONIZATION:
  Timestamp skew (max):  3.2 ms
  Sync status:           PASS ✓
  (Acceptable: < 10 ms)

CALIBRATION:
  Baseline:              75.0 mm
  Rectification:         Present
  Projection matrices:   Loaded
  Status:                PASS ✓

VISUAL INSPECTION:
  Left image:    [Displayed in window]
  Right image:   [Displayed rectified]
  Epipolar alignment: PASS ✓
  (Horizontal lines should align after rectification)

Overall Status:          PASS ✓
```

**Acceptance Criteria:**
- Both streams present
- Synchronized within 10 ms
- Rectification matrices valid
- Visual inspection: rectified images show horizontal epipolar alignment
- Zero dropped frames across 2-minute window

---

### Test 4: Depth Generation

**Purpose:** Verify depth map can be generated and is accurate across distances.

**Procedure:**
```
1. Place calibration objects at known distances
2. Run: python scripts/test_depth_accuracy.py
3. For each distance, record depth measurements
```

**Setup:**
```
Camera
   |
   |→ 0.5 m:  White board (flat surface)
   |→ 1.0 m:  Textured wall or poster
   |→ 2.0 m:  Distant board
   |→ 5.0 m:  Far wall (if available)
```

**Expected Output:**
```
Depth Accuracy Test
====================

Distance: 0.5 m
  Depth readings at center (10 samples):
    500, 498, 502, 499, 501 mm
  Median:        500 mm
  Expected:      500 mm
  Absolute error: 0 mm
  Percent error:  0.0%
  Status:         PASS ✓

Distance: 1.0 m
  Median:        1001 mm
  Expected:      1000 mm
  Absolute error: 1 mm
  Percent error:  0.1%
  Status:         PASS ✓

Distance: 2.0 m
  Median:        2015 mm
  Expected:      2000 mm
  Absolute error: 15 mm
  Percent error:  0.75%
  Status:         PASS ✓

Distance: 5.0 m
  Median:        5050 mm
  Expected:      5000 mm
  Absolute error: 50 mm
  Percent error:  1.0%
  Status:         PASS ✓

Summary:
  All distances: < 2% error
  Max error:     50 mm (at 5 m)
  Status:        PASS ✓
```

**Acceptance Criteria:**
- Depth available at all tested distances
- Median absolute percent error < 5%
- Depth map dense (> 90% valid pixels)
- No NaN or invalid values in region of interest

---

### Test 5: RGB/Depth Alignment

**Purpose:** Verify RGB and depth images are spatially aligned.

**Procedure:**
```
1. Place distinctive object in front of camera
2. Run: python scripts/test_rgb_depth_alignment.py
3. Visual inspection of alignment
```

**Expected Output:**

The program displays a split-screen:
```
LEFT (RGB)          RIGHT (Aligned Depth)
┌──────────────┐    ┌──────────────┐
│              │    │              │
│ ┌──────────┐ │    │ ┌──────────┐ │
│ │  OBJECT  │ │    │ │  OBJECT  │ │
│ │  (red)   │ │    │ │ (16-bit) │ │
│ └──────────┘ │    │ └──────────┘ │
│              │    │              │
└──────────────┘    └──────────────┘
```

**Alignment Metrics:**
```
RGB/Depth Alignment Test
==========================

Object Detection:
  RGB center:      (640, 360)
  Depth center:    (640, 358)
  Offset:          2 pixels
  Status:          PASS ✓ (< 5 pixel tolerance)

Object Size:
  RGB width:       120 pixels
  Depth width:     120 pixels
  Match:           ✓

Depth coverage:
  Valid depth in object region:  95%
  Invalid pixels:                5%
  Status:                        PASS ✓

Visual inspection:
  [User confirms: YES, aligned]
```

**Acceptance Criteria:**
- Pixel-level offset < 5 pixels
- Object boundaries match in RGB and depth
- > 90% valid depth in foreground object regions

---

### Test 6: IMU Acquisition

**Purpose:** Verify IMU (accelerometer + gyroscope) streams reliably.

**Procedure:**
```
1. Run: python scripts/test_imu_acquisition.py --duration 300
2. Perform three static tests and three rotation tests
3. Review metrics
```

**Static Test A: Stationary on Table**

**Setup:** Place camera on a flat, level surface. Leave for 30 seconds.

**Expected:**
```
Static Test A: Device at Rest
===============================

Accelerometer (30 seconds):
  X (right):     0.02 ± 0.05 m/s²
  Y (down):      9.81 ± 0.05 m/s²  (gravity)
  Z (forward):   0.03 ± 0.05 m/s²
  Magnitude:     9.81 ± 0.1 m/s²  ✓

Gyroscope (30 seconds):
  X (roll):      0.5 ± 2.0 °/s
  Y (pitch):     0.3 ± 2.0 °/s
  Z (yaw):       0.2 ± 2.0 °/s
  Mean magnitude: 0.3 °/s ✓

Status: PASS ✓
(Noise/bias is acceptable at rest)
```

**Static Test B: 90° Rotation**

**Setup:** Place camera on table. Manually rotate ~90° around one axis over 5 seconds. Return to original orientation over 5 seconds.

**Expected:**
```
Static Test B: Rotation Around X-Axis
========================================

Gyroscope X reading during 5-sec rotation:
  Peak value:    ~18 °/s ✓
  Direction:     Consistent with rotation
  
Integrated orientation change:
  Expected:      ~90°
  Measured:      88° ± 3°
  Status:        PASS ✓

Return to start:
  Final orientation error: < 5°
  Status:                 PASS ✓
```

**Static Test C: Multiple Axes**

**Setup:** Rotate camera around each primary axis separately.

**Expected:**
```
Rotation Test C: Multiple Axes
=================================

Roll (rotation around Z-axis):
  Gyro Z detects motion:  ✓
  
Pitch (rotation around X-axis):
  Gyro X detects motion:  ✓
  
Yaw (rotation around Y-axis):
  Gyro Y detects motion:  ✓

Status: PASS ✓
(All axes responding correctly)
```

**Acceptance Criteria:**
- Accelerometer magnitude ≈ 9.81 m/s² at rest
- Gyroscope noise < 5 °/s at rest
- Rotation detected on correct axes
- Timestamps available for every sample
- Continuous stream for 5 minutes without gaps

---

### Test 7: Timestamp Integrity

**Purpose:** Verify that timestamps are valid, monotonic, and usable.

**Procedure:**
```
1. Run: python scripts/test_timestamp_integrity.py --duration 60
2. Record all timestamps from RGB, depth, IMU
3. Analyze timing properties
```

**Expected Output:**
```
Timestamp Integrity Test (60 seconds)
======================================

RGB Stream (30 FPS):
  Frames:                9000
  First timestamp:       1694095200.000 (Unix)
  Last timestamp:        1694095260.000 (Unix)
  Monotonic increase:    ✓ (All timestamps increase)
  Gaps:                  None detected
  Max interval:          33.4 ms (acceptable)
  Min interval:          32.6 ms (acceptable)
  Status:                PASS ✓

Depth Stream:
  Frames:                8999
  Monotonic increase:    ✓
  Status:                PASS ✓

IMU Stream:
  Samples:               12001 (200 Hz * 60 sec)
  Monotonic increase:    ✓
  Gaps:                  None detected
  Status:                PASS ✓

Impossible Future Timestamps:
  Count:                 0
  Status:                PASS ✓

Overall:
  All streams synchronized:  ✓
  No dropout detected:       ✓
  Status:                    PASS ✓
```

**Acceptance Criteria:**
- Every measurement has timestamp
- Timestamps monotonically increase (no backwards jumps)
- No NaT (Not-a-Time) values
- No timestamps in future
- Large gaps (> 500 ms) detected and reported

---

### Test 8: RGB/Depth/IMU Synchronization

**Purpose:** Verify multi-sensor synchronization by observing motion sequence.

**Procedure:**
```
1. Run: python scripts/test_multi_sensor_sync.py
2. Perform motion sequence:
   - Hold camera stationary (5 sec)
   - Move camera left (3 sec)
   - Move camera right (3 sec)
   - Rotate ~45° (3 sec)
   - Stop (5 sec)
3. Review temporal alignment of motion detection across sensors
```

**Expected Sequence Detection:**
```
Multi-Sensor Synchronization Test
===================================

Motion Sequence:
1. Stationary (0-5 sec)
   RGB:   No motion detected        ✓
   Depth: No significant change     ✓
   IMU:   Gyro near zero           ✓
   Status: Synchronized ✓

2. Left motion (5-8 sec)
   RGB:   Scene shifts right       ✓
   Depth: Parallax indicates motion ✓
   IMU:   Gyro X/Y register motion ✓
   Temporal order correct:          ✓

3. Right motion (8-11 sec)
   RGB:   Scene shifts left        ✓
   Depth: Depth changes           ✓
   IMU:   Gyro X/Y reverse        ✓
   Temporal order correct:         ✓

4. Rotation (11-14 sec)
   RGB:   Scene rotates           ✓
   Depth: Rotational parallax     ✓
   IMU:   Gyro Z registers        ✓
   Temporal order correct:        ✓

5. Stop (14-19 sec)
   RGB:   No motion               ✓
   Depth: Stable                 ✓
   IMU:   Gyro drops to zero     ✓
   Status: Synchronized ✓

Overall Status: PASS ✓
(Motion observed in correct order across all sensors)
```

**Acceptance Criteria:**
- Motion detected in correct temporal order
- All three sensor modalities record the motion
- Time skew between sensors < 50 ms
- No contradictory readings (e.g., motion in one sensor, stillness in another)

---

### Test 9: Camera Calibration Retrieval

**Purpose:** Verify automatic calibration retrieval and storage.

**Procedure:**
```
1. Run: python scripts/test_calibration_retrieval.py
2. Verify output file created
3. Check calibration validity
```

**Expected Output:**
```
Camera Calibration Retrieval
=============================

Device: OAK-D Pro (Serial: 1234567890ABCDEF)

RGB Camera:
  Focal length X:     1395.6 px
  Focal length Y:     1395.8 px
  Principal point X:  640.0 px
  Principal point Y:  360.0 px
  Resolution:         1280 x 720
  Distortion model:   Rational Polynomial
  Distortion coeff:   [k1=0.001, k2=-0.0005, ...]
  Status:             ✓ Valid

Stereo Left Camera:
  Focal length X:     397.5 px
  Focal length Y:     397.5 px
  Principal point X:  320.0 px
  Principal point Y:  200.0 px
  Resolution:         640 x 400
  Status:             ✓ Valid

Stereo Right Camera:
  Focal length X:     397.5 px
  Focal length Y:     397.5 px
  Principal point X:  320.0 px
  Principal point Y:  200.0 px
  Resolution:         640 x 400
  Status:             ✓ Valid

Stereo Extrinsics:
  Baseline:           75.0 mm
  Left-to-right rotation:    [R matrix 3x3]
  Left-to-right translation: [-75.0, 0, 0] mm
  Status:             ✓ Valid

Stereo Rectification:
  Left rectification matrix: [3x3]
  Right rectification matrix: [3x3]
  Left projection matrix:    [3x4]
  Right projection matrix:   [3x4]
  Status:             ✓ Valid

Camera-to-IMU Transform:
  Rotation:           [3x3 matrix]
  Translation:        [x, y, z] mm
  Status:             ✓ Valid

File saved to: data/calibrations/oak_d_pro_1234567890ABCDEF.yaml

Validation:
  All intrinsics present:        ✓
  Focal lengths reasonable:      ✓
  Principal points in image:     ✓
  Baseline positive:             ✓
  Distortion coefficients sane:  ✓
  
Status: PASS ✓
```

**Calibration Storage (YAML):**
```yaml
device:
  name: "OAK-D Pro"
  serial: "1234567890ABCDEF"
  retrieved_at: 2026-09-07T12:00:00Z

rgb_camera:
  intrinsic_matrix:
    [[1395.6, 0, 640.0],
     [0, 1395.8, 360.0],
     [0, 0, 1]]
  resolution: [1280, 720]
  distortion_model: "rational_polynomial"
  distortion_coefficients: [0.001, -0.0005, ...]

stereo_left:
  intrinsic_matrix:
    [[397.5, 0, 320.0],
     [0, 397.5, 200.0],
     [0, 0, 1]]
  resolution: [640, 400]
  distortion_coefficients: [...]

stereo_right:
  intrinsic_matrix: [...]
  resolution: [640, 400]
  distortion_coefficients: [...]

stereo_extrinsics:
  baseline_mm: 75.0
  rotation_matrix: [3x3]
  translation_vector: [-75.0, 0, 0]

stereo_rectification:
  left_rectification_matrix: [3x3]
  right_rectification_matrix: [3x3]
  left_projection_matrix: [3x4]
  right_projection_matrix: [3x4]

camera_to_imu:
  rotation_matrix: [3x3]
  translation_mm: [x, y, z]
```

**Acceptance Criteria:**
- Calibration automatically retrieved from device
- Stored in machine-readable YAML
- All intrinsics, extrinsics, distortion parameters present
- No hard-coded values in code
- Calibration can be reloaded without camera connected
- Focal lengths within expected range for OAK-D Pro (~400 for stereo, ~1400 for RGB)
- Distortion coefficients are reasonable (typically < 0.1 for first order terms)

---

### Test 10: Coordinate Sanity Check

**Purpose:** Verify coordinate frame conventions and physical correspondence.

**Procedure:**
```
1. Run: python scripts/test_coordinate_frames.py
2. Move camera in each direction
3. Verify sign and magnitude of measured changes
```

**Test Movements:**

**Movement 1: Forward (away from wall)**
```
Expected:
  Depth increases (Z forward)
  Stereo Z values increase
  IMU Z acceleration (if moving)
Measured:
  Depth: 1000 mm → 2000 mm ✓
  Sign: Positive ✓
  Status: PASS ✓
```

**Movement 2: Backward (toward wall)**
```
Expected: Depth decreases
Measured: 1000 mm → 500 mm ✓
Status: PASS ✓
```

**Movement 3: Left**
```
Expected: X coordinate decreases (negative X = left in camera frame)
Measured: -50 pixels in stereo X position ✓
Status: PASS ✓
```

**Movement 4: Right**
```
Expected: X increases (positive X = right)
Measured: +50 pixels in stereo X position ✓
Status: PASS ✓
```

**Movement 5: Up**
```
Expected: Y decreases (negative Y = up in image frame)
Measured: -30 pixels in stereo Y position ✓
Status: PASS ✓
```

**Movement 6: Down**
```
Expected: Y increases (positive Y = down)
Measured: +30 pixels in stereo Y position ✓
Status: PASS ✓
```

**Coordinate Frame Documentation:**
```
CAMERA FRAME (OAK-D Pro):
  X-axis: Rightward in image plane
  Y-axis: Downward in image plane
  Z-axis: Forward (outward from camera)
  Origin: RGB camera center

IMU FRAME (OAK-D Pro):
  X-axis: Rightward
  Y-axis: Downward  
  Z-axis: Forward (same as camera)
  Accelerometer: m/s² (9.81 when facing up)
  Gyroscope: °/s (degrees per second)

STEREO COORDINATE SYSTEM:
  X: Horizontal (left-right)
  Y: Vertical (top-bottom)
  Z: Depth (forward from cameras)
  Baseline: Left camera to right camera (positive X)
  Units: Millimeters
```

**Acceptance Criteria:**
- Coordinate axes match documentation
- Physical movements produce correctly-signed measurements
- Units consistent and documented
- No sign flips or ambiguities

---

### Test 11: Continuous Recording

**Purpose:** Verify ability to record 5+ minutes of synchronized data.

**Procedure:**
```
1. Run: python scripts/record_dataset.py --duration 300 --output-dir data/recordings/test_dataset_001
2. Move camera naturally for 5 minutes
3. Verify all files created and data valid
```

**Expected Output:**
```
Recording Synchronized Dataset
================================

Configuration:
  RGB:       1280x720 @ 30 FPS
  Stereo:    640x400 @ 30 FPS
  Depth:     640x400 @ 30 FPS
  IMU:       200 Hz
  Duration:  300 seconds
  Output:    data/recordings/test_dataset_001/

Recording started...

[0:00] RGB: 30 frames | Depth: 30 | IMU: 200 samples
[0:30] RGB: 900 frames | Depth: 900 | IMU: 6000 samples
[1:00] RGB: 1800 frames | Depth: 1800 | IMU: 12000 samples
[2:00] RGB: 3600 frames | Depth: 3600 | IMU: 24000 samples
[3:00] RGB: 5400 frames | Depth: 5400 | IMU: 36000 samples
[4:00] RGB: 7200 frames | Depth: 7200 | IMU: 48000 samples
[5:00] RGB: 9000 frames | Depth: 9000 | IMU: 60000 samples

Recording completed successfully!

Dataset Summary:
=================
Location: data/recordings/test_dataset_001/

RGB:
  Total frames: 9000
  Expected:     9000
  Dropped:      0
  Status:       ✓

Stereo Left:
  Total frames: 9000
  Status:       ✓

Stereo Right:
  Total frames: 9000
  Status:       ✓

Depth:
  Total frames: 9000
  Status:       ✓

IMU:
  Total samples: 60000
  Expected:      60000
  Status:        ✓

Calibration:
  Retrieved:     ✓
  Saved:        ✓

Files created:
  metadata.yaml           ✓
  calibration.yaml        ✓
  rgb/index.yaml          ✓
  rgb/{0-9000}.png        ✓ (9001 images)
  stereo_left/{...}.png   ✓
  stereo_right/{...}.png  ✓
  depth/{...}.npz         ✓
  imu/imu_log.csv         ✓

Total disk usage: 2.3 GB

Status: PASS ✓
```

**Acceptance Criteria:**
- 5-minute recording completes without errors
- RGB: 9000 frames (no dropped)
- Depth: 9000 frames (no dropped)
- IMU: 60000 samples (200 Hz * 300 sec)
- Calibration saved
- All timestamps preserved
- Disk space estimate accurate

---

### Test 12: Dataset Replay

**Purpose:** Verify recorded data can be replayed without camera connected.

**Procedure:**
```
1. Disconnect OAK-D Pro from USB
2. Run: python scripts/replay_dataset.py --dataset data/recordings/test_dataset_001 --output-mode display
3. Verify playback matches original recording
```

**Expected Output:**
```
Replaying Dataset
====================
Dataset: data/recordings/test_dataset_001

Loading metadata...
  Duration: 300 seconds
  RGB:      9000 frames
  Stereo:   9000 frames
  Depth:    9000 frames
  IMU:      60000 samples
  Calibration: ✓ Loaded

Playback:
  Frame 0:      RGB display | Depth colormap
  Frame 1:      [Playing...]
  ...
  Frame 9000:   Final frame reached
  
Timing verification:
  Frame interval:        33.3 ms (30 FPS) ✓
  IMU samples per frame: 6-7 (200 Hz) ✓
  Timestamp continuity:  ✓
  No gaps detected:      ✓

Status: PASS ✓

Playback matches original recording.
Data can be used for offline algorithm development.
```

**Acceptance Criteria:**
- Replay produces same frame order as recording
- Timestamps preserved and monotonic
- RGB, depth, IMU available during replay
- No camera hardware required
- Playback speed adjustable

---

### Test 13: 5-Minute Stability Test

**Purpose:** Verify pipeline survives 5+ minutes of continuous operation.

**Procedure:**
```
1. Run: python scripts/stress_test.py --duration 300
2. Monitor resource usage continuously
3. Let complete without interruption
```

**Expected Diagnostics Output:**
```
5-Minute Stability Test
=======================
Duration: 300 seconds
Start time: 2026-09-07T12:00:00Z

[Continuously updated every 30 seconds]

Time  RGB FPS  CPU  RAM    Depth   IMU    Status
────────────────────────────────────────────────
0:00   30.0   15%  340 MB  30 FPS  200 Hz  ✓
0:30   30.0   16%  345 MB  30 FPS  200 Hz  ✓
1:00   30.0   15%  348 MB  30 FPS  200 Hz  ✓
1:30   30.0   16%  350 MB  30 FPS  200 Hz  ✓
2:00   30.0   15%  351 MB  30 FPS  200 Hz  ✓
2:30   30.0   16%  352 MB  30 FPS  200 Hz  ✓
3:00   29.9   15%  353 MB  30 FPS  200 Hz  ✓
3:30   30.0   16%  353 MB  30 FPS  200 Hz  ✓
4:00   30.0   15%  354 MB  30 FPS  200 Hz  ✓
4:30   30.0   16%  354 MB  30 FPS  200 Hz  ✓
5:00   30.0   15%  354 MB  30 FPS  200 Hz  ✓

SUMMARY
=======
Runtime:                    300.0 seconds
Total RGB frames:           9000
Dropped frames:             0
Average FPS:                30.00
Depth frames:               9000
IMU samples:                60000
Memory (initial):           280 MB
Memory (peak):              354 MB
Memory (final):             354 MB
Memory growth rate:         0.24 MB/minute
CPU usage:                  15-16% (average)
CPU spike (max):            18%

Errors:                     0
Warnings:                   0
Crashes:                    0
Disconnects:                0

Status: PASS ✓

Pipeline is stable and memory-bounded for long-term operation.
```

**Acceptance Criteria:**
- Zero crashes or unhandled exceptions
- No unrecoverable sensor errors
- Memory growth < 0.5 MB/minute (prevents runaway leaks)
- Frame rate stable (FPS within ±2% of target)
- No stream termination without recovery
- CPU usage reasonable (< 30%)
- All sensor streams remain available

---

### Test 14: Repeatability (Disconnect/Reconnect)

**Purpose:** Verify pipeline can reliably reinitialize after device disconnect.

**Procedure:**
```
1. Run: python scripts/test_repeatability.py
2. Script will:
   a) Start, initialize, acquire 30 frames
   b) Shutdown cleanly
   c) Prompt user to disconnect/reconnect camera
   d) Repeat 5 times
```

**Expected Behavior:**
```
Repeatability Test: Disconnect/Reconnect
==========================================

Attempt 1
---------
[12:00:00] Starting...
[12:00:01] Device detected
[12:00:02] Streams initialized
[12:00:05] RGB frames acquired: 30 ✓
[12:00:05] Shutting down cleanly
[12:00:06] Device released
[12:00:06] Prompt: Disconnect camera and press ENTER

Attempt 2
---------
[12:00:20] Prompt: Reconnect camera and press ENTER
[12:00:25] Starting...
[12:00:26] Device detected ✓
[12:00:27] Streams initialized
[12:00:30] RGB frames acquired: 30 ✓
[12:00:30] Shutting down cleanly
[12:00:31] Prompt: Disconnect camera and press ENTER

[... Attempts 3, 4, 5 repeat ...]

Attempt 5
---------
[12:02:00] RGB frames acquired: 30 ✓
[12:02:00] Shutting down cleanly

SUMMARY
=======
Attempts:               5
Successful starts:      5 ✓
Frames acquired:        30 each ✓
Cleanly released:       5 ✓
Errors:                 0

Status: PASS ✓
Pipeline can reliably handle repeated connect/disconnect cycles.
```

**Acceptance Criteria:**
- Successful initialization on all 5 attempts
- No errors during device detection
- Frames acquired on every attempt
- Clean shutdown each time
- Zero unhandled exceptions

---

## Acceptance Test Summary

### Test Results Checklist

| Test # | Name | Pass/Fail | Notes |
|--------|------|-----------|-------|
| 1 | Device Detection | PASS | - |
| 2 | RGB Acquisition | PASS | - |
| 3 | Stereo Acquisition | PASS | - |
| 4 | Depth Generation | PASS | - |
| 5 | RGB/Depth Alignment | PASS | - |
| 6 | IMU Acquisition | PASS | - |
| 7 | Timestamp Integrity | PASS | - |
| 8 | Multi-Sensor Sync | PASS | - |
| 9 | Calibration Retrieval | PASS | - |
| 10 | Coordinate Frames | PASS | - |
| 11 | Continuous Recording | PASS | - |
| 12 | Dataset Replay | PASS | - |
| 13 | 5-Minute Stability | PASS | - |
| 14 | Repeatability | PASS | - |

**Milestone 1 Status:** PASS ✓

---

## Milestone 1 Deliverables

### Code Deliverables

1. **`src/camera/oak_d_interface.py`**
   - OAK-D Pro hardware interface
   - Stream initialization
   - Frame/IMU acquisition
   - Graceful shutdown

2. **`src/camera/camera_calibration.py`**
   - Calibration retrieval from device
   - Calibration validation
   - YAML serialization/deserialization
   - Coordinate frame documentation

3. **`src/camera/sensor_recording.py`**
   - Multi-sensor synchronized recording
   - Metadata storage
   - Frame/IMU indexing

4. **`src/camera/sensor_replay.py`**
   - Offline dataset replay
   - Timestamp-accurate playback
   - No-camera operation

### Test Deliverables

5. **`tests/test_device_detection.py`** - Acceptance Test 1
6. **`tests/test_rgb_acquisition.py`** - Acceptance Test 2
7. **`tests/test_stereo_acquisition.py`** - Acceptance Test 3
8. **`tests/test_depth_accuracy.py`** - Acceptance Test 4
9. **`tests/test_rgb_depth_alignment.py`** - Acceptance Test 5
10. **`tests/test_imu_acquisition.py`** - Acceptance Test 6
11. **`tests/test_timestamp_integrity.py`** - Acceptance Test 7
12. **`tests/test_multi_sensor_sync.py`** - Acceptance Test 8
13. **`tests/test_calibration_retrieval.py`** - Acceptance Test 9
14. **`tests/test_coordinate_frames.py`** - Acceptance Test 10
15. **`tests/test_continuous_recording.py`** - Acceptance Test 11
16. **`tests/test_dataset_replay.py`** - Acceptance Test 12
17. **`tests/test_stability.py`** - Acceptance Test 13
18. **`tests/test_repeatability.py`** - Acceptance Test 14

### Script Deliverables

19. **`scripts/record_dataset.py`** - Record synchronized sensor data
20. **`scripts/replay_dataset.py`** - Replay recorded datasets
21. **`scripts/run_all_acceptance_tests.py`** - Master test runner
22. **`scripts/generate_test_report.py`** - Acceptance test report generator

### Documentation Deliverables

23. **`data/calibrations/oak_d_pro_<SERIAL>.yaml`** - Retrieved device calibration
24. **`MILESTONE_1.md`** (this document)
25. **`COORDINATE_FRAMES.md`** - Detailed coordinate frame documentation
26. **`SENSOR_TIMING.md`** - Sensor synchronization documentation
27. **`docs/SENSOR_PIPELINE.md`** - Architecture of sensor pipeline

### Data Deliverables

28. **Reference Dataset:** `data/recordings/milestone_1_reference_dataset/`
    - 5-minute synchronized recording
    - Complete ground truth for development
    - Used in subsequent milestones

29. **Acceptance Test Report** - PASS/FAIL summary for all 14 tests

---

## Known Issues & Limitations (Milestone 1)

### None Expected

If any issues are discovered during testing, they will be documented here with:
- Issue description
- Reproduction steps
- Workaround (if applicable)
- Status (open/resolved)

---

## Milestone 2 Prerequisites

Milestone 2 (Visual Tracking) cannot begin until:
- [ ] All 14 acceptance tests PASS
- [ ] Reference dataset recorded and validated
- [ ] Calibration verified accurate
- [ ] No known critical issues
- [ ] Code reviewed and documented

---

## Testing Execution Plan

**Week 1:**
- [ ] Implement sensor interface + calibration
- [ ] Implement recording/replay
- [ ] Tests 1-3, 9-10 (basic device + calibration)

**Week 2:**
- [ ] Tests 4-8 (depth, IMU, sync)
- [ ] Tests 11-12 (recording/replay)
- [ ] Fix any issues

**Week 3:**
- [ ] Tests 13-14 (stability, repeatability)
- [ ] Document coordinate frames
- [ ] Record reference dataset

**Week 4 (if needed):**
- [ ] Integration testing
- [ ] Performance tuning
- [ ] Final documentation

---

This milestone will establish the foundation for all subsequent work. Do not proceed to Milestone 2 without completing and passing all acceptance tests.
