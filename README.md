# GPS-Denied Visual Localization & AR Mapping System

A standalone proof-of-concept system for determining camera pose using visual + geometric reasoning without GPS, using an OAK-D Pro camera.

## Project Status

**Current Milestone:** Milestone 1 - OAK-D Pro Sensor Pipeline

This is a **research prototype** demonstrating:
- Reliable OAK-D Pro sensor acquisition
- Multi-sensor synchronization (RGB, stereo depth, IMU)
- Calibration management
- Dataset recording and replay

## Quick Start

### Prerequisites

- **Hardware:**
  - OAK-D Pro camera (USB 3.x)
  - Linux/Windows/macOS laptop
  - Python 3.8+

- **Software:**
  - Python 3.8+
  - pip/conda

### Installation

1. **Clone repository**
   ```bash
   git clone <repo>
   cd ar_display_autonomous
   ```

2. **Create virtual environment** (recommended)
   ```bash
   python3 -m venv venv
   source venv/bin/activate  # On Windows: venv\Scripts\activate
   ```

3. **Install dependencies**
   ```bash
   pip install -r requirements.txt
   ```

### Hardware Setup

1. Connect OAK-D Pro to USB 3.x port
2. Verify connection:
   ```bash
   python -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
   ```

### Running Milestone 1 Tests

**Run all 14 acceptance tests:**
```bash
python scripts/run_all_acceptance_tests.py
```

**Run individual tests:**
```bash
# Device detection
python tests/test_device_detection.py

# RGB acquisition (5 minutes)
python tests/test_rgb_acquisition.py --duration 300

# Stereo & depth
python tests/test_stereo_acquisition.py --duration 120
python tests/test_depth_accuracy.py

# IMU
python tests/test_imu_acquisition.py --duration 300

# Record dataset
python scripts/record_dataset.py --duration 300 --output data/recordings/my_dataset

# Replay dataset (no camera needed)
python scripts/replay_dataset.py --dataset data/recordings/my_dataset
```

### Configuration

Edit `src/config/camera_config.yaml` to adjust:
- Resolution, FPS
- Depth filtering
- IMU rate
- Recording parameters

## Project Structure

```
ar_display_autonomous/
├── ARCHITECTURE.md          # Complete system architecture
├── MILESTONE_1.md           # Milestone 1 detailed spec
├── README.md                # This file
├── requirements.txt         # Python dependencies
│
├── src/                     # Source code
│   ├── camera/             # Hardware interface
│   │   ├── oak_d_interface.py
│   │   └── camera_calibration.py
│   ├── config/             # Configuration management
│   ├── perception/         # Scene understanding (future)
│   ├── localization/       # GPS-denied positioning (future)
│   ├── mapping/            # Geographic data (future)
│   ├── tracking/           # Pose tracking (future)
│   ├── ar/                 # AR rendering (future)
│   ├── evaluation/         # Testing & validation
│   └── utils/              # Utilities
│
├── tests/                  # Test suite
├── scripts/                # Utility scripts
├── data/                   # Data storage
│   ├── calibrations/       # Camera calibration files
│   ├── recordings/         # Recorded datasets
│   ├── maps/               # Map cache
│   └── evaluation/         # Ground truth data
│
└── docs/                   # Documentation
```

## Architecture Overview

The system will eventually work like this:

```
OAK-D Pro Camera
    ↓
    ├─ RGB (1280×720 @ 30 FPS)
    ├─ Stereo Depth (640×400 @ 30 FPS)
    └─ IMU (200 Hz)
    ↓
Scene Perception
    ├─ Object Detection
    ├─ Feature Extraction
    └─ Scene Understanding
    ↓
Visual Place Recognition (Coarse Localization)
    ↓
Candidate Locations (with confidence scores)
    ↓
Geographic Map Loading
    ↓
Multi-Building Geometric Verification
    ↓
Global Pose Estimation
    ├─ Latitude
    ├─ Longitude
    ├─ Roll, Pitch, Yaw
    └─ Confidence
    ↓
Continuous Local Tracking
    ↓
AR Display
    ├─ Building Wireframes
    ├─ Road Boundaries
    ├─ Building/Street Labels
    ├─ Navigation Paths
    └─ Obstacle Detection
```

## Key Design Principles

### 1. GPS is NOT a Core Dependency

The system uses GPS **only for development/evaluation**:
- Ground truth collection
- Search region hints
- Algorithm validation

**Production mode operates completely GPS-denied.**

### 2. No Hard-Coded Data

- Camera calibration automatically retrieved from device
- Stored in machine-readable YAML
- Building locations loaded from map data
- No hard-coded coordinates or camera poses

### 3. Multi-Sensor Fusion

- RGB: Visual features, place recognition
- Stereo Depth: 3D geometry, distance estimation
- IMU: Motion tracking, frame alignment

### 4. Explicit Coordinate Frames

All operations maintain clear coordinate frame documentation:
```
GLOBAL (WGS84) → LOCAL (ENU) → CAMERA → IMAGE
```

### 5. Incremental Development

- Build minimum viable system first
- Validate each stage before proceeding
- Clear acceptance tests for each milestone

## Milestone Progress

- **Milestone 1** (Current): OAK-D Sensor Pipeline
  - ✓ Device detection & initialization
  - ✓ RGB, stereo, depth acquisition
  - ✓ IMU data collection
  - ✓ Multi-sensor synchronization
  - ✓ Calibration management
  - ✓ Dataset recording/replay
  - ⏳ Final testing & validation

- **Milestone 2**: Local Visual Tracking
- **Milestone 3**: Depth-based trajectory estimation
- **Milestone 4**: GPS-assisted AR projection
- **Milestone 5**: Road visualization
- **Milestone 6**: Building/street labels
- **Milestone 7**: Object detection + depth
- **Milestone 8**: Navigation path rendering
- **Milestone 9**: Visual place recognition
- **Milestone 10**: Geometric verification
- **Milestone 11**: Continuous tracking & relocalization
- **Milestone 12**: GPS-denied autonomous operation

## Technology Stack

| Component | Technology | Why |
|-----------|-----------|-----|
| Hardware Interface | DepthAI SDK | Official OAK-D API |
| Computer Vision | OpenCV 4.8+ | Standard, optimized |
| Geospatial | pyproj | Geographic transformations |
| Data Management | NumPy, Pandas | Numerical computing |
| 3D Processing | Open3D | Point cloud operations |
| Visualization | OpenCV | Direct, no heavy deps |
| Configuration | YAML | Human-readable config |
| Testing | pytest | Standard test framework |

## Performance Targets (Milestone 1)

- **RGB:** 30 FPS, zero drops
- **Depth:** 30 FPS, < 2% error at 5m
- **IMU:** 200 Hz, continuous sampling
- **Stability:** 5+ minutes without crash
- **CPU:** 15-25% laptop GPU

## Testing Methodology

All functionality is validated with **acceptance tests**, not assumptions:

- **Test 1:** Device detection
- **Test 2:** RGB acquisition (5 min)
- **Test 3:** Stereo acquisition
- **Test 4:** Depth accuracy at known distances
- **Test 5:** RGB/Depth alignment
- **Test 6:** IMU acquisition & calibration
- **Test 7:** Timestamp integrity
- **Test 8:** Multi-sensor synchronization
- **Test 9:** Calibration retrieval & storage
- **Test 10:** Coordinate frame verification
- **Test 11:** Continuous recording (5 min)
- **Test 12:** Dataset replay (no camera)
- **Test 13:** Stability test (5 min)
- **Test 14:** Repeated disconnect/reconnect

See `MILESTONE_1.md` for detailed acceptance criteria.

## Known Limitations (Milestone 1)

- Offline map requirement (loaded at startup)
- Development mode supports optional GPS
- No localization yet (sensor pipeline only)
- No AR display yet
- Depth limited to ~5m (stereo limitation)

## Documentation

- **ARCHITECTURE.md** - Complete system design, all 25 components
- **MILESTONE_1.md** - Detailed Milestone 1 spec with all 14 tests
- **src/camera/oak_d_interface.py** - Camera interface documentation
- **src/camera/camera_calibration.py** - Calibration management

## Contributing

This is a research prototype. For changes:

1. Update relevant documentation
2. Add acceptance tests for new functionality
3. Ensure all existing tests still pass
4. Validate coordinate frame assumptions

## License

[To be determined]

## Contact

GPS-Denied Visual Localization Team

## References

- [DepthAI SDK Documentation](https://docs.luxonis.com/)
- [OAK-D Pro Specifications](https://www.luxonis.com/oak-d-pro)
- [OpenStreetMap](https://www.openstreetmap.org)
- [pyproj Documentation](https://pyproj4.github.io/pyproj/stable/)

---

**Current Focus:** Complete all 14 Milestone 1 acceptance tests. Device detection and basic sensor acquisition implemented. Next: Depth accuracy, IMU, and synchronization tests.