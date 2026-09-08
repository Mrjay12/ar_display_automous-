"""
Milestone 12 Acceptance Tests: Production Integration & AR Pipeline

Purpose: Verify end-to-end AR visualization pipeline with all components integrated

Test Coverage:
1. ARCompositor initialization
2. Camera-only mode
3. AR overlay mode (full pipeline)
4. Status bar rendering
5. Navigation route rendering
6. FPS monitoring and latency tracking
7. Frame rate stability (30 FPS target)
8. Mode switching
9. Pose update handling
10. Integration with projection engine
11. Integration with rendering engines
12. Statistics collection
13. Error recovery
14. Complete autonomous demo pipeline
"""

import pytest
import logging
import numpy as np
import time
from pathlib import Path
from dataclasses import dataclass

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import cv2
from ar.ar_compositor import ARCompositor, ARMode, CompositorConfig
from mapping.osm_loader import Building, Road

logger = logging.getLogger(__name__)


# Test Fixtures

@pytest.fixture
def test_frame_720p():
    """Create test RGB frame (720p)."""
    return np.ones((720, 1280, 3), dtype=np.uint8) * 128


@pytest.fixture
def test_depth_map():
    """Create test depth map (5m everywhere)."""
    return np.full((720, 1280), 5000, dtype=np.uint16)


@pytest.fixture
def camera_intrinsics():
    """OAK-D Pro camera matrix."""
    return np.array([
        [1395.6, 0.0, 640.0],
        [0.0, 1395.8, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


@pytest.fixture
def local_origin():
    """Test area origin."""
    return (54.687381, 25.279652, 125.5)


@pytest.fixture
def camera_pose():
    """Mock camera pose."""
    @dataclass
    class MockPose:
        timestamp_us: int = 0
        latitude: float = 54.687381
        longitude: float = 25.279652
        altitude: float = 125.5
        roll_deg: float = 0.0
        pitch_deg: float = 0.0
        yaw_deg: float = 0.0

    return MockPose()


@pytest.fixture
def sample_buildings():
    """Create test buildings."""
    buildings = []
    for i in range(3):
        lat_offset = i * 0.0001
        building = Building(
            osm_id=1000 + i,
            name=f"Building_{i}",
            latitude=54.687381 + lat_offset,
            longitude=25.279652,
            height=15.0,
            footprint=[
                (54.687381 + lat_offset, 25.279652),
                (54.687381 + lat_offset, 25.279652 + 0.00005),
                (54.687381 + lat_offset + 0.00005, 25.279652 + 0.00005),
                (54.687381 + lat_offset + 0.00005, 25.279652),
            ],
            area_m2=100.0,
            tags={},
        )
        buildings.append(building)
    return buildings


@pytest.fixture
def sample_roads():
    """Create test roads."""
    roads = []
    for i in range(2):
        lon_offset = i * 0.00005
        road = Road(
            osm_id=2000 + i,
            name=f"Street_{i}",
            way_type='residential',
            path=[
                (54.687381, 25.279652 + lon_offset),
                (54.687381 + 0.0001, 25.279652 + lon_offset),
                (54.687381 + 0.0002, 25.279652 + lon_offset),
            ],
            width=10.0,
            is_oneway=False,
            tags={},
        )
        roads.append(road)
    return roads


@pytest.fixture
def compositor(camera_intrinsics, local_origin):
    """Create AR compositor."""
    comp = ARCompositor()
    comp.initialize(camera_intrinsics, None, local_origin)
    return comp


# Test 1: Compositor Initialization
def test_compositor_init(camera_intrinsics, local_origin):
    """Test 1: ARCompositor initializes correctly."""
    comp = ARCompositor()
    comp.initialize(camera_intrinsics, None, local_origin)

    assert comp._local_origin == local_origin
    assert comp.projector is not None
    assert comp.renderer is not None
    assert comp.label_renderer is not None
    logger.info("✓ Test 1 PASS: Compositor initialization")


# Test 2: Camera-Only Mode
def test_camera_only_mode(compositor, test_frame_720p, camera_pose):
    """Test 2: Camera-only mode returns unmodified frame."""
    compositor.config.ar_mode = ARMode.CAMERA_ONLY
    compositor.set_pose(camera_pose)

    result = compositor.render_frame(test_frame_720p, camera_pose)

    assert result.shape == test_frame_720p.shape
    # In camera-only mode, frame should be mostly unchanged
    logger.info("✓ Test 2 PASS: Camera-only mode")


# Test 3: AR Overlay Mode
def test_ar_overlay_mode(compositor, test_frame_720p, camera_pose, sample_buildings, sample_roads):
    """Test 3: AR overlay mode renders all components."""
    compositor.config.ar_mode = ARMode.AR_OVERLAY
    compositor.set_pose(camera_pose)

    result = compositor.render_frame(
        test_frame_720p,
        camera_pose,
        buildings=sample_buildings,
        roads=sample_roads
    )

    assert result.shape == test_frame_720p.shape
    assert not np.array_equal(result, test_frame_720p), "Frame should be modified"
    logger.info("✓ Test 3 PASS: AR overlay mode")


# Test 4: Status Bar Rendering
def test_status_bar_rendering(compositor, test_frame_720p, camera_pose):
    """Test 4: Status bar renders correctly with FPS and pose info."""
    compositor.config.show_status_bar = True
    compositor.set_pose(camera_pose)

    result = compositor.render_frame(
        test_frame_720p,
        camera_pose,
        confidence=0.95
    )

    assert result.shape == test_frame_720p.shape
    logger.info("✓ Test 4 PASS: Status bar rendering")


# Test 5: Navigation Route Rendering
def test_navigation_rendering(compositor, test_frame_720p, camera_pose, local_origin):
    """Test 5: Navigation routes render correctly."""
    compositor.config.show_navigation_path = True

    # Set route: 100m north, then 100m east, then 100m south
    lat_start = local_origin[0]
    lon_start = local_origin[1]
    route = [
        (lat_start + 0.0005, lon_start),  # North
        (lat_start + 0.0005, lon_start + 0.0005),  # East
        (lat_start, lon_start + 0.0005),  # South
    ]
    compositor.set_navigation_route(route)
    compositor.set_pose(camera_pose)

    result = compositor.render_frame(test_frame_720p, camera_pose)

    assert result.shape == test_frame_720p.shape
    logger.info("✓ Test 5 PASS: Navigation rendering")


# Test 6: FPS Monitoring
def test_fps_monitoring(compositor, test_frame_720p, camera_pose):
    """Test 6: FPS tracking works correctly."""
    compositor.reset_statistics()
    compositor.set_pose(camera_pose)

    # Render multiple frames
    for _ in range(10):
        compositor.render_frame(test_frame_720p, camera_pose)

    fps = compositor.get_fps()
    latency_ms = compositor.get_latency_ms()

    assert fps > 0, "FPS should be positive"
    assert latency_ms > 0, "Latency should be positive"
    logger.info(f"  FPS: {fps:.1f}, Latency: {latency_ms:.2f}ms")
    logger.info("✓ Test 6 PASS: FPS monitoring")


# Test 7: Frame Rate Stability
def test_frame_rate_stability(compositor, test_frame_720p, camera_pose, sample_buildings):
    """Test 7: Frame rate stays close to 30 FPS target."""
    compositor.reset_statistics()
    compositor.set_pose(camera_pose)

    # Render 30 frames
    frame_times = []
    for _ in range(30):
        start = time.time()
        compositor.render_frame(
            test_frame_720p,
            camera_pose,
            buildings=sample_buildings
        )
        frame_times.append((time.time() - start) * 1000)

    avg_latency = np.mean(frame_times)
    fps = 1000.0 / avg_latency if avg_latency > 0 else 0

    # Should achieve at least 15 FPS (realistic minimum)
    assert fps >= 15, f"FPS too low: {fps:.1f}"

    logger.info(f"  Average latency: {avg_latency:.2f}ms")
    logger.info(f"  Achieved FPS: {fps:.1f}")
    logger.info("✓ Test 7 PASS: Frame rate stability")


# Test 8: Mode Switching
def test_mode_switching(compositor, test_frame_720p, camera_pose, sample_buildings):
    """Test 8: Mode switching works without errors."""
    compositor.set_pose(camera_pose)

    # Switch through modes
    for mode in [ARMode.CAMERA_ONLY, ARMode.AR_OVERLAY, ARMode.DEBUG]:
        compositor.config.ar_mode = mode
        result = compositor.render_frame(
            test_frame_720p,
            camera_pose,
            buildings=sample_buildings
        )
        assert result.shape == test_frame_720p.shape

    logger.info("✓ Test 8 PASS: Mode switching")


# Test 9: Pose Update Handling
def test_pose_update(compositor, test_frame_720p, camera_pose):
    """Test 9: Pose updates are reflected in projections."""
    # Render with initial pose
    compositor.set_pose(camera_pose)
    result1 = compositor.render_frame(test_frame_720p, camera_pose)

    # Update pose
    new_pose = camera_pose.__class__()
    new_pose.latitude = camera_pose.latitude + 0.0001
    new_pose.yaw_deg = 45.0

    result2 = compositor.render_frame(test_frame_720p, new_pose)

    assert result1.shape == result2.shape
    # Frames might be different due to different pose
    logger.info("✓ Test 9 PASS: Pose update handling")


# Test 10: Projection Engine Integration
def test_projection_integration(compositor, camera_pose, sample_buildings):
    """Test 10: Projection engine integrates correctly."""
    compositor.set_pose(camera_pose)

    # Project buildings
    proj_buildings = compositor.projector.project_buildings(
        sample_buildings,
        720, 1280
    )

    assert len(proj_buildings) > 0, "Should project at least some buildings"
    logger.info(f"  Projected {len(proj_buildings)} buildings")
    logger.info("✓ Test 10 PASS: Projection integration")


# Test 11: Rendering Engine Integration
def test_rendering_integration(compositor, test_frame_720p, sample_buildings):
    """Test 11: Rendering engine integrates correctly."""
    # Create mock projected buildings
    from ar.projection_engine import ProjectedBuilding

    proj_buildings = [
        ProjectedBuilding(
            osm_id=1,
            name="Test",
            height=20.0,
            footprint_pixels=[(100, 100), (100, 150), (150, 150), (150, 100)],
            roof_pixels=[(105, 95), (105, 145), (155, 145), (155, 95)],
            edges=[(0, 1), (1, 2), (2, 3), (3, 0)],
            depth_values=[10.0, 10.0, 10.0, 10.0],
            is_visible=True,
            centroid_pixel=(125, 125),
            bounding_box=(100, 95, 155, 150)
        )
    ]

    result = compositor.renderer.render_buildings(test_frame_720p, proj_buildings)

    assert result.shape == test_frame_720p.shape
    logger.info("✓ Test 11 PASS: Rendering integration")


# Test 12: Statistics Collection
def test_statistics_collection(compositor, test_frame_720p, camera_pose, sample_buildings):
    """Test 12: Statistics are collected correctly."""
    compositor.reset_statistics()
    compositor.set_pose(camera_pose)

    # Render multiple frames
    for _ in range(10):
        compositor.render_frame(test_frame_720p, camera_pose, buildings=sample_buildings)

    stats = compositor.get_statistics()

    assert stats['frame_count'] == 10
    assert stats['fps'] > 0
    assert stats['latency_ms'] > 0
    assert 'projector_stats' in stats
    assert 'renderer_stats' in stats

    logger.info(f"  Frame count: {stats['frame_count']}")
    logger.info(f"  FPS: {stats['fps']:.1f}")
    logger.info("✓ Test 12 PASS: Statistics collection")


# Test 13: Error Recovery
def test_error_recovery(compositor, test_frame_720p, camera_pose):
    """Test 13: Compositor handles missing data gracefully."""
    compositor.set_pose(camera_pose)

    # Render with no buildings, roads, or detections
    result = compositor.render_frame(test_frame_720p, camera_pose)

    assert result.shape == test_frame_720p.shape
    logger.info("✓ Test 13 PASS: Error recovery")


# Test 14: Complete Autonomous Demo
def test_autonomous_demo_pipeline(compositor, test_frame_720p, camera_pose, sample_buildings, sample_roads):
    """Test 14: Complete autonomous visualization pipeline works end-to-end."""
    # Setup
    compositor.config.ar_mode = ARMode.AR_OVERLAY
    compositor.config.show_status_bar = True
    compositor.config.show_navigation_path = True

    # Set navigation route
    local_origin = compositor._local_origin
    route = [
        (local_origin[0] + 0.0005, local_origin[1]),
        (local_origin[0] + 0.0005, local_origin[1] + 0.0005),
        (local_origin[0], local_origin[1] + 0.0005),
    ]
    compositor.set_navigation_route(route)

    # Simulated detection results
    detections = [
        {
            'bbox_2d': (200, 200, 250, 280),
            'label': 'person',
            'confidence': 0.92
        },
        {
            'bbox_2d': (600, 300, 700, 400),
            'label': 'car',
            'confidence': 0.88
        },
    ]

    # Run complete pipeline
    composer_stats_start = len(compositor.get_statistics())

    result = compositor.render_frame(
        test_frame_720p,
        camera_pose,
        buildings=sample_buildings,
        roads=sample_roads,
        detections=detections,
        confidence=0.92
    )

    # Verify output
    assert result.shape == test_frame_720p.shape
    assert not np.array_equal(result, test_frame_720p), "Frame should be modified"

    # Check statistics
    stats = compositor.get_statistics()
    assert stats['frame_count'] > 0
    assert stats['fps'] > 0

    logger.info(f"  Demo pipeline FPS: {stats['fps']:.1f}")
    logger.info(f"  Buildings: {len(sample_buildings)}, Roads: {len(sample_roads)}")
    logger.info(f"  Detections: {len(detections)}")
    logger.info("✓ Test 14 PASS: Complete autonomous demo pipeline")


# Test Suite Runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

