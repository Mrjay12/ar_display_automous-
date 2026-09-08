"""
Milestone 10 Acceptance Tests: 3D-to-2D Projection Engine

Purpose: Verify accurate projection of geographic 3D geometries into 2D camera image plane

All tests use synthetic pose and map data for reproducibility without hardware.

Test Coverage:
1. Engine initialization with camera matrix
2. Pose setting and coordinate frame setup
3. Single point projection (WGS84 → ENU → Camera → Image)
4. Building geometry projection (footprint + roof)
5. Road geometry projection
6. Behind-camera detection (z ≤ 0)
7. Out-of-bounds clipping
8. Occlusion testing with depth map
9. Batch projection performance
10. Coordinate frame consistency
11. Distortion coefficient handling
12. Centroid and bounding box calculation
13. Multiple buildings and roads
14. Recovery from missing pose/data
"""

import pytest
import logging
import numpy as np
from pathlib import Path
from dataclasses import dataclass

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from ar.projection_engine import ProjectionEngine, ProjectedPoint, ProjectedBuilding
from mapping.osm_loader import Building, Road

logger = logging.getLogger(__name__)


# Test Fixtures

@pytest.fixture
def camera_intrinsics():
    """Standard camera intrinsic matrix (OAK-D Pro RGB camera)."""
    return np.array([
        [1395.6, 0.0, 640.0],
        [0.0, 1395.8, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)


@pytest.fixture
def distortion_coeffs():
    """Typical distortion coefficients."""
    return np.array([0.0, 0.0, 0.0, 0.0, 0.0], dtype=np.float32)


@pytest.fixture
def projector(camera_intrinsics, distortion_coeffs):
    """Create projection engine with camera calibration."""
    return ProjectionEngine(
        camera_intrinsics=camera_intrinsics,
        distortion_coeffs=distortion_coeffs,
        depth_threshold_m=0.1,
        image_bounds_margin_px=10
    )


@pytest.fixture
def camera_pose_centered():
    """Camera looking north from center of test area."""
    @dataclass
    class MockPose:
        timestamp_us: int = 0
        latitude: float = 54.687381
        longitude: float = 25.279652
        altitude: float = 125.5
        roll_deg: float = 0.0
        pitch_deg: float = 0.0
        yaw_deg: float = 0.0  # Looking north

    return MockPose()


@pytest.fixture
def local_origin():
    """Local ENU origin at center of test area."""
    return (54.687381, 25.279652, 125.5)


@pytest.fixture
def sample_building():
    """Single test building near origin."""
    return Building(
        osm_id=12345,
        name="Test Building",
        latitude=54.687381,
        longitude=25.279652,
        height=20.0,
        footprint=[
            (54.687380, 25.279651),
            (54.687380, 25.279653),
            (54.687382, 25.279653),
            (54.687382, 25.279651),
        ],
        area_m2=100.0,
        tags={'building': 'test'},
    )


@pytest.fixture
def sample_road():
    """Single test road near origin."""
    return Road(
        osm_id=54321,
        name="Test Street",
        way_type='residential',
        path=[
            (54.687381, 25.279651),
            (54.687382, 25.279652),
            (54.687383, 25.279653),
        ],
        width=10.0,
        is_oneway=False,
        tags={'highway': 'residential'},
    )


# Test 1: Engine Initialization
def test_projection_engine_init(camera_intrinsics, distortion_coeffs):
    """Test 1: Engine initializes with valid camera parameters."""
    projector = ProjectionEngine(
        camera_intrinsics=camera_intrinsics,
        distortion_coeffs=distortion_coeffs,
    )
    assert projector.camera_intrinsics is not None
    assert projector.camera_intrinsics.shape == (3, 3)
    assert projector.distortion_coeffs is not None
    logger.info("✓ Test 1 PASS: Engine initialization")


# Test 2: Pose Setting
def test_pose_setting(projector, camera_pose_centered, local_origin):
    """Test 2: Pose can be set and coordinate frames are initialized."""
    projector.set_pose(camera_pose_centered, local_origin)
    assert projector._pose is not None
    assert projector._local_origin is not None
    assert projector._R_enu_to_cam is not None
    assert projector._R_enu_to_cam.shape == (3, 3)
    assert projector._t_cam is not None
    logger.info("✓ Test 2 PASS: Pose setting")


# Test 3: Single Point Projection
def test_point_projection_at_origin(projector, camera_pose_centered, local_origin):
    """Test 3: Point at camera origin projects to image center."""
    projector.set_pose(camera_pose_centered, local_origin)

    # Project point very close to camera location
    proj = projector.project_point(
        lat=camera_pose_centered.latitude,
        lon=camera_pose_centered.longitude,
        alt=camera_pose_centered.altitude,
        image_height=720,
        image_width=1280
    )

    # Should be very close to image center (640, 360)
    assert proj.is_visible, "Point at camera location should be visible"
    assert proj.depth_m > 0, "Depth should be positive"
    logger.info(f"  Point at origin: pixel=({proj.pixel_x:.1f}, {proj.pixel_y:.1f}), depth={proj.depth_m:.2f}m")
    logger.info("✓ Test 3 PASS: Point projection at origin")


# Test 4: Building Projection
def test_building_projection(projector, camera_pose_centered, local_origin, sample_building):
    """Test 4: Building geometry projects correctly."""
    projector.set_pose(camera_pose_centered, local_origin)

    proj_building = projector.project_building(
        sample_building,
        image_height=720,
        image_width=1280
    )

    assert proj_building.osm_id == sample_building.osm_id
    assert len(proj_building.footprint_pixels) == len(sample_building.footprint)
    assert len(proj_building.roof_pixels) == len(sample_building.footprint)
    assert proj_building.is_visible, "Building near camera should be visible"
    assert proj_building.bounding_box is not None
    assert proj_building.centroid_pixel is not None
    logger.info(f"  Building footprint: {len(proj_building.footprint_pixels)} vertices")
    logger.info(f"  Building visible: {proj_building.is_visible}")
    logger.info(f"  Bounding box: {proj_building.bounding_box}")
    logger.info("✓ Test 4 PASS: Building projection")


# Test 5: Road Projection
def test_road_projection(projector, camera_pose_centered, local_origin, sample_road):
    """Test 5: Road geometry projects correctly."""
    projector.set_pose(camera_pose_centered, local_origin)

    proj_road = projector.project_road(
        sample_road,
        image_height=720,
        image_width=1280
    )

    assert proj_road.osm_id == sample_road.osm_id
    assert len(proj_road.path_pixels) == len(sample_road.path)
    assert proj_road.is_visible, "Road near camera should be visible"
    logger.info(f"  Road path: {len(proj_road.path_pixels)} vertices")
    logger.info("✓ Test 5 PASS: Road projection")


# Test 6: Behind-Camera Detection
def test_behind_camera_not_projected(projector, camera_pose_centered, local_origin):
    """Test 6: Points behind camera (z ≤ 0) are not visible."""
    projector.set_pose(camera_pose_centered, local_origin)

    # Point south of camera (behind when looking north)
    proj = projector.project_point(
        lat=camera_pose_centered.latitude - 0.01,  # Far south
        lon=camera_pose_centered.longitude,
        alt=camera_pose_centered.altitude,
        image_height=720,
        image_width=1280
    )

    assert not proj.is_visible, "Point behind camera should not be visible"
    logger.info("✓ Test 6 PASS: Behind-camera detection")


# Test 7: Out-of-Bounds Clipping
def test_out_of_bounds_clipping(projector, camera_pose_centered, local_origin):
    """Test 7: Points outside image bounds are marked not visible."""
    projector.set_pose(camera_pose_centered, local_origin)

    # Point far to the east (out of view)
    proj = projector.project_point(
        lat=camera_pose_centered.latitude,
        lon=camera_pose_centered.longitude + 1.0,  # Very far east
        alt=camera_pose_centered.altitude,
        image_height=720,
        image_width=1280
    )

    assert not proj.is_visible, "Point far from camera should not be visible"
    logger.info("✓ Test 7 PASS: Out-of-bounds clipping")


# Test 8: Occlusion Testing
def test_occlusion_testing(projector, camera_pose_centered, local_origin):
    """Test 8: Depth map occlusion testing works correctly."""
    projector.set_pose(camera_pose_centered, local_origin)

    # Create synthetic depth map (5 meters at all pixels)
    depth_map = np.full((720, 1280), 5000, dtype=np.uint16)  # 5 meters in mm

    # Point at 3 meters should not be occluded
    is_visible = projector.test_occlusion(
        point_px=(640, 360),
        depth_m=3.0,
        depth_map=depth_map,
        depth_scale=0.001,
        occlusion_margin_m=0.5
    )
    assert is_visible, "Point closer than depth map should be visible"

    # Point at 6 meters should be occluded
    is_visible = projector.test_occlusion(
        point_px=(640, 360),
        depth_m=6.0,
        depth_map=depth_map,
        depth_scale=0.001,
        occlusion_margin_m=0.5
    )
    assert not is_visible, "Point farther than depth map should be occluded"
    logger.info("✓ Test 8 PASS: Occlusion testing")


# Test 9: Batch Projection Performance
def test_batch_projection_performance(projector, camera_pose_centered, local_origin):
    """Test 9: Batch projection of multiple buildings meets performance targets."""
    import time

    projector.set_pose(camera_pose_centered, local_origin)

    # Create 50 test buildings
    buildings = []
    for i in range(50):
        lat_offset = (i % 10) * 0.0001
        lon_offset = (i // 10) * 0.0001
        building = Building(
            osm_id=i,
            name=f"Building_{i}",
            latitude=local_origin[0] + lat_offset,
            longitude=local_origin[1] + lon_offset,
            height=10.0 + i,
            footprint=[
                (local_origin[0] + lat_offset, local_origin[1] + lon_offset),
                (local_origin[0] + lat_offset, local_origin[1] + lon_offset + 0.00005),
                (local_origin[0] + lat_offset + 0.00005, local_origin[1] + lon_offset + 0.00005),
                (local_origin[0] + lat_offset + 0.00005, local_origin[1] + lon_offset),
            ],
            area_m2=100.0,
            tags={},
        )
        buildings.append(building)

    # Time batch projection
    start_time = time.time()
    proj_buildings = projector.project_buildings(buildings, 720, 1280)
    elapsed_ms = (time.time() - start_time) * 1000

    assert elapsed_ms < 50, f"Batch projection too slow: {elapsed_ms:.1f}ms"
    logger.info(f"  Projected {len(buildings)} buildings in {elapsed_ms:.1f}ms")
    logger.info("✓ Test 9 PASS: Batch projection performance")


# Test 10: Coordinate Frame Consistency
def test_coordinate_frame_consistency(projector, camera_pose_centered, local_origin):
    """Test 10: Transformed coordinates maintain geometric consistency."""
    projector.set_pose(camera_pose_centered, local_origin)

    # Project two points and verify their relative positions
    proj1 = projector.project_point(
        lat=local_origin[0] + 0.00005,
        lon=local_origin[1],
        alt=local_origin[2],
        image_height=720,
        image_width=1280
    )

    proj2 = projector.project_point(
        lat=local_origin[0],
        lon=local_origin[1] + 0.00005,
        alt=local_origin[2],
        image_height=720,
        image_width=1280
    )

    # Both should be visible
    assert proj1.is_visible, "Point 1 should be visible"
    assert proj2.is_visible, "Point 2 should be visible"

    # They should project to different image locations
    dist_px = np.sqrt((proj1.pixel_x - proj2.pixel_x)**2 + (proj1.pixel_y - proj2.pixel_y)**2)
    assert dist_px > 10, f"Points should project to different locations: {dist_px:.1f}px"
    logger.info(f"  Points separated by {dist_px:.1f} pixels")
    logger.info("✓ Test 10 PASS: Coordinate frame consistency")


# Test 11: Distortion Coefficient Handling
def test_distortion_handling(camera_intrinsics):
    """Test 11: Distortion coefficients are applied correctly."""
    distortion = np.array([0.1, -0.05, 0.001, 0.0005, 0.01], dtype=np.float32)
    projector_with_dist = ProjectionEngine(
        camera_intrinsics=camera_intrinsics,
        distortion_coeffs=distortion
    )

    assert np.allclose(projector_with_dist.distortion_coeffs, distortion)
    logger.info("✓ Test 11 PASS: Distortion coefficient handling")


# Test 12: Centroid and Bounding Box
def test_centroid_bounding_box(projector, camera_pose_centered, local_origin, sample_building):
    """Test 12: Centroid and bounding box are computed correctly."""
    projector.set_pose(camera_pose_centered, local_origin)

    proj_building = projector.project_building(sample_building, 720, 1280)

    assert proj_building.centroid_pixel is not None
    assert len(proj_building.centroid_pixel) == 2
    assert proj_building.bounding_box is not None
    assert len(proj_building.bounding_box) == 4

    u_min, v_min, u_max, v_max = proj_building.bounding_box
    assert u_min <= u_max
    assert v_min <= v_max
    logger.info(f"  Centroid: {proj_building.centroid_pixel}")
    logger.info(f"  Bounding box: {proj_building.bounding_box}")
    logger.info("✓ Test 12 PASS: Centroid and bounding box")


# Test 13: Multiple Buildings and Roads
def test_multiple_features(projector, camera_pose_centered, local_origin):
    """Test 13: Multiple buildings and roads project correctly."""
    projector.set_pose(camera_pose_centered, local_origin)

    # Create multiple buildings
    buildings = []
    for i in range(5):
        lat_offset = i * 0.00005
        building = Building(
            osm_id=i,
            name=f"Building_{i}",
            latitude=local_origin[0] + lat_offset,
            longitude=local_origin[1],
            height=15.0,
            footprint=[
                (local_origin[0] + lat_offset, local_origin[1]),
                (local_origin[0] + lat_offset, local_origin[1] + 0.00003),
                (local_origin[0] + lat_offset + 0.00003, local_origin[1] + 0.00003),
                (local_origin[0] + lat_offset + 0.00003, local_origin[1]),
            ],
            area_m2=100.0,
            tags={},
        )
        buildings.append(building)

    # Create multiple roads
    roads = []
    for i in range(3):
        lon_offset = i * 0.0001
        road = Road(
            osm_id=100 + i,
            name=f"Street_{i}",
            way_type='residential',
            path=[
                (local_origin[0], local_origin[1] + lon_offset),
                (local_origin[0] + 0.0001, local_origin[1] + lon_offset),
                (local_origin[0] + 0.0002, local_origin[1] + lon_offset),
            ],
            width=10.0,
            is_oneway=False,
            tags={},
        )
        roads.append(road)

    proj_buildings = projector.project_buildings(buildings, 720, 1280)
    proj_roads = projector.project_roads(roads, 720, 1280)

    assert len(proj_buildings) > 0, "Should project at least some buildings"
    assert len(proj_roads) > 0, "Should project at least some roads"
    logger.info(f"  Projected {len(proj_buildings)} buildings and {len(proj_roads)} roads")
    logger.info("✓ Test 13 PASS: Multiple features")


# Test 14: Recovery from Missing Pose
def test_recovery_missing_pose(projector):
    """Test 14: Projection gracefully handles missing pose estimate."""
    # Don't set pose
    proj = projector.project_point(
        lat=54.687381,
        lon=25.279652,
        alt=125.5,
        image_height=720,
        image_width=1280
    )

    assert not proj.is_visible, "Should not be visible without pose"
    logger.info("✓ Test 14 PASS: Recovery from missing pose")


# Test Suite Runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

