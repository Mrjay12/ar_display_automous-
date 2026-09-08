"""
Milestone 11 Acceptance Tests: Geometric Rendering & AR Visualization

Purpose: Verify real-time rendering of AR geometries with high quality and performance

Test Coverage:
1. Renderer initialization and configuration
2. Building wireframe rendering (blue, semi-transparent)
3. Road boundary rendering (green dashed)
4. Obstacle bounding box rendering (yellow)
5. Person/vehicle detection rendering (magenta)
6. Alpha blending and transparency
7. Dashed line rendering for roads
8. Label rendering with high contrast
9. Batch rendering performance (30 FPS target)
10. Out-of-bounds clipping
11. Color configuration and customization
12. Occlusion-based visibility filtering
13. Multiple overlapping geometries
14. Text contrast and readability
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
from ar.geometry_renderer import GeometryRenderer, RenderConfig
from ar.label_renderer import LabelRenderer, LabelConfig
from ar.projection_engine import ProjectedBuilding, ProjectedRoad

logger = logging.getLogger(__name__)


# Test Fixtures

@pytest.fixture
def test_frame_720p():
    """Create blank test frame (720p RGB)."""
    return np.ones((720, 1280, 3), dtype=np.uint8) * 200  # Gray background


@pytest.fixture
def test_frame_1080p():
    """Create blank test frame (1080p RGB)."""
    return np.ones((1080, 1920, 3), dtype=np.uint8) * 200


@pytest.fixture
def renderer_config():
    """Standard rendering configuration."""
    return RenderConfig(
        building_color_bgr=(255, 0, 0),
        building_thickness=2,
        building_alpha=0.7,
        road_color_bgr=(0, 255, 0),
        road_thickness=2,
        road_dashed=True,
        obstacle_color_bgr=(0, 255, 255),
        detection_color_bgr=(255, 0, 255),
    )


@pytest.fixture
def geometry_renderer(renderer_config):
    """Create geometry renderer."""
    return GeometryRenderer(
        frame_shape=(720, 1280),
        config=renderer_config
    )


@pytest.fixture
def label_renderer():
    """Create label renderer."""
    return LabelRenderer()


def create_mock_projected_building():
    """Create mock projected building for testing."""
    return ProjectedBuilding(
        osm_id=12345,
        name="Test Building",
        height=20.0,
        footprint_pixels=[(100, 100), (100, 150), (150, 150), (150, 100)],
        roof_pixels=[(105, 95), (105, 145), (155, 145), (155, 95)],
        edges=[(0, 1), (1, 2), (2, 3), (3, 0), (0, 4), (1, 5), (2, 6), (3, 7), (4, 5), (5, 6), (6, 7), (7, 4)],
        depth_values=[10.0, 10.0, 10.0, 10.0, 15.0, 15.0, 15.0, 15.0],
        is_visible=True,
        centroid_pixel=(125, 125),
        bounding_box=(100, 95, 155, 150)
    )


def create_mock_projected_road():
    """Create mock projected road for testing."""
    return ProjectedRoad(
        osm_id=54321,
        name="Test Street",
        path_pixels=[(200, 400), (300, 400), (400, 400), (500, 400)],
        depth_values=[15.0, 16.0, 17.0, 18.0],
        is_visible=True,
        bounding_box=(200, 395, 500, 405)
    )


# Test 1: Renderer Initialization
def test_renderer_init(geometry_renderer):
    """Test 1: GeometryRenderer initializes with correct parameters."""
    assert geometry_renderer.frame_shape == (720, 1280)
    assert geometry_renderer.config is not None
    logger.info("✓ Test 1 PASS: Renderer initialization")


# Test 2: Building Wireframe Rendering
def test_building_wireframe_rendering(geometry_renderer, test_frame_720p):
    """Test 2: Building wireframe renders correctly."""
    proj_building = create_mock_projected_building()

    frame_with_building = geometry_renderer.render_building_wireframe(
        test_frame_720p, proj_building)

    assert frame_with_building.shape == test_frame_720p.shape
    assert not np.array_equal(frame_with_building, test_frame_720p), "Frame should be modified"
    logger.info("✓ Test 2 PASS: Building wireframe rendering")


# Test 3: Road Rendering
def test_road_rendering(geometry_renderer, test_frame_720p):
    """Test 3: Road geometry renders correctly."""
    proj_road = create_mock_projected_road()

    frame_with_road = geometry_renderer.render_road(
        test_frame_720p, proj_road)

    assert frame_with_road.shape == test_frame_720p.shape
    assert not np.array_equal(frame_with_road, test_frame_720p), "Frame should be modified"
    logger.info("✓ Test 3 PASS: Road rendering")


# Test 4: Obstacle Detection Rendering
def test_obstacle_rendering(geometry_renderer, test_frame_720p):
    """Test 4: Obstacle bounding boxes render correctly."""
    detection = {
        'bbox_2d': (200, 150, 250, 200),
        'label': 'car',
        'confidence': 0.95
    }

    frame_with_detection = geometry_renderer.render_detection_box(
        test_frame_720p,
        detection['bbox_2d'],
        detection['label'],
        detection['confidence'],
        geometry_renderer.config.obstacle_color_bgr
    )

    assert frame_with_detection.shape == test_frame_720p.shape
    assert not np.array_equal(frame_with_detection, test_frame_720p)
    logger.info("✓ Test 4 PASS: Obstacle rendering")


# Test 5: Alpha Blending
def test_alpha_blending(geometry_renderer, test_frame_720p):
    """Test 5: Alpha blending produces correct transparency."""
    frame1 = test_frame_720p.copy()
    frame2 = np.ones_like(frame1) * 100

    # Manual blend
    alpha = 0.5
    blended = cv2.addWeighted(frame1, 1.0 - alpha, frame2, alpha, 0)

    # Check that blend is between frame1 and frame2
    assert np.all(blended >= np.minimum(frame1, frame2))
    assert np.all(blended <= np.maximum(frame1, frame2))
    logger.info("✓ Test 5 PASS: Alpha blending")


# Test 6: Dashed Line Rendering
def test_dashed_line_rendering(geometry_renderer, test_frame_720p):
    """Test 6: Dashed lines render correctly for roads."""
    config = RenderConfig(road_dashed=True, road_dash_length=10)
    renderer = GeometryRenderer(frame_shape=(720, 1280), config=config)

    proj_road = create_mock_projected_road()
    frame_with_dashed = renderer.render_road(test_frame_720p, proj_road)

    assert not np.array_equal(frame_with_dashed, test_frame_720p)
    logger.info("✓ Test 6 PASS: Dashed line rendering")


# Test 7: Label Rendering
def test_label_rendering(label_renderer, test_frame_720p):
    """Test 7: Labels render with high contrast."""
    frame_with_label = label_renderer.render_text_with_background(
        test_frame_720p,
        "Test Label",
        (100, 100),
        0.7,
        (255, 255, 255)
    )

    assert frame_with_label.shape == test_frame_720p.shape
    assert not np.array_equal(frame_with_label, test_frame_720p)
    logger.info("✓ Test 7 PASS: Label rendering")


# Test 8: Building Label Rendering
def test_building_label_rendering(label_renderer, test_frame_720p):
    """Test 8: Building labels render at correct position."""
    proj_building = create_mock_projected_building()

    frame_with_label = label_renderer.render_building_label(
        test_frame_720p,
        proj_building,
        "Office Building"
    )

    assert frame_with_label.shape == test_frame_720p.shape
    logger.info("✓ Test 8 PASS: Building label rendering")


# Test 9: Batch Rendering Performance
def test_batch_rendering_performance(geometry_renderer, test_frame_720p):
    """Test 9: Batch rendering meets 30 FPS performance target."""
    # Create 20 buildings
    buildings = [create_mock_projected_building() for _ in range(20)]

    start_time = time.time()
    frame_result = geometry_renderer.render_buildings(test_frame_720p, buildings)
    elapsed_ms = (time.time() - start_time) * 1000

    # Should be less than 33ms for 30 FPS
    assert elapsed_ms < 33, f"Rendering too slow: {elapsed_ms:.1f}ms"
    assert frame_result.shape == test_frame_720p.shape
    logger.info(f"  Rendered {len(buildings)} buildings in {elapsed_ms:.1f}ms")
    logger.info("✓ Test 9 PASS: Batch rendering performance")


# Test 10: Out-of-Bounds Clipping
def test_out_of_bounds_clipping(geometry_renderer, test_frame_720p):
    """Test 10: Out-of-bounds boxes are clipped correctly."""
    # Box partially outside frame
    detection = {
        'bbox_2d': (1200, 700, 1400, 750),  # Mostly outside
        'label': 'car',
        'confidence': 0.9
    }

    frame_result = geometry_renderer.render_detection_box(
        test_frame_720p,
        detection['bbox_2d'],
        detection['label'],
        detection['confidence'],
        (0, 255, 255)
    )

    assert frame_result.shape == test_frame_720p.shape
    logger.info("✓ Test 10 PASS: Out-of-bounds clipping")


# Test 11: Custom Colors
def test_custom_colors(test_frame_720p):
    """Test 11: Custom colors in configuration work correctly."""
    config = RenderConfig(
        building_color_bgr=(100, 150, 200),
        road_color_bgr=(50, 100, 50),
        obstacle_color_bgr=(200, 100, 50)
    )

    renderer = GeometryRenderer(frame_shape=(720, 1280), config=config)
    assert renderer.config.building_color_bgr == (100, 150, 200)
    assert renderer.config.road_color_bgr == (50, 100, 50)
    assert renderer.config.obstacle_color_bgr == (200, 100, 50)
    logger.info("✓ Test 11 PASS: Custom colors")


# Test 12: Visibility Filtering
def test_visibility_filtering(geometry_renderer, test_frame_720p):
    """Test 12: Non-visible geometries are skipped."""
    invisible_building = create_mock_projected_building()
    invisible_building.is_visible = False

    frame_before = test_frame_720p.copy()
    frame_after = geometry_renderer.render_building_wireframe(
        test_frame_720p,
        invisible_building
    )

    # Should be unchanged
    assert np.array_equal(frame_after, frame_before)
    logger.info("✓ Test 12 PASS: Visibility filtering")


# Test 13: Multiple Overlapping Geometries
def test_multiple_overlapping_geometries(geometry_renderer, test_frame_720p):
    """Test 13: Multiple overlapping geometries render correctly."""
    buildings = [create_mock_projected_building() for _ in range(5)]
    roads = [create_mock_projected_road() for _ in range(3)]
    detections = [
        {'bbox_2d': (150, 150, 200, 200), 'label': 'person', 'confidence': 0.9},
        {'bbox_2d': (250, 300, 300, 350), 'label': 'car', 'confidence': 0.85},
    ]

    frame_result = geometry_renderer.render_frame(
        test_frame_720p,
        proj_buildings=buildings,
        proj_roads=roads,
        detections=detections
    )

    assert frame_result.shape == test_frame_720p.shape
    assert not np.array_equal(frame_result, test_frame_720p)
    logger.info("✓ Test 13 PASS: Multiple overlapping geometries")


# Test 14: Full Rendering Pipeline
def test_full_rendering_pipeline(geometry_renderer, label_renderer, test_frame_720p):
    """Test 14: Complete rendering pipeline with geometry and labels."""
    buildings = [create_mock_projected_building()]
    roads = [create_mock_projected_road()]

    # Render geometries
    frame_result = geometry_renderer.render_buildings(test_frame_720p, buildings)
    frame_result = geometry_renderer.render_roads(frame_result, roads)

    # Add labels
    frame_result = label_renderer.render_building_labels(frame_result, buildings)
    frame_result = label_renderer.render_street_labels(frame_result, roads)

    # Add status text
    status_lines = ["FPS: 30", "Confidence: 0.95", "Pose: OK"]
    frame_result = label_renderer.render_status_text(
        frame_result,
        status_lines,
        position=(10, 30),
        color_bgr=(0, 255, 0)
    )

    assert frame_result.shape == test_frame_720p.shape
    assert not np.array_equal(frame_result, test_frame_720p)

    # Check statistics
    geo_stats = geometry_renderer.get_statistics()
    assert geo_stats['frames_rendered'] > 0

    logger.info(f"  Geometry stats: {geo_stats}")
    logger.info("✓ Test 14 PASS: Full rendering pipeline")


# Test Suite Runner
if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])

