"""
AR Compositor Module - Integrate all AR components into unified visualization pipeline

Responsibilities:
- Orchestrate complete AR pipeline: projection → rendering → labels
- Blend AR layer with live camera feed
- Render real-time status bar (FPS, confidence, pose info)
- Handle navigation path rendering and guidance
- Autonomous navigation demo mode
- Frame rate management and performance monitoring
- Mode switching (live camera, AR overlay, debug visualization)

Input:
  - Live RGB frame from camera
  - Camera pose estimate (position + orientation)
  - Map data (buildings, roads)
  - Detection results (obstacles, people, vehicles)
  - Depth map (optional, for occlusion)
  - Localization confidence
  - Navigation route (optional)

Output:
  - Composite RGB frame with AR overlay
  - Performance metrics (FPS, latency breakdown)
  - Navigation guidance (if applicable)

Performance:
  - Full pipeline: <33 ms per frame (30 FPS target)
  - Latency breakdown:
    - Projection: 5-10 ms
    - Rendering: 10-15 ms
    - Labels: 2-5 ms
    - Compositing: <5 ms

Modes:
  1. Camera Only - Raw camera feed
  2. AR Overlay - Camera + map geometries + detections
  3. Map View - Top-down map visualization
  4. Navigation - AR + route visualization + guidance

Failure Modes:
  - Missing pose: Show camera only or previous pose
  - No map data: Show camera + detections only
  - Low FPS: Drop secondary features (labels, confidence map)
  - Camera failure: Show error state

Example:
    >>> compositor = ARCompositor()
    >>> compositor.initialize(camera_calib, local_origin)
    >>> while True:
    ...     rgb_frame = camera.get_frame()
    ...     pose = localization.get_pose()
    ...     buildings = map_mgr.get_buildings()
    ...     output = compositor.render_frame(rgb_frame, pose, buildings, detections)
    ...     display(output)

Reference:
- Complete system architecture (ARCHITECTURE.md)
- Coordinate frame transformations (COORDINATE_FRAMES.md)
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class ARMode:
    """AR display modes."""
    CAMERA_ONLY = "camera_only"
    AR_OVERLAY = "ar_overlay"
    MAP_VIEW = "map_view"
    NAVIGATION = "navigation"
    DEBUG = "debug"


@dataclass
class CompositorConfig:
    """Configuration for AR compositor."""
    # Display modes
    ar_mode: str = ARMode.AR_OVERLAY

    # Performance
    target_fps: int = 30
    performance_budget_ms: float = 33.0

    # Status bar
    show_status_bar: bool = True
    status_bar_position: str = "top_left"  # top_left, top_right, bottom_left, bottom_right

    # Guidance
    show_navigation_path: bool = False
    show_confidence_map: bool = True

    # Colors
    status_text_color_bgr: Tuple[int, int, int] = (0, 255, 0)  # Green
    confidence_low_color: Tuple[int, int, int] = (0, 0, 255)  # Red
    confidence_high_color: Tuple[int, int, int] = (0, 255, 0)  # Green


class ARCompositor:
    """
    Integrate complete AR visualization pipeline.

    Orchestrates projection, rendering, and labeling into
    a unified real-time AR display with status information
    and navigation guidance.
    """

    def __init__(self, config: Optional[CompositorConfig] = None):
        """
        Initialize AR compositor.

        Args:
            config: CompositorConfig with display settings
        """
        self.config = config if config is not None else CompositorConfig()

        # Import components
        from .projection_engine import ProjectionEngine
        from .geometry_renderer import GeometryRenderer, RenderConfig
        from .label_renderer import LabelRenderer, LabelConfig

        self.projector = ProjectionEngine()
        self.renderer = GeometryRenderer(frame_shape=(720, 1280))
        self.label_renderer = LabelRenderer()

        # State
        self._current_pose = None
        self._local_origin = None
        self._frame_count = 0
        self._start_time = time.time()
        self._frame_times = []  # Rolling window of frame times
        self._max_history = 30  # Keep last 30 frames for FPS calculation

        # Navigation
        self._route_points = []  # Route waypoints in WGS84
        self._route_index = 0

        logger.info("ARCompositor initialized")

    def initialize(
        self,
        camera_intrinsics: np.ndarray,
        camera_distortion: Optional[np.ndarray],
        local_origin: Tuple[float, float, float],
    ):
        """
        Initialize compositor with camera parameters.

        Args:
            camera_intrinsics: 3x3 camera matrix K
            camera_distortion: Distortion coefficients
            local_origin: (latitude, longitude, altitude) reference point
        """
        self.projector = self._create_projector(camera_intrinsics, camera_distortion)
        self._local_origin = local_origin
        logger.info(f"Compositor initialized: local_origin={local_origin}")

    def _create_projector(self, intrinsics, distortion):
        """Create projection engine with camera parameters."""
        from .projection_engine import ProjectionEngine
        return ProjectionEngine(
            camera_intrinsics=intrinsics,
            distortion_coeffs=distortion
        )

    def set_pose(self, camera_pose):
        """Set current camera pose for projection."""
        self._current_pose = camera_pose
        if self._local_origin is not None:
            self.projector.set_pose(camera_pose, self._local_origin)

    def set_navigation_route(self, waypoints: List[Tuple[float, float]]):
        """
        Set navigation route as waypoints.

        Args:
            waypoints: List of (latitude, longitude) tuples
        """
        self._route_points = waypoints
        self._route_index = 0
        logger.info(f"Navigation route set: {len(waypoints)} waypoints")

    def render_frame(
        self,
        rgb_frame: np.ndarray,
        camera_pose,
        buildings: Optional[List] = None,
        roads: Optional[List] = None,
        detections: Optional[List[Dict]] = None,
        depth_map: Optional[np.ndarray] = None,
        confidence: Optional[float] = None,
    ) -> np.ndarray:
        """
        Render complete AR frame.

        Orchestrates full pipeline:
        1. Set pose for projection
        2. Project map geometries
        3. Render geometries to frame
        4. Add labels
        5. Add status bar
        6. Add navigation guidance

        Args:
            rgb_frame: Input RGB frame
            camera_pose: Current camera pose estimate
            buildings: List of buildings to render
            roads: List of roads to render
            detections: List of detection boxes
            depth_map: Optional depth for occlusion
            confidence: Localization confidence (0-1)

        Returns:
            Composite AR frame
        """
        frame_start_time = time.time()

        # Initialize if needed
        if self._local_origin is None and buildings:
            # Use first building location as origin
            self._local_origin = (buildings[0].latitude, buildings[0].longitude, 125.0)

        # Set pose
        self.set_pose(camera_pose)

        # Project geometries
        proj_buildings = []
        proj_roads = []

        if buildings:
            proj_buildings = self.projector.project_buildings(
                buildings,
                rgb_frame.shape[0],
                rgb_frame.shape[1]
            )

        if roads:
            proj_roads = self.projector.project_roads(
                roads,
                rgb_frame.shape[0],
                rgb_frame.shape[1]
            )

        # Render based on mode
        if self.config.ar_mode == ARMode.CAMERA_ONLY:
            result = rgb_frame.copy()
        elif self.config.ar_mode == ARMode.AR_OVERLAY:
            result = self.renderer.render_frame(
                rgb_frame,
                proj_buildings=proj_buildings,
                proj_roads=proj_roads,
                detections=detections,
                depth_map=depth_map
            )
            result = self.label_renderer.render_building_labels(result, proj_buildings)
            result = self.label_renderer.render_street_labels(result, proj_roads)
        elif self.config.ar_mode == ARMode.DEBUG:
            result = rgb_frame.copy()
            result = self.renderer.render_frame(result, proj_buildings, proj_roads, detections, depth_map)
        else:
            result = rgb_frame.copy()

        # Add navigation if enabled
        if self.config.show_navigation_path and self._route_points:
            result = self._render_navigation(result)

        # Add status bar
        if self.config.show_status_bar:
            result = self._render_status_bar(
                result,
                camera_pose,
                confidence,
                proj_buildings,
                proj_roads
            )

        # Record frame time for FPS calculation
        frame_time = time.time() - frame_start_time
        self._frame_times.append(frame_time)
        if len(self._frame_times) > self._max_history:
            self._frame_times.pop(0)

        self._frame_count += 1

        return result

    def _render_status_bar(
        self,
        frame: np.ndarray,
        camera_pose,
        confidence: Optional[float],
        proj_buildings,
        proj_roads,
    ) -> np.ndarray:
        """
        Render status bar with system information.

        Args:
            frame: Input frame
            camera_pose: Current pose
            confidence: Localization confidence
            proj_buildings: Projected buildings
            proj_roads: Projected roads

        Returns:
            Frame with status bar
        """
        # Calculate FPS
        fps = self.get_fps()

        # Build status lines
        status_lines = [
            f"FPS: {fps:.1f}",
            f"Lat: {camera_pose.latitude:.6f}",
            f"Lon: {camera_pose.longitude:.6f}",
            f"Alt: {camera_pose.altitude:.1f}m",
        ]

        if confidence is not None:
            status_lines.append(f"Conf: {confidence:.2f}")

        status_lines.extend([
            f"Buildings: {len(proj_buildings)}",
            f"Roads: {len(proj_roads)}",
        ])

        # Determine position
        if self.config.status_bar_position == "top_left":
            position = (10, 30)
        elif self.config.status_bar_position == "top_right":
            position = (frame.shape[1] - 300, 30)
        elif self.config.status_bar_position == "bottom_left":
            position = (10, frame.shape[0] - 200)
        else:  # bottom_right
            position = (frame.shape[1] - 300, frame.shape[0] - 200)

        # Render status text
        result = self.label_renderer.render_status_text(
            frame,
            status_lines,
            position=position,
            color_bgr=self.config.status_text_color_bgr
        )

        return result

    def _render_navigation(self, frame: np.ndarray) -> np.ndarray:
        """
        Render navigation path and guidance.

        Args:
            frame: Input frame

        Returns:
            Frame with navigation overlay
        """
        if not self._route_points or self._current_pose is None:
            return frame

        result = frame.copy()

        # Project route waypoints
        route_pixels = []
        for lat, lon in self._route_points:
            proj = self.projector.project_point(
                lat, lon,
                self._local_origin[2] if self._local_origin else 125.0,
                frame.shape[0],
                frame.shape[1]
            )
            if proj.is_visible:
                route_pixels.append((int(proj.pixel_x), int(proj.pixel_y)))

        # Draw route as white line
        if len(route_pixels) > 1:
            for i in range(len(route_pixels) - 1):
                cv2.line(
                    result,
                    route_pixels[i],
                    route_pixels[i + 1],
                    (255, 255, 255),  # White
                    3,
                    lineType=cv2.LINE_AA
                )

        # Draw next waypoint as circle
        if self._route_index < len(route_pixels):
            next_wp = route_pixels[self._route_index]
            cv2.circle(result, next_wp, 10, (0, 255, 0), 2)  # Green circle

        return result

    def get_fps(self) -> float:
        """Calculate current FPS from frame time history."""
        if not self._frame_times:
            return 0.0

        avg_frame_time = np.mean(self._frame_times)
        return 1.0 / avg_frame_time if avg_frame_time > 0 else 0.0

    def get_latency_ms(self) -> float:
        """Get average frame latency in milliseconds."""
        if not self._frame_times:
            return 0.0
        return np.mean(self._frame_times) * 1000.0

    def get_statistics(self) -> Dict[str, Any]:
        """Get compositor statistics."""
        return {
            'frame_count': self._frame_count,
            'fps': self.get_fps(),
            'latency_ms': self.get_latency_ms(),
            'ar_mode': self.config.ar_mode,
            'projector_stats': self.projector.get_statistics(),
            'renderer_stats': self.renderer.get_statistics(),
            'label_stats': self.label_renderer.get_statistics(),
        }

    def reset_statistics(self):
        """Reset performance statistics."""
        self._frame_times.clear()
        self._frame_count = 0

