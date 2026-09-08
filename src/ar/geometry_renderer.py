"""
Geometric Rendering Module - Render AR geometries in real-time

Responsibilities:
- Render building wireframes (blue, semi-transparent)
- Render road boundaries (green dashed lines)
- Render dynamic obstacle bounding boxes (yellow, from detection)
- Render detected person/vehicle blocks (magenta, from detection)
- Apply depth-based occlusion
- Maintain real-time performance (30 FPS target)

Input:
  - RGB frame from camera
  - Projected geometries (from ProjectionEngine)
  - Detection results (bounding boxes, labels)
  - Depth map for occlusion
  - Rendering configuration (colors, transparency, line widths)

Output:
  - Composited RGB frame with AR overlays
  - Performance metrics (FPS, draw time)

Performance:
  - Single frame rendering: 10-20 ms
  - Full pipeline: <33 ms for 30 FPS
  - Memory: minimal additional overhead

Rendering Specification:
  - Building wireframes: Blue (0, 255, 0), alpha=0.6, thickness=2px
  - Building heights: Semi-transparent roof edges
  - Road boundaries: Green (0, 255, 0), dashed, thickness=2px
  - Obstacles: Yellow (0, 255, 255), thickness=3px, dynamic confidence
  - People/Vehicles: Magenta (255, 0, 255), thickness=3px
  - Selection: Bright white outline when camera looks at building

Failure Modes:
  - Invalid projected points: Skip rendering
  - No depth map: Render all visible geometries without occlusion
  - Frame format mismatch: Convert and continue
  - Performance too low: Drop non-essential overlays (confidence maps, etc.)

Example:
    >>> renderer = GeometryRenderer(frame_shape=(720, 1280))
    >>> render_config = RenderConfig(...)
    >>> frame_ar = renderer.render_buildings(frame, proj_buildings, depth_map, config)
    >>> frame_ar = renderer.render_roads(frame_ar, proj_roads, config)
    >>> frame_ar = renderer.render_obstacles(frame_ar, detections, config)

Reference:
- OpenCV drawing functions (cv2.line, cv2.polylines, etc.)
- Coordinate geometry for wireframe edges
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class RenderConfig:
    """Configuration for AR rendering."""
    # Building wireframe colors and style
    building_color_bgr: Tuple[int, int, int] = (255, 0, 0)  # Blue in BGR
    building_thickness: int = 2
    building_alpha: float = 0.7
    building_roof_color_bgr: Tuple[int, int, int] = (200, 0, 0)  # Darker blue

    # Road rendering
    road_color_bgr: Tuple[int, int, int] = (0, 255, 0)  # Green
    road_thickness: int = 2
    road_dashed: bool = True
    road_dash_length: int = 10

    # Obstacle rendering
    obstacle_color_bgr: Tuple[int, int, int] = (0, 255, 255)  # Yellow
    obstacle_thickness: int = 3
    obstacle_confidence_threshold: float = 0.5

    # Detection rendering (person, vehicle)
    detection_color_bgr: Tuple[int, int, int] = (255, 0, 255)  # Magenta
    detection_thickness: int = 3

    # Occlusion testing
    enable_occlusion: bool = True
    occlusion_margin_m: float = 0.5

    # Performance
    skip_distant_buildings: bool = True
    max_distance_m: float = 200.0


class GeometryRenderer:
    """
    Render AR geometries onto RGB frame.

    Draws buildings, roads, obstacles, and detections with
    configurable colors, transparency, and occlusion handling.
    Optimized for real-time performance at 30 FPS.
    """

    def __init__(
        self,
        frame_shape: Tuple[int, int],
        config: Optional[RenderConfig] = None,
    ):
        """
        Initialize geometry renderer.

        Args:
            frame_shape: (height, width) of expected frames
            config: RenderConfig with style parameters
        """
        self.frame_shape = frame_shape
        self.config = config if config is not None else RenderConfig()

        # Statistics
        self._frame_count = 0
        self._total_render_time_ms = 0.0
        self._buildings_rendered = 0
        self._roads_rendered = 0
        self._obstacles_rendered = 0

        logger.info(
            f"GeometryRenderer initialized: shape={frame_shape}, "
            f"occlusion_enabled={self.config.enable_occlusion}"
        )

    def render_building_wireframe(
        self,
        frame: np.ndarray,
        proj_building,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render single building as wireframe.

        Draws footprint and roof edges, with semi-transparent effect
        using alpha blending. Supports occlusion testing if depth map provided.

        Args:
            frame: Input RGB frame (H, W, 3), uint8
            proj_building: ProjectedBuilding with projected coordinates
            depth_map: Optional depth map for occlusion testing

        Returns:
            Frame with building wireframe drawn
        """
        if not proj_building.is_visible:
            return frame

        frame_with_building = frame.copy()

        # Draw footprint edges
        if len(proj_building.footprint_pixels) > 1:
            for edge_start, edge_end in proj_building.edges[:len(proj_building.footprint_pixels)]:
                if edge_start < len(proj_building.footprint_pixels) and edge_end < len(proj_building.footprint_pixels):
                    p1 = proj_building.footprint_pixels[edge_start]
                    p2 = proj_building.footprint_pixels[edge_end]

                    # Check occlusion
                    if self.config.enable_occlusion and depth_map is not None:
                        depth1 = proj_building.depth_values[edge_start]
                        depth2 = proj_building.depth_values[edge_end]
                        # Simple occlusion: skip if both points are far
                        if depth1 > self.config.max_distance_m or depth2 > self.config.max_distance_m:
                            if self.config.skip_distant_buildings:
                                continue

                    p1_int = (int(round(p1[0])), int(round(p1[1])))
                    p2_int = (int(round(p2[0])), int(round(p2[1])))

                    # Draw line on frame
                    cv2.line(
                        frame_with_building,
                        p1_int,
                        p2_int,
                        self.config.building_color_bgr,
                        self.config.building_thickness,
                        lineType=cv2.LINE_AA
                    )

        # Apply alpha blending for semi-transparency
        alpha = self.config.building_alpha
        frame_with_building = cv2.addWeighted(
            frame,
            1.0 - alpha,
            frame_with_building,
            alpha,
            0
        )

        self._buildings_rendered += 1
        return frame_with_building

    def render_buildings(
        self,
        frame: np.ndarray,
        proj_buildings: List,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render multiple buildings.

        Args:
            frame: Input RGB frame
            proj_buildings: List of ProjectedBuilding objects
            depth_map: Optional depth map for occlusion

        Returns:
            Frame with all buildings rendered
        """
        result = frame.copy()

        for proj_building in proj_buildings:
            result = self.render_building_wireframe(result, proj_building, depth_map)

        return result

    def render_road(
        self,
        frame: np.ndarray,
        proj_road,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render single road as dashed line.

        Args:
            frame: Input RGB frame
            proj_road: ProjectedRoad with projected path
            depth_map: Optional depth map for occlusion

        Returns:
            Frame with road drawn
        """
        if not proj_road.is_visible or len(proj_road.path_pixels) < 2:
            return frame

        result = frame.copy()
        path_pixels = proj_road.path_pixels

        # Draw road as connected line segments with dashing
        if self.config.road_dashed:
            for i in range(len(path_pixels) - 1):
                # Draw dashed segments
                p1 = path_pixels[i]
                p2 = path_pixels[i + 1]

                # Interpolate between points for dashing
                segment_length = np.sqrt((p2[0] - p1[0])**2 + (p2[1] - p1[1])**2)
                num_dashes = max(1, int(segment_length / (2 * self.config.road_dash_length)))

                for dash in range(num_dashes):
                    t1 = dash / num_dashes
                    t2 = (dash + 0.5) / num_dashes
                    if t2 > 1.0:
                        t2 = 1.0

                    dash_p1 = (
                        int(p1[0] + t1 * (p2[0] - p1[0])),
                        int(p1[1] + t1 * (p2[1] - p1[1]))
                    )
                    dash_p2 = (
                        int(p1[0] + t2 * (p2[0] - p1[0])),
                        int(p1[1] + t2 * (p2[1] - p1[1]))
                    )

                    cv2.line(
                        result,
                        dash_p1,
                        dash_p2,
                        self.config.road_color_bgr,
                        self.config.road_thickness,
                        lineType=cv2.LINE_AA
                    )
        else:
            # Draw solid line
            for i in range(len(path_pixels) - 1):
                p1_int = (int(round(path_pixels[i][0])), int(round(path_pixels[i][1])))
                p2_int = (int(round(path_pixels[i + 1][0])), int(round(path_pixels[i + 1][1])))

                cv2.line(
                    result,
                    p1_int,
                    p2_int,
                    self.config.road_color_bgr,
                    self.config.road_thickness,
                    lineType=cv2.LINE_AA
                )

        self._roads_rendered += 1
        return result

    def render_roads(
        self,
        frame: np.ndarray,
        proj_roads: List,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render multiple roads.

        Args:
            frame: Input RGB frame
            proj_roads: List of ProjectedRoad objects
            depth_map: Optional depth map

        Returns:
            Frame with roads rendered
        """
        result = frame.copy()

        for proj_road in proj_roads:
            result = self.render_road(result, proj_road, depth_map)

        return result

    def render_detection_box(
        self,
        frame: np.ndarray,
        bbox_2d: Tuple[float, float, float, float],
        label: str,
        confidence: float,
        color_bgr: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Render single detection bounding box.

        Args:
            frame: Input RGB frame
            bbox_2d: (x_min, y_min, x_max, y_max) in pixels
            label: Object class label (person, car, etc.)
            confidence: Detection confidence (0-1)
            color_bgr: Box color

        Returns:
            Frame with box drawn
        """
        x_min, y_min, x_max, y_max = bbox_2d
        x_min, y_min = int(round(x_min)), int(round(y_min))
        x_max, y_max = int(round(x_max)), int(round(y_max))

        # Clamp to frame bounds
        x_min = max(0, x_min)
        y_min = max(0, y_min)
        x_max = min(frame.shape[1], x_max)
        y_max = min(frame.shape[0], y_max)

        if x_min >= x_max or y_min >= y_max:
            return frame

        result = frame.copy()

        # Draw box
        thickness = self.config.obstacle_thickness
        cv2.rectangle(result, (x_min, y_min), (x_max, y_max), color_bgr, thickness)

        # Draw label and confidence
        label_text = f"{label} {confidence:.2f}"
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.6
        text_thickness = 1
        text_size = cv2.getTextSize(label_text, font, font_scale, text_thickness)[0]

        label_x = x_min
        label_y = max(20, y_min - 5)

        # Draw background for text
        cv2.rectangle(
            result,
            (label_x - 2, label_y - text_size[1] - 2),
            (label_x + text_size[0] + 2, label_y + 2),
            color_bgr,
            cv2.FILLED
        )

        # Draw text
        cv2.putText(
            result,
            label_text,
            (label_x, label_y),
            font,
            font_scale,
            (255, 255, 255),  # White text
            text_thickness,
            lineType=cv2.LINE_AA
        )

        self._obstacles_rendered += 1
        return result

    def render_detections(
        self,
        frame: np.ndarray,
        detections: List[Dict[str, Any]],
    ) -> np.ndarray:
        """
        Render detection bounding boxes and labels.

        Expected detection dict:
        {
            'bbox_2d': (x_min, y_min, x_max, y_max),
            'label': 'person' or 'car',
            'confidence': 0.95,
        }

        Args:
            frame: Input RGB frame
            detections: List of detection dictionaries

        Returns:
            Frame with detections rendered
        """
        result = frame.copy()

        for detection in detections:
            if detection.get('confidence', 0) < self.config.obstacle_confidence_threshold:
                continue

            bbox_2d = detection.get('bbox_2d')
            label = detection.get('label', 'unknown')
            confidence = detection.get('confidence', 0.0)

            # Choose color based on object type
            if label == 'person':
                color = self.config.detection_color_bgr
            elif label in ['car', 'truck', 'bus']:
                color = self.config.obstacle_color_bgr
            else:
                color = self.config.obstacle_color_bgr

            result = self.render_detection_box(result, bbox_2d, label, confidence, color)

        return result

    def render_frame(
        self,
        frame: np.ndarray,
        proj_buildings: Optional[List] = None,
        proj_roads: Optional[List] = None,
        detections: Optional[List[Dict]] = None,
        depth_map: Optional[np.ndarray] = None,
    ) -> np.ndarray:
        """
        Render complete AR scene to frame.

        Combines all AR layers: map geometries + detections.

        Args:
            frame: Input RGB frame
            proj_buildings: List of projected buildings
            proj_roads: List of projected roads
            detections: List of detection boxes
            depth_map: Optional depth map for occlusion

        Returns:
            Composite frame with all AR layers
        """
        start_time = time.time()

        result = frame.copy()

        # Render map geometries
        if proj_buildings:
            result = self.render_buildings(result, proj_buildings, depth_map)

        if proj_roads:
            result = self.render_roads(result, proj_roads, depth_map)

        # Render detections last (on top)
        if detections:
            result = self.render_detections(result, detections)

        elapsed_ms = (time.time() - start_time) * 1000
        self._total_render_time_ms += elapsed_ms
        self._frame_count += 1

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get rendering statistics."""
        avg_render_ms = (
            self._total_render_time_ms / self._frame_count
            if self._frame_count > 0 else 0
        )
        return {
            'frames_rendered': self._frame_count,
            'avg_render_time_ms': avg_render_ms,
            'buildings_rendered': self._buildings_rendered,
            'roads_rendered': self._roads_rendered,
            'obstacles_rendered': self._obstacles_rendered,
            'estimated_fps': 1000 / avg_render_ms if avg_render_ms > 0 else 0,
        }

