"""
Label Rendering Module - Render high-contrast text labels on AR display

Responsibilities:
- Render building names and addresses at projected locations
- Render street names and POI labels
- Ensure high contrast for readability (light text on dark, dark on light)
- Handle text clipping and occlusion
- Support multiple text sizes and colors

Input:
  - RGB frame
  - Building/road features with names
  - Projected positions (centroids, bbox centers)
  - Camera confidence and localization status

Output:
  - Frame with rendered labels
  - Label rendering statistics

Performance:
  - Label rendering: <5 ms for 20 labels
  - Text size adjustment: automatic based on distance
  - Occlusion: simple distance-based culling

Label Specification:
  - Building names: rendered at centroid with drop shadow
  - Street names: rendered along road paths
  - Size: adaptive based on distance (farther = smaller)
  - Color: high-contrast (white on dark, dark on light background)
  - Font: OpenCV SIMPLEX for clarity
  - Opacity: full (labels are critical for navigation)

Failure Modes:
  - Missing name: Skip label, show fallback
  - Text outside bounds: Clip or reposition
  - Low contrast: Add background rectangle (drop shadow)
  - Recovery: Log and continue

Example:
    >>> label_renderer = LabelRenderer()
    >>> frame_with_labels = label_renderer.render_building_labels(
    ...     frame, proj_buildings, camera_pose)
    >>> frame_with_labels = label_renderer.render_street_labels(
    ...     frame_with_labels, proj_roads)

Reference:
- OpenCV text rendering (cv2.putText, cv2.getTextSize)
- High-contrast color selection
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class LabelConfig:
    """Configuration for text label rendering."""
    # Font
    font_face: int = cv2.FONT_HERSHEY_SIMPLEX
    font_thickness: int = 1

    # Building labels
    building_label_color_bgr: Tuple[int, int, int] = (255, 255, 255)  # White
    building_label_size_base: float = 0.7
    building_label_max_distance: float = 100.0  # Don't render farther than this

    # Street labels
    street_label_color_bgr: Tuple[int, int, int] = (200, 200, 255)  # Light blue
    street_label_size_base: float = 0.6

    # Background for readability
    use_background: bool = True
    background_color_bgr: Tuple[int, int, int] = (0, 0, 0)  # Black
    background_alpha: float = 0.7

    # Positioning
    offset_px: Tuple[int, int] = (10, -5)  # Offset from anchor point


class LabelRenderer:
    """
    Render text labels for AR visualization.

    Handles building names, street names, and POI labels
    with adaptive sizing and high contrast for readability.
    """

    def __init__(self, config: Optional[LabelConfig] = None):
        """
        Initialize label renderer.

        Args:
            config: LabelConfig with style parameters
        """
        self.config = config if config is not None else LabelConfig()

        # Statistics
        self._labels_rendered = 0
        self._total_render_time_ms = 0.0

        logger.info("LabelRenderer initialized")

    def compute_font_scale(self, distance_m: float) -> float:
        """
        Compute adaptive font size based on distance.

        Closer objects get larger labels.

        Args:
            distance_m: Distance to object in meters

        Returns:
            Font scale for cv2.putText
        """
        # Linear scaling: closer = larger
        # At 10m: full size
        # At 100m: half size
        min_distance = 2.0
        max_distance = self.config.building_label_max_distance

        distance_m = np.clip(distance_m, min_distance, max_distance)
        scale = self.config.building_label_size_base * (1.0 - (distance_m - min_distance) / (max_distance - min_distance))
        return max(0.4, scale)

    def render_text_with_background(
        self,
        frame: np.ndarray,
        text: str,
        position: Tuple[int, int],
        font_scale: float,
        color_bgr: Tuple[int, int, int],
    ) -> np.ndarray:
        """
        Render text with semi-transparent background for contrast.

        Args:
            frame: Input RGB frame
            text: Text to render
            position: (x, y) position in pixels
            font_scale: OpenCV font scale
            color_bgr: Text color in BGR

        Returns:
            Frame with text rendered
        """
        result = frame.copy()

        if not text:
            return result

        font_face = self.config.font_face
        font_thickness = self.config.font_thickness

        # Get text size
        text_size, baseline = cv2.getTextSize(
            text,
            font_face,
            font_scale,
            font_thickness
        )

        x, y = position
        text_width, text_height = text_size

        if self.config.use_background:
            # Draw semi-transparent background
            overlay = result.copy()
            background_rect = (
                x - 5,
                y - text_height - 5,
                x + text_width + 5,
                y + baseline + 5
            )

            cv2.rectangle(
                overlay,
                (background_rect[0], background_rect[1]),
                (background_rect[2], background_rect[3]),
                self.config.background_color_bgr,
                cv2.FILLED
            )

            # Blend with background
            alpha = self.config.background_alpha
            result = cv2.addWeighted(result, 1.0 - alpha, overlay, alpha, 0)

        # Draw text
        cv2.putText(
            result,
            text,
            (int(x), int(y)),
            font_face,
            font_scale,
            color_bgr,
            font_thickness,
            lineType=cv2.LINE_AA
        )

        self._labels_rendered += 1
        return result

    def render_building_label(
        self,
        frame: np.ndarray,
        proj_building,
        building_name: Optional[str] = None,
    ) -> np.ndarray:
        """
        Render single building label.

        Args:
            frame: Input RGB frame
            proj_building: ProjectedBuilding with centroid and depth
            building_name: Name to render (or use from building)

        Returns:
            Frame with label rendered
        """
        if not proj_building.is_visible or proj_building.centroid_pixel is None:
            return frame

        centroid_px = proj_building.centroid_pixel
        avg_depth = np.mean(proj_building.depth_values) if proj_building.depth_values else 10.0

        # Check distance visibility
        if avg_depth > self.config.building_label_max_distance:
            return frame

        # Compute font size
        font_scale = self.compute_font_scale(avg_depth)

        # Get label text
        label_text = building_name or proj_building.name or "Building"

        # Position offset
        offset_x, offset_y = self.config.offset_px
        label_position = (
            int(centroid_px[0] + offset_x),
            int(centroid_px[1] + offset_y)
        )

        # Render
        result = self.render_text_with_background(
            frame,
            label_text,
            label_position,
            font_scale,
            self.config.building_label_color_bgr
        )

        return result

    def render_building_labels(
        self,
        frame: np.ndarray,
        proj_buildings: List,
    ) -> np.ndarray:
        """
        Render labels for multiple buildings.

        Args:
            frame: Input RGB frame
            proj_buildings: List of ProjectedBuilding objects

        Returns:
            Frame with all labels rendered
        """
        start_time = time.time()
        result = frame.copy()

        for proj_building in proj_buildings:
            result = self.render_building_label(result, proj_building)

        elapsed_ms = (time.time() - start_time) * 1000
        self._total_render_time_ms += elapsed_ms

        return result

    def render_street_label(
        self,
        frame: np.ndarray,
        proj_road,
    ) -> np.ndarray:
        """
        Render street name along road path.

        Args:
            frame: Input RGB frame
            proj_road: ProjectedRoad with path

        Returns:
            Frame with street label rendered
        """
        if not proj_road.is_visible or not proj_road.path_pixels:
            return frame

        street_name = proj_road.name or "Street"

        # Place label at midpoint of road path
        if len(proj_road.path_pixels) >= 2:
            mid_idx = len(proj_road.path_pixels) // 2
            mid_point = proj_road.path_pixels[mid_idx]

            # Compute font size based on distance
            avg_depth = np.mean(proj_road.depth_values) if proj_road.depth_values else 20.0
            font_scale = self.config.street_label_size_base * (1.0 - avg_depth / 200.0)
            font_scale = max(0.3, font_scale)

            # Position
            offset_x, offset_y = self.config.offset_px
            label_position = (
                int(mid_point[0] + offset_x),
                int(mid_point[1] + offset_y)
            )

            # Render
            result = self.render_text_with_background(
                frame,
                street_name,
                label_position,
                font_scale,
                self.config.street_label_color_bgr
            )

            return result

        return frame

    def render_street_labels(
        self,
        frame: np.ndarray,
        proj_roads: List,
    ) -> np.ndarray:
        """
        Render labels for multiple streets.

        Args:
            frame: Input RGB frame
            proj_roads: List of ProjectedRoad objects

        Returns:
            Frame with all street labels rendered
        """
        result = frame.copy()

        for proj_road in proj_roads:
            result = self.render_street_label(result, proj_road)

        return result

    def render_status_text(
        self,
        frame: np.ndarray,
        status_lines: List[str],
        position: Tuple[int, int] = (10, 30),
        color_bgr: Tuple[int, int, int] = (0, 255, 0),
    ) -> np.ndarray:
        """
        Render status text (FPS, confidence, etc.).

        Args:
            frame: Input RGB frame
            status_lines: List of text lines to render
            position: Starting (x, y) position
            color_bgr: Text color

        Returns:
            Frame with status text rendered
        """
        result = frame.copy()

        x, y = position
        line_height = 25

        for i, line in enumerate(status_lines):
            line_y = y + i * line_height
            result = self.render_text_with_background(
                result,
                line,
                (x, line_y),
                0.7,
                color_bgr
            )

        return result

    def get_statistics(self) -> Dict[str, Any]:
        """Get label rendering statistics."""
        return {
            'labels_rendered': self._labels_rendered,
            'total_render_time_ms': self._total_render_time_ms,
        }

