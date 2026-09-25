"""
Spatial Visualizer - Simple 3D spatial visualization with ground plane boxes.

Converts stereo depth data into 3D points and renders them as boxes on a ground
plane, showing object positions and obstacles in space relative to the rover.

Pipeline:
  Stereo Depth → Point Cloud → Ground Plane Projection → Box Drawing
"""

import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SpatialObject:
    """Detected object in 3D space."""
    x: float  # meters
    y: float  # meters
    z: float  # meters (height above ground)
    width: float  # meters
    depth: float  # meters
    height: float  # meters
    confidence: float  # 0-1
    label: str = "object"


class SpatialVisualizer:
    """Render 3D spatial visualization with ground plane reference."""

    def __init__(self, camera_matrix: np.ndarray, image_width: int = 640, image_height: int = 360):
        """
        Initialize spatial visualizer.

        Args:
            camera_matrix: Camera intrinsic matrix K (3x3)
            image_width: Image width in pixels
            image_height: Image height in pixels
        """
        self.K = camera_matrix
        self.width = image_width
        self.height = image_height

        # Default local origin (Minsk, Belarus example from current system)
        self.local_origin = (53.9045, 27.5615, 125.5)

        # Ground plane configuration
        self.ground_plane_distance = 5.0  # Look 5m ahead
        self.grid_size = 0.5  # 0.5m grid cells
        self.max_range = 10.0  # Max detection range

        # Visualization parameters
        self.fov_degrees = 60.0
        self.show_grid = True
        self.show_debug = False

        logger.info("SpatialVisualizer initialized")

    def process_depth_frame(self, depth_frame: np.ndarray) -> List[SpatialObject]:
        """
        Process depth frame and detect objects in space.

        Args:
            depth_frame: Depth map (H, W) in meters

        Returns:
            List of detected spatial objects
        """
        if depth_frame is None or depth_frame.size == 0:
            logger.warning("Empty depth frame")
            return []

        try:
            # Convert depth to point cloud
            points_3d = self._depth_to_points(depth_frame)

            # Detect objects in 3D space
            objects = self._detect_spatial_objects(points_3d, depth_frame)

            return objects

        except Exception as e:
            logger.error(f"Depth processing failed: {e}")
            return []

    def render_frame(
        self,
        rgb_frame: np.ndarray,
        depth_frame: np.ndarray,
        objects: Optional[List[SpatialObject]] = None
    ) -> np.ndarray:
        """
        Render RGB frame with spatial visualization overlay.

        Args:
            rgb_frame: RGB image (H, W, 3)
            depth_frame: Depth map (H, W) in meters
            objects: List of spatial objects to draw

        Returns:
            Rendered frame with visualization
        """
        if rgb_frame is None:
            return None

        try:
            # Make a copy to avoid modifying original
            canvas = rgb_frame.copy()

            if objects is None:
                objects = []

            # Draw ground plane grid
            if self.show_grid:
                canvas = self._draw_ground_grid(canvas, depth_frame)

            # Draw detected objects
            for obj in objects:
                canvas = self._draw_object_box(canvas, obj, depth_frame)

            # Draw info text
            canvas = self._draw_info_overlay(canvas, len(objects), depth_frame)

            return canvas

        except Exception as e:
            logger.error(f"Frame rendering failed: {e}")
            return rgb_frame

    def _depth_to_points(self, depth_frame: np.ndarray) -> np.ndarray:
        """Convert depth map to 3D point cloud."""
        h, w = depth_frame.shape

        # Create pixel coordinate grids
        u = np.arange(w, dtype=np.float32)
        v = np.arange(h, dtype=np.float32)
        uu, vv = np.meshgrid(u, v)

        # Get camera intrinsics
        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # Compute 3D points: X = Z*(u-cx)/fx, Y = Z*(v-cy)/fy
        z = depth_frame
        x = z * (uu - cx) / fx
        y = z * (vv - cy) / fy

        # Stack into N x 3 array
        points = np.stack([x, y, z], axis=-1)
        points = points.reshape(-1, 3)

        # Filter valid points (Z > 0, not NaN/Inf)
        valid = (z.flatten() > 0.1) & np.isfinite(z.flatten())
        points = points[valid]

        return points

    def _detect_spatial_objects(
        self,
        points_3d: np.ndarray,
        depth_frame: np.ndarray
    ) -> List[SpatialObject]:
        """Detect objects in 3D space using depth analysis."""
        objects = []

        if len(points_3d) == 0:
            return objects

        try:
            depth_map = depth_frame.copy()

            # Create binary mask of valid depth pixels in range
            valid_mask = (depth_map > 0.2) & (depth_map < self.max_range) & np.isfinite(depth_map)

            if not np.any(valid_mask):
                return objects

            depth_binary = (valid_mask).astype(np.uint8) * 255

            # Morphological operations to clean up depth map
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (7, 7))
            depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, kernel)
            depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_OPEN, kernel)

            # Find contours (potential objects)
            contours, _ = cv2.findContours(depth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Use smaller area threshold to catch more objects
                area = cv2.contourArea(contour)
                if area < 50:  # Lowered from 100
                    continue

                # Get bounding box
                x_min, y_min, w, h = cv2.boundingRect(contour)
                x_max = x_min + w
                y_max = y_min + h

                # Ensure bounds are valid
                if x_max <= x_min or y_max <= y_min:
                    continue

                # Get depth range in this region
                roi_depth = depth_map[y_min:y_max, x_min:x_max]
                roi_valid = valid_mask[y_min:y_max, x_min:x_max]
                valid_depths = roi_depth[roi_valid]

                if len(valid_depths) == 0:
                    continue

                mean_depth = np.mean(valid_depths)
                depth_std = np.std(valid_depths)

                # Skip if depth is invalid
                if not np.isfinite(mean_depth):
                    continue

                # Project corners to 3D to estimate size
                corners_2d = np.array([
                    [x_min, y_min],
                    [x_max, y_min],
                    [x_min, y_max],
                    [x_max, y_max]
                ], dtype=np.float32)

                corners_3d = self._pixel_to_3d(corners_2d, mean_depth)

                # Estimate object dimensions
                width = np.linalg.norm(corners_3d[1] - corners_3d[0])
                depth_dim = np.linalg.norm(corners_3d[2] - corners_3d[0])

                # Center in 3D
                center_2d = np.array([(x_min + x_max) / 2, (y_min + y_max) / 2], dtype=np.float32)
                center_3d = self._pixel_to_3d(center_2d.reshape(1, 2), mean_depth)[0]

                # Skip very small objects
                if width < 0.05 or depth_dim < 0.05:
                    continue

                # Confidence based on depth consistency
                confidence = 0.5 + min(0.5, 1.0 - depth_std / mean_depth) if mean_depth > 0 else 0.7

                # Create spatial object
                obj = SpatialObject(
                    x=center_3d[0],
                    y=center_3d[1],
                    z=center_3d[2],
                    width=width,
                    depth=depth_dim,
                    height=0.5,  # Assume standard height
                    confidence=confidence,
                    label="obstacle"
                )

                objects.append(obj)

            logger.debug(f"Detected {len(objects)} objects")
            return objects

        except Exception as e:
            logger.warning(f"Object detection failed: {e}")
            return objects

    def _pixel_to_3d(self, pixel_coords: np.ndarray, depth: float) -> np.ndarray:
        """Convert pixel coordinates to 3D points."""
        # pixel_coords: (N, 2) array of (u, v)
        # depth: scalar or (N,) array

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        u = pixel_coords[:, 0]
        v = pixel_coords[:, 1]

        # Ensure depth is an array
        if np.isscalar(depth):
            z_vals = np.full_like(u, float(depth), dtype=np.float32)
        else:
            z_vals = np.asarray(depth, dtype=np.float32)

        x = (u - cx) * z_vals / fx
        y = (v - cy) * z_vals / fy

        points = np.stack([x, y, z_vals], axis=-1)
        return points

    def _draw_ground_grid(self, canvas: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
        """Draw 3D perspective grid on ground plane."""
        try:
            h, w = canvas.shape[:2]
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]

            grid_spacing = 0.5  # 0.5m grid cells
            grid_width = 5.0    # ±2.5m width
            max_dist = self.max_range

            # Draw depth lines (parallel to camera, receding into distance)
            for dist in np.arange(0.5, max_dist + 0.5, grid_spacing):
                points_2d = []

                # Generate line from -grid_width to +grid_width at this distance
                for x in np.linspace(-grid_width / 2, grid_width / 2, 20):
                    # Project 3D point (x, 0, dist) to 2D image
                    px = cx + (x / dist) * fx
                    py = cy - (0 / dist) * fy  # y is 0 (on ground)

                    if 0 <= px < w:
                        points_2d.append([px, py])

                # Draw line
                if len(points_2d) > 1:
                    points_array = np.array(points_2d, dtype=np.int32)
                    # Fade color with distance
                    intensity = int(200 * (1 - dist / max_dist))
                    color = (intensity // 2, intensity, intensity // 2)
                    cv2.polylines(canvas, [points_array], False, color, 1)

            # Draw width lines (perpendicular to camera, at different distances)
            for x in np.linspace(-grid_width / 2, grid_width / 2, 10):
                points_2d = []

                # Generate line from 0.5m to max_dist
                for dist in np.linspace(0.5, max_dist, 20):
                    # Project 3D point (x, 0, dist) to 2D image
                    px = cx + (x / dist) * fx
                    py = cy - (0 / dist) * fy

                    if 0 <= px < w and 0 <= py < h:
                        points_2d.append([px, py])

                # Draw line
                if len(points_2d) > 1:
                    points_array = np.array(points_2d, dtype=np.int32)
                    color = (80, 120, 80)
                    cv2.polylines(canvas, [points_array], False, color, 1)

            # Draw center line (forward direction)
            cv2.line(canvas, (int(cx), h), (int(cx), h // 2), (0, 255, 0), 2)

            return canvas
        except Exception as e:
            logger.warning(f"Grid drawing failed: {e}")
            return canvas

    def _draw_object_box(
        self,
        canvas: np.ndarray,
        obj: SpatialObject,
        depth_frame: np.ndarray
    ) -> np.ndarray:
        """Draw a 3D wireframe box for detected object."""
        try:
            h, w = canvas.shape[:2]
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]

            if obj.z <= 0:
                return canvas

            # Define 3D bounding box corners (in camera frame)
            # Object center at (obj.x, obj.y, obj.z)
            # Dimensions: width, depth, height
            half_w = obj.width / 2
            half_d = obj.depth / 2
            half_h = obj.height / 2

            corners_3d = np.array([
                # Bottom face (y = obj.y - half_h)
                [obj.x - half_w, obj.y - half_h, obj.z + half_d],
                [obj.x + half_w, obj.y - half_h, obj.z + half_d],
                [obj.x + half_w, obj.y - half_h, obj.z - half_d],
                [obj.x - half_w, obj.y - half_h, obj.z - half_d],
                # Top face (y = obj.y + half_h)
                [obj.x - half_w, obj.y + half_h, obj.z + half_d],
                [obj.x + half_w, obj.y + half_h, obj.z + half_d],
                [obj.x + half_w, obj.y + half_h, obj.z - half_d],
                [obj.x - half_w, obj.y + half_h, obj.z - half_d],
            ], dtype=np.float32)

            # Project to 2D
            points_2d = []
            for corner in corners_3d:
                x, y, z = corner
                if z <= 0:  # Behind camera
                    points_2d.append(None)
                else:
                    px = cx + (x / z) * fx
                    py = cy - (y / z) * fy
                    points_2d.append((int(px), int(py)))

            # Draw edges if all points are valid
            color = (0, 255, 0) if obj.confidence > 0.5 else (0, 165, 255)

            if all(p is not None for p in points_2d):
                # Bottom face
                cv2.line(canvas, points_2d[0], points_2d[1], color, 2)
                cv2.line(canvas, points_2d[1], points_2d[2], color, 2)
                cv2.line(canvas, points_2d[2], points_2d[3], color, 2)
                cv2.line(canvas, points_2d[3], points_2d[0], color, 2)

                # Top face
                cv2.line(canvas, points_2d[4], points_2d[5], color, 2)
                cv2.line(canvas, points_2d[5], points_2d[6], color, 2)
                cv2.line(canvas, points_2d[6], points_2d[7], color, 2)
                cv2.line(canvas, points_2d[7], points_2d[4], color, 2)

                # Vertical edges
                cv2.line(canvas, points_2d[0], points_2d[4], color, 2)
                cv2.line(canvas, points_2d[1], points_2d[5], color, 2)
                cv2.line(canvas, points_2d[2], points_2d[6], color, 2)
                cv2.line(canvas, points_2d[3], points_2d[7], color, 2)

            # Draw center point and label
            center_2d = np.mean([p for p in points_2d if p is not None], axis=0)
            if len([p for p in points_2d if p is not None]) > 0:
                cv2.circle(canvas, tuple(center_2d.astype(int)), 5, color, -1)

                # Label with distance and confidence
                label = f"{obj.label} {obj.z:.1f}m (conf: {obj.confidence:.1f})"
                cv2.putText(canvas, label, (int(center_2d[0]) - 50, int(center_2d[1]) - 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)

            return canvas
        except Exception as e:
            logger.warning(f"Box drawing failed: {e}")
            return canvas

    def _draw_info_overlay(
        self,
        canvas: np.ndarray,
        num_objects: int,
        depth_frame: np.ndarray
    ) -> np.ndarray:
        """Draw information overlay on canvas."""
        try:
            h, w = canvas.shape[:2]

            # Background for text
            cv2.rectangle(canvas, (5, 5), (400, 120), (0, 0, 0), -1)
            cv2.rectangle(canvas, (5, 5), (400, 120), (0, 255, 0), 2)

            # Info text
            lines = [
                f"Spatial Visualization - Objects: {num_objects}",
                f"Range: 0.2m - {self.max_range}m",
                f"FOV: {self.fov_degrees}°",
                f"Grid: {self.grid_size}m cells",
            ]

            if depth_frame is not None:
                valid_depths = depth_frame[depth_frame > 0.1]
                if len(valid_depths) > 0:
                    mean_depth = np.mean(valid_depths)
                    lines.append(f"Mean Depth: {mean_depth:.2f}m")

            for i, line in enumerate(lines):
                cv2.putText(canvas, line, (15, 25 + i * 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

            return canvas
        except Exception as e:
            logger.warning(f"Info overlay failed: {e}")
            return canvas
