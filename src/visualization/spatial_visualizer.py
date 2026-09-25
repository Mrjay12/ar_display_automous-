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
    contour: Optional[np.ndarray] = None  # 2D contour points in image space
    bbox: Optional[tuple] = None  # Bounding box (x_min, y_min, x_max, y_max)


class SpatialVisualizer:
    """Render 3D spatial visualization with octomap-style voxel grid."""

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

        # Octomap voxel grid configuration
        self.voxel_size = 0.1  # 0.1m voxel size for detailed grid
        self.grid_size = 0.5  # 0.5m grid cells for visualization
        self.max_range = 10.0  # Max detection range
        self.grid_width = 4.0  # ±2m width

        # Occupancy grid: tracks which voxels are occupied
        self.occupancy_grid = {}  # {(x, y, z): occupancy_probability}

        # Visualization parameters
        self.fov_degrees = 60.0
        self.show_grid = True
        self.show_debug = False
        self._last_point_cloud_count = 0

        logger.info("SpatialVisualizer initialized with octomap voxel grid")

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

    def _build_occupancy_grid(self, objects: List[SpatialObject]) -> None:
        """Build 3D occupancy grid from detected objects."""
        self.occupancy_grid.clear()

        for obj in objects:
            if obj.z <= 0:
                continue

            # Mark voxels occupied within object bounding box
            half_w = obj.width / 2
            half_d = obj.depth / 2
            half_h = obj.height / 2

            # Voxel grid from object bounds
            x_min = obj.x - half_w
            x_max = obj.x + half_w
            z_min = obj.z - half_d
            z_max = obj.z + half_d
            y_min = obj.y - half_h
            y_max = obj.y + half_h

            # Quantize to voxel grid
            for x in np.arange(x_min, x_max, self.voxel_size):
                for z in np.arange(z_min, z_max, self.voxel_size):
                    for y in np.arange(y_min, y_max, self.voxel_size):
                        voxel_key = (round(x / self.voxel_size) * self.voxel_size,
                                    round(y / self.voxel_size) * self.voxel_size,
                                    round(z / self.voxel_size) * self.voxel_size)
                        self.occupancy_grid[voxel_key] = 1.0  # Occupied

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

            # Draw depth-based point cloud (octomap-style visualization)
            canvas = self._draw_depth_point_cloud(canvas, depth_frame)

            # Draw detected objects
            for obj in objects:
                canvas = self._draw_object_box(canvas, obj, depth_frame)

            # Draw info text
            canvas = self._draw_info_overlay(canvas, len(objects), depth_frame)

            return canvas

        except Exception as e:
            logger.error(f"Frame rendering failed: {e}")
            return rgb_frame

    def _voxel_to_2d(self, voxel_3d: np.ndarray) -> Optional[Tuple[int, int]]:
        """
        Project a 3D voxel coordinate to 2D image pixel.

        Args:
            voxel_3d: 3D point (x, y, z) in meters

        Returns:
            Tuple of (u, v) pixel coordinates or None if out of frame
        """
        x, y, z = voxel_3d

        # Skip points behind camera
        if z <= 0.1:
            return None

        fx = self.K[0, 0]
        fy = self.K[1, 1]
        cx = self.K[0, 2]
        cy = self.K[1, 2]

        # Project to image plane: (u,v) = (cx + fx*x/z, cy - fy*y/z)
        u = cx + (x / z) * fx
        v = cy - (y / z) * fy

        # Check bounds
        if 0 <= u < self.width and 0 <= v < self.height:
            return (int(u), int(v))

        return None

    def _draw_depth_point_cloud(self, canvas: np.ndarray, depth_frame: np.ndarray) -> np.ndarray:
        """Draw point cloud from depth data, colored by height (octomap style)."""
        try:
            if depth_frame is None or depth_frame.size == 0:
                return canvas

            h, w = canvas.shape[:2]

            # Get valid depth points
            valid_mask = (depth_frame > 0.1) & (depth_frame < self.max_range) & np.isfinite(depth_frame)
            valid_indices = np.where(valid_mask)

            if len(valid_indices[0]) == 0:
                return canvas

            # Convert depth pixels to 3D points
            v_pixels = valid_indices[0]  # row (y)
            u_pixels = valid_indices[1]  # col (x)
            depths = depth_frame[valid_mask]

            # Project to 3D using camera matrix
            fx = self.K[0, 0]
            fy = self.K[1, 1]
            cx = self.K[0, 2]
            cy = self.K[1, 2]

            x_3d = (u_pixels - cx) * depths / fx
            y_3d = (v_pixels - cy) * depths / fy
            z_3d = depths

            # Get height range for coloring
            min_height = np.min(y_3d)
            max_height = np.max(y_3d)
            height_range = max_height - min_height

            # If all points at same height, use depth for coloring instead
            if height_range < 0.01:
                min_depth = np.min(z_3d)
                max_depth = np.max(z_3d)
                height_range = max_depth - min_depth if max_depth > min_depth else 1.0
                use_depth_color = True
            else:
                use_depth_color = False

            # Downsample for performance (render every Nth point) - render all for dense cloud
            downsample = 1
            indices = np.arange(0, len(x_3d), downsample)
            point_count = len(indices)

            # Draw each point colored by height
            for idx in indices:
                ux, uy, uz = u_pixels[idx], v_pixels[idx], z_3d[idx]
                hy = y_3d[idx]

                # Skip if outside image
                if not (0 <= ux < w and 0 <= uy < h):
                    continue

                # Height-based rainbow coloring (or depth-based if height too uniform)
                if use_depth_color:
                    normalized_value = (uz - min_depth) / height_range
                else:
                    normalized_value = (hy - min_height) / height_range

                if normalized_value < 0.25:  # Blue (lowest)
                    b = int(255 * (1.0 - normalized_value / 0.25))
                    g = int(255 * (normalized_value / 0.25))
                    r = 0
                elif normalized_value < 0.5:  # Cyan to Green
                    b = int(255 * (1.0 - (normalized_value - 0.25) / 0.25))
                    g = 255
                    r = 0
                elif normalized_value < 0.75:  # Green to Yellow
                    b = 0
                    g = 255
                    r = int(255 * ((normalized_value - 0.5) / 0.25))
                else:  # Yellow to Red (highest)
                    b = 0
                    g = int(255 * (1.0 - (normalized_value - 0.75) / 0.25))
                    r = 255

                color = (b, g, r)

                # Draw point at pixel location (larger radius for visibility)
                cv2.circle(canvas, (int(ux), int(uy)), 4, color, -1)

            # Store for debug
            self._last_point_cloud_count = point_count

            return canvas

        except Exception as e:
            logger.warning(f"Depth point cloud rendering failed: {e}")
            return canvas

    def _draw_occupancy_grid(self, canvas: np.ndarray) -> np.ndarray:
        """Draw octomap-style point cloud - individual voxels colored by height."""
        try:
            if not self.occupancy_grid:
                return canvas

            h, w = canvas.shape[:2]
            voxels = list(self.occupancy_grid.keys())

            # Get height range for color mapping
            all_heights = [vy for vx, vy, vz in voxels]
            if not all_heights:
                return canvas

            min_height = min(all_heights)
            max_height = max(all_heights)
            height_range = max_height - min_height if max_height > min_height else 1.0

            # Group by depth for proper rendering (far to near)
            voxels_by_depth = {}
            for vx, vy, vz in voxels:
                depth_bin = int(vz / 0.2)  # Finer depth grouping
                if depth_bin not in voxels_by_depth:
                    voxels_by_depth[depth_bin] = []
                voxels_by_depth[depth_bin].append((vx, vy, vz))

            # Draw voxels from far to near (depth sorting for proper occlusion)
            for depth_bin in sorted(voxels_by_depth.keys(), reverse=True):
                voxel_group = voxels_by_depth[depth_bin]

                for vx, vy, vz in voxel_group:
                    # Project voxel center to 2D image
                    voxel_3d = np.array([vx, vy, vz])
                    voxel_2d = self._voxel_to_2d(voxel_3d)

                    if voxel_2d is not None:
                        # Color based on HEIGHT (Y coordinate) - rainbow gradient
                        # Blue (low) → Cyan → Green → Yellow → Red (high)
                        normalized_height = (vy - min_height) / height_range

                        if normalized_height < 0.25:  # Blue (lowest)
                            b = int(255 * (1.0 - normalized_height / 0.25))
                            g = int(255 * (normalized_height / 0.25))
                            r = 0
                        elif normalized_height < 0.5:  # Cyan to Green
                            b = int(255 * (1.0 - (normalized_height - 0.25) / 0.25))
                            g = 255
                            r = 0
                        elif normalized_height < 0.75:  # Green to Yellow
                            b = 0
                            g = 255
                            r = int(255 * ((normalized_height - 0.5) / 0.25))
                        else:  # Yellow to Red (highest)
                            b = 0
                            g = int(255 * (1.0 - (normalized_height - 0.75) / 0.25))
                            r = 255

                        color = (b, g, r)

                        # Draw voxel as colored point
                        occupancy = self.occupancy_grid[(vx, vy, vz)]
                        radius = max(2, int(3 * occupancy))  # 2-3px radius
                        cv2.circle(canvas, voxel_2d, radius, color, -1)

            return canvas

        except Exception as e:
            logger.warning(f"Occupancy grid drawing failed: {e}")
            return canvas

    def _get_voxel_corners(self, vx: float, vy: float, vz: float) -> List[np.ndarray]:
        """Get 8 corner points of a voxel in 3D."""
        half_size = self.voxel_size / 2
        corners = []
        for dx in [-half_size, half_size]:
            for dy in [-half_size, half_size]:
                for dz in [-half_size, half_size]:
                    corners.append(np.array([vx + dx, vy + dy, vz + dz]))
        return corners

    def _draw_voxel_wireframe(
        self,
        canvas: np.ndarray,
        corners_3d: List[np.ndarray],
        color: Tuple[int, int, int],
        thickness: int = 1
    ) -> None:
        """Draw wireframe edges of a voxel cube."""
        # Define 12 edges of a cube (pairs of corner indices)
        edges = [
            (0, 1), (1, 3), (3, 2), (2, 0),  # Bottom face
            (4, 5), (5, 7), (7, 6), (6, 4),  # Top face
            (0, 4), (1, 5), (2, 6), (3, 7)   # Vertical edges
        ]

        for i, j in edges:
            p1_2d = self._voxel_to_2d(corners_3d[i])
            p2_2d = self._voxel_to_2d(corners_3d[j])

            if p1_2d is not None and p2_2d is not None:
                cv2.line(canvas, p1_2d, p2_2d, color, thickness)

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
            # More lenient range to catch nearby and distant objects
            valid_mask = (depth_map > 0.1) & (depth_map < self.max_range) & np.isfinite(depth_map)

            if not np.any(valid_mask):
                return objects

            depth_binary = (valid_mask).astype(np.uint8) * 255

            # Minimal morphological operations - only light smoothing
            kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))

            # Just light closing, skip opening to preserve thin objects
            depth_binary = cv2.morphologyEx(depth_binary, cv2.MORPH_CLOSE, kernel, iterations=1)

            # Find contours (potential objects)
            contours, _ = cv2.findContours(depth_binary, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

            for contour in contours:
                # Very low area threshold to catch thin objects like cables
                area = cv2.contourArea(contour)
                if area < 10:  # Detect objects down to ~3x3 pixel regions
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

                # Don't filter by minimum size - allow all detected contours
                # Clamp very small dimensions to at least 1cm for visualization
                width = max(width, 0.01)
                depth_dim = max(depth_dim, 0.01)

                # Confidence based on depth consistency
                confidence = 0.5 + min(0.5, 1.0 - depth_std / mean_depth) if mean_depth > 0 else 0.7

                # Create spatial object with contour
                obj = SpatialObject(
                    x=center_3d[0],
                    y=center_3d[1],
                    z=center_3d[2],
                    width=width,
                    depth=depth_dim,
                    height=0.5,  # Assume standard height
                    confidence=confidence,
                    label="obstacle",
                    contour=contour,  # Store original contour
                    bbox=(x_min, y_min, x_max, y_max)  # Store bounding box
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
            grid_width = 4.0    # ±2m width
            max_dist = self.max_range
            grid_height = 0.5   # Assume camera height 0.5m above ground

            # Draw depth lines (parallel to camera, receding into distance)
            for dist in np.arange(grid_spacing, max_dist, grid_spacing):
                points_2d = []

                # Generate line from -grid_width to +grid_width at this distance
                for x in np.linspace(-grid_width / 2, grid_width / 2, 15):
                    # Project 3D point (x, -grid_height, dist) to 2D image
                    # Negative Y because ground is below camera
                    z = dist
                    x_3d = x
                    y_3d = -grid_height

                    if z > 0.1:
                        px = cx + (x_3d / z) * fx
                        py = cy - (y_3d / z) * fy

                        if 0 <= px < w and 0 <= py < h:
                            points_2d.append([px, py])

                # Draw line
                if len(points_2d) > 1:
                    points_array = np.array(points_2d, dtype=np.int32)
                    # Fade color with distance
                    intensity = int(150 * (1 - dist / max_dist)) + 50
                    color = (intensity // 2, intensity, intensity // 2)
                    cv2.polylines(canvas, [points_array], False, color, 2)

            # Draw width lines (perpendicular to camera, at different distances)
            for x in np.linspace(-grid_width / 2, grid_width / 2, 9):
                points_2d = []

                # Generate line from 0.3m to max_dist
                for dist in np.linspace(0.3, max_dist, 20):
                    z = dist
                    x_3d = x
                    y_3d = -grid_height

                    if z > 0.1:
                        px = cx + (x_3d / z) * fx
                        py = cy - (y_3d / z) * fy

                        if 0 <= px < w and 0 <= py < h:
                            points_2d.append([px, py])

                # Draw line
                if len(points_2d) > 1:
                    points_array = np.array(points_2d, dtype=np.int32)
                    color = (100, 150, 100)
                    cv2.polylines(canvas, [points_array], False, color, 1)

            # Draw center line (forward direction)
            cv2.line(canvas, (int(cx), int(cy)), (int(cx), int(cy - 200)), (0, 255, 0), 2)

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
        """Draw contour and ground plane label for detected object."""
        try:
            h, w = canvas.shape[:2]

            if obj.z <= 0:
                return canvas

            color = (0, 255, 0) if obj.confidence > 0.5 else (0, 165, 255)

            # Draw the actual contour (covers entire detected region)
            if obj.contour is not None:
                cv2.drawContours(canvas, [obj.contour], 0, color, 3)

            # Draw filled overlay with transparency to highlight entire region
            if obj.contour is not None:
                overlay = canvas.copy()
                cv2.drawContours(overlay, [obj.contour], 0, color, -1)
                # Blend: semi-transparent fill
                cv2.addWeighted(overlay, 0.15, canvas, 0.85, 0, canvas)

            # Project object center to ground plane (Y=-0.5m, where ground is)
            ground_height = -0.5
            ground_point_3d = np.array([obj.x, ground_height, obj.z], dtype=np.float32)
            ground_point_2d = self._voxel_to_2d(ground_point_3d)

            if ground_point_2d is not None:
                gx, gy = ground_point_2d

                # Draw label at ground plane position
                label = f"{obj.label} {obj.z:.1f}m"

                # Draw text with background for visibility
                font = cv2.FONT_HERSHEY_SIMPLEX
                font_scale = 0.5
                thickness = 1

                text_size = cv2.getTextSize(label, font, font_scale, thickness)[0]

                # Background rect
                padding = 4
                cv2.rectangle(canvas,
                             (gx - padding, gy - text_size[1] - padding),
                             (gx + text_size[0] + padding, gy + padding),
                             (0, 0, 0), -1)

                # Text
                cv2.putText(canvas, label, (gx, gy),
                           font, font_scale, color, thickness)

                # Confidence indicator below label
                conf_label = f"conf: {obj.confidence:.2f}"
                cv2.putText(canvas, conf_label, (gx, gy + 15),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.35, color, 1)

                # Draw point on ground plane
                cv2.circle(canvas, (gx, gy), 4, color, -1)

            # Draw center point in image space
            if obj.bbox is not None:
                x_min, y_min, x_max, y_max = obj.bbox
                center_x = (x_min + x_max) // 2
                center_y = (y_min + y_max) // 2
                cv2.circle(canvas, (center_x, center_y), 3, color, -1)

            return canvas
        except Exception as e:
            logger.warning(f"Contour drawing failed: {e}")
            return canvas

    def _draw_info_overlay(
        self,
        canvas: np.ndarray,
        num_objects: int,
        depth_frame: np.ndarray
    ) -> np.ndarray:
        """Draw information overlay on canvas (semi-transparent, minimal)."""
        try:
            h, w = canvas.shape[:2]

            # Semi-transparent background for text (minimal size)
            overlay = canvas.copy()
            cv2.rectangle(overlay, (5, 5), (320, 85), (0, 0, 0), -1)
            cv2.addWeighted(overlay, 0.3, canvas, 0.7, 0, canvas)

            # Border
            cv2.rectangle(canvas, (5, 5), (320, 85), (0, 255, 0), 1)

            # Info text - compact format
            lines = [
                f"Objects: {num_objects} | Points: {self._last_point_cloud_count:,}",
                f"Range: 0.2m-{self.max_range}m | FOV: {self.fov_degrees}°",
            ]

            if depth_frame is not None:
                valid_depths = depth_frame[depth_frame > 0.1]
                if len(valid_depths) > 0:
                    mean_depth = np.mean(valid_depths)
                    lines.append(f"Mean Depth: {mean_depth:.2f}m")

            for i, line in enumerate(lines):
                cv2.putText(canvas, line, (15, 25 + i * 18),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.4, (0, 255, 0), 1)

            return canvas
        except Exception as e:
            logger.warning(f"Info overlay failed: {e}")
            return canvas
