"""
3D-to-2D Projection Engine - Project geographic geometries into camera view

Responsibilities:
- Transform 3D map geometries through complete coordinate frame chain
- Project building edges, road boundaries, and features into image plane
- Handle occlusion testing using depth information
- Compute visibility and screen boundaries for clipping
- Maintain numeric stability across all transformations

Input:
  - Geographic features (buildings, roads) with WGS84 coordinates
  - Camera pose estimate (position + orientation in WGS84)
  - Camera calibration matrix (intrinsics)
  - Depth map for occlusion testing
  - Current RGB frame for bounds checking

Output:
  - Projected 2D pixel coordinates
  - Depth values at projected locations
  - Visibility flags (in front of camera, within bounds)
  - Transformed 3D coordinates in camera frame (for rendering)

Performance:
  - Feature projection: ~1-5 ms for 100+ buildings
  - Occlusion testing: ~5-10 ms per frame
  - Total: <20 ms for typical scene (target 30 FPS)

Failure Modes:
  - Features behind camera (z ≤ 0): Mark as not visible
  - Features outside image bounds: Mark as out of frame
  - No pose estimate: Gracefully return empty projection
  - Invalid depth data: Skip occlusion test, render anyway
  - Recovery: Log, continue with best available data

Example:
    >>> projector = ProjectionEngine(camera_intrinsics=K)
    >>> projector.set_pose(camera_pose)
    >>> buildings_2d = projector.project_buildings(buildings_list, origin)
    >>> roads_2d = projector.project_roads(roads_list, origin)
    >>> visible = projector.filter_visible(buildings_2d, rgb_frame.shape)

Reference:
- COORDINATE_FRAMES.md for frame definitions
- ARCHITECTURE.md section 8 for transformation math
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

logger = logging.getLogger(__name__)


@dataclass
class ProjectedPoint:
    """Single 3D point projected to 2D image plane."""
    pixel_x: float              # Image column (u coordinate)
    pixel_y: float              # Image row (v coordinate)
    depth_m: float              # Distance from camera (meters)
    is_visible: bool            # True if in front of camera and in bounds
    world_x: float              # Original world X coordinate
    world_y: float              # Original world Y coordinate
    world_z: float              # Original world Z coordinate
    camera_x: float             # Camera frame X coordinate
    camera_y: float             # Camera frame Y coordinate
    camera_z: float             # Camera frame Z coordinate


@dataclass
class ProjectedBuilding:
    """Projected building geometry (wireframe edges)."""
    osm_id: int
    name: Optional[str]
    height: float
    footprint_pixels: List[Tuple[float, float]]  # [[(u, v), (u, v), ...]]
    roof_pixels: List[Tuple[float, float]]
    edges: List[Tuple[int, int]]  # Pairs of vertex indices for edges
    depth_values: List[float]     # Depth at each vertex
    is_visible: bool              # At least one vertex in view
    centroid_pixel: Optional[Tuple[float, float]]
    bounding_box: Optional[Tuple[float, float, float, float]]  # (u_min, v_min, u_max, v_max)


@dataclass
class ProjectedRoad:
    """Projected road geometry (path line)."""
    osm_id: int
    name: Optional[str]
    path_pixels: List[Tuple[float, float]]  # [(u, v), (u, v), ...]
    depth_values: List[float]
    is_visible: bool
    bounding_box: Optional[Tuple[float, float, float, float]]


class ProjectionEngine:
    """
    Project 3D geographic geometries into 2D camera image plane.

    Handles full coordinate frame transformation chain:
    GLOBAL (WGS84) → LOCAL (ENU) → CAMERA → IMAGE (pixels)

    Uses camera intrinsics and pose estimate to accurately project
    map features into the current camera view.
    """

    def __init__(
        self,
        camera_intrinsics: Optional[np.ndarray] = None,
        distortion_coeffs: Optional[np.ndarray] = None,
        depth_threshold_m: float = 0.1,
        image_bounds_margin_px: int = 10,
    ):
        """
        Initialize projection engine.

        Args:
            camera_intrinsics: 3x3 camera matrix K
            distortion_coeffs: Distortion coefficients [k1, k2, p1, p2, k3, ...]
            depth_threshold_m: Minimum depth for visibility (avoid near-plane clipping)
            image_bounds_margin_px: Pixel margin for bounds checking
        """
        self.camera_intrinsics = camera_intrinsics
        self.distortion_coeffs = distortion_coeffs if distortion_coeffs is not None else np.zeros(5)
        self.depth_threshold_m = depth_threshold_m
        self.image_bounds_margin_px = image_bounds_margin_px

        # Current pose estimate
        self._pose = None  # CameraPose object
        self._local_origin = None  # (lat, lon, alt) in WGS84
        self._R_enu_to_cam = None  # Rotation matrix
        self._t_cam = None  # Translation in camera frame

        # Statistics
        self._projection_count = 0
        self._points_projected = 0
        self._points_visible = 0

        logger.info(
            f"ProjectionEngine initialized: K={camera_intrinsics.shape if camera_intrinsics is not None else 'None'}, "
            f"distortion={self.distortion_coeffs is not None}"
        )

    def set_pose(
        self,
        camera_pose,
        local_origin: Tuple[float, float, float],
    ):
        """
        Set camera pose for all subsequent projections.

        INPUT FRAMES:
        - Camera pose: position (lat, lon, alt) in WGS84
        - Orientation: (roll, pitch, yaw) in degrees
        - Local origin: (lat, lon, alt) in WGS84 for ENU frame

        OUTPUT:
        - Stores rotation matrix R_enu_to_cam
        - Stores translation t_cam
        - Ready for projection operations

        Args:
            camera_pose: CameraPose object with lat, lon, alt, roll, pitch, yaw
            local_origin: (latitude, longitude, altitude) as reference point
        """
        self._pose = camera_pose
        self._local_origin = local_origin

        # Compute rotation matrix from Euler angles (ZYX order)
        roll_rad = np.radians(camera_pose.roll_deg)
        pitch_rad = np.radians(camera_pose.pitch_deg)
        yaw_rad = np.radians(camera_pose.yaw_deg)

        # Create rotation: Rz(yaw) * Ry(pitch) * Rx(roll)
        rot = Rotation.from_euler('ZYX', [yaw_rad, pitch_rad, roll_rad])
        self._R_enu_to_cam = rot.as_matrix().T  # Transpose for camera-to-world transform

        # Compute camera position in ENU frame (simplified)
        # TODO: Use pyproj for accurate geographic to ENU conversion
        lat_offset = (camera_pose.latitude - local_origin[0]) * 111000.0  # ~111 km per degree
        lon_offset = (camera_pose.longitude - local_origin[1]) * 111000.0 * np.cos(np.radians(local_origin[0]))
        alt_offset = camera_pose.altitude - local_origin[2]
        pos_enu = np.array([lon_offset, lat_offset, alt_offset])  # [east, north, up]

        # Camera translation
        self._t_cam = -self._R_enu_to_cam @ pos_enu

        logger.debug(
            f"Pose updated: lat={camera_pose.latitude:.6f}, lon={camera_pose.longitude:.6f}, "
            f"alt={camera_pose.altitude:.1f}m, roll={camera_pose.roll_deg:.1f}°, "
            f"pitch={camera_pose.pitch_deg:.1f}°, yaw={camera_pose.yaw_deg:.1f}°"
        )

    def project_point(
        self,
        lat: float,
        lon: float,
        alt: float,
        image_height: int = 720,
        image_width: int = 1280,
    ) -> ProjectedPoint:
        """
        Project single geographic point to image plane.

        INPUT FRAME (GLOBAL - WGS84):
        - Point: (latitude, longitude, altitude_msl)

        OUTPUT FRAME (IMAGE - pixels):
        - Pixel coordinates: (u, v)
        - Depth: distance from camera

        TRANSFORMATION CHAIN:
        1. WGS84 → ENU (using local_origin)
        2. ENU → Camera frame (using pose rotation/translation)
        3. Camera → Image (using intrinsic matrix K)

        Args:
            lat: Latitude in degrees
            lon: Longitude in degrees
            alt: Altitude in meters (MSL)
            image_height: Image height in pixels
            image_width: Image width in pixels

        Returns:
            ProjectedPoint with pixel coordinates and visibility flags
        """
        if self._pose is None or self._local_origin is None:
            logger.warning("Pose not set; cannot project")
            return ProjectedPoint(
                pixel_x=0, pixel_y=0, depth_m=0,
                is_visible=False,
                world_x=lon, world_y=lat, world_z=alt,
                camera_x=0, camera_y=0, camera_z=0
            )

        # Step 1: Convert WGS84 to ENU
        lat_offset = (lat - self._local_origin[0]) * 111000.0
        lon_offset = (lon - self._local_origin[1]) * 111000.0 * np.cos(np.radians(self._local_origin[0]))
        alt_offset = alt - self._local_origin[2]
        p_enu = np.array([lon_offset, lat_offset, alt_offset])  # [east, north, up]

        # Step 2: Transform ENU to camera frame
        p_camera = self._R_enu_to_cam @ p_enu + self._t_cam

        # Step 3: Project to image plane
        x, y, z = p_camera
        depth_m = z

        # Check depth threshold (behind camera or too close)
        is_behind_camera = z <= self.depth_threshold_m
        if is_behind_camera:
            return ProjectedPoint(
                pixel_x=0, pixel_y=0, depth_m=depth_m,
                is_visible=False,
                world_x=lon, world_y=lat, world_z=alt,
                camera_x=x, camera_y=y, camera_z=z
            )

        # Perspective projection
        u_norm = x / z
        v_norm = y / z

        # Apply distortion if coefficients provided
        if self.distortion_coeffs is not None and np.any(self.distortion_coeffs != 0):
            r2 = u_norm**2 + v_norm**2
            k1, k2, p1, p2, k3 = self.distortion_coeffs[:5]
            radial = 1 + k1*r2 + k2*r2**2 + k3*r2**3
            tangential_u = 2*p1*u_norm*v_norm + p2*(r2 + 2*u_norm**2)
            tangential_v = p1*(r2 + 2*v_norm**2) + 2*p2*u_norm*v_norm
            u_dist = u_norm * radial + tangential_u
            v_dist = v_norm * radial + tangential_v
        else:
            u_dist, v_dist = u_norm, v_norm

        # Apply intrinsic matrix
        if self.camera_intrinsics is not None:
            fx, fy = self.camera_intrinsics[0, 0], self.camera_intrinsics[1, 1]
            cx, cy = self.camera_intrinsics[0, 2], self.camera_intrinsics[1, 2]
            pixel_x = fx * u_dist + cx
            pixel_y = fy * v_dist + cy
        else:
            pixel_x, pixel_y = u_dist, v_dist

        # Check image bounds
        in_bounds = (
            0 - self.image_bounds_margin_px <= pixel_x <= image_width + self.image_bounds_margin_px and
            0 - self.image_bounds_margin_px <= pixel_y <= image_height + self.image_bounds_margin_px
        )

        self._points_projected += 1
        if in_bounds:
            self._points_visible += 1

        return ProjectedPoint(
            pixel_x=pixel_x,
            pixel_y=pixel_y,
            depth_m=depth_m,
            is_visible=in_bounds,
            world_x=lon,
            world_y=lat,
            world_z=alt,
            camera_x=x,
            camera_y=y,
            camera_z=z
        )

    def project_building(
        self,
        building,
        image_height: int = 720,
        image_width: int = 1280,
    ) -> ProjectedBuilding:
        """
        Project building geometry (footprint + roof) to 2D.

        INPUT: Building object with footprint (lat/lon vertices) and height

        OUTPUT: ProjectedBuilding with:
        - Footprint edges as pixel coordinates
        - Roof edges (raised by height)
        - Visibility flags
        - Bounding box in image space

        Args:
            building: Building object with footprint and height
            image_height: Image height in pixels
            image_width: Image width in pixels

        Returns:
            ProjectedBuilding with all geometries projected to 2D
        """
        # Project footprint vertices
        footprint_pixels = []
        depth_values = []

        for lat, lon in building.footprint:
            proj = self.project_point(lat, lon, building.latitude, image_height, image_width)
            footprint_pixels.append((proj.pixel_x, proj.pixel_y))
            depth_values.append(proj.depth_m)

        # Project roof vertices (elevated by building height)
        roof_pixels = []
        for lat, lon in building.footprint:
            proj = self.project_point(
                lat, lon,
                building.latitude + building.height,  # Altitude + building height
                image_height, image_width
            )
            roof_pixels.append((proj.pixel_x, proj.pixel_y))
            depth_values.append(proj.depth_m)

        # Determine visibility
        is_visible = any(
            p[0] > 0 and p[0] < image_width and
            p[1] > 0 and p[1] < image_height
            for p in footprint_pixels + roof_pixels
        )

        # Compute bounding box
        if footprint_pixels + roof_pixels:
            xs = [p[0] for p in footprint_pixels + roof_pixels]
            ys = [p[1] for p in footprint_pixels + roof_pixels]
            bounding_box = (min(xs), min(ys), max(xs), max(ys))
        else:
            bounding_box = None

        # Compute centroid
        if footprint_pixels:
            centroid_x = np.mean([p[0] for p in footprint_pixels])
            centroid_y = np.mean([p[1] for p in footprint_pixels])
            centroid_pixel = (centroid_x, centroid_y)
        else:
            centroid_pixel = None

        # Create edges between consecutive footprint vertices
        n_vertices = len(building.footprint)
        edges = [(i, (i + 1) % n_vertices) for i in range(n_vertices)]
        # Add vertical edges from footprint to roof
        edges.extend([(i, n_vertices + i) for i in range(n_vertices)])
        # Add roof edges
        edges.extend([(n_vertices + i, n_vertices + (i + 1) % n_vertices) for i in range(n_vertices)])

        return ProjectedBuilding(
            osm_id=building.osm_id,
            name=building.name,
            height=building.height,
            footprint_pixels=footprint_pixels,
            roof_pixels=roof_pixels,
            edges=edges,
            depth_values=depth_values,
            is_visible=is_visible,
            centroid_pixel=centroid_pixel,
            bounding_box=bounding_box
        )

    def project_buildings(
        self,
        buildings: List,
        image_height: int = 720,
        image_width: int = 1280,
    ) -> List[ProjectedBuilding]:
        """
        Project multiple buildings to 2D.

        Args:
            buildings: List of Building objects
            image_height: Image height in pixels
            image_width: Image width in pixels

        Returns:
            List of ProjectedBuilding objects
        """
        projected = []
        self._projection_count += len(buildings)

        for building in buildings:
            proj_building = self.project_building(building, image_height, image_width)
            if proj_building.is_visible:
                projected.append(proj_building)

        logger.debug(f"Projected {len(projected)}/{len(buildings)} buildings visible")
        return projected

    def project_road(
        self,
        road,
        image_height: int = 720,
        image_width: int = 1280,
    ) -> ProjectedRoad:
        """
        Project road path to 2D line.

        Args:
            road: Road object with path (list of lat/lon)
            image_height: Image height in pixels
            image_width: Image width in pixels

        Returns:
            ProjectedRoad with path projected to 2D
        """
        path_pixels = []
        depth_values = []

        for lat, lon in road.path:
            # Use road center altitude
            proj = self.project_point(lat, lon, self._local_origin[2], image_height, image_width)
            path_pixels.append((proj.pixel_x, proj.pixel_y))
            depth_values.append(proj.depth_m)

        # Determine visibility
        is_visible = any(
            p[0] > 0 and p[0] < image_width and
            p[1] > 0 and p[1] < image_height
            for p in path_pixels
        )

        # Compute bounding box
        if path_pixels:
            xs = [p[0] for p in path_pixels]
            ys = [p[1] for p in path_pixels]
            bounding_box = (min(xs), min(ys), max(xs), max(ys))
        else:
            bounding_box = None

        return ProjectedRoad(
            osm_id=road.osm_id,
            name=road.name,
            path_pixels=path_pixels,
            depth_values=depth_values,
            is_visible=is_visible,
            bounding_box=bounding_box
        )

    def project_roads(
        self,
        roads: List,
        image_height: int = 720,
        image_width: int = 1280,
    ) -> List[ProjectedRoad]:
        """
        Project multiple roads to 2D.

        Args:
            roads: List of Road objects
            image_height: Image height in pixels
            image_width: Image width in pixels

        Returns:
            List of ProjectedRoad objects
        """
        projected = []

        for road in roads:
            proj_road = self.project_road(road, image_height, image_width)
            if proj_road.is_visible:
                projected.append(proj_road)

        logger.debug(f"Projected {len(projected)}/{len(roads)} roads visible")
        return projected

    def test_occlusion(
        self,
        point_px: Tuple[float, float],
        depth_m: float,
        depth_map: np.ndarray,
        depth_scale: float = 0.001,
        occlusion_margin_m: float = 0.5,
    ) -> bool:
        """
        Test if projected point is occluded by depth map.

        INPUT:
        - Pixel location in image: (u, v)
        - Expected depth from projection: depth_m
        - Depth map from sensor: depth_map (uint16, mm units)

        OUTPUT:
        - True if point is visible (not occluded)
        - False if depth map shows closer object

        Args:
            point_px: (u, v) pixel coordinates
            depth_m: Expected depth from projection (meters)
            depth_map: Depth map from camera (uint16, typically millimeters)
            depth_scale: Scale factor for depth map (0.001 for mm→m conversion)
            occlusion_margin_m: Margin for occlusion test (objects closer than this are occluding)

        Returns:
            True if point is visible, False if occluded
        """
        u, v = int(round(point_px[0])), int(round(point_px[1]))

        # Check bounds
        if u < 0 or u >= depth_map.shape[1] or v < 0 or v >= depth_map.shape[0]:
            return True  # Out of depth map bounds, assume visible

        # Get depth value at pixel
        depth_sensor = depth_map[v, u] * depth_scale  # Convert to meters

        if depth_sensor == 0:
            return True  # No depth data, assume visible

        # Test occlusion
        is_occluded = depth_sensor < (depth_m - occlusion_margin_m)
        return not is_occluded

    def get_statistics(self) -> Dict[str, Any]:
        """Get projection statistics."""
        return {
            'projection_count': self._projection_count,
            'points_projected': self._points_projected,
            'points_visible': self._points_visible,
            'visibility_ratio': (
                self._points_visible / self._points_projected
                if self._points_projected > 0 else 0
            ),
        }

