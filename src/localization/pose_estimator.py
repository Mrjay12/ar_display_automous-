"""
Pose Estimator - Estimate 6-DoF camera pose from features and depth.

Outputs:
- Geographic position (latitude, longitude, altitude)
- Camera orientation (roll, pitch, yaw in degrees)

Method:
1. Match frame features to 3D building geometry
2. Establish 2D-3D correspondences
3. Solve PnP (Perspective-n-Point) to get camera pose
4. Transform to geographic coordinates
"""

import logging
from typing import Optional
from dataclasses import dataclass

import numpy as np
import cv2

logger = logging.getLogger(__name__)


@dataclass
class CameraPose:
    """Camera pose in world frame."""
    latitude: float
    longitude: float
    altitude: float
    roll_deg: float
    pitch_deg: float
    yaw_deg: float
    timestamp_us: int = 0


class PoseEstimator:
    """Estimate 6-DoF camera pose from features and depth."""

    def __init__(self):
        logger.info("PoseEstimator initialized")

    def estimate(
        self,
        features,
        depth_map,
        building,
        calibration,
        origin: tuple = (53.9045, 27.5615, 125.5)
    ) -> Optional[CameraPose]:
        """
        Estimate camera pose.

        Args:
            features: Extracted features from frame
            depth_map: Depth frame
            building: 3D building candidate
            calibration: Camera calibration
            origin: (lat, lon, alt) reference point

        Returns:
            CameraPose or None on failure
        """
        try:
            if features is None or len(features.keypoints) == 0:
                logger.warning("No features provided for pose estimation")
                return None

            if depth_map is None:
                logger.warning("No depth map provided for pose estimation")
                return None

            # Simple pose: return building center as estimated position
            # In production: use PnP + depth to estimate actual camera pose
            pose = CameraPose(
                latitude=building.latitude if building else 0.0,
                longitude=building.longitude if building else 0.0,
                altitude=building.altitude if building else 0.0,
                roll_deg=0.0,
                pitch_deg=0.0,
                yaw_deg=0.0
            )

            return pose

        except Exception as e:
            logger.error(f"Pose estimation failed: {e}")
            return None

    def _get_camera_matrix(self, calibration):
        """Extract camera matrix from calibration."""
        try:
            if calibration is None:
                # Default matrix for 640x400 resolution
                fx = 300.0  # Focal length in x
                fy = 300.0  # Focal length in y
                cx = 320.0  # Principal point x
                cy = 200.0  # Principal point y
                return np.array([
                    [fx, 0, cx],
                    [0, fy, cy],
                    [0, 0, 1]
                ], dtype=np.float32)

            # Extract from depthai calibration if available
            return calibration.getCameraIntrinsics() if hasattr(
                calibration, 'getCameraIntrinsics'
            ) else None

        except Exception as e:
            logger.warning(f"Could not extract camera matrix: {e}")
            return None

    def _solve_pnp(
        self,
        object_points,
        image_points,
        camera_matrix,
        dist_coeffs=None
    ) -> Optional[tuple]:
        """
        Solve PnP to estimate camera pose.

        Returns:
            (rvec, tvec) or None on failure
        """
        try:
            if len(object_points) < 4 or len(image_points) < 4:
                return None

            if dist_coeffs is None:
                dist_coeffs = np.zeros(5)

            success, rvec, tvec = cv2.solvePnP(
                object_points,
                image_points,
                camera_matrix,
                dist_coeffs,
                useExtrinsicGuess=False,
                flags=cv2.SOLVEPNP_EPNP
            )

            if success:
                return (rvec, tvec)
            return None

        except Exception as e:
            logger.error(f"PnP solving failed: {e}")
            return None

    def _rotation_vector_to_euler(self, rvec) -> tuple:
        """Convert rotation vector to Euler angles (roll, pitch, yaw in degrees)."""
        try:
            rotation_matrix, _ = cv2.Rodrigues(rvec)

            # Extract Euler angles
            sin_pitch = -rotation_matrix[2, 0]
            sin_pitch = np.clip(sin_pitch, -1.0, 1.0)
            pitch = np.arcsin(sin_pitch)

            cos_pitch = np.cos(pitch)
            if abs(cos_pitch) > 1e-6:
                roll = np.arctan2(rotation_matrix[2, 1], rotation_matrix[2, 2])
                yaw = np.arctan2(rotation_matrix[1, 0], rotation_matrix[0, 0])
            else:
                roll = 0.0
                yaw = np.arctan2(-rotation_matrix[0, 1], rotation_matrix[1, 1])

            return (
                np.degrees(roll),
                np.degrees(pitch),
                np.degrees(yaw)
            )

        except Exception as e:
            logger.warning(f"Euler angle conversion failed: {e}")
            return (0.0, 0.0, 0.0)
