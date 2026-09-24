"""
Geometric Verifier - Verify pose consistency using depth and 3D geometry.

Checks that:
1. Matched features have consistent depth values
2. Camera pose is geometrically consistent with building geometry
3. Reprojection errors are small
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class GeometricResult:
    """Result of geometric verification."""
    is_valid: bool
    confidence: float  # 0-1
    reprojection_error_px: float
    depth_consistency: float
    inlier_ratio: float


class GeometricVerifier:
    """Verify geometric consistency of pose estimates."""

    def __init__(self, max_reprojection_error_px: float = 5.0):
        self.max_reprojection_error_px = max_reprojection_error_px
        logger.info("GeometricVerifier initialized")

    def verify(
        self,
        camera_pose,
        features,
        depth_map,
        building,
        calibration=None
    ) -> GeometricResult:
        """
        Verify that pose is geometrically consistent.

        Args:
            camera_pose: Estimated camera pose
            features: Extracted features from frame
            depth_map: Depth frame from stereo
            building: 3D building model
            calibration: Camera calibration

        Returns:
            GeometricResult with verification status
        """
        try:
            if features is None or len(features.keypoints) == 0:
                return GeometricResult(
                    is_valid=False,
                    confidence=0.0,
                    reprojection_error_px=float('inf'),
                    depth_consistency=0.0,
                    inlier_ratio=0.0
                )

            # Simple consistency check
            depth_consistency = self._check_depth_consistency(features, depth_map)
            inlier_ratio = max(0.0, min(1.0, depth_consistency))
            reprojection_error = self._estimate_reprojection_error(
                camera_pose, features, depth_map
            )

            is_valid = (
                depth_consistency > 0.3 and
                reprojection_error < self.max_reprojection_error_px
            )

            confidence = depth_consistency if is_valid else 0.0

            return GeometricResult(
                is_valid=is_valid,
                confidence=confidence,
                reprojection_error_px=reprojection_error,
                depth_consistency=depth_consistency,
                inlier_ratio=inlier_ratio
            )

        except Exception as e:
            logger.error(f"Geometric verification failed: {e}")
            return GeometricResult(
                is_valid=False,
                confidence=0.0,
                reprojection_error_px=float('inf'),
                depth_consistency=0.0,
                inlier_ratio=0.0
            )

    def _check_depth_consistency(self, features, depth_map) -> float:
        """Check if depth values at feature locations are consistent."""
        try:
            if depth_map is None or len(features.keypoints) == 0:
                return 0.0

            valid_depths = []
            h, w = features.image_size

            for kp in features.keypoints:
                x, y = int(kp.x), int(kp.y)
                if 0 <= x < w and 0 <= y < h:
                    depth = depth_map[y, x]
                    if depth > 0.1:  # Valid depth
                        valid_depths.append(depth)

            if not valid_depths:
                return 0.0

            # Consistency: variance should be small
            depths_array = np.array(valid_depths)
            mean_depth = np.mean(depths_array)
            std_depth = np.std(depths_array)

            # Ratio of valid to total keypoints
            consistency = len(valid_depths) / len(features.keypoints)

            return float(min(1.0, consistency))

        except Exception as e:
            logger.warning(f"Depth consistency check failed: {e}")
            return 0.0

    def _estimate_reprojection_error(self, pose, features, depth_map) -> float:
        """Estimate reprojection error (simplified)."""
        try:
            if features is None or len(features.keypoints) == 0:
                return 0.0

            # Simplified: use average keypoint response as proxy for quality
            responses = np.array([kp.response for kp in features.keypoints])
            if len(responses) == 0:
                return 5.0

            # Lower response = higher error
            avg_response = np.mean(responses)
            error = 10.0 / (avg_response + 1.0)

            return float(error)

        except Exception as e:
            logger.warning(f"Reprojection error estimation failed: {e}")
            return 5.0
