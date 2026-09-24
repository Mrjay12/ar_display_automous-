"""
Confidence Estimator - Fuse multiple signals into single confidence score.

Combines:
- VPR confidence (visual place recognition similarity)
- Geometric confidence (depth consistency, reprojection error)
- Pose confidence (pose estimation quality)
- Depth quality (fraction of valid depth pixels)
"""

import logging
from typing import Optional

import numpy as np

logger = logging.getLogger(__name__)


class ConfidenceEstimator:
    """Estimate overall localization confidence."""

    def __init__(self):
        # Fusion weights
        self.w_vpr = 0.4  # Visual place recognition
        self.w_geometric = 0.3  # Geometric consistency
        self.w_pose = 0.2  # Pose estimation quality
        self.w_depth = 0.1  # Depth map quality

        logger.info("ConfidenceEstimator initialized")

    def estimate(
        self,
        vpr_confidence: float = 0.0,
        geometric_confidence: float = 0.0,
        pose_confidence: float = 0.0,
        depth_quality: float = 0.0
    ) -> float:
        """
        Estimate overall localization confidence.

        Args:
            vpr_confidence: VPR match score (0-1)
            geometric_confidence: Geometric consistency (0-1)
            pose_confidence: Pose quality (0-1)
            depth_quality: Depth map quality (0-1)

        Returns:
            Overall confidence score (0-1)
        """
        try:
            # Normalize inputs to [0, 1]
            vpr_conf = float(np.clip(vpr_confidence, 0, 1))
            geom_conf = float(np.clip(geometric_confidence, 0, 1))
            pose_conf = float(np.clip(pose_confidence, 0, 1))
            depth_qual = float(np.clip(depth_quality, 0, 1))

            # Weighted fusion
            confidence = (
                self.w_vpr * vpr_conf +
                self.w_geometric * geom_conf +
                self.w_pose * pose_conf +
                self.w_depth * depth_qual
            )

            # Final clipping
            confidence = float(np.clip(confidence, 0, 1))

            logger.debug(
                f"Confidence: {confidence:.3f} "
                f"(VPR={vpr_conf:.2f}, Geom={geom_conf:.2f}, "
                f"Pose={pose_conf:.2f}, Depth={depth_qual:.2f})"
            )

            return confidence

        except Exception as e:
            logger.error(f"Confidence estimation failed: {e}")
            return 0.0

    def estimate_vpr_confidence(self, top_confidence: float, ambiguity: float) -> float:
        """
        Estimate VPR confidence from top match and ambiguity.

        Args:
            top_confidence: Similarity of top candidate (0-1)
            ambiguity: Ratio of 2nd to 1st candidate confidence

        Returns:
            VPR confidence (0-1)
        """
        # High confidence if:
        # - Top match has high similarity
        # - Second candidate is much weaker (low ambiguity)
        try:
            ambiguity_penalty = min(1.0, ambiguity)  # High ambiguity = low confidence
            confidence = top_confidence * (1.0 - 0.5 * ambiguity_penalty)
            return float(np.clip(confidence, 0, 1))
        except:
            return 0.0

    def estimate_pose_confidence(self, reprojection_error: float) -> float:
        """
        Estimate pose quality from reprojection error.

        Args:
            reprojection_error: Average pixel error

        Returns:
            Pose confidence (0-1)
        """
        # Reprojection error of 5px = 0.5 confidence
        # Error of 0px = 1.0 confidence
        # Error > 10px = 0.0 confidence
        try:
            error = float(np.clip(reprojection_error, 0, 10))
            confidence = 1.0 - (error / 10.0)
            return float(np.clip(confidence, 0, 1))
        except:
            return 0.0

    def estimate_depth_quality(self, depth_map) -> float:
        """
        Estimate quality of depth map.

        Args:
            depth_map: Depth frame from stereo

        Returns:
            Quality score (0-1), fraction of valid pixels
        """
        try:
            if depth_map is None:
                return 0.0

            # Valid depth: 0.1m to 50m
            valid = (depth_map > 0.1) & (depth_map < 50.0)
            quality = float(np.sum(valid)) / float(depth_map.size)

            return float(np.clip(quality, 0, 1))

        except Exception as e:
            logger.warning(f"Depth quality estimation failed: {e}")
            return 0.0
