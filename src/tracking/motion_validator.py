"""
Motion Validation and Tracking Failure Detection Module

Responsibilities:
- Detect tracking loss and degradation
- Validate estimated motion for physical plausibility
- Trigger relocalization on tracking failure
- Monitor pose confidence and uncertainty

Input:
  - Sequence of estimated poses
  - Frame-to-frame motion vectors
  - Feature tracking confidence scores
  - Scene depth statistics

Output:
  - Tracking status (valid, degraded, lost)
  - Confidence scores
  - Relocalization trigger signals
  - Diagnostic information

Performance:
  - Per-frame check: ~1-2 ms
  - Memory: ~5 MB for moving statistics
  - Part of real-time pipeline

Failure Modes:
  - Abrupt motion: May be detected as invalid (user jerked camera)
  - Textureless region: Tracking confidence drops naturally
  - Occlusion: Brief loss of tracking recoverable
  - Persistent loss: Triggers relocalization

Example:
    >>> validator = MotionValidator()
    >>> status = validator.validate_motion(pose, motion, depth_frame)
    >>> if status.tracking_lost:
    ...     print("Trigger global relocalization!")
"""

import logging
from typing import Optional, List
from dataclasses import dataclass
from collections import deque
from enum import Enum
import time

import numpy as np

from tracking.trajectory_estimator import Pose, FrameMotion

logger = logging.getLogger(__name__)


class TrackingStatus(Enum):
    """Tracking state enumeration."""
    TRACKING_OK = 'ok'           # Normal tracking, high confidence
    TRACKING_DEGRADED = 'degraded'  # Tracking works but lower confidence
    TRACKING_LOST = 'lost'       # Complete loss of tracking


@dataclass
class ValidationResult:
    """Result of motion and pose validation."""
    status: TrackingStatus        # Overall tracking status
    confidence: float             # Confidence (0-1)
    is_valid: bool                # Whether motion passes all checks
    trigger_relocalization: bool  # Whether to trigger global relocalization

    # Diagnostic flags
    low_feature_count: bool       # Fewer features than threshold
    high_reprojection_error: bool # Large reprojection error
    fast_motion: bool             # Motion seems too fast (possible jerk)
    scale_degradation: bool       # Scale estimation unreliable
    pose_instability: bool        # Pose changed too much
    depth_dropout: bool           # Insufficient depth coverage

    # Numerical diagnostics
    feature_count: int
    reprojection_error: float
    motion_magnitude: float       # ||t|| in meters
    confidence_drop: float        # Confidence change from previous frame
    diagnostic_msg: str           # Human-readable diagnostic message


class MotionValidator:
    """
    Validate motion estimates and detect tracking failures.

    Monitors multiple indicators to determine when tracking is lost
    and needs global relocalization.
    """

    def __init__(
        self,
        max_motion_per_frame_m: float = 0.5,
        min_feature_count: int = 20,
        max_reprojection_error_px: float = 2.5,
        confidence_drop_threshold: float = 0.2,
        lost_frame_threshold: int = 30,  # Frames before declaring lost
        depth_dropout_ratio: float = 0.3,
    ):
        """
        Initialize motion validator.

        Args:
            max_motion_per_frame_m: Maximum expected translation per frame (meters)
            min_feature_count: Minimum tracked features for valid motion
            max_reprojection_error_px: Maximum acceptable reprojection error
            confidence_drop_threshold: Max confidence decrease before flagging
            lost_frame_threshold: How many bad frames before declaring tracking lost
            depth_dropout_ratio: Max ratio of invalid depth before dropout
        """
        self.max_motion_per_frame_m = max_motion_per_frame_m
        self.min_feature_count = min_feature_count
        self.max_reprojection_error_px = max_reprojection_error_px
        self.confidence_drop_threshold = confidence_drop_threshold
        self.lost_frame_threshold = lost_frame_threshold
        self.depth_dropout_ratio = depth_dropout_ratio

        # State tracking
        self._confidence_history: deque = deque(maxlen=10)
        self._bad_frame_count = 0
        self._last_valid_pose: Optional[Pose] = None
        self._last_valid_motion: Optional[FrameMotion] = None
        self._frame_count = 0

        logger.info("MotionValidator initialized")

    def validate_motion(
        self,
        pose: Pose,
        motion: FrameMotion,
        depth_frame: Optional[np.ndarray] = None,
    ) -> ValidationResult:
        """
        Validate motion and pose estimate.

        INPUT:
        - Current estimated pose
        - Frame-to-frame motion vector
        - Optional depth map for coverage check

        OUTPUT:
        - ValidationResult with status and diagnostic information

        CHECKS:
        1. Feature count: >= min_feature_count
        2. Reprojection error: <= threshold
        3. Motion magnitude: <= max per frame
        4. Confidence change: smooth over time
        5. Depth coverage: not too much dropout
        6. Pose stability: not jittering

        Args:
            pose: Current camera pose
            motion: Frame-to-frame motion estimate
            depth_frame: Optional depth map for coverage analysis

        Returns:
            ValidationResult with detailed diagnostics
        """
        self._frame_count += 1
        diagnostics = []

        # Check 1: Feature count
        low_features = motion.num_inlier_matches < self.min_feature_count
        if low_features:
            diagnostics.append(f"Low features: {motion.num_inlier_matches}/{self.min_feature_count}")

        # Check 2: Reprojection error
        high_error = motion.reprojection_error > self.max_reprojection_error_px
        if high_error:
            diagnostics.append(f"High reprojection error: {motion.reprojection_error:.2f} px")

        # Check 3: Motion magnitude (unrealistic speed)
        fast_motion = motion.motion_magnitude > self.max_motion_per_frame_m
        if fast_motion:
            diagnostics.append(f"Fast motion: {motion.motion_magnitude:.3f} m/frame")

        # Check 4: Confidence trend
        self._confidence_history.append(pose.confidence)
        confidence_drop = 0.0
        if len(self._confidence_history) >= 2:
            confidence_drop = (
                self._confidence_history[-2] - self._confidence_history[-1]
            )

        confidence_degraded = confidence_drop > self.confidence_drop_threshold
        if confidence_degraded:
            diagnostics.append(f"Confidence drop: {confidence_drop:.3f}")

        # Check 5: Depth coverage
        depth_dropout = False
        if depth_frame is not None:
            valid_depth_ratio = np.sum(depth_frame > 0) / depth_frame.size
            depth_dropout = valid_depth_ratio < (1.0 - self.depth_dropout_ratio)
            if depth_dropout:
                diagnostics.append(
                    f"Depth dropout: only {valid_depth_ratio:.1%} valid"
                )

        # Check 6: Pose instability
        pose_instability = False
        if self._last_valid_pose is not None:
            pos_change = np.linalg.norm(
                pose.position - self._last_valid_pose.position
            )
            if pos_change > self.max_motion_per_frame_m:
                pose_instability = True
                diagnostics.append(f"Pose instability: {pos_change:.3f} m jump")

        # Determine status
        all_checks_pass = (
            not low_features and
            not high_error and
            not fast_motion and
            not confidence_degraded and
            not depth_dropout and
            not pose_instability
        )

        if all_checks_pass:
            status = TrackingStatus.TRACKING_OK
            self._bad_frame_count = 0
            is_valid = True
            confidence = min(1.0, pose.confidence)
        elif not (high_error or fast_motion or low_features):
            status = TrackingStatus.TRACKING_DEGRADED
            self._bad_frame_count = max(0, self._bad_frame_count - 1)
            is_valid = True
            confidence = 0.5 * pose.confidence
        else:
            status = TrackingStatus.TRACKING_DEGRADED
            self._bad_frame_count += 1
            is_valid = False
            confidence = 0.2 * pose.confidence

        # Trigger relocalization if tracking is lost
        trigger_relocalization = (
            self._bad_frame_count >= self.lost_frame_threshold or
            (fast_motion and motion.motion_magnitude > self.max_motion_per_frame_m * 2)
        )

        if trigger_relocalization:
            status = TrackingStatus.TRACKING_LOST
            logger.warning("Tracking lost: triggering relocalization")
        elif self._bad_frame_count >= self.lost_frame_threshold // 2:
            status = TrackingStatus.TRACKING_LOST

        # Build diagnostic message
        if diagnostics:
            diagnostic_msg = "; ".join(diagnostics)
        else:
            diagnostic_msg = "OK"

        # Update state
        if is_valid:
            self._last_valid_pose = pose
            self._last_valid_motion = motion

        result = ValidationResult(
            status=status,
            confidence=float(confidence),
            is_valid=is_valid,
            trigger_relocalization=trigger_relocalization,
            low_feature_count=low_features,
            high_reprojection_error=high_error,
            fast_motion=fast_motion,
            scale_degradation=confidence_drop > self.confidence_drop_threshold,
            pose_instability=pose_instability,
            depth_dropout=depth_dropout,
            feature_count=motion.num_inlier_matches,
            reprojection_error=motion.reprojection_error,
            motion_magnitude=motion.motion_magnitude,
            confidence_drop=confidence_drop,
            diagnostic_msg=diagnostic_msg
        )

        # Log based on status
        if result.status == TrackingStatus.TRACKING_OK:
            if self._frame_count % 60 == 0:
                logger.debug(f"[Frame {self._frame_count}] Tracking OK: {diagnostic_msg}")
        elif result.status == TrackingStatus.TRACKING_DEGRADED:
            logger.warning(f"[Frame {self._frame_count}] Tracking degraded: {diagnostic_msg}")
        else:
            logger.error(f"[Frame {self._frame_count}] Tracking LOST: {diagnostic_msg}")

        return result

    def reset(self):
        """Reset validator state (e.g., after relocalization)."""
        self._confidence_history.clear()
        self._bad_frame_count = 0
        self._last_valid_pose = None
        self._last_valid_motion = None
        logger.info("Motion validator reset")

    def get_diagnostics(self) -> dict:
        """Return current diagnostic state."""
        return {
            'frames_processed': self._frame_count,
            'bad_frames': self._bad_frame_count,
            'avg_confidence': (
                float(np.mean(list(self._confidence_history)))
                if self._confidence_history else 0.0
            ),
            'last_good_pose': self._last_valid_pose is not None,
        }
