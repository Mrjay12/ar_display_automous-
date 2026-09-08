"""
Confidence Estimation Module - Localization Quality Assessment

Responsibilities:
- Score localization quality (0-1)
- Estimate position uncertainty
- Estimate orientation uncertainty
- Propagate confidence through pipeline stages
- Detect confidence degradation (tracking loss)

Input:
  - VPR confidence scores
  - Geometric verification scores
  - Pose estimation quality metrics
  - Tracking quality (inlier ratio, feature count)
  - Map coverage/density

Output:
  - Global localization confidence (0-1)
  - Local tracking confidence (0-1)
  - Position uncertainty (meters)
  - Orientation uncertainty (degrees)
  - Relocalization trigger (boolean)

Performance:
  - Computation: ~5-10 ms
  - Negligible latency overhead

Failure Modes:
  - Conflicting scores: Use weighted average
  - Low confidence: Log and flag for relocalization
  - Recovery: Trigger full global re-localization

Example:
    >>> scorer = ConfidenceEstimator()
    >>> confidence = scorer.estimate(vpr_conf, geom_conf, pose_quality, tracking_quality)
    >>> print(f"Global confidence: {confidence.global_conf:.2f}")
    >>> if confidence.should_relocalize:
    ...     print("Triggering relocalization")

Reference:
- Fusion of multiple confidence sources
- Uncertainty propagation through pipeline stages
"""

import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import time

import numpy as np

logger = logging.getLogger(__name__)


class LocalizationState(Enum):
    """Localization state machine."""
    UNINITIALIZED = 0
    GLOBAL_LOCALIZING = 1
    TRACKING = 2
    TRACKING_DEGRADED = 3
    RELOCALIZATION = 4
    LOST = 5


@dataclass
class ConfidenceMetrics:
    """Comprehensive confidence and uncertainty metrics."""
    timestamp_us: int
    global_localization_conf: float  # 0-1, VPR + geometric + pose
    tracking_conf: float             # 0-1, feature tracking quality
    overall_conf: float              # 0-1, combined confidence

    position_uncertainty_m: float    # 1-sigma position uncertainty
    orientation_uncertainty_deg: float  # 1-sigma orientation uncertainty

    state: LocalizationState
    should_relocalize: bool          # Trigger full re-localization
    tracking_lost_duration_sec: float  # Time since tracking lost

    # Component scores (for debugging)
    vpr_component_score: float
    geometric_component_score: float
    pose_component_score: float
    tracking_component_score: float

    @property
    def is_confident(self) -> bool:
        """High confidence localization."""
        return self.overall_conf >= 0.7

    @property
    def is_uncertain(self) -> bool:
        """Low confidence, but not lost."""
        return 0.3 <= self.overall_conf < 0.7

    @property
    def is_lost(self) -> bool:
        """Localization lost."""
        return self.overall_conf < 0.3


class ConfidenceEstimator:
    """
    Estimate localization confidence and uncertainty.

    Fuses multiple confidence sources and detects when
    to trigger relocalization.
    """

    def __init__(
        self,
        vpr_weight: float = 0.2,
        geometric_weight: float = 0.3,
        pose_weight: float = 0.3,
        tracking_weight: float = 0.2,
        relocalization_threshold: float = 0.4,
        tracking_loss_timeout_sec: float = 5.0,
    ):
        """
        Initialize confidence estimator.

        Args:
            vpr_weight: Weight for VPR confidence (0-1)
            geometric_weight: Weight for geometric verification (0-1)
            pose_weight: Weight for pose estimation quality (0-1)
            tracking_weight: Weight for tracking quality (0-1)
            relocalization_threshold: Confidence threshold for relocalization trigger
            tracking_loss_timeout_sec: Duration before declaring tracking lost
        """
        # Normalize weights
        total = vpr_weight + geometric_weight + pose_weight + tracking_weight
        self.vpr_weight = vpr_weight / total
        self.geometric_weight = geometric_weight / total
        self.pose_weight = pose_weight / total
        self.tracking_weight = tracking_weight / total

        self.relocalization_threshold = relocalization_threshold
        self.tracking_loss_timeout_sec = tracking_loss_timeout_sec

        self._state = LocalizationState.UNINITIALIZED
        self._tracking_loss_time = None
        self._frame_count = 0

        logger.info(
            f"ConfidenceEstimator initialized: "
            f"weights=(vpr={self.vpr_weight:.2f}, geom={self.geometric_weight:.2f}, "
            f"pose={self.pose_weight:.2f}, track={self.tracking_weight:.2f}), "
            f"threshold={relocalization_threshold}"
        )

    def estimate(
        self,
        vpr_confidence: float = 0.0,
        geometric_confidence: float = 0.0,
        pose_quality: Optional[dict] = None,
        tracking_quality: Optional[dict] = None,
        timestamp_us: int = 0,
    ) -> ConfidenceMetrics:
        """
        Estimate overall localization confidence.

        INPUTS:
        - vpr_confidence: VPR candidate match score (0-1)
        - geometric_confidence: Geometric verification score (0-1)
        - pose_quality: Dict with PnP inlier_ratio, reprojection_error, num_matched
        - tracking_quality: Dict with feature_count, inlier_ratio, match_ratio

        OUTPUTS:
        - Combined confidence score
        - Uncertainty estimates
        - Relocalization trigger decision
        - Current state

        Args:
            vpr_confidence: Confidence from VPR stage (0-1)
            geometric_confidence: Confidence from geometric verification (0-1)
            pose_quality: Dict with pose estimation metrics
            tracking_quality: Dict with tracking metrics
            timestamp_us: Frame timestamp

        Returns:
            ConfidenceMetrics with all confidence and uncertainty information
        """
        start_time = time.time()
        self._frame_count += 1

        # Clamp inputs to [0, 1]
        vpr_conf = np.clip(vpr_confidence, 0.0, 1.0)
        geom_conf = np.clip(geometric_confidence, 0.0, 1.0)

        # Extract pose quality metrics
        pose_inlier_ratio = 0.0
        reprojection_error = float('inf')
        if pose_quality:
            pose_inlier_ratio = pose_quality.get('inlier_ratio', 0.0)
            reprojection_error = pose_quality.get('reprojection_error_px', float('inf'))

        # Convert pose quality to confidence (higher inlier ratio + lower error = higher confidence)
        pose_conf = np.clip(pose_inlier_ratio - (reprojection_error / 100.0), 0.0, 1.0)

        # Extract tracking quality metrics
        tracking_conf = 0.0
        if tracking_quality:
            feature_count = tracking_quality.get('feature_count', 0)
            inlier_ratio = tracking_quality.get('inlier_ratio', 0.0)
            # Confidence based on feature count and inlier ratio
            tracking_conf = np.clip(inlier_ratio * min(feature_count / 50.0, 1.0), 0.0, 1.0)

        # Compute overall confidence as weighted average
        overall_conf = (
            self.vpr_weight * vpr_conf +
            self.geometric_weight * geom_conf +
            self.pose_weight * pose_conf +
            self.tracking_weight * tracking_conf
        )

        # Update state machine
        old_state = self._state
        self._update_state(overall_conf, tracking_conf)

        # Estimate uncertainties based on confidence
        position_uncertainty = self._estimate_position_uncertainty(overall_conf)
        orientation_uncertainty = self._estimate_orientation_uncertainty(overall_conf)

        # Determine if relocalization is needed
        should_relocalize = (
            overall_conf < self.relocalization_threshold or
            self._state == LocalizationState.TRACKING_DEGRADED
        )

        # Track tracking loss duration
        tracking_lost_duration = 0.0
        if self._state in [LocalizationState.TRACKING_DEGRADED, LocalizationState.LOST]:
            if self._tracking_loss_time is None:
                self._tracking_loss_time = time.time()
            tracking_lost_duration = time.time() - self._tracking_loss_time
        else:
            self._tracking_loss_time = None

        elapsed_ms = (time.time() - start_time) * 1000.0

        metrics = ConfidenceMetrics(
            timestamp_us=timestamp_us,
            global_localization_conf=max(vpr_conf, geom_conf),
            tracking_conf=tracking_conf,
            overall_conf=overall_conf,
            position_uncertainty_m=position_uncertainty,
            orientation_uncertainty_deg=orientation_uncertainty,
            state=self._state,
            should_relocalize=should_relocalize,
            tracking_lost_duration_sec=tracking_lost_duration,
            vpr_component_score=vpr_conf,
            geometric_component_score=geom_conf,
            pose_component_score=pose_conf,
            tracking_component_score=tracking_conf,
        )

        # Log state transitions
        if old_state != self._state:
            logger.info(
                f"[Frame {self._frame_count}] State transition: "
                f"{old_state.name} → {self._state.name} "
                f"(confidence={overall_conf:.2f})"
            )

        if should_relocalize and old_state != LocalizationState.RELOCALIZATION:
            logger.warning(
                f"[Frame {self._frame_count}] Relocalization triggered "
                f"(confidence={overall_conf:.2f})"
            )

        logger.debug(
            f"[Frame {self._frame_count}] Confidence estimation: "
            f"vpr={vpr_conf:.2f}, geom={geom_conf:.2f}, pose={pose_conf:.2f}, "
            f"track={tracking_conf:.2f}, overall={overall_conf:.2f} ({elapsed_ms:.2f}ms)"
        )

        return metrics

    def _update_state(self, overall_conf: float, tracking_conf: float):
        """Update localization state machine."""
        if self._state == LocalizationState.UNINITIALIZED:
            if overall_conf > 0.5:
                self._state = LocalizationState.TRACKING
        elif self._state == LocalizationState.GLOBAL_LOCALIZING:
            if overall_conf > 0.7:
                self._state = LocalizationState.TRACKING
        elif self._state == LocalizationState.TRACKING:
            if overall_conf < 0.4:
                self._state = LocalizationState.TRACKING_DEGRADED
        elif self._state == LocalizationState.TRACKING_DEGRADED:
            if overall_conf > 0.6:
                self._state = LocalizationState.TRACKING
            elif overall_conf < 0.2:
                self._state = LocalizationState.LOST
        elif self._state == LocalizationState.RELOCALIZATION:
            if overall_conf > 0.7:
                self._state = LocalizationState.TRACKING
        elif self._state == LocalizationState.LOST:
            if overall_conf > 0.5:
                self._state = LocalizationState.GLOBAL_LOCALIZING

    def _estimate_position_uncertainty(self, confidence: float) -> float:
        """Estimate position uncertainty from confidence."""
        # Higher confidence → lower uncertainty
        # At confidence=1.0, uncertainty = 1 meter
        # At confidence=0.0, uncertainty = 100 meters
        return 1.0 + 99.0 * (1.0 - confidence)

    def _estimate_orientation_uncertainty(self, confidence: float) -> float:
        """Estimate orientation uncertainty from confidence."""
        # Higher confidence → lower uncertainty
        # At confidence=1.0, uncertainty = 0.5 degrees
        # At confidence=0.0, uncertainty = 30 degrees
        return 0.5 + 29.5 * (1.0 - confidence)

    def get_statistics(self) -> dict:
        """Return confidence estimation statistics."""
        return {
            'frames_processed': self._frame_count,
            'current_state': self._state.name,
            'vpr_weight': self.vpr_weight,
            'geometric_weight': self.geometric_weight,
            'pose_weight': self.pose_weight,
            'tracking_weight': self.tracking_weight,
        }
