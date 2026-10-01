"""
Confidence Estimator - Tracks localization confidence and detects tracking loss

Responsibilities:
- Track localization confidence over time
- Detect when tracking is lost (confidence below threshold)
- Determine when global relocalization should be triggered
- Manage confidence state machine
- Provide metrics for visualization

Input:
  - Pose estimates from tracking
  - Pose covariance (uncertainty)
  - Feature match quality
  - Motion consistency
  - Re-localization success/failure

Output:
  - Current confidence score (0-1)
  - Confidence trend (increasing/stable/decreasing)
  - Should_relocalize flag
  - Time since last good localization

State Machine:
  TRACKING → tracking confidence high, pose estimates consistent
  UNCERTAIN → confidence dropping, pose uncertainty increasing
  LOST → confidence below threshold, tracking disabled
  RELOCALIZATION → global relocalization in progress

Performance:
  - Update: <1ms per frame
  - State machine transitions: instantaneous
"""

import logging
from typing import Optional
from dataclasses import dataclass
from enum import Enum
import numpy as np
import time

logger = logging.getLogger(__name__)


class TrackingState(Enum):
    """Tracking state machine."""
    INITIALIZED = 0
    TRACKING = 1
    UNCERTAIN = 2
    LOST = 3
    RELOCALIZATION = 4


@dataclass
class ConfidenceMetrics:
    """Confidence metrics for current frame."""
    overall_confidence: float  # 0-1
    pose_confidence: float  # From pose estimator
    feature_confidence: float  # From feature tracking
    motion_confidence: float  # From motion consistency
    confidence_trend: str  # "increasing", "stable", "decreasing"
    state: TrackingState
    time_in_state_sec: float
    frames_since_good_localization: int


class ConfidenceEstimator:
    """
    Tracks localization confidence and detects tracking loss.

    Manages the confidence state machine that determines when
    tracking is lost and global relocalization should be triggered.
    """

    def __init__(
        self,
        high_confidence_threshold: float = 0.7,
        low_confidence_threshold: float = 0.3,
        uncertainty_threshold_m: float = 5.0,
        min_features_threshold: int = 10,
        relocalization_period_sec: float = 5.0,
    ):
        """
        Initialize confidence estimator.

        Args:
            high_confidence_threshold: Threshold for entering TRACKING state
            low_confidence_threshold: Threshold for entering LOST state
            uncertainty_threshold_m: Max position uncertainty before uncertain
            min_features_threshold: Minimum features for good tracking
            relocalization_period_sec: How often to attempt relocalization when lost
        """
        self.high_confidence_threshold = high_confidence_threshold
        self.low_confidence_threshold = low_confidence_threshold
        self.uncertainty_threshold_m = uncertainty_threshold_m
        self.min_features_threshold = min_features_threshold
        self.relocalization_period_sec = relocalization_period_sec

        # State tracking
        self._state = TrackingState.INITIALIZED
        self._state_start_time = time.time()
        self._confidence_history = []
        self._max_history = 100

        # Metrics
        self._current_confidence = 0.5
        self._last_relocalization_time = time.time()
        self._frames_processed = 0
        self._frames_since_good_localization = 0

        logger.info(
            f"ConfidenceEstimator initialized: high_thresh={high_confidence_threshold}, "
            f"low_thresh={low_confidence_threshold}, "
            f"uncertainty_thresh={uncertainty_threshold_m}m"
        )

    def update(
        self,
        pose_confidence: float = 0.5,
        pose_uncertainty_m: Optional[float] = None,
        feature_count: int = 0,
        feature_quality: float = 0.5,
        motion_consistency: float = 0.5,
    ) -> ConfidenceMetrics:
        """
        Update confidence based on tracking metrics.

        Args:
            pose_confidence: Confidence from pose estimator (0-1)
            pose_uncertainty_m: Position uncertainty in meters
            feature_count: Number of tracked features
            feature_quality: Average feature match quality (0-1)
            motion_consistency: How consistent motion is (0-1)

        Returns:
            ConfidenceMetrics for current frame
        """
        self._frames_processed += 1

        # Compute component confidences
        feature_confidence = min(
            1.0, feature_count / self.min_features_threshold
        ) * feature_quality

        motion_confidence = motion_consistency

        # Compute uncertainty confidence
        if pose_uncertainty_m is not None:
            uncertainty_confidence = max(
                0.0, 1.0 - (pose_uncertainty_m / self.uncertainty_threshold_m)
            )
        else:
            uncertainty_confidence = 0.5

        # Combine confidences (weighted average)
        overall_confidence = (
            pose_confidence * 0.4 +
            feature_confidence * 0.3 +
            motion_confidence * 0.2 +
            uncertainty_confidence * 0.1
        )

        overall_confidence = np.clip(overall_confidence, 0.0, 1.0)

        # Update confidence history and compute trend
        self._confidence_history.append(overall_confidence)
        if len(self._confidence_history) > self._max_history:
            self._confidence_history.pop(0)

        confidence_trend = self._compute_confidence_trend()

        # Update current confidence
        self._current_confidence = overall_confidence

        # Update state machine
        self._update_state(overall_confidence, feature_count, pose_uncertainty_m)

        # Update frame counter
        if overall_confidence >= self.high_confidence_threshold:
            self._frames_since_good_localization = 0
        else:
            self._frames_since_good_localization += 1

        # Compute time in current state
        time_in_state = time.time() - self._state_start_time

        metrics = ConfidenceMetrics(
            overall_confidence=overall_confidence,
            pose_confidence=pose_confidence,
            feature_confidence=feature_confidence,
            motion_confidence=motion_confidence,
            confidence_trend=confidence_trend,
            state=self._state,
            time_in_state_sec=time_in_state,
            frames_since_good_localization=self._frames_since_good_localization,
        )

        if self._frames_processed % 30 == 0:
            logger.debug(
                f"Confidence {self._frames_processed}: overall={overall_confidence:.2f}, "
                f"state={self._state.name}, trend={confidence_trend}"
            )

        return metrics

    def should_relocalize(self) -> bool:
        """
        Check if global relocalization should be triggered.

        Returns:
            True if relocalization should be attempted
        """
        # Relocalize if tracking is lost
        if self._state == TrackingState.LOST:
            # But throttle relocalization attempts
            time_since_last = time.time() - self._last_relocalization_time
            if time_since_last >= self.relocalization_period_sec:
                self._last_relocalization_time = time.time()
                return True

        return False

    def on_relocalization_success(self):
        """Called when relocalization succeeds."""
        self._state = TrackingState.TRACKING
        self._state_start_time = time.time()
        self._frames_since_good_localization = 0
        logger.info("Relocalization successful, back to tracking")

    def on_relocalization_failure(self):
        """Called when relocalization fails."""
        logger.warning("Relocalization failed, remaining in LOST state")

    def _update_state(
        self,
        overall_confidence: float,
        feature_count: int,
        pose_uncertainty_m: Optional[float],
    ):
        """Update the tracking state machine."""
        previous_state = self._state

        if self._state == TrackingState.INITIALIZED:
            # Transition to TRACKING if confidence is good
            if overall_confidence >= self.high_confidence_threshold and feature_count > 0:
                self._state = TrackingState.TRACKING
                logger.info("Transitioning to TRACKING state")

        elif self._state == TrackingState.TRACKING:
            # Transition to UNCERTAIN if confidence drops
            if overall_confidence < self.high_confidence_threshold:
                self._state = TrackingState.UNCERTAIN
                logger.warning("Transitioning to UNCERTAIN state")

        elif self._state == TrackingState.UNCERTAIN:
            # Transition back to TRACKING if confidence recovers
            if overall_confidence >= self.high_confidence_threshold:
                self._state = TrackingState.TRACKING
                logger.info("Confidence recovered, back to TRACKING")
            # Or go to LOST if confidence drops further
            elif overall_confidence < self.low_confidence_threshold or feature_count < 3:
                self._state = TrackingState.LOST
                logger.error("Transitioning to LOST state - tracking confidence too low")

        elif self._state == TrackingState.LOST:
            # Remain in LOST until relocalization succeeds
            # (on_relocalization_success will transition back)
            pass

        # Log state transition
        if previous_state != self._state:
            self._state_start_time = time.time()
            logger.info(
                f"State transition: {previous_state.name} → {self._state.name}, "
                f"confidence={overall_confidence:.2f}"
            )

    def _compute_confidence_trend(self) -> str:
        """
        Compute trend in confidence (increasing/stable/decreasing).

        Returns:
            "increasing", "stable", or "decreasing"
        """
        if len(self._confidence_history) < 10:
            return "stable"

        recent = np.array(self._confidence_history[-10:])
        older = np.array(self._confidence_history[-20:-10])

        recent_mean = np.mean(recent)
        older_mean = np.mean(older)
        diff = recent_mean - older_mean

        threshold = 0.05

        if diff > threshold:
            return "increasing"
        elif diff < -threshold:
            return "decreasing"
        else:
            return "stable"

    def get_statistics(self) -> dict:
        """Return confidence estimator statistics."""
        if len(self._confidence_history) == 0:
            avg_confidence = 0.0
            confidence_std = 0.0
        else:
            avg_confidence = np.mean(self._confidence_history)
            confidence_std = np.std(self._confidence_history)

        return {
            "frames_processed": self._frames_processed,
            "current_state": self._state.name,
            "current_confidence": self._current_confidence,
            "average_confidence": float(avg_confidence),
            "confidence_std": float(confidence_std),
            "frames_since_good_localization": self._frames_since_good_localization,
        }
