"""
Frame-to-Frame Visual Tracking Module

Responsibilities:
- Track feature correspondences across consecutive frames
- Estimate frame-to-frame optical flow
- Detect tracking failures and confidence degradation
- Maintain feature track history for trajectory estimation

Input:
  - RGB frame sequence
  - Extracted feature sets from each frame
  - Matched features between frames

Output:
  - Frame-to-frame motion vectors
  - Tracking confidence scores
  - Feature track identities (linking features across multiple frames)

Performance:
  - Typical latency: ~10-20 ms per frame
  - Memory: ~5-20 MB per tracked frame
  - Target: < 50 ms total for M2, < 33 ms for M3

Failure Modes:
  - Tracking lost: Confidence drops below threshold
  - Feature drift: Accumulated reprojection error too large
  - Insufficient features: Cannot estimate reliable motion
  - Recovery: Trigger global relocalization

Example:
    >>> tracker = FrameTracker(max_track_length=10)
    >>> matches = feature_matcher.match_features(feat1, feat2)
    >>> motion = tracker.estimate_motion(matches, frame1_kpts, frame2_kpts)
    >>> print(f"Motion: translation={motion.translation}, rotation={motion.rotation}")
"""

import logging
from typing import Optional, List, Tuple, Dict
from dataclasses import dataclass, field
from collections import defaultdict
import time

import cv2
import numpy as np
from scipy.spatial.transform import Rotation

from tracking.feature_matcher import FeatureMatch, MatchResult
from perception.feature_extractor import Feature, FeatureSet

logger = logging.getLogger(__name__)


@dataclass
class FrameMotion:
    """Estimated motion between two frames."""
    translation: np.ndarray        # Translation vector (3,) in meters/pixels
    rotation: np.ndarray           # Rotation matrix (3, 3) or quaternion
    confidence: float              # Motion confidence (0-1)
    num_inlier_matches: int        # Number of good matches used
    reprojection_error: float      # RMS reprojection error (pixels)
    timestamp_frame1_us: int
    timestamp_frame2_us: int
    frame_id_1: int
    frame_id_2: int

    @property
    def motion_magnitude(self) -> float:
        """Euclidean norm of translation."""
        return float(np.linalg.norm(self.translation))

    @property
    def time_elapsed_sec(self) -> float:
        """Time elapsed between frames (seconds)."""
        return (self.timestamp_frame2_us - self.timestamp_frame1_us) / 1_000_000.0


@dataclass
class TrackPoint:
    """Single point tracked across multiple frames."""
    track_id: int                  # Unique identifier for this track
    keypoint_positions: List[Tuple[float, float]] = field(default_factory=list)  # (u, v) per frame
    frame_ids: List[int] = field(default_factory=list)     # Frame ID for each position
    confidence_history: List[float] = field(default_factory=list)  # Confidence per frame
    last_seen_frame_id: int = -1
    creation_frame_id: int = -1

    @property
    def track_length(self) -> int:
        """Number of frames this point has been tracked."""
        return len(self.keypoint_positions)

    @property
    def is_active(self) -> bool:
        """Whether this track is still being updated."""
        return self.track_length > 0

    def get_displacement(self) -> Optional[Tuple[float, float]]:
        """Get displacement from first to last position."""
        if self.track_length < 2:
            return None
        x0, y0 = self.keypoint_positions[0]
        x1, y1 = self.keypoint_positions[-1]
        return (x1 - x0, y1 - y0)


class FrameTracker:
    """
    Track visual features across consecutive frames.

    Maintains feature correspondences and estimates frame-to-frame motion.
    """

    def __init__(
        self,
        max_track_length: int = 10,
        min_track_confidence: float = 0.3,
        reprojection_error_threshold: float = 2.0,
    ):
        """
        Initialize frame tracker.

        Args:
            max_track_length: Maximum number of frames to track a feature
            min_track_confidence: Minimum confidence to consider track valid
            reprojection_error_threshold: Max pixel error for tracking (pixels)
        """
        self.max_track_length = max_track_length
        self.min_track_confidence = min_track_confidence
        self.reprojection_error_threshold = reprojection_error_threshold

        self._track_id_counter = 0
        self._active_tracks: Dict[int, TrackPoint] = {}
        self._closed_tracks: List[TrackPoint] = []
        self._frame_count = 0

        logger.info("FrameTracker initialized")

    def estimate_motion(
        self,
        match_result: MatchResult,
        keypoints_frame1: List[Feature],
        keypoints_frame2: List[Feature],
        camera_intrinsics: Optional[np.ndarray] = None,
        depth_frame1: Optional[np.ndarray] = None,
    ) -> FrameMotion:
        """
        Estimate motion between two frames from matched features.

        INPUT MATCHES:
        - MatchResult with inlier correspondences
        - Keypoints from both frames
        - Optional: Camera intrinsics, depth map for scale

        OUTPUT MOTION:
        - Translation vector
        - Rotation matrix
        - Confidence score
        - Reprojection error

        FAILURE MODES:
        - Insufficient inliers: Returns zero motion with low confidence
        - Degenerate configuration: RANSAC fails, fallback to weak estimate
        - Large error: Motion estimation unreliable

        Args:
            match_result: Result from feature_matcher.match_features()
            keypoints_frame1: List of Feature objects from frame 1
            keypoints_frame2: List of Feature objects from frame 2
            camera_intrinsics: Optional 3x3 camera matrix for metric scale
            depth_frame1: Optional depth map for scale recovery

        Returns:
            FrameMotion with estimated transformation
        """
        self._frame_count += 1

        # Validate inputs
        if len(match_result.inliers) < 4:
            logger.warning(f"Insufficient inliers ({len(match_result.inliers)}) for motion estimation")
            return FrameMotion(
                translation=np.zeros(3),
                rotation=np.eye(3),
                confidence=0.0,
                num_inlier_matches=len(match_result.inliers),
                reprojection_error=float('inf'),
                timestamp_frame1_us=0,
                timestamp_frame2_us=match_result.timestamp_us,
                frame_id_1=-1,
                frame_id_2=-1
            )

        try:
            # Extract inlier points
            inlier_matches = [match_result.matches[i] for i in match_result.inliers]

            pts1 = np.float32([[m.x1, m.y1] for m in inlier_matches])
            pts2 = np.float32([[m.x2, m.y2] for m in inlier_matches])

            # Estimate homography as fallback
            if match_result.homography is not None:
                # Decompose homography into rotation and translation
                H = match_result.homography
                translation, rotation = self._decompose_homography(H, camera_intrinsics)
            else:
                # Fallback: compute motion from point correspondence
                translation, rotation = self._estimate_motion_direct(
                    pts1, pts2, camera_intrinsics
                )

            # Compute reprojection error
            reprojection_error = self._compute_reprojection_error(
                pts1, pts2, rotation, translation
            )

            # Estimate confidence
            inlier_ratio = len(match_result.inliers) / len(match_result.matches)
            error_confidence = max(0.0, 1.0 - reprojection_error / self.reprojection_error_threshold)
            confidence = inlier_ratio * error_confidence

            motion = FrameMotion(
                translation=translation,
                rotation=rotation,
                confidence=float(confidence),
                num_inlier_matches=len(match_result.inliers),
                reprojection_error=float(reprojection_error),
                timestamp_frame1_us=0,
                timestamp_frame2_us=match_result.timestamp_us,
                frame_id_1=-1,
                frame_id_2=-1
            )

            logger.debug(
                f"Motion: translation={motion.translation}, "
                f"confidence={confidence:.3f}, error={reprojection_error:.2f} px"
            )

            return motion

        except Exception as e:
            logger.error(f"Motion estimation failed: {e}")
            return FrameMotion(
                translation=np.zeros(3),
                rotation=np.eye(3),
                confidence=0.0,
                num_inlier_matches=0,
                reprojection_error=float('inf'),
                timestamp_frame1_us=0,
                timestamp_frame2_us=match_result.timestamp_us,
                frame_id_1=-1,
                frame_id_2=-1
            )

    def _decompose_homography(
        self,
        H: np.ndarray,
        camera_intrinsics: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Decompose homography into rotation and translation.

        For planar scenes or distant objects, homography encodes motion.
        """
        if camera_intrinsics is None:
            camera_intrinsics = np.array([
                [1000, 0, 640],
                [0, 1000, 360],
                [0, 0, 1]
            ], dtype=np.float32)

        try:
            # Use OpenCV's decomposeHomographyMat
            K_inv = np.linalg.inv(camera_intrinsics)
            H_normalized = K_inv @ H @ camera_intrinsics

            # Decompose
            rotations, translations, normals = cv2.decomposeHomographyMat(H_normalized, camera_intrinsics)

            if len(rotations) > 0:
                R = rotations[0]
                t = translations[0].flatten()
                return t, R
        except Exception as e:
            logger.debug(f"Homography decomposition failed: {e}")

        # Fallback
        return np.zeros(3), np.eye(3)

    def _estimate_motion_direct(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        camera_intrinsics: Optional[np.ndarray]
    ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Estimate motion directly from point correspondence.

        Uses Kabsch algorithm or SVD-based methods.
        """
        try:
            # Center points
            centroid1 = np.mean(pts1, axis=0)
            centroid2 = np.mean(pts2, axis=0)

            pts1_centered = pts1 - centroid1
            pts2_centered = pts2 - centroid2

            # SVD for optimal rotation
            H = pts1_centered.T @ pts2_centered
            U, S, Vt = np.linalg.svd(H)
            R = Vt.T @ U.T

            # Ensure proper rotation (det = +1)
            if np.linalg.det(R) < 0:
                Vt[-1, :] *= -1
                R = Vt.T @ U.T

            # Translation
            t = centroid2 - R @ centroid1

            return t, R

        except Exception as e:
            logger.debug(f"Direct motion estimation failed: {e}")
            return np.zeros(3), np.eye(3)

    def _compute_reprojection_error(
        self,
        pts1: np.ndarray,
        pts2: np.ndarray,
        R: np.ndarray,
        t: np.ndarray
    ) -> float:
        """
        Compute RMS reprojection error.

        For planar motion, measure error in reprojecting pts1 to pts2 location.
        """
        try:
            # Transform points1 using estimated motion
            pts1_transformed = (R @ pts1.T).T + t

            # Compute error (in pixel space)
            errors = np.linalg.norm(pts2 - pts1_transformed[:, :2], axis=1)
            rms_error = float(np.sqrt(np.mean(errors**2)))
            return rms_error

        except Exception:
            return float('inf')

    def create_track(self, feature: Feature, frame_id: int) -> TrackPoint:
        """Create new feature track."""
        track_id = self._track_id_counter
        self._track_id_counter += 1

        track = TrackPoint(
            track_id=track_id,
            creation_frame_id=frame_id,
            keypoint_positions=[(feature.x, feature.y)],
            frame_ids=[frame_id],
            confidence_history=[feature.confidence],
            last_seen_frame_id=frame_id
        )
        self._active_tracks[track_id] = track
        return track

    def update_track(self, track_id: int, feature: Feature, frame_id: int) -> bool:
        """Update existing track with new position."""
        if track_id not in self._active_tracks:
            return False

        track = self._active_tracks[track_id]

        # Check if track exceeds max length
        if track.track_length >= self.max_track_length:
            self._close_track(track_id)
            return False

        track.keypoint_positions.append((feature.x, feature.y))
        track.frame_ids.append(frame_id)
        track.confidence_history.append(feature.confidence)
        track.last_seen_frame_id = frame_id

        return True

    def _close_track(self, track_id: int):
        """Move track from active to closed."""
        if track_id in self._active_tracks:
            track = self._active_tracks.pop(track_id)
            self._closed_tracks.append(track)

    def get_active_tracks(self) -> List[TrackPoint]:
        """Get all currently active tracks."""
        return list(self._active_tracks.values())

    def get_closed_tracks(self) -> List[TrackPoint]:
        """Get all completed tracks."""
        return self._closed_tracks

    def get_statistics(self) -> dict:
        """Return tracking statistics."""
        return {
            'frames_processed': self._frame_count,
            'active_tracks': len(self._active_tracks),
            'closed_tracks': len(self._closed_tracks),
            'total_tracks': self._track_id_counter,
            'avg_track_length': (
                np.mean([t.track_length for t in self.get_active_tracks()])
                if self._active_tracks else 0
            )
        }
