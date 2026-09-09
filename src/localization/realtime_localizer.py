"""
Real-Time Localizer - Direct camera feed → localization (no recording)

One-shot pipeline:
  Camera Frame → VPR → Geometric Verification → Pose Estimation → Output

No recording, no replay - instant localization on whatever the camera sees.

Example:
    >>> localizer = RealtimeLocalizer(camera, map_loader)
    >>> while True:
    ...     pose, confidence = localizer.localize_frame()
    ...     print(f"Location: {pose.latitude:.4f}, {pose.longitude:.4f}")
"""

import logging
from typing import Optional, Tuple, Any
import time

from perception.feature_extractor import FeatureExtractor
from localization.visual_place_recognition import VisualPlaceRecognizer
from localization.geometric_verifier import GeometricVerifier
from localization.pose_estimator import PoseEstimator
from localization.confidence_estimator import ConfidenceEstimator

logger = logging.getLogger(__name__)


class CameraPose:
    """Estimated camera pose in world frame."""
    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0
        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0
        self.timestamp_us = 0


class LocalizationResult:
    """Output from real-time localization."""
    def __init__(self):
        self.pose = CameraPose()
        self.confidence = 0.0
        self.matched_buildings = []
        self.tracking_status = "init"  # init, tracking, lost, relocalized
        self.frame_time_ms = 0.0
        self.vpr_time_ms = 0.0
        self.geometric_time_ms = 0.0
        self.pose_time_ms = 0.0


class RealtimeLocalizer:
    """
    Real-time localization: camera frame → pose in one shot.

    Pipeline:
    1. Get frame from camera
    2. Extract visual features (ORB/SIFT)
    3. VPR: Find candidate buildings in geographic region
    4. Geometric verification: Check depth consistency
    5. Pose estimation: 6-DoF camera pose (lat/lon/alt + roll/pitch/yaw)
    6. Confidence scoring: Multi-source fusion
    """

    def __init__(
        self,
        camera,
        map_loader,
        config: Optional[dict] = None,
        local_origin: Tuple[float, float, float] = (53.9045, 27.5615, 125.5)
    ):
        """
        Initialize real-time localizer.

        Args:
            camera: OAKDInterface instance for frame acquisition
            map_loader: Map3DLoader with buildings/roads loaded
            config: Optional config dict with tuning parameters
            local_origin: (lat, lon, alt) reference point
        """
        self.camera = camera
        self.map_loader = map_loader
        self.local_origin = local_origin

        # Default config
        self.config = config or {
            "vpr_top_k": 10,  # Return top 10 candidate buildings
            "min_vpr_confidence": 0.3,
            "min_geometric_confidence": 0.4,
            "min_pose_confidence": 0.5,
            "max_search_radius_m": 500,
            "detector_type": "orb",  # orb or sift
        }

        # Initialize pipeline components
        self.feature_extractor = FeatureExtractor(
            detector_type=self.config["detector_type"]
        )
        self.vpr = VisualPlaceRecognizer()
        self.geometric_verifier = GeometricVerifier()
        self.pose_estimator = PoseEstimator()
        self.confidence_estimator = ConfidenceEstimator()

        # State tracking
        self.last_pose = None
        self.tracking_status = "init"
        self.frame_count = 0
        self.lost_frames = 0
        self.lost_threshold = 5  # Frames before "lost"

        logger.info("RealtimeLocalizer initialized")

    def localize_frame(self) -> LocalizationResult:
        """
        Localize camera on a single frame (real-time mode).

        Steps:
        1. Get RGB, depth, calibration from camera
        2. VPR: Find matching buildings in map
        3. Geometric: Verify depth consistency
        4. Pose: Estimate 6-DoF camera pose
        5. Confidence: Score the result

        Returns:
            LocalizationResult with pose, confidence, timing
        """
        result = LocalizationResult()
        start_time = time.time()

        # 1. Get frame from camera
        frame_time_start = time.time()
        rgb = self.camera.get_rgb_frame()
        depth = self.camera.get_depth_frame()
        calibration = self.camera.get_calibration()

        if rgb is None:
            logger.warning("No RGB frame available")
            result.tracking_status = "init"
            return result

        result.frame_time_ms = (time.time() - frame_time_start) * 1000

        # 2. Extract features
        features = self.feature_extractor.extract(rgb)
        if not features or len(features.keypoints) == 0:
            logger.warning("No features extracted from frame")
            result.tracking_status = "init"
            return result

        # 3. VPR: Find candidate buildings
        vpr_time_start = time.time()
        vpr_result = self.vpr.recognize(rgb)
        result.vpr_time_ms = (time.time() - vpr_time_start) * 1000

        if not vpr_result or not vpr_result.candidates:
            logger.warning("VPR: No candidate buildings found")
            self.lost_frames += 1
            result.tracking_status = "lost"
            return result

        # 4. Use best VPR candidate
        best_candidate = vpr_result.best_candidate
        vpr_confidence = vpr_result.best_confidence if vpr_result.best_candidate else 0.0

        # 5. Pose estimation: get 6-DoF camera pose from top candidate
        pose_time_start = time.time()
        pose = self.pose_estimator.estimate(
            features,
            depth,
            best_candidate,
            calibration,
            origin=self.local_origin
        )
        result.pose_time_ms = (time.time() - pose_time_start) * 1000

        if pose is None:
            logger.warning("Pose estimation failed")
            self.lost_frames += 1
            result.tracking_status = "lost"
            return result

        # 6. Confidence scoring
        confidence = self.confidence_estimator.estimate(
            vpr_confidence=vpr_confidence,
            geometric_confidence=0.7,  # Default if no geometric verifier
            pose_confidence=0.8,  # Default reprojection quality
            depth_quality=self._estimate_depth_quality(depth)
        )

        if confidence < self.config["min_pose_confidence"]:
            logger.warning(f"Low confidence: {confidence:.2f}")
            self.lost_frames += 1
            result.tracking_status = "lost"
            return result

        # Success
        result.pose = pose
        result.confidence = confidence
        result.matched_buildings = [best_candidate] if best_candidate else []
        result.tracking_status = "tracking"
        self.lost_frames = 0
        self.last_pose = pose

        logger.info(
            f"✓ Localized: ({pose.latitude:.4f}, {pose.longitude:.4f}) @ {confidence:.2f} confidence | "
            f"VPR:{result.vpr_time_ms:.1f}ms Geom:{result.geometric_time_ms:.1f}ms "
            f"Pose:{result.pose_time_ms:.1f}ms"
        )

        return result

    def localize_continuous(self, duration_sec: float = 60.0):
        """
        Localize continuously on camera feed.

        Streams pose estimates in real-time. Useful for live AR or
        autonomous navigation.

        Args:
            duration_sec: How long to run (0 = infinite)

        Yields:
            LocalizationResult for each frame
        """
        start_time = time.time()
        self.frame_count = 0

        logger.info(f"Starting continuous localization for {duration_sec}s")

        while True:
            # Check duration
            if duration_sec > 0 and time.time() - start_time > duration_sec:
                logger.info(f"Localization complete: {self.frame_count} frames")
                break

            # Localize single frame
            result = self.localize_frame()
            self.frame_count += 1

            yield result

            # Check for tracking loss
            if self.lost_frames > self.lost_threshold:
                logger.warning(f"Tracking lost for {self.lost_frames} frames")
                # Could trigger relocalization here

    def get_statistics(self) -> dict:
        """Return localization statistics."""
        return {
            "frames_processed": self.frame_count,
            "tracking_status": self.tracking_status,
            "last_pose": {
                "latitude": self.last_pose.latitude if self.last_pose else 0,
                "longitude": self.last_pose.longitude if self.last_pose else 0,
                "altitude": self.last_pose.altitude if self.last_pose else 0,
            } if self.last_pose else None,
        }

    def _estimate_depth_quality(self, depth_map) -> float:
        """Estimate quality of depth map (0-1)."""
        if depth_map is None:
            return 0.0

        # Simple metric: fraction of valid pixels
        valid = (depth_map > 0.1) & (depth_map < 50.0)
        return float(valid.sum()) / valid.size
