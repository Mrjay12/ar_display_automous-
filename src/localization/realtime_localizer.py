"""
Real-Time Visual Localizer.

Pipeline:

    OAK-D synchronized RGB + depth
                |
                v
        Feature Extraction
                |
                v
       Visual Place Recognition
                |
                v
      Geometric Verification
                |
                v
         Pose Estimation
                |
                v
       Confidence Estimation
                |
                v
          LocalizationResult

The camera interface is expected to provide:

    camera.get_rgbd_frame()

returning an RGBDFrame containing:

    frame.rgb
    frame.depth
    frame.timestamp_us

No recording or replay is performed.
"""

from __future__ import annotations

import logging
import time
from typing import Optional, Tuple, Any

from perception.feature_extractor import FeatureExtractor
from localization.visual_place_recognition import VisualPlaceRecognizer
from localization.geometric_verifier import GeometricVerifier
from localization.pose_estimator import PoseEstimator
from localization.confidence_estimator import ConfidenceEstimator


logger = logging.getLogger(__name__)


# ============================================================================
# CAMERA POSE
# ============================================================================

class CameraPose:
    """Estimated camera pose in world coordinates."""

    def __init__(self):
        self.latitude = 0.0
        self.longitude = 0.0
        self.altitude = 0.0

        self.roll_deg = 0.0
        self.pitch_deg = 0.0
        self.yaw_deg = 0.0

        self.timestamp_us = 0


# ============================================================================
# LOCALIZATION RESULT
# ============================================================================

class LocalizationResult:
    """Result produced by one localization iteration."""

    def __init__(self):
        self.pose = CameraPose()

        self.confidence = 0.0

        self.matched_buildings = []

        self.tracking_status = "init"
        # init
        # tracking
        # lost
        # relocalized

        self.frame_time_ms = 0.0
        self.vpr_time_ms = 0.0
        self.geometric_time_ms = 0.0
        self.pose_time_ms = 0.0
        self.total_time_ms = 0.0

        self.timestamp_us = 0

        self.feature_count = 0
        self.depth_quality = 0.0
        self.vpr_confidence = 0.0
        self.geometric_confidence = 0.0
        self.pose_confidence = 0.0


# ============================================================================
# REAL-TIME LOCALIZER
# ============================================================================

class RealtimeLocalizer:
    """
    Real-time GPS-denied visual localization.

    Processing flow:

        Camera
          |
          +---- RGB
          |
          +---- aligned depth
          |
          v
        ORB/SIFT
          |
          v
        VPR
          |
          v
        Geometry
          |
          v
        PnP / pose
          |
          v
        Confidence
          |
          v
        CameraPose
    """

    def __init__(
        self,
        camera,
        map_loader,
        config: Optional[dict] = None,
        local_origin: Tuple[
            float,
            float,
            float
        ] = (53.9045, 27.5615, 125.5),
    ):
        """
        Args:
            camera:
                OAKDInterface instance.

            map_loader:
                Map3DLoader containing the geographic/3D map.

            config:
                Optional localization configuration.

            local_origin:
                Reference latitude, longitude and altitude.
        """

        self.camera = camera
        self.map_loader = map_loader
        self.local_origin = local_origin

        # --------------------------------------------------------------------
        # Configuration
        # --------------------------------------------------------------------

        default_config = {
            "vpr_top_k": 10,

            "min_vpr_confidence": 0.30,

            "min_geometric_confidence": 0.40,

            "min_pose_confidence": 0.50,

            "max_search_radius_m": 500.0,

            "detector_type": "orb",

            # Depth quality requirements.
            "min_depth_quality": 0.05,

            # Maximum depth considered useful for geometric verification.
            "max_depth_m": 50.0,

            # Minimum number of visual features.
            "min_features": 20,
        }

        if config:
            default_config.update(config)

        self.config = default_config

        # --------------------------------------------------------------------
        # Components
        # --------------------------------------------------------------------

        self.feature_extractor = FeatureExtractor(
            detector_type=self.config["detector_type"]
        )

        self.vpr = VisualPlaceRecognizer(
            top_k=self.config["vpr_top_k"]
        )

        self.geometric_verifier = GeometricVerifier()

        self.pose_estimator = PoseEstimator()

        self.confidence_estimator = ConfidenceEstimator()

        # --------------------------------------------------------------------
        # State
        # --------------------------------------------------------------------

        self.last_pose: Optional[CameraPose] = None

        self.tracking_status = "init"

        self.frame_count = 0

        self.lost_frames = 0

        self.lost_threshold = 5

        logger.info(
            "RealtimeLocalizer initialized"
        )

    # ========================================================================
    # SINGLE FRAME LOCALIZATION
    # ========================================================================

    def localize_frame(self) -> LocalizationResult:
        """
        Localize one synchronized RGB/depth frame.

        Returns:
            LocalizationResult
        """

        result = LocalizationResult()

        total_start = time.perf_counter()

        # ====================================================================
        # 1. GET SYNCHRONIZED RGB + DEPTH
        # ====================================================================

        frame_start = time.perf_counter()

        rgbd = self.camera.get_rgbd_frame()

        if rgbd is None:
            logger.warning(
                "No synchronized RGB/depth frame available"
            )

            result.tracking_status = "init"

            result.frame_time_ms = (
                time.perf_counter() - frame_start
            ) * 1000.0

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        rgb = rgbd.rgb
        depth = rgbd.depth

        result.timestamp_us = rgbd.timestamp_us

        result.frame_time_ms = (
            time.perf_counter() - frame_start
        ) * 1000.0

        # Validate RGB.
        if rgb is None or getattr(rgb, "size", 0) == 0:
            logger.warning(
                "Received empty RGB frame"
            )

            result.tracking_status = "init"

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        # Validate depth.
        if depth is None or getattr(depth, "size", 0) == 0:
            logger.warning(
                "Received empty depth frame"
            )

            result.tracking_status = "init"

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        # ====================================================================
        # 2. CAMERA CALIBRATION
        # ====================================================================

        calibration = self.camera.get_calibration()

        if calibration is None:
            logger.warning(
                "Camera calibration unavailable"
            )

        # ====================================================================
        # 3. DEPTH QUALITY
        # ====================================================================

        result.depth_quality = (
            self._estimate_depth_quality(depth)
        )

        if result.depth_quality < self.config[
            "min_depth_quality"
        ]:
            logger.warning(
                "Poor depth quality: %.3f",
                result.depth_quality,
            )

        # ====================================================================
        # 4. FEATURE EXTRACTION
        # ====================================================================

        try:
            features = self.feature_extractor.extract(
                rgb
            )

        except Exception:
            logger.exception(
                "Feature extraction failed"
            )

            result.tracking_status = "lost"

            self.lost_frames += 1

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        if features is None:
            logger.warning(
                "Feature extractor returned None"
            )

            result.tracking_status = "lost"

            self.lost_frames += 1

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        keypoints = getattr(
            features,
            "keypoints",
            None,
        )

        feature_count = (
            len(keypoints)
            if keypoints is not None
            else 0
        )

        result.feature_count = feature_count

        if feature_count < self.config[
            "min_features"
        ]:
            logger.warning(
                "Insufficient features: %d",
                feature_count,
            )

            result.tracking_status = "lost"

            self.lost_frames += 1

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        # ====================================================================
        # 5. VISUAL PLACE RECOGNITION
        # ====================================================================

        vpr_start = time.perf_counter()

        try:
            vpr_result = self.vpr.recognize(
                rgb
            )

        except Exception:
            logger.exception(
                "VPR failed"
            )

            result.tracking_status = "lost"

            self.lost_frames += 1

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        result.vpr_time_ms = (
            time.perf_counter() - vpr_start
        ) * 1000.0

        if (
            vpr_result is None
            or not getattr(
                vpr_result,
                "candidates",
                None,
            )
        ):
            logger.warning(
                "VPR: no candidate buildings found"
            )

            self.lost_frames += 1

            result.tracking_status = "lost"

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        # ====================================================================
        # 6. BEST VPR CANDIDATE
        # ====================================================================

        best_candidate = getattr(
            vpr_result,
            "best_candidate",
            None,
        )

        vpr_confidence = float(
            getattr(
                vpr_result,
                "best_confidence",
                0.0,
            )
            or 0.0
        )

        result.vpr_confidence = vpr_confidence

        if best_candidate is None:
            logger.warning(
                "VPR produced candidates but no best candidate"
            )

            self.lost_frames += 1

            result.tracking_status = "lost"

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        if vpr_confidence < self.config[
            "min_vpr_confidence"
        ]:
            logger.warning(
                "VPR confidence too low: %.3f",
                vpr_confidence,
            )

            self.lost_frames += 1

            result.tracking_status = "lost"

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        # ====================================================================
        # 7. GEOMETRIC VERIFICATION
        # ====================================================================

        geometric_start = time.perf_counter()

        geometric_result = None

        try:
            geometric_result = (
                self._run_geometric_verification(
                    rgb=rgb,
                    depth=depth,
                    features=features,
                    candidate=best_candidate,
                    calibration=calibration,
                )
            )

        except Exception:
            logger.exception(
                "Geometric verification failed"
            )

        result.geometric_time_ms = (
            time.perf_counter() - geometric_start
        ) * 1000.0

        geometric_confidence = (
            self._extract_geometric_confidence(
                geometric_result
            )
        )

        result.geometric_confidence = (
            geometric_confidence
        )

        if geometric_result is not None:
            if geometric_confidence < self.config[
                "min_geometric_confidence"
            ]:
                logger.warning(
                    "Geometric verification confidence too low: %.3f",
                    geometric_confidence,
                )

                self.lost_frames += 1

                result.tracking_status = "lost"

                result.total_time_ms = (
                    time.perf_counter() - total_start
                ) * 1000.0

                return result

        # ====================================================================
        # 8. POSE ESTIMATION
        # ====================================================================

        pose_start = time.perf_counter()

        try:
            pose = self.pose_estimator.estimate(
                features,
                depth,
                best_candidate,
                calibration,
                origin=self.local_origin,
            )

        except Exception:
            logger.exception(
                "Pose estimation failed"
            )

            pose = None

        result.pose_time_ms = (
            time.perf_counter() - pose_start
        ) * 1000.0

        if pose is None:
            logger.warning(
                "Pose estimation returned no pose"
            )

            self.lost_frames += 1

            result.tracking_status = "lost"

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        # Attach camera timestamp to pose.
        try:
            pose.timestamp_us = rgbd.timestamp_us
        except Exception:
            pass

        # ====================================================================
        # 9. POSE CONFIDENCE
        # ====================================================================

        pose_confidence = (
            self._extract_pose_confidence(
                pose
            )
        )

        result.pose_confidence = (
            pose_confidence
        )

        # ====================================================================
        # 10. FINAL CONFIDENCE
        # ====================================================================

        try:
            confidence = (
                self.confidence_estimator.estimate(
                    vpr_confidence=vpr_confidence,
                    geometric_confidence=geometric_confidence,
                    pose_confidence=pose_confidence,
                    depth_quality=result.depth_quality,
                )
            )

        except Exception:
            logger.exception(
                "Confidence estimation failed"
            )

            confidence = 0.0

        confidence = float(
            max(
                0.0,
                min(
                    1.0,
                    confidence,
                ),
            )
        )

        # ====================================================================
        # 11. CONFIDENCE GATE
        # ====================================================================

        if confidence < self.config[
            "min_pose_confidence"
        ]:
            logger.warning(
                "Localization rejected: confidence %.3f < %.3f",
                confidence,
                self.config[
                    "min_pose_confidence"
                ],
            )

            self.lost_frames += 1

            result.tracking_status = "lost"

            result.confidence = confidence

            result.total_time_ms = (
                time.perf_counter() - total_start
            ) * 1000.0

            return result

        # ====================================================================
        # 12. SUCCESS
        # ====================================================================

        previous_pose = self.last_pose

        result.pose = pose

        result.confidence = confidence

        result.matched_buildings = [
            best_candidate
        ]

        if previous_pose is None:
            result.tracking_status = "relocalized"
        else:
            result.tracking_status = "tracking"

        self.tracking_status = (
            result.tracking_status
        )

        self.lost_frames = 0

        self.last_pose = pose

        result.total_time_ms = (
            time.perf_counter() - total_start
        ) * 1000.0

        logger.info(
            "Localized: "
            "(%.6f, %.6f, %.2fm) "
            "confidence=%.3f "
            "features=%d "
            "VPR=%.1fms "
            "Geom=%.1fms "
            "Pose=%.1fms "
            "Total=%.1fms",
            pose.latitude,
            pose.longitude,
            pose.altitude,
            confidence,
            feature_count,
            result.vpr_time_ms,
            result.geometric_time_ms,
            result.pose_time_ms,
            result.total_time_ms,
        )

        return result

    # ========================================================================
    # GEOMETRIC VERIFICATION
    # ========================================================================

    def _run_geometric_verification(
        self,
        rgb,
        depth,
        features,
        candidate,
        calibration,
    ):
        """
        Run geometric verification.

        The exact GeometricVerifier API can differ between implementations.
        Try the common interfaces without fabricating a result when the
        verifier cannot actually perform verification.
        """

        verifier = self.geometric_verifier

        # ---------------------------------------------------------------
        # Preferred interface
        # ---------------------------------------------------------------

        if hasattr(verifier, "verify"):
            return verifier.verify(
                features=features,
                depth=depth,
                candidate=candidate,
                calibration=calibration,
            )

        # ---------------------------------------------------------------
        # Alternative interface
        # ---------------------------------------------------------------

        if hasattr(verifier, "verify_candidate"):
            return verifier.verify_candidate(
                features=features,
                depth=depth,
                candidate=candidate,
                calibration=calibration,
            )

        # ---------------------------------------------------------------
        # If no compatible API exists, do not invent a score.
        # ---------------------------------------------------------------

        logger.warning(
            "GeometricVerifier has no supported verification method; "
            "geometric verification skipped."
        )

        return None

    # ========================================================================
    # GEOMETRIC CONFIDENCE
    # ========================================================================

    @staticmethod
    def _extract_geometric_confidence(
        geometric_result,
    ) -> float:
        """
        Extract geometric confidence from verifier result.

        If the verifier does not return a confidence value,
        return 0 rather than inventing 0.7.
        """

        if geometric_result is None:
            return 0.0

        if isinstance(
            geometric_result,
            (float, int),
        ):
            return float(
                max(
                    0.0,
                    min(
                        1.0,
                        float(geometric_result),
                    ),
                )
            )

        for name in (
            "confidence",
            "geometric_confidence",
            "score",
        ):
            value = getattr(
                geometric_result,
                name,
                None,
            )

            if value is not None:
                try:
                    return float(
                        max(
                            0.0,
                            min(
                                1.0,
                                float(value),
                            ),
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        return 0.0

    # ========================================================================
    # POSE CONFIDENCE
    # ========================================================================

    @staticmethod
    def _extract_pose_confidence(
        pose,
    ) -> float:
        """
        Extract pose confidence.

        Do not automatically claim 0.8 confidence.
        If PoseEstimator provides an explicit confidence, use it.
        Otherwise use a conservative value derived from available
        reprojection-error information when available.
        """

        for name in (
            "confidence",
            "pose_confidence",
        ):
            value = getattr(
                pose,
                name,
                None,
            )

            if value is not None:
                try:
                    return float(
                        max(
                            0.0,
                            min(
                                1.0,
                                float(value),
                            ),
                        )
                    )
                except (
                    TypeError,
                    ValueError,
                ):
                    pass

        # Some pose estimators expose reprojection error.
        reprojection_error = getattr(
            pose,
            "reprojection_error",
            None,
        )

        if reprojection_error is not None:
            try:
                error = float(
                    reprojection_error
                )

                # 0 px error -> 1.0 confidence.
                # 10+ px error -> 0.0 confidence.
                return max(
                    0.0,
                    min(
                        1.0,
                        1.0 - error / 10.0,
                    ),
                )

            except (
                TypeError,
                ValueError,
            ):
                pass

        # No actual pose confidence is available.
        return 0.0

    # ========================================================================
    # CONTINUOUS LOCALIZATION
    # ========================================================================

    def localize_continuous(
        self,
        duration_sec: float = 60.0,
    ):
        """
        Continuously localize from the live camera.

        Args:
            duration_sec:
                Runtime in seconds.
                0 means run indefinitely.

        Yields:
            LocalizationResult
        """

        start_time = time.time()

        self.frame_count = 0

        logger.info(
            "Starting continuous localization for %ss",
            duration_sec if duration_sec > 0 else "infinite",
        )

        while True:

            # ---------------------------------------------------------------
            # Duration
            # ---------------------------------------------------------------

            if (
                duration_sec > 0
                and time.time() - start_time
                > duration_sec
            ):
                logger.info(
                    "Localization complete: %d frames",
                    self.frame_count,
                )
                break

            # ---------------------------------------------------------------
            # Process frame
            # ---------------------------------------------------------------

            result = self.localize_frame()

            self.frame_count += 1

            yield result

            # ---------------------------------------------------------------
            # Tracking loss
            # ---------------------------------------------------------------

            if (
                self.lost_frames
                > self.lost_threshold
            ):
                if self.tracking_status != "lost":
                    logger.warning(
                        "Tracking lost for %d frames",
                        self.lost_frames,
                    )

                self.tracking_status = "lost"

    # ========================================================================
    # DEPTH QUALITY
    # ========================================================================

    def _estimate_depth_quality(
        self,
        depth_map,
    ) -> float:
        """
        Estimate fraction of usable depth pixels.

        DepthAI depth is normally expressed in millimeters.
        Convert to meters before applying the configured range.
        """

        if depth_map is None:
            return 0.0

        try:
            if getattr(
                depth_map,
                "size",
                0,
            ) == 0:
                return 0.0

            depth = depth_map.astype(
                "float32",
                copy=False,
            )

            # DepthAI stereo depth is normally millimeters.
            depth_m = depth / 1000.0

            max_depth = float(
                self.config[
                    "max_depth_m"
                ]
            )

            valid = (
                np.isfinite(depth_m)
                & (depth_m > 0.1)
                & (depth_m < max_depth)
            )

            return float(
                np.count_nonzero(valid)
            ) / float(
                depth_m.size
            )

        except Exception:
            logger.exception(
                "Depth quality calculation failed"
            )
            return 0.0

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(self) -> dict:
        """Return localization statistics."""

        return {
            "frames_processed": self.frame_count,
            "tracking_status": self.tracking_status,
            "lost_frames": self.lost_frames,
            "last_pose": (
                {
                    "latitude": self.last_pose.latitude,
                    "longitude": self.last_pose.longitude,
                    "altitude": self.last_pose.altitude,
                    "roll_deg": self.last_pose.roll_deg,
                    "pitch_deg": self.last_pose.pitch_deg,
                    "yaw_deg": self.last_pose.yaw_deg,
                    "timestamp_us": self.last_pose.timestamp_us,
                }
                if self.last_pose is not None
                else None
            ),
        }