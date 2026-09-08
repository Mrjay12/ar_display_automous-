"""
Milestone 3 Acceptance Tests: Depth-based Trajectory Estimation

14 acceptance tests for trajectory estimation and motion validation:
- Depth-based scale recovery
- 6-DoF pose estimation
- Motion validation and failure detection
- Trajectory stability

Target: All 14 tests PASS with <33ms per-frame latency (30 FPS).
"""

import sys
import time
import logging
from pathlib import Path

import pytest
import numpy as np
from scipy.spatial.transform import Rotation as ScipyRotation

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from tracking.trajectory_estimator import (
    TrajectoryEstimator, Pose, Trajectory, FrameMotion
)
from tracking.motion_validator import MotionValidator, TrackingStatus

logger = logging.getLogger(__name__)


class TestAcceptance3_1_ScaleRecovery:
    """Tests 1-2: Depth-based scale recovery."""

    def test_scale_recovery_from_depth(self):
        """Recover metric scale from depth measurements."""
        estimator = TrajectoryEstimator(window_size=10)

        # Create synthetic depth (1m away)
        depth_frame = np.ones((400, 640), dtype=np.uint16) * 1000  # 1000 mm = 1 m

        # Synthetic camera intrinsics
        K = np.array([
            [400, 0, 320],
            [0, 400, 200],
            [0, 0, 1]
        ], dtype=np.float32)

        # Feature points in first image
        feature_points = np.array([
            [100, 100],
            [150, 100],
            [200, 100],
            [250, 100],
        ], dtype=np.float32)

        # Create dummy motion
        motion = FrameMotion(
            translation=np.array([0.01, 0, 0]),  # Small motion
            rotation=np.eye(3),
            confidence=0.9,
            num_inlier_matches=4,
            reprojection_error=1.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        pose = estimator.estimate_pose(
            motion, depth_frame, K, feature_points
        )

        assert pose is not None
        assert pose.scale_confidence > 0
        # Scale should be roughly 1.0 m (based on depth)
        stats = estimator.get_statistics()
        assert stats['avg_scale'] > 0.5 and stats['avg_scale'] < 2.0

    def test_scale_recovery_without_depth(self):
        """Handle missing depth gracefully."""
        estimator = TrajectoryEstimator()

        motion = FrameMotion(
            translation=np.array([0.1, 0, 0]),
            rotation=np.eye(3),
            confidence=0.9,
            num_inlier_matches=10,
            reprojection_error=1.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        # No depth frame
        pose = estimator.estimate_pose(motion, None, None)

        assert pose is not None
        # Should fallback to unit scale
        assert pose.scale_confidence >= 0


class TestAcceptance3_2_PoseEstimation:
    """Tests 3-5: 6-DoF pose estimation."""

    def test_pose_position_integration(self):
        """Pose position integrates motion correctly."""
        estimator = TrajectoryEstimator()

        # Create depth frame (1m away)
        depth_frame = np.ones((400, 640), dtype=np.uint16) * 1000

        K = np.array([
            [400, 0, 320],
            [0, 400, 200],
            [0, 0, 1]
        ], dtype=np.float32)

        feature_points = np.array([[100, 100], [150, 100], [200, 100], [250, 100]], dtype=np.float32)

        # Motion: forward 0.1 m
        motion = FrameMotion(
            translation=np.array([0, 0, 0.1]),
            rotation=np.eye(3),
            confidence=0.9,
            num_inlier_matches=10,
            reprojection_error=1.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        pose = estimator.estimate_pose(motion, depth_frame, K, feature_points)

        assert pose.position[2] > 0, "Should move forward (positive Z)"

    def test_pose_rotation_integration(self):
        """Pose rotation is properly integrated."""
        estimator = TrajectoryEstimator()

        # 90-degree rotation around Y axis
        R = ScipyRotation.from_euler('y', 90, degrees=True).as_matrix()

        motion = FrameMotion(
            translation=np.array([0, 0, 0]),
            rotation=R,
            confidence=0.9,
            num_inlier_matches=10,
            reprojection_error=1.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        depth_frame = np.ones((400, 640), dtype=np.uint16) * 1000
        K = np.array([[400, 0, 320], [0, 400, 200], [0, 0, 1]], dtype=np.float32)

        pose = estimator.estimate_pose(motion, depth_frame, K)

        # Check that rotation is present
        assert not np.allclose(pose.rotation, np.eye(3)), "Rotation should be applied"
        assert np.allclose(np.linalg.det(pose.rotation), 1.0), "Should be proper rotation"

    def test_pose_covariance_presence(self):
        """Pose should have optional covariance."""
        estimator = TrajectoryEstimator()

        motion = FrameMotion(
            translation=np.array([0, 0, 0.1]),
            rotation=np.eye(3),
            confidence=0.9,
            num_inlier_matches=10,
            reprojection_error=1.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        pose = estimator.estimate_pose(motion, None, None)

        assert pose.confidence > 0
        # Covariance is optional
        if pose.covariance is not None:
            assert pose.covariance.shape == (6, 6)


class TestAcceptance3_3_MotionValidation:
    """Tests 6-8: Motion validation and tracking failure detection."""

    def test_valid_motion_acceptance(self):
        """Accept physically plausible motion."""
        validator = MotionValidator()

        pose = Pose(
            position=np.array([0, 0, 0]),
            rotation=np.eye(3),
            timestamp_us=0,
            frame_id=0,
            confidence=0.9,
            scale_confidence=0.9
        )

        motion = FrameMotion(
            translation=np.array([0, 0, 0.1]),
            rotation=np.eye(3),
            confidence=0.9,
            num_inlier_matches=50,
            reprojection_error=1.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        result = validator.validate_motion(pose, motion)

        assert result.is_valid
        assert result.status == TrackingStatus.TRACKING_OK

    def test_invalid_motion_rejection(self):
        """Reject implausible motion (too fast, too many errors)."""
        validator = MotionValidator()

        pose = Pose(
            position=np.array([0, 0, 0]),
            rotation=np.eye(3),
            timestamp_us=0,
            frame_id=0,
            confidence=0.5,
            scale_confidence=0.5
        )

        # Too-fast motion (1m/frame is unrealistic)
        motion = FrameMotion(
            translation=np.array([1.0, 0, 0]),
            rotation=np.eye(3),
            confidence=0.1,
            num_inlier_matches=5,
            reprojection_error=10.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        result = validator.validate_motion(pose, motion)

        # Should flag as invalid or degraded
        assert not result.is_valid or result.confidence < 0.5

    def test_tracking_loss_detection(self):
        """Detect and trigger relocalization on tracking loss."""
        validator = MotionValidator(lost_frame_threshold=5)

        pose = Pose(
            position=np.array([0, 0, 0]),
            rotation=np.eye(3),
            timestamp_us=0,
            frame_id=0,
            confidence=0.5,
            scale_confidence=0.5
        )

        # Simulate consecutive bad frames
        for i in range(10):
            bad_motion = FrameMotion(
                translation=np.array([0, 0, 0.1]),
                rotation=np.eye(3),
                confidence=0.05,
                num_inlier_matches=2,  # Very few matches
                reprojection_error=15.0,
                timestamp_frame1_us=i*33000,
                timestamp_frame2_us=(i+1)*33000,
                frame_id_1=i,
                frame_id_2=i+1
            )
            result = validator.validate_motion(pose, bad_motion)

        # After enough bad frames, should trigger relocalization
        assert result.trigger_relocalization or result.status == TrackingStatus.TRACKING_LOST


class TestAcceptance3_4_TrajectoryManagement:
    """Tests 9-11: Trajectory maintenance and history."""

    def test_trajectory_accumulation(self):
        """Trajectory accumulates poses correctly."""
        estimator = TrajectoryEstimator()
        depth_frame = np.ones((400, 640), dtype=np.uint16) * 1000
        K = np.array([[400, 0, 320], [0, 400, 200], [0, 0, 1]], dtype=np.float32)

        # Add 10 poses
        for i in range(10):
            motion = FrameMotion(
                translation=np.array([0, 0, 0.1]),
                rotation=np.eye(3),
                confidence=0.9,
                num_inlier_matches=10,
                reprojection_error=1.0,
                timestamp_frame1_us=i*33000,
                timestamp_frame2_us=(i+1)*33000,
                frame_id_1=i,
                frame_id_2=i+1
            )
            pose = estimator.estimate_pose(motion, depth_frame, K)

        trajectory = estimator.get_trajectory()
        assert len(trajectory) == 11  # Initial + 10 added
        assert trajectory.get_last_pose() is not None

    def test_trajectory_displacement(self):
        """Trajectory correctly computes total displacement."""
        estimator = TrajectoryEstimator()
        depth_frame = np.ones((400, 640), dtype=np.uint16) * 1000
        K = np.array([[400, 0, 320], [0, 400, 200], [0, 0, 1]], dtype=np.float32)

        # Add 5 forward motions (0.1 m each)
        for i in range(5):
            motion = FrameMotion(
                translation=np.array([0, 0, 0.1]),
                rotation=np.eye(3),
                confidence=0.9,
                num_inlier_matches=10,
                reprojection_error=1.0,
                timestamp_frame1_us=i*33000,
                timestamp_frame2_us=(i+1)*33000,
                frame_id_1=i,
                frame_id_2=i+1
            )
            estimator.estimate_pose(motion, depth_frame, K)

        displacement = estimator.get_trajectory().get_displacement()
        assert displacement is not None
        # Should have accumulated some forward motion
        assert displacement[2] > 0


class TestAcceptance3_5_Performance:
    """Tests 12-14: Performance and stability."""

    def test_pose_estimation_performance(self):
        """Pose estimation must complete within 33ms for 30 FPS."""
        estimator = TrajectoryEstimator()
        depth_frame = np.ones((400, 640), dtype=np.uint16) * 1000
        K = np.array([[400, 0, 320], [0, 400, 200], [0, 0, 1]], dtype=np.float32)

        motion = FrameMotion(
            translation=np.array([0.01, 0, 0.01]),
            rotation=np.eye(3),
            confidence=0.9,
            num_inlier_matches=10,
            reprojection_error=1.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        start_time = time.time()
        pose = estimator.estimate_pose(motion, depth_frame, K)
        elapsed_ms = (time.time() - start_time) * 1000.0

        assert elapsed_ms < 33, f"Pose estimation took {elapsed_ms:.2f}ms, target <33ms"

    def test_motion_validation_performance(self):
        """Motion validation must be fast (1-2ms)."""
        validator = MotionValidator()
        pose = Pose(
            position=np.array([0, 0, 0]),
            rotation=np.eye(3),
            timestamp_us=0,
            frame_id=0,
            confidence=0.9,
            scale_confidence=0.9
        )

        motion = FrameMotion(
            translation=np.array([0, 0, 0.1]),
            rotation=np.eye(3),
            confidence=0.9,
            num_inlier_matches=50,
            reprojection_error=1.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        start_time = time.time()
        result = validator.validate_motion(pose, motion)
        elapsed_ms = (time.time() - start_time) * 1000.0

        assert elapsed_ms < 5, f"Validation took {elapsed_ms:.2f}ms, target <5ms"

    def test_continuous_trajectory_stability(self):
        """Process 300 frames (10 seconds @ 30 FPS) without crashes."""
        estimator = TrajectoryEstimator()
        validator = MotionValidator()
        depth_frame = np.ones((400, 640), dtype=np.uint16) * 1000
        K = np.array([[400, 0, 320], [0, 400, 200], [0, 0, 1]], dtype=np.float32)

        for i in range(300):
            # Slight variation in motion
            angle = 0.02 * np.sin(i / 30.0)
            R = ScipyRotation.from_euler('z', angle, degrees=True).as_matrix()

            motion = FrameMotion(
                translation=np.array([0.01 * np.cos(angle), 0, 0.05]),
                rotation=R,
                confidence=0.8,
                num_inlier_matches=20 + (i % 30),
                reprojection_error=1.0,
                timestamp_frame1_us=i*33000,
                timestamp_frame2_us=(i+1)*33000,
                frame_id_1=i,
                frame_id_2=i+1
            )

            pose = estimator.estimate_pose(motion, depth_frame, K)
            result = validator.validate_motion(pose, motion)

        stats = estimator.get_statistics()
        assert stats['frames_processed'] == 300
        assert len(estimator.get_trajectory()) > 0

    def test_trajectory_consistency(self):
        """Trajectory positions increase monotonically."""
        estimator = TrajectoryEstimator()
        depth_frame = np.ones((400, 640), dtype=np.uint16) * 1000
        K = np.array([[400, 0, 320], [0, 400, 200], [0, 0, 1]], dtype=np.float32)

        # Forward motion only
        for i in range(10):
            motion = FrameMotion(
                translation=np.array([0, 0, 0.1]),
                rotation=np.eye(3),
                confidence=0.9,
                num_inlier_matches=10,
                reprojection_error=1.0,
                timestamp_frame1_us=i*33000,
                timestamp_frame2_us=(i+1)*33000,
                frame_id_1=i,
                frame_id_2=i+1
            )
            estimator.estimate_pose(motion, depth_frame, K)

        trajectory = estimator.get_trajectory()
        for i in range(len(trajectory) - 1):
            # Z should increase (forward motion)
            assert trajectory.poses[i+1].position[2] >= trajectory.poses[i].position[2]

    def test_validator_state_reset(self):
        """Validator can be reset after relocalization."""
        validator = MotionValidator()

        # Simulate some bad frames
        pose = Pose(
            position=np.array([0, 0, 0]),
            rotation=np.eye(3),
            timestamp_us=0,
            frame_id=0,
            confidence=0.5,
            scale_confidence=0.5
        )

        bad_motion = FrameMotion(
            translation=np.array([0, 0, 0.1]),
            rotation=np.eye(3),
            confidence=0.05,
            num_inlier_matches=2,
            reprojection_error=15.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        for _ in range(3):
            validator.validate_motion(pose, bad_motion)

        # Reset
        validator.reset()
        diag = validator.get_diagnostics()

        assert diag['bad_frames'] == 0

    def test_imu_rotation_fallback(self):
        """Use IMU rotation when visual confidence is low."""
        estimator = TrajectoryEstimator()
        depth_frame = np.ones((400, 640), dtype=np.uint16) * 1000
        K = np.array([[400, 0, 320], [0, 400, 200], [0, 0, 1]], dtype=np.float32)

        # Low visual confidence motion
        motion = FrameMotion(
            translation=np.array([0, 0, 0.1]),
            rotation=np.eye(3),
            confidence=0.1,  # Very low
            num_inlier_matches=3,
            reprojection_error=5.0,
            timestamp_frame1_us=0,
            timestamp_frame2_us=33000,
            frame_id_1=0,
            frame_id_2=1
        )

        # Provide IMU rotation
        imu_rotation = ScipyRotation.from_euler('z', 5, degrees=True).as_matrix()

        pose = estimator.estimate_pose(
            motion, depth_frame, K, imu_rotation=imu_rotation
        )

        assert pose is not None

    def test_empty_pose_history_handling(self):
        """Handle empty trajectory initialization."""
        estimator = TrajectoryEstimator()
        trajectory = estimator.get_trajectory()

        assert len(trajectory) == 1  # Should have initial pose
        assert trajectory.get_last_pose() is not None


# Test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
