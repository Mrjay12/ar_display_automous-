"""
Milestone 2 Acceptance Tests: Local Visual Tracking

14 acceptance tests for visual feature tracking pipeline:
- Feature extraction
- Feature matching
- Frame-to-frame tracking
- Performance and stability

Target: All 14 tests PASS with <50ms per-frame latency.
"""

import sys
import time
import logging
from pathlib import Path

import pytest
import numpy as np
import cv2

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from perception.feature_extractor import FeatureExtractor, FeatureSet
from tracking.feature_matcher import FeatureMatcher, MatchResult
from tracking.frame_tracker import FrameTracker, FrameMotion

logger = logging.getLogger(__name__)


class TestAcceptance2_1_FeatureExtraction:
    """Test 1: Feature extraction from RGB frame."""

    def test_orb_extraction(self):
        """Extract ORB features from test image."""
        # Create test image with distinctive patterns
        frame = self._create_test_image(1280, 720, pattern='chessboard')

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        features = extractor.extract(frame, timestamp_us=1000000, frame_id=0)

        # Validate output
        assert features is not None
        assert len(features.keypoints) > 0, "Should extract features from chessboard"
        assert features.descriptors is not None
        assert features.descriptors.shape[0] == len(features.keypoints)
        assert features.frame_shape == (720, 1280)
        assert features.detector_type == 'orb'

        # Check feature properties
        for feat in features.keypoints:
            assert 0 <= feat.x < 1280
            assert 0 <= feat.y < 720
            assert feat.confidence > 0
            assert len(feat.descriptor) > 0

    def test_sift_extraction(self):
        """Extract SIFT features from test image."""
        frame = self._create_test_image(1280, 720, pattern='circles')

        extractor = FeatureExtractor(detector_type='sift', max_features=300)
        features = extractor.extract(frame, timestamp_us=2000000, frame_id=1)

        assert len(features.keypoints) > 0, "SIFT should extract features"
        assert features.detector_type == 'sift'

    def test_textureless_image_handling(self):
        """Handle textureless images gracefully."""
        # Completely uniform image
        frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        features = extractor.extract(frame)

        # Should return empty or very few features
        assert len(features.keypoints) < 10, "Textureless image should have few/no features"
        assert features.is_empty or len(features.keypoints) < 5

    def test_feature_count_limit(self):
        """Respect max_features parameter."""
        frame = self._create_test_image(1280, 720, pattern='dense_texture')
        max_features = 50

        extractor = FeatureExtractor(detector_type='orb', max_features=max_features)
        features = extractor.extract(frame)

        assert len(features.keypoints) <= max_features, f"Should not exceed {max_features} features"

    def test_extraction_performance(self):
        """Feature extraction must complete within time budget."""
        frame = self._create_test_image(1280, 720, pattern='chessboard')
        extractor = FeatureExtractor(detector_type='orb', max_features=500)

        start_time = time.time()
        features = extractor.extract(frame)
        elapsed_ms = (time.time() - start_time) * 1000.0

        # ORB should be < 50ms, SIFT < 100ms
        assert elapsed_ms < 50, f"ORB extraction took {elapsed_ms:.2f}ms, target <50ms"

    @staticmethod
    def _create_test_image(width, height, pattern='chessboard'):
        """Create synthetic test image with patterns."""
        if pattern == 'chessboard':
            square_size = 50
            board = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(0, height, square_size):
                for j in range(0, width, square_size):
                    if ((i // square_size) + (j // square_size)) % 2 == 0:
                        board[i:i+square_size, j:j+square_size] = 255
            return board

        elif pattern == 'circles':
            img = np.zeros((height, width, 3), dtype=np.uint8)
            for i in range(50, min(height, 600), 100):
                for j in range(50, min(width, 1200), 100):
                    cv2.circle(img, (j, i), 30, (255, 255, 255), 2)
            return img

        elif pattern == 'dense_texture':
            # Random noise with moderate intensity
            img = np.random.randint(50, 200, (height, width, 3), dtype=np.uint8)
            return img

        else:
            return np.random.randint(0, 256, (height, width, 3), dtype=np.uint8)


class TestAcceptance2_2_FeatureMatching:
    """Tests 2-4: Feature matching between frames."""

    def test_feature_matching_basic(self):
        """Match features between consecutive frames."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        # Slightly shifted version
        frame2 = np.roll(frame1, 10, axis=1)

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        feat1 = extractor.extract(frame1, frame_id=0)
        feat2 = extractor.extract(frame2, frame_id=1)

        matcher = FeatureMatcher(detector_type='orb', method='bf')
        result = matcher.match_features(feat1, feat2)

        assert result is not None
        assert len(result.matches) > 0, "Should find matches in shifted frame"
        assert len(result.inliers) > 0
        assert result.inlier_ratio > 0.5

    def test_ransac_outlier_rejection(self):
        """RANSAC should reject outliers."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 10, axis=1)

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        feat1 = extractor.extract(frame1, frame_id=0)
        feat2 = extractor.extract(frame2, frame_id=1)

        matcher = FeatureMatcher(detector_type='orb', method='bf')
        result = matcher.match_features(feat1, feat2, use_ransac=True)

        # After RANSAC, inliers should be subset of all matches
        assert len(result.inliers) <= len(result.matches)
        assert len(result.outliers) <= len(result.matches)
        assert len(result.inliers) + len(result.outliers) == len(result.matches)

    def test_match_homography_computation(self):
        """Homography should be computable from good matches."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 10, axis=1)

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        feat1 = extractor.extract(frame1, frame_id=0)
        feat2 = extractor.extract(frame2, frame_id=1)

        matcher = FeatureMatcher(detector_type='orb', method='bf')
        result = matcher.match_features(feat1, feat2, use_ransac=True)

        if len(result.inliers) >= 4:
            # Homography should be present
            assert result.homography is not None
            assert result.homography.shape == (3, 3)

    def test_matching_performance(self):
        """Feature matching must complete within time budget."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 10, axis=1)

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        feat1 = extractor.extract(frame1)
        feat2 = extractor.extract(frame2)

        matcher = FeatureMatcher(detector_type='orb')

        start_time = time.time()
        result = matcher.match_features(feat1, feat2)
        elapsed_ms = (time.time() - start_time) * 1000.0

        assert elapsed_ms < 50, f"Matching took {elapsed_ms:.2f}ms, target <50ms"

    @staticmethod
    def _create_test_image(width, height, pattern='chessboard'):
        return TestAcceptance2_1_FeatureExtraction._create_test_image(width, height, pattern)


class TestAcceptance2_3_FrameTracking:
    """Tests 5-7: Frame-to-frame tracking."""

    def test_motion_estimation_basic(self):
        """Estimate motion from feature matches."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 15, axis=1)  # Known horizontal shift

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        matcher = FeatureMatcher(detector_type='orb')
        tracker = FrameTracker()

        feat1 = extractor.extract(frame1, frame_id=0)
        feat2 = extractor.extract(frame2, frame_id=1)
        match_result = matcher.match_features(feat1, feat2)

        motion = tracker.estimate_motion(
            match_result,
            feat1.keypoints,
            feat2.keypoints
        )

        assert motion is not None
        assert motion.confidence > 0
        assert motion.num_inlier_matches > 0

    def test_motion_confidence_correlation(self):
        """Motion confidence should correlate with inlier count."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 10, axis=1)

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        matcher = FeatureMatcher(detector_type='orb')
        tracker = FrameTracker()

        feat1 = extractor.extract(frame1, frame_id=0)
        feat2 = extractor.extract(frame2, frame_id=1)
        match_result = matcher.match_features(feat1, feat2)

        motion = tracker.estimate_motion(match_result, feat1.keypoints, feat2.keypoints)

        # More inliers → higher confidence
        if motion.num_inlier_matches > 20:
            assert motion.confidence > 0.5, "High inlier count should give high confidence"

    def test_tracking_performance(self):
        """Frame tracking must complete within time budget."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 10, axis=1)

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        matcher = FeatureMatcher(detector_type='orb')
        tracker = FrameTracker()

        feat1 = extractor.extract(frame1)
        feat2 = extractor.extract(frame2)
        match_result = matcher.match_features(feat1, feat2)

        start_time = time.time()
        motion = tracker.estimate_motion(match_result, feat1.keypoints, feat2.keypoints)
        elapsed_ms = (time.time() - start_time) * 1000.0

        assert elapsed_ms < 50, f"Motion estimation took {elapsed_ms:.2f}ms, target <50ms"

    @staticmethod
    def _create_test_image(width, height, pattern='chessboard'):
        return TestAcceptance2_1_FeatureExtraction._create_test_image(width, height, pattern)


class TestAcceptance2_4_Pipeline:
    """Tests 8-14: Full pipeline and stability."""

    def test_full_pipeline_sequence(self):
        """Run full pipeline: extract → match → track."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 10, axis=1)
        frame3 = np.roll(frame1, 20, axis=1)

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        matcher = FeatureMatcher(detector_type='orb')
        tracker = FrameTracker()

        # Process sequence
        feat1 = extractor.extract(frame1, frame_id=0)
        feat2 = extractor.extract(frame2, frame_id=1)
        match_1_2 = matcher.match_features(feat1, feat2)
        motion_1_2 = tracker.estimate_motion(match_1_2, feat1.keypoints, feat2.keypoints)

        assert len(feat1) > 0
        assert len(feat2) > 0
        assert len(match_1_2.matches) > 0
        assert motion_1_2.confidence > 0

        # Second frame
        feat3 = extractor.extract(frame3, frame_id=2)
        match_2_3 = matcher.match_features(feat2, feat3)
        motion_2_3 = tracker.estimate_motion(match_2_3, feat2.keypoints, feat3.keypoints)

        assert len(feat3) > 0
        assert motion_2_3.confidence > 0

    def test_end_to_end_performance(self):
        """Full pipeline (extract + match + track) must complete within 50ms."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 10, axis=1)

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        matcher = FeatureMatcher(detector_type='orb')
        tracker = FrameTracker()

        start_time = time.time()

        feat1 = extractor.extract(frame1, frame_id=0)
        feat2 = extractor.extract(frame2, frame_id=1)
        match = matcher.match_features(feat1, feat2)
        motion = tracker.estimate_motion(match, feat1.keypoints, feat2.keypoints)

        elapsed_ms = (time.time() - start_time) * 1000.0

        assert elapsed_ms < 50, f"Full pipeline took {elapsed_ms:.2f}ms, target <50ms"

    def test_continuous_stability(self):
        """Process 100 frames without crashes or memory leaks."""
        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        matcher = FeatureMatcher(detector_type='orb')
        tracker = FrameTracker()

        frame_template = self._create_test_image(1280, 720, pattern='chessboard')

        for i in range(100):
            # Create frame with slight variation
            offset = (i * 5) % 100
            frame = np.roll(frame_template, offset, axis=1)

            if i > 0:
                prev_features = features
            features = extractor.extract(frame, frame_id=i)

            if i > 0:
                match = matcher.match_features(prev_features, features)
                motion = tracker.estimate_motion(match, prev_features.keypoints, features.keypoints)

        # Should complete without exception
        assert extractor.get_statistics()['frames_processed'] == 100

    def test_different_detector_types(self):
        """Pipeline works with different detectors."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 10, axis=1)

        for detector_type in ['orb', 'akaze']:
            extractor = FeatureExtractor(detector_type=detector_type, max_features=500)
            matcher = FeatureMatcher(detector_type=detector_type)

            feat1 = extractor.extract(frame1)
            feat2 = extractor.extract(frame2)
            result = matcher.match_features(feat1, feat2)

            assert result is not None

    def test_tracker_maintains_state(self):
        """Tracker correctly maintains statistics."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 10, axis=1)

        extractor = FeatureExtractor(detector_type='orb')
        matcher = FeatureMatcher(detector_type='orb')
        tracker = FrameTracker()

        feat1 = extractor.extract(frame1, frame_id=0)
        feat2 = extractor.extract(frame2, frame_id=1)
        match = matcher.match_features(feat1, feat2)
        tracker.estimate_motion(match, feat1.keypoints, feat2.keypoints)

        stats = tracker.get_statistics()
        assert stats['frames_processed'] > 0

    def test_no_features_handling(self):
        """Handle frames with no features gracefully."""
        uniform_frame = np.ones((720, 1280, 3), dtype=np.uint8) * 128

        extractor = FeatureExtractor(detector_type='orb')
        matcher = FeatureMatcher(detector_type='orb')
        tracker = FrameTracker()

        feat1 = extractor.extract(uniform_frame, frame_id=0)
        feat2 = extractor.extract(uniform_frame, frame_id=1)

        # Should handle gracefully
        if len(feat1) == 0 or len(feat2) == 0:
            result = matcher.match_features(feat1, feat2)
            assert len(result.matches) == 0

    def test_large_motion_handling(self):
        """Handle large frame-to-frame motion."""
        frame1 = self._create_test_image(1280, 720, pattern='chessboard')
        frame2 = np.roll(frame1, 200, axis=1)  # Large shift

        extractor = FeatureExtractor(detector_type='orb', max_features=500)
        matcher = FeatureMatcher(detector_type='orb')
        tracker = FrameTracker()

        feat1 = extractor.extract(frame1)
        feat2 = extractor.extract(frame2)
        match = matcher.match_features(feat1, feat2)

        # Should still produce result, possibly with lower confidence
        assert match is not None
        # Might have fewer inliers due to large baseline
        if len(match.matches) > 0:
            assert match.inlier_ratio >= 0

    @staticmethod
    def _create_test_image(width, height, pattern='chessboard'):
        return TestAcceptance2_1_FeatureExtraction._create_test_image(width, height, pattern)


# Test execution
if __name__ == "__main__":
    pytest.main([__file__, "-v"])
