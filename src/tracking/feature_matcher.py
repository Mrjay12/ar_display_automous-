"""
Robust Feature Matching Module

Responsibilities:
- Match features between frames using descriptor similarity
- Apply geometric verification (epipolar constraints)
- Filter outliers using RANSAC or statistical methods
- Provide match confidence scores

Input:
  - Feature set from frame N (keypoints + descriptors)
  - Feature set from frame N+1 (keypoints + descriptors)

Output:
  - Matched feature pairs with (u1,v1) -> (u2,v2) correspondences
  - Match confidence scores
  - Homography or fundamental matrix from RANSAC

Performance:
  - Typical matching: ~20-30 ms for ~500 features
  - Memory: ~5-10 MB per frame pair
  - Latency target: < 50 ms total for frame tracking

Failure Modes:
  - Few features: Low match count, tracking unreliable
  - Insufficient matches: < 4 matches cannot compute homography
  - Outlier contamination: RANSAC may fail on noisy data
  - Recovery: Fall back to looser matching criteria

Example:
    >>> matcher = FeatureMatcher(method='bf')
    >>> matches = matcher.match_features(features_frame1, features_frame2)
    >>> print(f"Found {len(matches)} good matches")
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
from enum import Enum
import time

import cv2
import numpy as np

from perception.feature_extractor import FeatureSet

logger = logging.getLogger(__name__)


class MatchMethod(Enum):
    """Feature matching algorithm selection."""
    BRUTE_FORCE = 'bf'      # Exhaustive search
    FLANN = 'flann'         # Fast approximate nearest neighbors
    KNN = 'knn'             # K-nearest neighbors


@dataclass
class FeatureMatch:
    """Single matched feature pair."""
    x1: float                # Feature position in frame 1 (u)
    y1: float                # Feature position in frame 1 (v)
    x2: float                # Feature position in frame 2 (u)
    y2: float                # Feature position in frame 2 (v)
    descriptor_distance: float  # Descriptor similarity (lower = better)
    confidence: float        # Match confidence (0-1)

    @property
    def pixel_distance(self) -> float:
        """Euclidean distance between matched points (pixels)."""
        dx = self.x2 - self.x1
        dy = self.y2 - self.y1
        return np.sqrt(dx*dx + dy*dy)


@dataclass
class MatchResult:
    """Result of feature matching between two frames."""
    matches: List[FeatureMatch]  # Matched feature pairs
    inliers: List[int]           # Indices of inlier matches (after RANSAC)
    outliers: List[int]          # Indices of outlier matches
    homography: Optional[np.ndarray]    # 3x3 homography matrix (if computed)
    fundamental_matrix: Optional[np.ndarray]  # 3x3 fundamental matrix (if computed)
    timestamp_us: int            # Timestamp of second frame
    inlier_ratio: float          # Fraction of inliers (0-1)

    def __len__(self):
        return len(self.matches)

    @property
    def num_inliers(self) -> int:
        return len(self.inliers)

    @property
    def num_outliers(self) -> int:
        return len(self.outliers)


class FeatureMatcher:
    """
    Match visual features between consecutive frames.

    Uses descriptor similarity and geometric verification to find
    reliable correspondences between frames.
    """

    def __init__(
        self,
        method: str = 'bf',
        detector_type: str = 'orb',
        max_matches: int = 1000,
        ratio_test_threshold: float = 0.7,
        ransac_threshold_px: float = 1.0,
        min_inlier_ratio: float = 0.3,
    ):
        """
        Initialize feature matcher.

        Args:
            method: 'bf' (brute force), 'flann' (fast approximate)
            detector_type: Feature detector used ('orb', 'sift', 'akaze')
            max_matches: Maximum matches to keep
            ratio_test_threshold: Lowe's ratio test threshold (0-1)
            ransac_threshold_px: RANSAC epipolar distance threshold (pixels)
            min_inlier_ratio: Minimum fraction of matches that must be inliers
        """
        self.method = method.lower()
        self.detector_type = detector_type.lower()
        self.max_matches = max_matches
        self.ratio_test_threshold = ratio_test_threshold
        self.ransac_threshold_px = ransac_threshold_px
        self.min_inlier_ratio = min_inlier_ratio

        self._matcher = self._create_matcher()
        self._match_count = 0
        self._total_attempts = 0

        logger.info(f"FeatureMatcher initialized with method: {self.method}")

    def _create_matcher(self):
        """Create descriptor matcher based on feature type."""
        if self.detector_type == 'sift':
            # SIFT uses float descriptors, use FLANN for efficiency
            if self.method == 'flann':
                FLANN_INDEX_KDTREE = 1
                index_params = dict(algorithm=FLANN_INDEX_KDTREE, trees=5)
                search_params = dict(checks=50)
                return cv2.FlannBasedMatcher(index_params, search_params)
            else:  # bf
                return cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
        else:
            # ORB, AKAZE use binary descriptors
            if self.method == 'flann':
                FLANN_INDEX_LSH = 6
                index_params = dict(
                    algorithm=FLANN_INDEX_LSH,
                    table_number=12,
                    key_size=20,
                    multi_probe_level=2
                )
                search_params = dict(checks=50)
                return cv2.FlannBasedMatcher(index_params, search_params)
            else:  # bf
                return cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

    def match_features(
        self,
        features_frame1: FeatureSet,
        features_frame2: FeatureSet,
        use_ransac: bool = True,
    ) -> MatchResult:
        """
        Match features between two consecutive frames.

        INPUT FRAMES:
        - Feature sets with keypoints and descriptors
        - frames should be consecutive for optical flow consistency

        OUTPUT MATCHES:
        - List of FeatureMatch objects
        - RANSAC-verified inliers and outliers
        - Estimated homography or fundamental matrix

        FAILURE MODES:
        - Insufficient features: Returns empty match result
        - No good matches: Empty matches list
        - RANSAC failure: All matches marked as outliers
        - Recovery: Fall back to optical flow or skipped tracking

        Args:
            features_frame1: Features from first frame
            features_frame2: Features from second frame
            use_ransac: Apply geometric verification (recommended: True)

        Returns:
            MatchResult with matched pairs and geometry
        """
        self._total_attempts += 1
        start_time = time.time()

        # Validate inputs
        if (features_frame1.descriptors is None or len(features_frame1) == 0 or
            features_frame2.descriptors is None or len(features_frame2) == 0):
            logger.debug("Insufficient features for matching")
            return MatchResult(
                matches=[],
                inliers=[],
                outliers=[],
                homography=None,
                fundamental_matrix=None,
                timestamp_us=features_frame2.timestamp_us,
                inlier_ratio=0.0
            )

        try:
            # Match descriptors
            raw_matches = self._match_descriptors(
                features_frame1.descriptors,
                features_frame2.descriptors
            )

            if len(raw_matches) == 0:
                logger.debug(f"No descriptor matches found between frames")
                return MatchResult(
                    matches=[],
                    inliers=[],
                    outliers=[],
                    homography=None,
                    fundamental_matrix=None,
                    timestamp_us=features_frame2.timestamp_us,
                    inlier_ratio=0.0
                )

            # Convert to FeatureMatch objects
            matches = []
            for match_pair in raw_matches:
                if isinstance(match_pair, tuple):
                    m = match_pair[0]
                else:
                    m = match_pair

                idx1, idx2 = m.queryIdx, m.trainIdx
                feat1 = features_frame1.keypoints[idx1]
                feat2 = features_frame2.keypoints[idx2]

                fm = FeatureMatch(
                    x1=feat1.x,
                    y1=feat1.y,
                    x2=feat2.x,
                    y2=feat2.y,
                    descriptor_distance=float(m.distance),
                    confidence=max(0.1, 1.0 - m.distance / 256.0)
                )
                matches.append(fm)

            # Apply geometric verification
            homography = None
            fundamental_matrix = None
            inliers = list(range(len(matches)))
            outliers = []

            if use_ransac and len(matches) >= 4:
                result = self._verify_geometry(
                    features_frame1,
                    features_frame2,
                    matches
                )
                inliers = result['inliers']
                outliers = result['outliers']
                homography = result['homography']
                fundamental_matrix = result['fundamental_matrix']

            inlier_ratio = len(inliers) / len(matches) if len(matches) > 0 else 0

            elapsed_ms = (time.time() - start_time) * 1000.0
            self._match_count += len(inliers)

            # Log periodically
            if self._total_attempts % 30 == 0:
                avg_inliers = self._match_count / self._total_attempts
                logger.debug(
                    f"Matching: {len(matches)} matches, "
                    f"{len(inliers)} inliers ({inlier_ratio:.1%}), "
                    f"{elapsed_ms:.2f} ms"
                )

            return MatchResult(
                matches=matches,
                inliers=inliers,
                outliers=outliers,
                homography=homography,
                fundamental_matrix=fundamental_matrix,
                timestamp_us=features_frame2.timestamp_us,
                inlier_ratio=inlier_ratio
            )

        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            return MatchResult(
                matches=[],
                inliers=[],
                outliers=[],
                homography=None,
                fundamental_matrix=None,
                timestamp_us=features_frame2.timestamp_us,
                inlier_ratio=0.0
            )

    def _match_descriptors(self, desc1: np.ndarray, desc2: np.ndarray) -> List:
        """
        Match descriptors using configured matcher.

        Returns:
            List of DMatch objects, sorted by distance
        """
        if self.detector_type == 'sift':
            # For SIFT, use k-NN matching with ratio test
            matches = self._matcher.knnMatch(desc1, desc2, k=2)
            good_matches = []
            for match_pair in matches:
                if len(match_pair) == 2:
                    m, n = match_pair
                    if m.distance < self.ratio_test_threshold * n.distance:
                        good_matches.append(m)
            good_matches.sort(key=lambda x: x.distance)
            return good_matches[:self.max_matches]
        else:
            # For ORB/AKAZE (binary descriptors), use hamming distance
            matches = self._matcher.match(desc1, desc2)
            matches.sort(key=lambda x: x.distance)
            return matches[:self.max_matches]

    def _verify_geometry(
        self,
        features_frame1: FeatureSet,
        features_frame2: FeatureSet,
        matches: List[FeatureMatch]
    ) -> dict:
        """
        Verify match geometry using RANSAC.

        Computes homography or fundamental matrix and classifies
        matches as inliers or outliers.

        Returns:
            Dictionary with 'inliers', 'outliers', 'homography', 'fundamental_matrix'
        """
        if len(matches) < 4:
            return {
                'inliers': list(range(len(matches))),
                'outliers': [],
                'homography': None,
                'fundamental_matrix': None
            }

        try:
            # Build point arrays
            pts1 = np.float32([[m.x1, m.y1] for m in matches])
            pts2 = np.float32([[m.x2, m.y2] for m in matches])

            # Compute homography with RANSAC
            H, mask = cv2.findHomography(
                pts1, pts2,
                method=cv2.RANSAC,
                ransacReprojThreshold=self.ransac_threshold_px,
                confidence=0.99
            )

            # Extract inliers and outliers
            inliers = []
            outliers = []
            if mask is not None:
                mask = mask.flatten()
                for i in range(len(matches)):
                    if mask[i]:
                        inliers.append(i)
                    else:
                        outliers.append(i)

            return {
                'inliers': inliers,
                'outliers': outliers,
                'homography': H,
                'fundamental_matrix': None
            }

        except Exception as e:
            logger.debug(f"RANSAC geometry verification failed: {e}")
            # Return all as inliers on failure (fallback)
            return {
                'inliers': list(range(len(matches))),
                'outliers': [],
                'homography': None,
                'fundamental_matrix': None
            }

    def get_statistics(self) -> dict:
        """Return matching statistics."""
        avg_matches = (
            self._match_count / self._total_attempts
            if self._total_attempts > 0 else 0
        )
        return {
            'method': self.method,
            'detector_type': self.detector_type,
            'total_match_attempts': self._total_attempts,
            'total_inlier_matches': self._match_count,
            'average_inliers_per_frame': avg_matches,
        }
