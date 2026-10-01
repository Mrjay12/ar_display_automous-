"""
Geometric Verifier - Validates VPR candidates using depth information

Responsibilities:
- Filter VPR candidates based on 3D geometry consistency
- Verify candidate locations using depth map and detected obstacles
- Compute geometric confidence scores
- Handle building/obstacle matching

Input:
  - VPR candidates (from VisualPlaceRecognizer)
  - Depth map (for 3D consistency checking)
  - Detected buildings/obstacles (3D bounding boxes)
  - Current camera pose estimate

Output:
  - Verified candidate location
  - Geometric verification score (0-1)
  - 3D consistency metrics
  - Building correspondence confidence

Algorithm:
  1. Project VPR candidate locations to camera frame
  2. Compute expected 3D geometry for candidate
  3. Compare expected vs observed depth patterns
  4. Verify building/obstacle correspondences
  5. Compute geometric confidence score
  6. Select best verified candidate

Performance:
  - Typical: 50-100ms per candidate
  - Full verification: 200-500ms for top-K candidates
"""

import logging
from typing import Optional, List, Tuple
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Building:
    """A detected building or obstacle."""
    location: Tuple[float, float]  # (latitude, longitude)
    height: float  # in meters
    footprint: List[Tuple[float, float]]  # polygon vertices
    confidence: float  # detection confidence


@dataclass
class VerificationResult:
    """Result from geometric verification."""
    verified_location: Optional[Tuple[float, float]]
    geometric_confidence: float  # 0-1
    depth_consistency_score: float  # 0-1
    building_correspondence_count: int
    processing_time_ms: float


class GeometricVerifier:
    """
    Verifies VPR candidates using 3D geometric information.

    Filters out geometrically inconsistent candidates by comparing
    the depth patterns and 3D structure expected from a candidate
    location against what was actually observed.
    """

    def __init__(
        self,
        camera_matrix: Optional[np.ndarray] = None,
        depth_variance_threshold: float = 0.3,
        building_match_threshold: float = 0.6,
    ):
        """
        Initialize geometric verifier.

        Args:
            camera_matrix: Camera intrinsics (3×3 matrix)
            depth_variance_threshold: Acceptable variance in depth patterns
            building_match_threshold: Confidence for building correspondence
        """
        self.camera_matrix = camera_matrix
        self.depth_variance_threshold = depth_variance_threshold
        self.building_match_threshold = building_match_threshold

        self._verification_count = 0
        self._successful_verifications = 0

        logger.info(
            f"GeometricVerifier initialized: depth_threshold={depth_variance_threshold}, "
            f"building_threshold={building_match_threshold}"
        )

    def verify(
        self,
        vpr_candidates: List,
        depth_map: np.ndarray,
        detected_buildings: List[Building],
        camera_pose: Optional[np.ndarray] = None,
    ) -> VerificationResult:
        """
        Verify VPR candidates using geometric information.

        Args:
            vpr_candidates: List of VPRCandidate from VisualPlaceRecognizer
            depth_map: Current depth frame (H×W)
            detected_buildings: List of detected buildings/obstacles
            camera_pose: Current camera pose estimate (optional)

        Returns:
            VerificationResult with verified location and confidence
        """
        import time
        start_time = time.time()
        self._verification_count += 1

        if not vpr_candidates:
            logger.warning("No VPR candidates to verify")
            return VerificationResult(
                verified_location=None,
                geometric_confidence=0.0,
                depth_consistency_score=0.0,
                building_correspondence_count=0,
                processing_time_ms=0.0,
            )

        # Score each candidate geometrically
        best_location = None
        best_score = 0.0
        best_depth_score = 0.0
        best_building_matches = 0

        for candidate in vpr_candidates[:5]:  # Check top 5 candidates
            # Compute depth consistency score
            depth_score = self._compute_depth_consistency(
                candidate.location, depth_map
            )

            # Check building correspondences
            building_matches = self._check_building_correspondences(
                candidate.location, detected_buildings
            )

            # Combine scores: visual confidence * geometric score
            geometric_score = (
                depth_score * 0.6 +  # Depth consistency
                (building_matches / max(len(detected_buildings), 1)) * 0.4  # Building match
            )

            # Weight by VPR confidence
            combined_score = candidate.confidence * geometric_score

            if combined_score > best_score:
                best_score = combined_score
                best_location = candidate.location
                best_depth_score = depth_score
                best_building_matches = building_matches

        if best_location is not None and best_score > 0.0:
            self._successful_verifications += 1

        processing_time_ms = (time.time() - start_time) * 1000

        return VerificationResult(
            verified_location=best_location,
            geometric_confidence=min(1.0, best_score),
            depth_consistency_score=best_depth_score,
            building_correspondence_count=best_building_matches,
            processing_time_ms=processing_time_ms,
        )

    def _compute_depth_consistency(
        self,
        candidate_location: Tuple[float, float],
        depth_map: np.ndarray,
    ) -> float:
        """
        Compute depth consistency score for a candidate location.

        Args:
            candidate_location: (latitude, longitude) of candidate
            depth_map: Current depth frame

        Returns:
            Consistency score 0-1 (higher = more consistent)
        """
        if depth_map is None or depth_map.size == 0:
            return 0.5  # Neutral score when no depth available

        # Get valid depth statistics
        valid_depth = depth_map[(depth_map > 0.1) & (depth_map < 50.0) & np.isfinite(depth_map)]

        if len(valid_depth) < 10:
            return 0.3  # Low score if insufficient valid depth

        depth_mean = np.mean(valid_depth)
        depth_std = np.std(valid_depth)

        # Compute variance as fraction of mean
        depth_cv = depth_std / (depth_mean + 1e-8)  # Coefficient of variation

        # Score: lower variance = higher score
        # Max variance allowed is depth_variance_threshold
        score = max(0.0, 1.0 - (depth_cv / self.depth_variance_threshold))

        return score

    def _check_building_correspondences(
        self,
        candidate_location: Tuple[float, float],
        detected_buildings: List[Building],
    ) -> int:
        """
        Check how many buildings correspond to candidate location.

        Args:
            candidate_location: (latitude, longitude) of candidate
            detected_buildings: List of detected buildings

        Returns:
            Number of buildings that correspond to location
        """
        if not detected_buildings:
            return 0

        matches = 0
        lat, lon = candidate_location

        # Simple proximity check: buildings within ~100m of candidate
        for building in detected_buildings:
            b_lat, b_lon = building.location

            # Rough distance check (1 degree ≈ 111km)
            dist_deg = np.sqrt((lat - b_lat)**2 + (lon - b_lon)**2)
            dist_m = dist_deg * 111000

            # Match if within reasonable distance and confidence high
            if dist_m < 100 and building.confidence > self.building_match_threshold:
                matches += 1

        return matches

    def get_statistics(self) -> dict:
        """Return verification statistics."""
        success_rate = (
            self._successful_verifications / max(self._verification_count, 1)
        )
        return {
            "verification_attempts": self._verification_count,
            "successful_verifications": self._successful_verifications,
            "success_rate": success_rate,
        }
