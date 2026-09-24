"""
Feature Extractor - Extract visual keypoints and descriptors from images.

Supports:
- ORB (Oriented FAST and Rotated BRIEF) - fast, rotation-invariant
- SIFT (Scale-Invariant Feature Transform) - slower but more robust
- AKAZE (Accelerated-KAZE) - efficient alternative

Returns keypoints, descriptors, and metadata for geometric verification.
"""

import logging
from typing import Optional, List
from dataclasses import dataclass

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Keypoint:
    """Visual keypoint (corner, edge, etc.)."""
    x: float
    y: float
    size: float
    angle: float
    response: float  # Strength
    octave: int


@dataclass
class Features:
    """Extracted features from an image."""
    keypoints: List[Keypoint]
    descriptors: np.ndarray  # (N, D) descriptor matrix
    image_size: tuple  # (height, width)
    detector_type: str


class FeatureExtractor:
    """Extract visual features from images."""

    def __init__(self, detector_type: str = "orb", n_features: int = 500):
        """
        Initialize feature extractor.

        Args:
            detector_type: "orb", "sift", or "akaze"
            n_features: Number of features to extract
        """
        self.detector_type = detector_type.lower()
        self.n_features = n_features

        # Initialize detector
        if self.detector_type == "orb":
            self.detector = cv2.ORB_create(nfeatures=n_features)
            logger.info(f"ORB detector initialized (n_features={n_features})")

        elif self.detector_type == "sift":
            try:
                self.detector = cv2.SIFT_create()
                logger.info("SIFT detector initialized")
            except AttributeError:
                logger.warning("SIFT not available in OpenCV, falling back to ORB")
                self.detector = cv2.ORB_create(nfeatures=n_features)
                self.detector_type = "orb"

        elif self.detector_type == "akaze":
            self.detector = cv2.AKAZE_create()
            logger.info("AKAZE detector initialized")

        else:
            logger.warning(f"Unknown detector {detector_type}, using ORB")
            self.detector = cv2.ORB_create(nfeatures=n_features)
            self.detector_type = "orb"

    def extract(self, image: np.ndarray) -> Optional[Features]:
        """
        Extract features from image.

        Args:
            image: RGB image (H, W, 3) uint8

        Returns:
            Features object or None on failure
        """
        try:
            if image is None or image.size == 0:
                logger.warning("Empty image provided")
                return None

            # Convert to grayscale if needed
            if len(image.shape) == 3:
                gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
            else:
                gray = image

            # Detect keypoints and descriptors
            kp, desc = self.detector.detectAndCompute(gray, None)

            if len(kp) == 0:
                logger.warning(f"No features detected in image {gray.shape}")
                return Features(
                    keypoints=[],
                    descriptors=np.array([]),
                    image_size=gray.shape[:2],
                    detector_type=self.detector_type
                )

            # Convert cv2 keypoints to our format
            keypoints = []
            for k in kp:
                keypoints.append(Keypoint(
                    x=k.pt[0],
                    y=k.pt[1],
                    size=k.size,
                    angle=k.angle,
                    response=k.response,
                    octave=k.octave
                ))

            # Ensure descriptors are float32 for distance computation
            if desc is not None:
                desc = desc.astype(np.float32)

            logger.debug(f"Extracted {len(keypoints)} features from {gray.shape} image")

            return Features(
                keypoints=keypoints,
                descriptors=desc if desc is not None else np.array([]),
                image_size=gray.shape[:2],
                detector_type=self.detector_type
            )

        except Exception as e:
            logger.error(f"Feature extraction failed: {e}")
            return None

    def match_features(
        self,
        desc1: np.ndarray,
        desc2: np.ndarray,
        ratio_test: float = 0.7
    ) -> List[tuple]:
        """
        Match descriptors between two images using Lowe's ratio test.

        Args:
            desc1: Descriptors from first image (N, D)
            desc2: Descriptors from second image (M, D)
            ratio_test: Ratio threshold (0.7 is typical)

        Returns:
            List of matched indices (idx1, idx2)
        """
        try:
            if len(desc1) == 0 or len(desc2) == 0:
                return []

            # Create matcher
            if self.detector_type == "sift":
                bf = cv2.BFMatcher(cv2.NORM_L2, crossCheck=False)
            else:
                bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=False)

            # Find k-nearest neighbors
            matches = bf.knnMatch(desc1, desc2, k=2)

            # Apply Lowe's ratio test
            good_matches = []
            for match_pair in matches:
                if len(match_pair) < 2:
                    continue
                m, n = match_pair
                if m.distance < ratio_test * n.distance:
                    good_matches.append((m.queryIdx, m.trainIdx))

            logger.debug(f"Matched {len(good_matches)} feature pairs")
            return good_matches

        except Exception as e:
            logger.error(f"Feature matching failed: {e}")
            return []
