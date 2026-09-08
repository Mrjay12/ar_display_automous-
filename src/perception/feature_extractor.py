"""
Visual Feature Extraction Module

Responsibilities:
- Extract robust visual features from RGB frames
- Compute feature descriptors
- Support multiple feature detectors (ORB, SIFT)
- Provide feature confidence scores

Input:
  - RGB frame (uint8, H x W x 3)
  - Feature detector type (ORB, SIFT)

Output:
  - Keypoints (u, v in pixels)
  - Descriptors (128-D or 256-D vectors)
  - Per-keypoint confidence scores

Performance:
  - ORB: ~10-15 ms per frame (efficient)
  - SIFT: ~30-50 ms per frame (more robust)
  - Typical latency: < 50 ms for ORB, < 100 ms for SIFT
  - Memory usage: ~10-50 MB per frame

Failure Modes:
  - Textureless images: Few or no features extracted
  - Motion blur: Features may be unreliable
  - Low light: Feature detection fails, fallback to fewer features
  - Recovery: Log and continue with best available features

Example:
    >>> extractor = FeatureExtractor(detector_type='orb', max_features=500)
    >>> frame = cv2.imread('image.png')
    >>> features = extractor.extract(frame)
    >>> print(f"Found {len(features.keypoints)} keypoints")
"""

import logging
from typing import Optional, Tuple, List
from dataclasses import dataclass
from pathlib import Path
import time

import cv2
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Feature:
    """Single visual feature."""
    x: float                  # Pixel column (u coordinate)
    y: float                  # Pixel row (v coordinate)
    descriptor: np.ndarray    # Feature descriptor (1D vector)
    confidence: float         # Confidence score (0-1)
    size: float               # Feature size (in pixels, from detector)
    angle: float              # Feature orientation in degrees


@dataclass
class FeatureSet:
    """Collection of features from a single frame."""
    timestamp_us: int         # Frame timestamp (microseconds)
    frame_id: int             # Frame sequence number
    keypoints: List[Feature]  # List of extracted features
    descriptors: Optional[np.ndarray]  # All descriptors as (N, D) matrix
    frame_shape: Tuple[int, int]  # (height, width) of source frame
    detector_type: str        # 'orb', 'sift', 'akaze', etc.

    def __len__(self):
        return len(self.keypoints)

    @property
    def is_empty(self) -> bool:
        """Check if no features were extracted."""
        return len(self.keypoints) == 0

    @property
    def descriptor_dim(self) -> int:
        """Dimension of feature descriptors."""
        if self.descriptors is None or len(self.descriptors) == 0:
            return 0
        return self.descriptors.shape[1]


class FeatureExtractor:
    """
    Extract and manage visual features from RGB frames.

    Supports ORB (fast) and SIFT (robust) feature detectors.
    Optimized for real-time operation on laptop GPUs.
    """

    def __init__(
        self,
        detector_type: str = 'orb',
        max_features: int = 500,
        min_confidence: float = 0.01,
    ):
        """
        Initialize feature extractor.

        Args:
            detector_type: 'orb' (fast, default) or 'sift' (robust)
            max_features: Maximum number of features to extract per frame
            min_confidence: Minimum response strength threshold
        """
        self.detector_type = detector_type.lower()
        self.max_features = max_features
        self.min_confidence = min_confidence
        self._frame_count = 0
        self._total_features = 0

        # Validate detector type
        if self.detector_type not in ['orb', 'sift', 'akaze']:
            logger.warning(f"Unknown detector type: {detector_type}, using ORB")
            self.detector_type = 'orb'

        # Initialize detector
        self._detector = self._create_detector()
        logger.info(f"FeatureExtractor initialized with detector: {self.detector_type.upper()}")

    def _create_detector(self):
        """Create and configure feature detector."""
        if self.detector_type == 'orb':
            # ORB: Fast, rotation-invariant, binary descriptors
            return cv2.ORB_create(
                nfeatures=self.max_features,
                scaleFactor=1.2,
                nlevels=8,
                edgeThreshold=15,
                firstLevel=0,
                WTA_K=2,
                scoreType=cv2.ORB_HARRIS_SCORE,
                patchSize=31,
                fastThreshold=20
            )
        elif self.detector_type == 'sift':
            # SIFT: More robust, 128-D float descriptors
            # Note: SIFT requires opencv-contrib-python
            return cv2.SIFT_create(
                nfeatures=self.max_features,
                nOctaveLayers=3,
                contrastThreshold=0.04,
                edgeThreshold=10.0,
                sigma=1.6
            )
        elif self.detector_type == 'akaze':
            # AKAZE: Fast, binary descriptors, rotation-invariant
            return cv2.AKAZE_create(
                descriptor_type=cv2.AKAZE_DESCRIPTOR_MLDB,
                descriptor_size=0,
                descriptor_channels=3,
                threshold=0.001,
                nOctaves=4,
                nOctaveLayers=4,
                flags=0
            )
        else:
            return cv2.ORB_create(nfeatures=self.max_features)

    def extract(
        self,
        frame: np.ndarray,
        timestamp_us: int = 0,
        frame_id: int = 0
    ) -> FeatureSet:
        """
        Extract features from RGB frame.

        INPUT FRAME:
        - Format: uint8, shape (H, W, 3) for color or (H, W) for grayscale
        - Range: 0-255
        - Size: typical 1280x720 or smaller

        OUTPUT FEATURES:
        - Format: FeatureSet object
        - Keypoints: (u, v) in pixels
        - Descriptors: Binary (ORB) or float (SIFT)

        FAILURE MODES:
        - Textureless frame: Returns empty FeatureSet
        - Invalid frame: Logs error, returns empty set
        - Detector failure: Logs and continues

        Args:
            frame: RGB or grayscale image (uint8, H x W x [3])
            timestamp_us: Frame timestamp in microseconds
            frame_id: Frame sequence number

        Returns:
            FeatureSet with extracted keypoints and descriptors
        """
        start_time = time.time()
        self._frame_count += 1

        # Validate input
        if frame is None or frame.size == 0:
            logger.warning(f"[Frame {frame_id}] Invalid frame received")
            return FeatureSet(
                timestamp_us=timestamp_us,
                frame_id=frame_id,
                keypoints=[],
                descriptors=None,
                frame_shape=frame.shape[:2] if frame is not None else (0, 0),
                detector_type=self.detector_type
            )

        frame_height, frame_width = frame.shape[:2]

        # Convert to grayscale if needed
        if len(frame.shape) == 3:
            gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        else:
            gray = frame

        try:
            # Detect keypoints and compute descriptors
            kps, descriptors = self._detector.detectAndCompute(gray, None)

            if kps is None or len(kps) == 0:
                logger.debug(f"[Frame {frame_id}] No features detected (textureless image)")
                return FeatureSet(
                    timestamp_us=timestamp_us,
                    frame_id=frame_id,
                    keypoints=[],
                    descriptors=None,
                    frame_shape=(frame_height, frame_width),
                    detector_type=self.detector_type
                )

            # Convert OpenCV keypoints to Feature objects
            features = []
            for i, kp in enumerate(kps):
                # OpenCV KeyPoint has: pt (x, y), size, angle, response, octave, class_id
                descriptor = descriptors[i] if descriptors is not None else np.array([])

                feature = Feature(
                    x=kp.pt[0],
                    y=kp.pt[1],
                    descriptor=descriptor.astype(np.float32) if len(descriptor) > 0 else descriptor,
                    confidence=float(kp.response) if kp.response > 0 else 0.1,
                    size=float(kp.size),
                    angle=float(kp.angle)
                )
                features.append(feature)

            # Build descriptor matrix
            if descriptors is not None and len(features) > 0:
                descriptor_matrix = np.array([f.descriptor for f in features], dtype=np.float32)
            else:
                descriptor_matrix = None

            elapsed_ms = (time.time() - start_time) * 1000.0
            self._total_features += len(features)

            # Log statistics periodically
            if self._frame_count % 30 == 0:
                avg_features = self._total_features / self._frame_count
                logger.debug(
                    f"[Frame {frame_id}] Extracted {len(features)} features "
                    f"({avg_features:.1f} avg) in {elapsed_ms:.2f} ms"
                )

            return FeatureSet(
                timestamp_us=timestamp_us,
                frame_id=frame_id,
                keypoints=features,
                descriptors=descriptor_matrix,
                frame_shape=(frame_height, frame_width),
                detector_type=self.detector_type
            )

        except Exception as e:
            logger.error(f"[Frame {frame_id}] Feature extraction failed: {e}")
            return FeatureSet(
                timestamp_us=timestamp_us,
                frame_id=frame_id,
                keypoints=[],
                descriptors=None,
                frame_shape=(frame_height, frame_width),
                detector_type=self.detector_type
            )

    def get_statistics(self) -> dict:
        """Return extraction statistics."""
        avg_features = (
            self._total_features / self._frame_count
            if self._frame_count > 0 else 0
        )
        return {
            'detector_type': self.detector_type,
            'frames_processed': self._frame_count,
            'total_features': self._total_features,
            'average_features_per_frame': avg_features,
            'max_features_configured': self.max_features
        }
