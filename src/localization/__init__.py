"""
Localization Module - GPS-Denied Global and Local Pose Estimation

Submodules:
  - visual_place_recognition: Coarse global localization using visual embeddings
  - geometric_verifier: Multi-building geometric constraint verification
  - pose_estimator: 6-DoF camera pose estimation (PnP + RANSAC)
  - confidence_estimator: Localization quality assessment and state machine
  - relocalization: Recovery from tracking loss
"""

from .visual_place_recognition import (
    VisualPlaceRecognizer,
    PlaceRecognitionResult,
    Candidate,
)
from .geometric_verifier import (
    GeometricVerifier,
    VerificationResult,
    DetectedBuilding,
)
from .pose_estimator import (
    PoseEstimator,
    CameraPose,
)
from .confidence_estimator import (
    ConfidenceEstimator,
    ConfidenceMetrics,
    LocalizationState,
)
from .relocalization import (
    RelocalizationHandler,
    RelocalizationResult,
    RelocalizationState,
)

__all__ = [
    'VisualPlaceRecognizer',
    'PlaceRecognitionResult',
    'Candidate',
    'GeometricVerifier',
    'VerificationResult',
    'DetectedBuilding',
    'PoseEstimator',
    'CameraPose',
    'ConfidenceEstimator',
    'ConfidenceMetrics',
    'LocalizationState',
    'RelocalizationHandler',
    'RelocalizationResult',
    'RelocalizationState',
]
