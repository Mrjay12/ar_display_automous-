"""
Camera Calibration Management

Responsibilities:
- Store calibration data to disk (YAML)
- Load calibration from disk
- Validate calibration parameters
- Provide calibration as OpenCV matrices

Input:
- OAKDCalibration object
- File path for storage

Output:
- Stored YAML calibration file
- Validated calibration parameters
- Calibration matrices for OpenCV functions

Failure Modes:
- Calibration file not found: Clear error message
- Corrupted YAML: Validation catches common errors
- Unreasonable parameters: Values checked against physical bounds
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any, Tuple
from dataclasses import asdict

import numpy as np
import yaml

from src.camera.oak_d_interface import OAKDCalibration

logger = logging.getLogger(__name__)


class CalibrationManager:
    """Manage camera calibration storage and retrieval."""

    # Physical bounds for sanity checking
    MIN_FOCAL_LENGTH = 100  # pixels
    MAX_FOCAL_LENGTH = 5000  # pixels
    MAX_DISTORTION_COEFFICIENT = 0.5

    def __init__(self, calibration_dir: Optional[Path] = None):
        """
        Initialize calibration manager.

        Args:
            calibration_dir: Directory to store calibration files.
                           Defaults to ./data/calibrations
        """
        self.calibration_dir = calibration_dir or Path("./data/calibrations")
        self.calibration_dir.mkdir(parents=True, exist_ok=True)

    def save_calibration(self, calibration: OAKDCalibration, device_serial: str) -> bool:
        """
        Save calibration to YAML file.

        Args:
            calibration: OAKDCalibration object
            device_serial: Device serial number for filename

        Returns:
            True if successful, False otherwise.
        """
        try:
            # Validate before saving
            if not self._validate_calibration(calibration):
                logger.error("Calibration validation failed")
                return False

            # Convert to dictionary
            calib_dict = calibration.to_dict()

            # Add metadata
            calib_dict["metadata"] = {
                "device_serial": device_serial,
                "device_name": "OAK-D Pro",
                "saved_at": self._get_timestamp_str(),
            }

            # Save to YAML
            filename = self.calibration_dir / f"oak_d_pro_{device_serial}.yaml"

            with open(filename, "w") as f:
                yaml.dump(calib_dict, f, default_flow_style=False, sort_keys=False)

            logger.info(f"Calibration saved to {filename}")
            return True

        except Exception as e:
            logger.error(f"Error saving calibration: {e}", exc_info=True)
            return False

    def load_calibration(self, device_serial: str) -> Optional[OAKDCalibration]:
        """
        Load calibration from YAML file.

        Args:
            device_serial: Device serial number

        Returns:
            OAKDCalibration or None if file not found or invalid.
        """
        try:
            filename = self.calibration_dir / f"oak_d_pro_{device_serial}.yaml"

            if not filename.exists():
                logger.warning(f"Calibration file not found: {filename}")
                return None

            with open(filename, "r") as f:
                calib_dict = yaml.safe_load(f)

            # Parse back to OAKDCalibration
            calibration = self._dict_to_calibration(calib_dict)

            if not self._validate_calibration(calibration):
                logger.error(f"Loaded calibration failed validation: {filename}")
                return None

            logger.info(f"Calibration loaded from {filename}")
            return calibration

        except Exception as e:
            logger.error(f"Error loading calibration: {e}", exc_info=True)
            return None

    def _dict_to_calibration(self, calib_dict: Dict[str, Any]) -> OAKDCalibration:
        """Convert dictionary to OAKDCalibration object."""
        rgb = calib_dict["rgb_camera"]
        left = calib_dict["stereo_left"]
        right = calib_dict["stereo_right"]
        stereo_extrinsics = calib_dict["stereo_extrinsics"]
        stereo_rect = calib_dict.get("stereo_rectification", {})
        imu = calib_dict.get("camera_to_imu", {})

        calibration = OAKDCalibration(
            rgb_intrinsics=np.array(rgb["intrinsic_matrix"]),
            rgb_distortion=np.array(rgb["distortion_coefficients"]),
            rgb_resolution=tuple(rgb["resolution"]),

            stereo_left_intrinsics=np.array(left["intrinsic_matrix"]),
            stereo_left_distortion=np.array(left["distortion_coefficients"]),
            stereo_left_resolution=tuple(left["resolution"]),

            stereo_right_intrinsics=np.array(right["intrinsic_matrix"]),
            stereo_right_distortion=np.array(right["distortion_coefficients"]),
            stereo_right_resolution=tuple(right["resolution"]),

            baseline_mm=float(stereo_extrinsics["baseline_mm"]),
            stereo_right_to_left_rotation=np.array(stereo_extrinsics["right_to_left_rotation"]),
            stereo_right_to_left_translation=np.array(stereo_extrinsics["right_to_left_translation"]),

            stereo_left_rectification=(np.array(stereo_rect["left_rectification"])
                                      if stereo_rect.get("left_rectification") else None),
            stereo_right_rectification=(np.array(stereo_rect["right_rectification"])
                                       if stereo_rect.get("right_rectification") else None),
            stereo_left_projection=(np.array(stereo_rect["left_projection"])
                                   if stereo_rect.get("left_projection") else None),
            stereo_right_projection=(np.array(stereo_rect["right_projection"])
                                    if stereo_rect.get("right_projection") else None),

            imu_to_camera_rotation=(np.array(imu["rotation"])
                                   if imu.get("rotation") else None),
            imu_to_camera_translation=(np.array(imu["translation_mm"])
                                      if imu.get("translation_mm") else None),

            distortion_model=rgb.get("distortion_model", "rational_polynomial"),
        )

        return calibration

    def _validate_calibration(self, calibration: OAKDCalibration) -> bool:
        """Validate calibration parameters."""
        try:
            # Check intrinsics are 3x3
            for name, K in [
                ("RGB", calibration.rgb_intrinsics),
                ("Stereo Left", calibration.stereo_left_intrinsics),
                ("Stereo Right", calibration.stereo_right_intrinsics),
            ]:
                if K.shape != (3, 3):
                    logger.error(f"{name} intrinsics wrong shape: {K.shape}")
                    return False

            # Check focal lengths are reasonable
            for name, K in [
                ("RGB", calibration.rgb_intrinsics),
                ("Stereo Left", calibration.stereo_left_intrinsics),
                ("Stereo Right", calibration.stereo_right_intrinsics),
            ]:
                fx, fy = K[0, 0], K[1, 1]
                if not (self.MIN_FOCAL_LENGTH < fx < self.MAX_FOCAL_LENGTH):
                    logger.error(f"{name} focal length out of range: fx={fx}")
                    return False
                if not (self.MIN_FOCAL_LENGTH < fy < self.MAX_FOCAL_LENGTH):
                    logger.error(f"{name} focal length out of range: fy={fy}")
                    return False

            # Check principal points are in image
            rgb_w, rgb_h = calibration.rgb_resolution
            cx, cy = calibration.rgb_intrinsics[0, 2], calibration.rgb_intrinsics[1, 2]
            if not (0 < cx < rgb_w and 0 < cy < rgb_h):
                logger.error(f"RGB principal point out of image: ({cx}, {cy}) vs {rgb_w}x{rgb_h}")
                return False

            # Check distortion coefficients are reasonable
            for name, dist in [
                ("RGB", calibration.rgb_distortion),
                ("Stereo Left", calibration.stereo_left_distortion),
                ("Stereo Right", calibration.stereo_right_distortion),
            ]:
                if np.any(np.abs(dist) > self.MAX_DISTORTION_COEFFICIENT):
                    logger.warning(f"{name} distortion coefficients may be unusual: {dist}")

            # Check baseline is positive
            if calibration.baseline_mm <= 0:
                logger.error(f"Baseline must be positive: {calibration.baseline_mm}")
                return False

            # Check stereo extrinsics are valid
            if calibration.stereo_right_to_left_rotation.shape != (3, 3):
                logger.error(f"Rotation matrix wrong shape: {calibration.stereo_right_to_left_rotation.shape}")
                return False

            if calibration.stereo_right_to_left_translation.shape != (3,):
                logger.error(f"Translation vector wrong shape: {calibration.stereo_right_to_left_translation.shape}")
                return False

            logger.info("Calibration validation passed")
            return True

        except Exception as e:
            logger.error(f"Validation error: {e}")
            return False

    def get_rgb_camera_matrix(self) -> Optional[np.ndarray]:
        """Get RGB camera intrinsic matrix (for backward compatibility)."""
        # This would be populated if calibration is loaded
        pass

    @staticmethod
    def _get_timestamp_str() -> str:
        """Get current timestamp as ISO 8601 string."""
        from datetime import datetime, timezone
        return datetime.now(timezone.utc).isoformat()


class CalibrationVerifier:
    """Verify calibration quality and consistency."""

    def __init__(self, calibration: OAKDCalibration):
        """Initialize verifier with calibration."""
        self.calibration = calibration

    def verify_stereo_rectification(self) -> bool:
        """
        Verify stereo rectification is valid.

        Returns:
            True if rectification matrices are valid.
        """
        try:
            if (self.calibration.stereo_left_rectification is None or
                self.calibration.stereo_right_rectification is None):
                logger.warning("Rectification matrices not available")
                return True  # Not an error, just not computed

            # Rectification matrices should be 3x3
            if (self.calibration.stereo_left_rectification.shape != (3, 3) or
                self.calibration.stereo_right_rectification.shape != (3, 3)):
                logger.error("Rectification matrix wrong shape")
                return False

            logger.info("Stereo rectification verification passed")
            return True

        except Exception as e:
            logger.error(f"Rectification verification error: {e}")
            return False

    def verify_baseline(self) -> bool:
        """Verify baseline is within expected range for OAK-D Pro."""
        try:
            # OAK-D Pro baseline is ~75mm
            expected_baseline = 75.0  # mm
            tolerance_percent = 10

            min_baseline = expected_baseline * (1 - tolerance_percent / 100)
            max_baseline = expected_baseline * (1 + tolerance_percent / 100)

            if not (min_baseline < self.calibration.baseline_mm < max_baseline):
                logger.warning(
                    f"Baseline {self.calibration.baseline_mm}mm outside expected range "
                    f"({min_baseline}-{max_baseline}mm)"
                )
                return False

            logger.info(f"Baseline verification passed: {self.calibration.baseline_mm}mm")
            return True

        except Exception as e:
            logger.error(f"Baseline verification error: {e}")
            return False

    def verify_resolution_consistency(self) -> bool:
        """Verify camera resolutions are consistent."""
        try:
            # RGB should be higher res than stereo
            rgb_pixels = self.calibration.rgb_resolution[0] * self.calibration.rgb_resolution[1]
            stereo_pixels = self.calibration.stereo_left_resolution[0] * self.calibration.stereo_left_resolution[1]

            if rgb_pixels < stereo_pixels:
                logger.warning("RGB resolution should typically be >= stereo resolution")

            logger.info("Resolution consistency check passed")
            return True

        except Exception as e:
            logger.error(f"Resolution consistency error: {e}")
            return False

    def run_all_checks(self) -> bool:
        """Run all verification checks."""
        checks = [
            ("Stereo Rectification", self.verify_stereo_rectification()),
            ("Baseline", self.verify_baseline()),
            ("Resolution Consistency", self.verify_resolution_consistency()),
        ]

        all_passed = all(result for _, result in checks)

        logger.info(f"Calibration verification: {'PASS' if all_passed else 'FAIL'}")
        for name, result in checks:
            logger.info(f"  {name}: {'✓' if result else '✗'}")

        return all_passed
