"""
Offline Dataset Replay

Responsibilities:
- Replay recorded sensor data without hardware
- Maintain original timestamps and synchronization
- Provide same interface as live camera for seamless testing
- Load calibration from dataset

Input:
- Recorded dataset directory

Output:
- Frames and IMU data with original timestamps
- Transparent replay for algorithm development
"""

import logging
import json
from pathlib import Path
from typing import Optional, List, Dict, Any
import time

import numpy as np
import cv2
import yaml

from src.camera.oak_d_interface import CameraFrame, StereoDepth, IMUSample, OAKDCalibration

logger = logging.getLogger(__name__)


class SensorReplayer:
    """Replay recorded sensor data without OAK-D Pro."""

    def __init__(self, dataset_dir: Path):
        """
        Initialize replayer.

        Args:
            dataset_dir: Directory containing recorded data
        """
        self.dataset_dir = Path(dataset_dir)

        # Load indices
        self.indices = self._load_indices()
        self.calibration = self._load_calibration()

        # Playback state
        self.current_rgb_idx = 0
        self.current_depth_idx = 0
        self.current_imu_idx = 0
        self.replay_start_time = None
        self.reference_time = None
        self.playback_rate = 1.0

    def _load_indices(self) -> Dict[str, List[Dict[str, Any]]]:
        """Load frame indices."""
        try:
            indices_file = self.dataset_dir / "indices.json"

            if not indices_file.exists():
                logger.warning(f"Indices file not found: {indices_file}")
                return {"rgb": [], "depth": [], "imu": []}

            with open(indices_file, "r") as f:
                indices = json.load(f)

            logger.info(f"Loaded indices: {len(indices.get('rgb', []))} RGB, "
                       f"{len(indices.get('imu', []))} IMU")
            return indices

        except Exception as e:
            logger.error(f"Error loading indices: {e}")
            return {"rgb": [], "depth": [], "imu": []}

    def _load_calibration(self) -> Optional[OAKDCalibration]:
        """Load camera calibration from dataset."""
        try:
            calib_file = self.dataset_dir / "calibration.yaml"

            if not calib_file.exists():
                logger.warning(f"Calibration file not found: {calib_file}")
                return None

            with open(calib_file, "r") as f:
                calib_dict = yaml.safe_load(f)

            # Parse calibration
            if not calib_dict:
                return None

            rgb = calib_dict.get("rgb_camera", {})
            left = calib_dict.get("stereo_left", {})
            right = calib_dict.get("stereo_right", {})
            extrinsics = calib_dict.get("stereo_extrinsics", {})

            calibration = OAKDCalibration(
                rgb_intrinsics=np.array(rgb.get("intrinsic_matrix", [])),
                rgb_distortion=np.array(rgb.get("distortion_coefficients", [])),
                rgb_resolution=tuple(rgb.get("resolution", [0, 0])),

                stereo_left_intrinsics=np.array(left.get("intrinsic_matrix", [])),
                stereo_left_distortion=np.array(left.get("distortion_coefficients", [])),
                stereo_left_resolution=tuple(left.get("resolution", [0, 0])),

                stereo_right_intrinsics=np.array(right.get("intrinsic_matrix", [])),
                stereo_right_distortion=np.array(right.get("distortion_coefficients", [])),
                stereo_right_resolution=tuple(right.get("resolution", [0, 0])),

                baseline_mm=float(extrinsics.get("baseline_mm", 75)),
                stereo_right_to_left_rotation=np.array(extrinsics.get("right_to_left_rotation", [])),
                stereo_right_to_left_translation=np.array(extrinsics.get("right_to_left_translation", [])),
            )

            logger.info("Calibration loaded from dataset")
            return calibration

        except Exception as e:
            logger.error(f"Error loading calibration: {e}")
            return None

    def get_rgb_frame(self) -> Optional[CameraFrame]:
        """Get next RGB frame."""
        try:
            if self.current_rgb_idx >= len(self.indices.get("rgb", [])):
                return None

            frame_info = self.indices["rgb"][self.current_rgb_idx]
            filename = self.dataset_dir / "rgb" / frame_info["filename"]

            if not filename.exists():
                logger.warning(f"Frame file not found: {filename}")
                self.current_rgb_idx += 1
                return None

            frame_data = cv2.imread(str(filename))

            camera_frame = CameraFrame(
                timestamp_us=frame_info["timestamp_us"],
                frame=frame_data,
                frame_id=frame_info["frame_id"],
            )

            self.current_rgb_idx += 1
            return camera_frame

        except Exception as e:
            logger.error(f"Error reading RGB frame: {e}")
            self.current_rgb_idx += 1
            return None

    def get_depth_frame(self) -> Optional[StereoDepth]:
        """Get next depth frame."""
        try:
            if self.current_depth_idx >= len(self.indices.get("depth", [])):
                return None

            frame_info = self.indices["depth"][self.current_depth_idx]
            filename = self.dataset_dir / "depth" / frame_info["filename"]

            if not filename.exists():
                logger.warning(f"Depth file not found: {filename}")
                self.current_depth_idx += 1
                return None

            data = np.load(filename)
            depth_map = data["depth"]

            stereo_depth = StereoDepth(
                timestamp_us=frame_info["timestamp_us"],
                depth_map=depth_map,
            )

            self.current_depth_idx += 1
            return stereo_depth

        except Exception as e:
            logger.error(f"Error reading depth frame: {e}")
            self.current_depth_idx += 1
            return None

    def get_imu_data(self) -> Optional[List[IMUSample]]:
        """Get next batch of IMU samples."""
        try:
            imu_file = self.dataset_dir / "imu" / "imu_log.csv"

            if not imu_file.exists():
                logger.warning(f"IMU file not found: {imu_file}")
                return None

            # Read and parse IMU log (simplified - reads all at once)
            samples = []

            with open(imu_file, "r") as f:
                for i, line in enumerate(f):
                    if i < self.current_imu_idx:
                        continue

                    parts = line.strip().split(",")
                    if len(parts) != 7:
                        continue

                    timestamp_us = int(parts[0])
                    accel_xyz = np.array([float(parts[1]), float(parts[2]), float(parts[3])])
                    gyro_xyz = np.array([float(parts[4]), float(parts[5]), float(parts[6])])

                    samples.append(IMUSample(
                        timestamp_us=timestamp_us,
                        accel_xyz=accel_xyz,
                        gyro_xyz=gyro_xyz,
                    ))

                    self.current_imu_idx = i + 1

                    # Return batch of samples (simplified)
                    if len(samples) >= 10:
                        break

            return samples if samples else None

        except Exception as e:
            logger.error(f"Error reading IMU data: {e}")
            return None

    def reset(self) -> None:
        """Reset playback to beginning."""
        self.current_rgb_idx = 0
        self.current_depth_idx = 0
        self.current_imu_idx = 0
        self.replay_start_time = None
        logger.info("Replay reset to beginning")

    def get_calibration(self) -> Optional[OAKDCalibration]:
        """Get camera calibration."""
        return self.calibration

    def get_dataset_info(self) -> Dict[str, Any]:
        """Get dataset information."""
        try:
            metadata_file = self.dataset_dir / "metadata.yaml"

            if not metadata_file.exists():
                return {}

            with open(metadata_file, "r") as f:
                metadata = yaml.safe_load(f)

            return metadata or {}

        except Exception as e:
            logger.error(f"Error reading metadata: {e}")
            return {}

    def is_complete(self) -> bool:
        """Check if replay has reached end."""
        rgb_complete = self.current_rgb_idx >= len(self.indices.get("rgb", []))
        imu_complete = self.current_imu_idx >= sum(1 for _ in
                                                   open(self.dataset_dir / "imu" / "imu_log.csv", "r"))
        return rgb_complete or imu_complete
