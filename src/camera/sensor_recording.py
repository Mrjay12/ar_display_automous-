"""
Synchronized Multi-Sensor Recording

Responsibilities:
- Record RGB, stereo, depth, and IMU streams
- Maintain temporal alignment across sensors
- Index frames for random access
- Store calibration with dataset
- Compress data where appropriate

Input:
- OAKDInterface stream
- Output directory
- Duration

Output:
- Synchronized dataset directory with all sensor data
- Metadata and calibration files
- Frame indices for temporal alignment
"""

import logging
import json
from pathlib import Path
from typing import Optional, Dict, Any
from datetime import datetime
import time
import threading

import numpy as np
import cv2
import yaml

from src.camera.oak_d_interface import OAKDInterface, OAKDCalibration

logger = logging.getLogger(__name__)


class SensorRecorder:
    """Record synchronized multi-sensor data."""

    def __init__(self, output_dir: Path, oak_d: OAKDInterface, config: Optional[Dict[str, Any]] = None):
        """
        Initialize recorder.

        Args:
            output_dir: Directory to store recorded data
            oak_d: OAKDInterface instance (already initialized)
            config: Optional recording configuration
        """
        self.output_dir = Path(output_dir)
        self.oak_d = oak_d
        self.config = config or {}

        # Create output structure
        self.rgb_dir = self.output_dir / "rgb"
        self.stereo_left_dir = self.output_dir / "stereo_left"
        self.stereo_right_dir = self.output_dir / "stereo_right"
        self.depth_dir = self.output_dir / "depth"
        self.imu_dir = self.output_dir / "imu"

        for d in [self.rgb_dir, self.stereo_left_dir, self.stereo_right_dir, self.depth_dir, self.imu_dir]:
            d.mkdir(parents=True, exist_ok=True)

        # Tracking
        self.frames_recorded = 0
        self.imu_samples_recorded = 0
        self.start_time = None
        self.stop_event = threading.Event()

        # Indices
        self.rgb_index = []
        self.stereo_left_index = []
        self.stereo_right_index = []
        self.depth_index = []
        self.imu_index = []

    def record(self, duration_sec: float) -> bool:
        """
        Record synchronized data for specified duration.

        Args:
            duration_sec: Recording duration in seconds

        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info(f"Starting {duration_sec}s recording to {self.output_dir}")

            self.start_time = time.time()
            self.stop_event.clear()

            # Recording loop
            while time.time() - self.start_time < duration_sec and not self.stop_event.is_set():
                # Get frames
                rgb_frame = self.oak_d.get_rgb_frame()
                if rgb_frame:
                    self._save_rgb_frame(rgb_frame)

                depth_frame = self.oak_d.get_depth_frame()
                if depth_frame:
                    self._save_depth_frame(depth_frame)

                imu_data = self.oak_d.get_imu_data()
                if imu_data:
                    self._save_imu_data(imu_data)

                # Small sleep to avoid busy-waiting
                time.sleep(0.001)

            # Save metadata
            self._save_metadata()
            self._save_calibration()
            self._save_indices()

            elapsed = time.time() - self.start_time
            logger.info(
                f"Recording completed: {elapsed:.1f}s, "
                f"{self.frames_recorded} RGB frames, "
                f"{self.imu_samples_recorded} IMU samples"
            )
            return True

        except Exception as e:
            logger.error(f"Recording error: {e}", exc_info=True)
            return False

    def _save_rgb_frame(self, frame) -> None:
        """Save RGB frame to disk."""
        try:
            filename = self.rgb_dir / f"{frame.frame_id:06d}.png"
            cv2.imwrite(str(filename), frame.frame)

            self.rgb_index.append({
                "frame_id": frame.frame_id,
                "timestamp_us": frame.timestamp_us,
                "filename": filename.name,
            })

            self.frames_recorded += 1

            if self.frames_recorded % 300 == 0:  # Log every 10 seconds at 30 FPS
                logger.debug(f"RGB: {self.frames_recorded} frames recorded")

        except Exception as e:
            logger.error(f"Error saving RGB frame: {e}")

    def _save_depth_frame(self, frame) -> None:
        """Save depth frame to disk (compressed)."""
        try:
            filename = self.depth_dir / f"{frame.timestamp_us:016d}.npz"
            np.savez_compressed(filename, depth=frame.depth_map)

            self.depth_index.append({
                "timestamp_us": frame.timestamp_us,
                "filename": filename.name,
                "shape": frame.depth_map.shape,
                "dtype": str(frame.depth_map.dtype),
            })

        except Exception as e:
            logger.error(f"Error saving depth frame: {e}")

    def _save_imu_data(self, samples) -> None:
        """Save IMU samples to disk."""
        try:
            # Append to IMU log
            imu_file = self.imu_dir / "imu_log.csv"

            with open(imu_file, "a") as f:
                for sample in samples:
                    line = (
                        f"{sample.timestamp_us},"
                        f"{sample.accel_xyz[0]:.6f},"
                        f"{sample.accel_xyz[1]:.6f},"
                        f"{sample.accel_xyz[2]:.6f},"
                        f"{sample.gyro_xyz[0]:.6f},"
                        f"{sample.gyro_xyz[1]:.6f},"
                        f"{sample.gyro_xyz[2]:.6f}\n"
                    )
                    f.write(line)

                    self.imu_samples_recorded += 1

                    # Add to index (only sample first one per second)
                    if self.imu_samples_recorded % 200 == 0:  # 200 Hz
                        self.imu_index.append({
                            "timestamp_us": sample.timestamp_us,
                            "sample_count": self.imu_samples_recorded,
                        })

        except Exception as e:
            logger.error(f"Error saving IMU data: {e}")

    def _save_metadata(self) -> None:
        """Save recording metadata."""
        try:
            elapsed = time.time() - self.start_time

            metadata = {
                "recording_info": {
                    "start_time": datetime.now().isoformat(),
                    "duration_sec": elapsed,
                    "camera_config": self.config,
                },
                "sensor_stats": {
                    "rgb_frames": self.frames_recorded,
                    "imu_samples": self.imu_samples_recorded,
                    "rgb_fps": self.frames_recorded / elapsed if elapsed > 0 else 0,
                    "imu_hz": self.imu_samples_recorded / elapsed if elapsed > 0 else 0,
                },
                "file_structure": {
                    "rgb": "RGB images (PNG format)",
                    "stereo_left": "Left stereo images",
                    "stereo_right": "Right stereo images",
                    "depth": "Depth maps (NPZ format, uint16 millimeters)",
                    "imu": "IMU data (CSV: timestamp_us, ax, ay, az, gx, gy, gz)",
                    "calibration.yaml": "Camera calibration",
                    "metadata.yaml": "This file",
                    "indices": "Frame indices for temporal alignment",
                },
            }

            with open(self.output_dir / "metadata.yaml", "w") as f:
                yaml.dump(metadata, f, default_flow_style=False)

            logger.info(f"Metadata saved")

        except Exception as e:
            logger.error(f"Error saving metadata: {e}")

    def _save_calibration(self) -> None:
        """Save camera calibration."""
        try:
            calibration = self.oak_d.get_calibration()
            if calibration:
                calib_dict = calibration.to_dict()

                with open(self.output_dir / "calibration.yaml", "w") as f:
                    yaml.dump(calib_dict, f, default_flow_style=False)

                logger.info("Calibration saved")

        except Exception as e:
            logger.error(f"Error saving calibration: {e}")

    def _save_indices(self) -> None:
        """Save frame indices for temporal alignment."""
        try:
            indices = {
                "rgb": self.rgb_index,
                "depth": self.depth_index,
                "imu": self.imu_index,
            }

            with open(self.output_dir / "indices.json", "w") as f:
                json.dump(indices, f, indent=2)

            logger.info("Frame indices saved")

        except Exception as e:
            logger.error(f"Error saving indices: {e}")

    def stop(self) -> None:
        """Stop recording."""
        self.stop_event.set()
        logger.info("Recording stop requested")

    def get_statistics(self) -> Dict[str, Any]:
        """Get recording statistics."""
        elapsed = time.time() - self.start_time if self.start_time else 0

        return {
            "elapsed_sec": elapsed,
            "rgb_frames": self.frames_recorded,
            "imu_samples": self.imu_samples_recorded,
            "rgb_fps": self.frames_recorded / elapsed if elapsed > 0 else 0,
            "output_dir": str(self.output_dir),
        }
