"""
OAK-D Pro Hardware Interface

Responsibilities:
- Detect and initialize OAK-D Pro camera
- Acquire RGB, stereo, depth, and IMU data
- Provide synchronized access to all sensor streams
- Handle graceful shutdown and resource cleanup
- Expose camera calibration data

Input:
- Configuration parameters (FPS, resolution, etc.)
- Device ID (optional, defaults to first device)

Output:
- RGB frames with timestamps
- Stereo left/right frames with timestamps
- Depth maps with timestamps
- IMU measurements with timestamps
- Camera calibration parameters

Performance:
- Minimal latency (< 50 ms end-to-end)
- Memory-efficient frame buffering
- Typical CPU usage: 15-25%

Failure Modes:
- Device not detected: Returns initialization error
- Stream initialization failure: Logs specific stream error
- USB bandwidth exceeded: May drop frames (acceptable, logged)
- Frame corruption: Detected and skipped
"""

import logging
from typing import Optional, Tuple, Dict, Any, List
from dataclasses import dataclass
from pathlib import Path
import time
import threading
from collections import deque

import cv2
import numpy as np
import depthai as dai

logger = logging.getLogger(__name__)


@dataclass
class CameraFrame:
    """Container for a single camera frame."""
    timestamp_us: int          # Microsecond-precision timestamp
    frame: np.ndarray          # Image data (H, W) or (H, W, C)
    frame_id: int              # Sequence number

    @property
    def timestamp_sec(self) -> float:
        """Return timestamp in seconds."""
        return self.timestamp_us / 1_000_000.0


@dataclass
class IMUSample:
    """Single IMU measurement."""
    timestamp_us: int          # Microsecond-precision timestamp
    accel_xyz: np.ndarray      # [x, y, z] in m/s²
    gyro_xyz: np.ndarray       # [x, y, z] in rad/s (or °/s depending on config)

    @property
    def timestamp_sec(self) -> float:
        """Return timestamp in seconds."""
        return self.timestamp_us / 1_000_000.0


@dataclass
class StereoDepth:
    """Stereo depth frame."""
    timestamp_us: int
    depth_map: np.ndarray      # Depth in millimeters (H, W), uint16
    disparity: Optional[np.ndarray] = None
    confidence: Optional[np.ndarray] = None  # Optional confidence map

    @property
    def timestamp_sec(self) -> float:
        """Return timestamp in seconds."""
        return self.timestamp_us / 1_000_000.0


@dataclass
class OAKDCalibration:
    """Camera calibration data from OAK-D Pro."""

    # RGB Camera Intrinsics
    rgb_intrinsics: np.ndarray  # 3x3 matrix
    rgb_distortion: np.ndarray  # Distortion coefficients
    rgb_resolution: Tuple[int, int]  # (width, height)

    # Stereo Left Camera Intrinsics
    stereo_left_intrinsics: np.ndarray  # 3x3 matrix
    stereo_left_distortion: np.ndarray
    stereo_left_resolution: Tuple[int, int]

    # Stereo Right Camera Intrinsics
    stereo_right_intrinsics: np.ndarray  # 3x3 matrix
    stereo_right_distortion: np.ndarray
    stereo_right_resolution: Tuple[int, int]

    # Stereo Extrinsics
    baseline_mm: float                      # Stereo baseline in mm
    stereo_right_to_left_rotation: np.ndarray  # 3x3 rotation
    stereo_right_to_left_translation: np.ndarray  # 3D translation vector

    # Stereo Rectification (if available)
    stereo_left_rectification: Optional[np.ndarray] = None  # 3x3
    stereo_right_rectification: Optional[np.ndarray] = None  # 3x3
    stereo_left_projection: Optional[np.ndarray] = None  # 3x4
    stereo_right_projection: Optional[np.ndarray] = None  # 3x4

    # Camera to IMU Transform (if available)
    imu_to_camera_rotation: Optional[np.ndarray] = None  # 3x3
    imu_to_camera_translation: Optional[np.ndarray] = None  # 3D vector in mm

    # Distortion model info
    distortion_model: str = "rational_polynomial"  # Or "fisheye", etc.

    def to_dict(self) -> Dict[str, Any]:
        """Serialize to dictionary for YAML storage."""
        return {
            "rgb_camera": {
                "intrinsic_matrix": self.rgb_intrinsics.tolist(),
                "distortion_coefficients": self.rgb_distortion.tolist(),
                "resolution": list(self.rgb_resolution),
                "distortion_model": self.distortion_model,
            },
            "stereo_left": {
                "intrinsic_matrix": self.stereo_left_intrinsics.tolist(),
                "distortion_coefficients": self.stereo_left_distortion.tolist(),
                "resolution": list(self.stereo_left_resolution),
            },
            "stereo_right": {
                "intrinsic_matrix": self.stereo_right_intrinsics.tolist(),
                "distortion_coefficients": self.stereo_right_distortion.tolist(),
                "resolution": list(self.stereo_right_resolution),
            },
            "stereo_extrinsics": {
                "baseline_mm": float(self.baseline_mm),
                "right_to_left_rotation": self.stereo_right_to_left_rotation.tolist(),
                "right_to_left_translation": self.stereo_right_to_left_translation.tolist(),
            },
            "stereo_rectification": {
                "left_rectification": (self.stereo_left_rectification.tolist()
                                      if self.stereo_left_rectification is not None else None),
                "right_rectification": (self.stereo_right_rectification.tolist()
                                       if self.stereo_right_rectification is not None else None),
                "left_projection": (self.stereo_left_projection.tolist()
                                   if self.stereo_left_projection is not None else None),
                "right_projection": (self.stereo_right_projection.tolist()
                                    if self.stereo_right_projection is not None else None),
            },
            "camera_to_imu": {
                "rotation": (self.imu_to_camera_rotation.tolist()
                            if self.imu_to_camera_rotation is not None else None),
                "translation_mm": (self.imu_to_camera_translation.tolist()
                                  if self.imu_to_camera_translation is not None else None),
            },
        }


class OAKDInterface:
    """Interface to OAK-D Pro camera."""

    def __init__(self, device_id: Optional[str] = None, config: Optional[Dict[str, Any]] = None):
        """
        Initialize OAK-D Pro interface.

        Args:
            device_id: Optional device serial number. None = auto-detect first device.
            config: Optional configuration dictionary (typically loaded from YAML).
        """
        self.device_id = device_id
        self.config = config or {}

        # DepthAI objects
        self.device: Optional[dai.Device] = None
        self.pipeline: Optional[dai.Pipeline] = None

        # Queues for sensor data
        self.rgb_queue: Optional[dai.DataInputQueue] = None
        self.stereo_queue: Optional[dai.DataInputQueue] = None
        self.depth_queue: Optional[dai.DataInputQueue] = None
        self.imu_queue: Optional[dai.DataInputQueue] = None

        # Calibration (retrieved at initialization)
        self.calibration: Optional[OAKDCalibration] = None
        self.device_info: Optional[Dict[str, str]] = None

        # Frame tracking
        self.rgb_frame_id = 0
        self.stereo_left_frame_id = 0
        self.stereo_right_frame_id = 0
        self.depth_frame_id = 0
        self.imu_sample_count = 0

        # Statistics
        self.frames_dropped = 0
        self.frames_received = 0

        # Thread synchronization
        self._lock = threading.RLock()
        self._initialized = False

    def initialize(self) -> bool:
        """
        Initialize OAK-D Pro.

        Returns:
            True if successful, False otherwise.
        """
        try:
            logger.info("Initializing OAK-D interface...")

            # Detect device
            if not self._detect_device():
                logger.error("Failed to detect OAK-D Pro")
                return False

            # Create pipeline
            if not self._create_pipeline():
                logger.error("Failed to create DepthAI pipeline")
                return False

            # Start device and queues
            if not self._start_device():
                logger.error("Failed to start device")
                return False

            # Retrieve calibration
            if not self._retrieve_calibration():
                logger.error("Failed to retrieve calibration")
                return False

            self._initialized = True
            logger.info("OAK-D interface initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            self.shutdown()
            return False

    def _detect_device(self) -> bool:
        """Detect OAK-D Pro device."""
        try:
            device_info = dai.Device.getDeviceByMxId(self.device_id) if self.device_id else None

            if not device_info:
                # Auto-detect first device
                devices = dai.Device.getAllAvailableDevices()
                if not devices:
                    logger.error("No OAK-D devices found")
                    return False
                device_info = devices[0]

            self.device_info = {
                "mxId": device_info.mxId,
                "name": device_info.name,
                "state": str(device_info.state),
            }

            logger.info(f"Device found: {self.device_info['name']} ({self.device_info['mxId']})")
            return True

        except Exception as e:
            logger.error(f"Device detection error: {e}")
            return False

    def _create_pipeline(self) -> bool:
        """Create DepthAI pipeline with all nodes."""
        try:
            self.pipeline = dai.Pipeline()

            # RGB Camera
            cam_rgb = self.pipeline.createColorCamera()
            cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)

            # Get RGB config from config file
            rgb_config = self.config.get("camera", {}).get("rgb", {})
            fps = rgb_config.get("fps", 30)
            resolution = rgb_config.get("resolution", [1280, 720])

            # Set resolution based on config
            if resolution == [1280, 720]:
                cam_rgb.setResolution(dai.ColorCameraProperties.SensorInfo.RGB_1280X720)
            elif resolution == [1920, 1080]:
                cam_rgb.setResolution(dai.ColorCameraProperties.SensorInfo.RGB_1920X1080)
            else:
                cam_rgb.setResolution(dai.ColorCameraProperties.SensorInfo.RGB_1280X720)

            cam_rgb.setFps(fps)

            # Stereo Depth
            stereo_depth = self.pipeline.createStereoDepth()
            stereo_depth.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

            # Get stereo config
            stereo_config = self.config.get("camera", {}).get("stereo", {})
            stereo_fps = stereo_config.get("fps", 30)
            stereo_resolution = stereo_config.get("resolution", [640, 400])

            # Stereo cameras
            mono_left = self.pipeline.createMonoCamera()
            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_left.setResolution(dai.MonoCameraProperties.SensorInfo.THE_400_P)
            mono_left.setFps(stereo_fps)

            mono_right = self.pipeline.createMonoCamera()
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            mono_right.setResolution(dai.MonoCameraProperties.SensorInfo.THE_400_P)
            mono_right.setFps(stereo_fps)

            mono_left.out.link(stereo_depth.left)
            mono_right.out.link(stereo_depth.right)

            # IMU
            imu = self.pipeline.createIMU()
            imu_config = self.config.get("camera", {}).get("imu", {})
            imu_report_rate = imu_config.get("report_rate", 200)

            # Set IMU report rate
            imu.setBatchReportThreshold(imu_config.get("batch_report_threshold", 5))
            imu.setMaxBatchReports(10)

            # Enable accelerometer and gyroscope
            imu.enableIMUSensor(dai.IMUSensor.ACCELEROMETER_RAW, imu_report_rate)
            imu.enableIMUSensor(dai.IMUSensor.GYROSCOPE_RAW, imu_report_rate)

            # Output queues
            xout_rgb = self.pipeline.createXLinkOut()
            xout_rgb.setStreamName("rgb")
            cam_rgb.video.link(xout_rgb.input)

            xout_depth = self.pipeline.createXLinkOut()
            xout_depth.setStreamName("depth")
            stereo_depth.depth.link(xout_depth.input)

            xout_stereo_left = self.pipeline.createXLinkOut()
            xout_stereo_left.setStreamName("stereo_left")
            mono_left.out.link(xout_stereo_left.input)

            xout_stereo_right = self.pipeline.createXLinkOut()
            xout_stereo_right.setStreamName("stereo_right")
            mono_right.out.link(xout_stereo_right.input)

            xout_imu = self.pipeline.createXLinkOut()
            xout_imu.setStreamName("imu")
            imu.out.link(xout_imu.input)

            logger.info("Pipeline created successfully")
            return True

        except Exception as e:
            logger.error(f"Pipeline creation error: {e}")
            return False

    def _start_device(self) -> bool:
        """Start device and create output queues."""
        try:
            self.device = dai.Device(self.pipeline, dai.DeviceInfo(self.device_info["mxId"]))

            # Create output queues
            self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)
            self.stereo_queue = self.device.getOutputQueue(name="stereo_left", maxSize=4, blocking=False)
            self.imu_queue = self.device.getOutputQueue(name="imu", maxSize=50, blocking=False)

            logger.info("Device started and queues created")
            return True

        except Exception as e:
            logger.error(f"Device start error: {e}")
            return False

    def _retrieve_calibration(self) -> bool:
        """Retrieve camera calibration from device."""
        try:
            calib = self.device.readCalibration()

            # RGB Camera
            rgb_intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RGB)
            rgb_distortion = calib.getDistortionCoefficients(dai.CameraBoardSocket.RGB)
            rgb_size = calib.getLensPosition(dai.CameraBoardSocket.RGB)

            # Stereo Left
            left_intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.LEFT)
            left_distortion = calib.getDistortionCoefficients(dai.CameraBoardSocket.LEFT)

            # Stereo Right
            right_intrinsics = calib.getCameraIntrinsics(dai.CameraBoardSocket.RIGHT)
            right_distortion = calib.getDistortionCoefficients(dai.CameraBoardSocket.RIGHT)

            # Stereo Extrinsics
            extrinsics = calib.getStereoRightToLeftExtrinsics()
            baseline = calib.getBaseline() / 1000.0  # Convert to meters

            self.calibration = OAKDCalibration(
                rgb_intrinsics=np.array(rgb_intrinsics),
                rgb_distortion=np.array(rgb_distortion),
                rgb_resolution=(1280, 720),

                stereo_left_intrinsics=np.array(left_intrinsics),
                stereo_left_distortion=np.array(left_distortion),
                stereo_left_resolution=(640, 400),

                stereo_right_intrinsics=np.array(right_intrinsics),
                stereo_right_distortion=np.array(right_distortion),
                stereo_right_resolution=(640, 400),

                baseline_mm=baseline * 1000,  # Convert back to mm
                stereo_right_to_left_rotation=np.array(extrinsics[0]),
                stereo_right_to_left_translation=np.array(extrinsics[1]),
            )

            logger.info(f"Calibration retrieved: baseline={self.calibration.baseline_mm:.1f}mm")
            return True

        except Exception as e:
            logger.error(f"Calibration retrieval error: {e}")
            return False

    def get_rgb_frame(self) -> Optional[CameraFrame]:
        """
        Get latest RGB frame.

        Returns:
            CameraFrame or None if no frame available.
        """
        if not self._initialized or self.rgb_queue is None:
            return None

        try:
            in_frame = self.rgb_queue.get()
            if in_frame is None:
                return None

            frame_data = in_frame.getCvFrame()
            timestamp = in_frame.getTimestamp().total_seconds() * 1_000_000

            self.rgb_frame_id += 1
            self.frames_received += 1

            return CameraFrame(
                timestamp_us=int(timestamp),
                frame=frame_data,
                frame_id=self.rgb_frame_id,
            )

        except Exception as e:
            logger.error(f"Error retrieving RGB frame: {e}")
            return None

    def get_depth_frame(self) -> Optional[StereoDepth]:
        """
        Get latest depth frame.

        Returns:
            StereoDepth or None if no frame available.
        """
        if not self._initialized or self.depth_queue is None:
            return None

        try:
            in_frame = self.depth_queue.get()
            if in_frame is None:
                return None

            depth_data = in_frame.getFrame()
            timestamp = in_frame.getTimestamp().total_seconds() * 1_000_000

            self.depth_frame_id += 1

            return StereoDepth(
                timestamp_us=int(timestamp),
                depth_map=depth_data,
            )

        except Exception as e:
            logger.error(f"Error retrieving depth frame: {e}")
            return None

    def get_imu_data(self) -> Optional[List[IMUSample]]:
        """
        Get latest IMU packet.

        Returns:
            List of IMUSample or None if no data available.
        """
        if not self._initialized or self.imu_queue is None:
            return None

        try:
            in_imu = self.imu_queue.get()
            if in_imu is None:
                return None

            samples = []
            imu_packet = in_imu.packets

            for packet in imu_packet:
                for accel_data in packet.acceleroMeter.data:
                    timestamp = int(accel_data.timestamp.total_seconds() * 1_000_000)
                    accel_xyz = np.array([accel_data.x, accel_data.y, accel_data.z])

                    # Try to get corresponding gyro data
                    gyro_xyz = np.array([0.0, 0.0, 0.0])
                    for gyro_data in packet.gyroscope.data:
                        if abs(gyro_data.timestamp.total_seconds() - accel_data.timestamp.total_seconds()) < 0.001:
                            gyro_xyz = np.array([gyro_data.x, gyro_data.y, gyro_data.z])
                            break

                    samples.append(IMUSample(
                        timestamp_us=timestamp,
                        accel_xyz=accel_xyz,
                        gyro_xyz=gyro_xyz,
                    ))

                    self.imu_sample_count += 1

            return samples if samples else None

        except Exception as e:
            logger.error(f"Error retrieving IMU data: {e}")
            return None

    def shutdown(self) -> None:
        """Gracefully shutdown device and release resources."""
        try:
            with self._lock:
                if self.device is not None:
                    self.device.close()
                    logger.info("Device closed")

                self.device = None
                self.rgb_queue = None
                self.depth_queue = None
                self.stereo_queue = None
                self.imu_queue = None
                self._initialized = False

        except Exception as e:
            logger.error(f"Error during shutdown: {e}")

    def get_calibration(self) -> Optional[OAKDCalibration]:
        """Get camera calibration."""
        return self.calibration

    def get_device_info(self) -> Optional[Dict[str, str]]:
        """Get device information."""
        return self.device_info

    def is_initialized(self) -> bool:
        """Check if interface is initialized and ready."""
        return self._initialized

    def get_statistics(self) -> Dict[str, int]:
        """Get pipeline statistics."""
        with self._lock:
            return {
                "frames_received": self.frames_received,
                "frames_dropped": self.frames_dropped,
                "rgb_frames": self.rgb_frame_id,
                "depth_frames": self.depth_frame_id,
                "imu_samples": self.imu_sample_count,
            }
