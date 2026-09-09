"""
OAK-D Pro Hardware Interface - Simplified, robust version

Works with multiple depthai API versions.
"""

import logging
from typing import Optional, Tuple
from dataclasses import dataclass
import numpy as np

try:
    import depthai as dai
except ImportError:
    dai = None

logger = logging.getLogger(__name__)


@dataclass
class CameraFrame:
    """Container for a single camera frame."""
    timestamp_us: int
    frame: np.ndarray
    frame_id: int


@dataclass
class StereoDepth:
    """Stereo depth frame."""
    timestamp_us: int
    depth_map: np.ndarray


class OAKDInterface:
    """Simplified OAK-D Pro interface."""

    def __init__(self, device_id: Optional[str] = None, config: Optional[dict] = None):
        self.device_id = device_id
        self.config = config or {}

        self.device: Optional[dai.Device] = None
        self.pipeline: Optional[dai.Pipeline] = None

        self.rgb_queue = None
        self.depth_queue = None

        self.calibration = None
        self.device_info = None

        self.rgb_frame_id = 0
        self.depth_frame_id = 0
        self._initialized = False

    def initialize(self) -> bool:
        """Initialize OAK-D Pro."""
        try:
            if dai is None:
                logger.error("depthai not installed")
                return False

            logger.info("Initializing OAK-D interface...")

            # Step 1: Detect device
            if not self._detect_device():
                logger.error("Failed to detect OAK-D Pro")
                return False

            # Step 2: Create and configure pipeline
            if not self._create_pipeline():
                logger.error("Failed to create pipeline")
                return False

            # Step 3: Start device
            if not self._start_device():
                logger.error("Failed to start device")
                return False

            # Step 4: Get calibration
            self._get_calibration()

            self._initialized = True
            logger.info("✓ OAK-D interface initialized successfully")
            return True

        except Exception as e:
            logger.error(f"Initialization error: {e}", exc_info=True)
            self.shutdown()
            return False

    def _detect_device(self) -> bool:
        """Detect OAK-D device."""
        try:
            devices = dai.Device.getAllAvailableDevices()
            if not devices:
                logger.error("No OAK-D devices found")
                return False

            device = devices[0]
            self.device_info = {
                "device": str(device),
                "count": len(devices),
            }
            logger.info(f"Device found: {str(device)}")
            return True

        except Exception as e:
            logger.error(f"Device detection error: {e}")
            return False

    def _create_pipeline(self) -> bool:
        """Create DepthAI pipeline."""
        try:
            self.pipeline = dai.Pipeline()

            # Try different API versions to create color camera
            cam_rgb = None
            try:
                # Try camelCase
                cam_rgb = self.pipeline.createColorCamera()
            except AttributeError:
                try:
                    # Try snake_case
                    cam_rgb = self.pipeline.create_color_camera()
                except AttributeError:
                    logger.error("Could not create color camera - unknown API")
                    return False

            cam_rgb.setBoardSocket(dai.CameraBoardSocket.RGB)
            cam_rgb.setResolution(dai.ColorCameraProperties.SensorInfo.RGB_1280X720)
            cam_rgb.setFps(30)

            # Create stereo depth
            try:
                stereo = self.pipeline.createStereoDepth()
            except AttributeError:
                stereo = self.pipeline.create_stereo_depth()

            stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)

            # Mono cameras
            try:
                mono_left = self.pipeline.createMonoCamera()
                mono_right = self.pipeline.createMonoCamera()
            except AttributeError:
                mono_left = self.pipeline.create_mono_camera()
                mono_right = self.pipeline.create_mono_camera()

            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_left.setResolution(dai.MonoCameraProperties.SensorInfo.THE_400_P)
            mono_left.setFps(30)

            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            mono_right.setResolution(dai.MonoCameraProperties.SensorInfo.THE_400_P)
            mono_right.setFps(30)

            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            # Output queues
            try:
                xout_rgb = self.pipeline.createXLinkOut()
                xout_depth = self.pipeline.createXLinkOut()
            except AttributeError:
                xout_rgb = self.pipeline.create_xlink_out()
                xout_depth = self.pipeline.create_xlink_out()

            xout_rgb.setStreamName("rgb")
            xout_depth.setStreamName("depth")

            cam_rgb.video.link(xout_rgb.input)
            stereo.depth.link(xout_depth.input)

            logger.info("✓ Pipeline created successfully")
            return True

        except Exception as e:
            logger.error(f"Pipeline creation error: {e}", exc_info=True)
            return False

    def _start_device(self) -> bool:
        """Start device and queues."""
        try:
            self.device = dai.Device(self.pipeline)

            self.rgb_queue = self.device.getOutputQueue(name="rgb", maxSize=4, blocking=False)
            self.depth_queue = self.device.getOutputQueue(name="depth", maxSize=4, blocking=False)

            logger.info("✓ Device started and queues created")
            return True

        except Exception as e:
            logger.error(f"Device start error: {e}", exc_info=True)
            return False

    def _get_calibration(self):
        """Get camera calibration."""
        try:
            calib = self.device.readCalibration()
            self.calibration = calib
            logger.info("✓ Calibration retrieved")
        except Exception as e:
            logger.warning(f"Could not retrieve calibration: {e}")

    def get_rgb_frame(self) -> Optional[CameraFrame]:
        """Get latest RGB frame."""
        if not self._initialized or self.rgb_queue is None:
            return None

        try:
            in_frame = self.rgb_queue.get()
            if in_frame is None:
                return None

            frame_data = in_frame.getCvFrame()
            timestamp = int(in_frame.getTimestamp().total_seconds() * 1_000_000)

            self.rgb_frame_id += 1

            return CameraFrame(
                timestamp_us=timestamp,
                frame=frame_data,
                frame_id=self.rgb_frame_id,
            )

        except Exception as e:
            logger.error(f"Error getting RGB frame: {e}")
            return None

    def get_depth_frame(self) -> Optional[StereoDepth]:
        """Get latest depth frame."""
        if not self._initialized or self.depth_queue is None:
            return None

        try:
            in_frame = self.depth_queue.get()
            if in_frame is None:
                return None

            depth_data = in_frame.getFrame()
            timestamp = int(in_frame.getTimestamp().total_seconds() * 1_000_000)

            self.depth_frame_id += 1

            return StereoDepth(
                timestamp_us=timestamp,
                depth_map=depth_data,
            )

        except Exception as e:
            logger.error(f"Error getting depth frame: {e}")
            return None

    def get_calibration(self):
        """Get camera calibration."""
        return self.calibration

    def get_device_info(self) -> Optional[dict]:
        """Get device information."""
        return self.device_info

    def is_initialized(self) -> bool:
        """Check if initialized."""
        return self._initialized

    def shutdown(self) -> None:
        """Shutdown device."""
        try:
            if self.device is not None:
                self.device.close()
                logger.info("✓ Device closed")
            self._initialized = False
        except Exception as e:
            logger.error(f"Shutdown error: {e}")

    def get_statistics(self) -> dict:
        """Get statistics."""
        return {
            "rgb_frames": self.rgb_frame_id,
            "depth_frames": self.depth_frame_id,
        }
