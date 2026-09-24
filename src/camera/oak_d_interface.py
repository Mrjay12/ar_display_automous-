"""
OAK-D Pro Hardware Interface - depthai 3.x API
"""

import logging
from typing import Optional
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
    """OAK-D Pro interface for depthai 3.x using factory pattern."""

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

            if not self._detect_device():
                logger.error("Failed to detect OAK-D Pro")
                return False

            if not self._create_pipeline():
                logger.error("Failed to create pipeline")
                return False

            if not self._start_device():
                logger.error("Failed to start device")
                return False

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
            self.device_info = {"device": str(device), "count": len(devices)}
            logger.info(f"Device found: {str(device)}")
            return True

        except Exception as e:
            logger.error(f"Device detection error: {e}")
            return False

    def _create_pipeline(self) -> bool:
        """Create DepthAI pipeline using depthai 3.x factory pattern."""
        try:
            self.pipeline = dai.Pipeline()

            # Mono cameras for stereo-based localization (no RGB color camera)
            mono_left = self.pipeline.create(dai.node.MonoCamera)
            mono_left.setBoardSocket(dai.CameraBoardSocket.LEFT)
            mono_left.setFps(30)

            # Try to set mono resolution - different depthai versions use different APIs
            try:
                mono_left.setResolution(dai.MonoCameraProperties.SensorInfo.THE_400_P)
            except AttributeError:
                try:
                    mono_left.setResolution(640, 400)
                except:
                    logger.warning("Could not set mono_left resolution - using default")

            mono_right = self.pipeline.create(dai.node.MonoCamera)
            mono_right.setBoardSocket(dai.CameraBoardSocket.RIGHT)
            mono_right.setFps(30)

            # Same resolution setting for mono_right
            try:
                mono_right.setResolution(dai.MonoCameraProperties.SensorInfo.THE_400_P)
            except AttributeError:
                try:
                    mono_right.setResolution(640, 400)
                except:
                    logger.warning("Could not set mono_right resolution - using default")

            # Create stereo depth from mono cameras
            stereo = self.pipeline.create(dai.node.StereoDepth)

            # Try to set preset - PresetMode may vary by depthai version
            try:
                stereo.setDefaultProfilePreset(dai.node.StereoDepth.PresetMode.HIGH_DENSITY)
            except AttributeError:
                try:
                    stereo.setDefaultProfilePreset(dai.node.StereoDepth.HIGH_DENSITY)
                except AttributeError:
                    try:
                        for preset_name in ['HIGH_DENSITY', 'HIGH_PRECISION', 'MEDIUM_DENSITY']:
                            if hasattr(dai.node.StereoDepth, preset_name):
                                stereo.setDefaultProfilePreset(getattr(dai.node.StereoDepth, preset_name))
                                break
                    except:
                        logger.warning("Could not set stereo preset - using defaults")

            # Link stereo cameras to stereo depth node
            mono_left.out.link(stereo.left)
            mono_right.out.link(stereo.right)

            # Stereo left output (for visual features/localization)
            xout_left = self.pipeline.create(dai.node.XLinkOut)
            xout_left.setStreamName("rgb")  # Use "rgb" stream name for compatibility
            mono_left.out.link(xout_left.input)

            # Depth output
            xout_depth = self.pipeline.create(dai.node.XLinkOut)
            xout_depth.setStreamName("depth")
            stereo.depth.link(xout_depth.input)

            logger.info("✓ Pipeline created successfully (stereo cameras only)")
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
