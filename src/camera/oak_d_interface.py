"""
OAK-D Pro Hardware Interface
DepthAI 3.x

Provides:
    - RGB frames
    - Stereo depth frames
    - RGB/depth synchronization
    - Depth aligned to RGB
    - Camera calibration
    - Device information
    - Runtime statistics

Designed for OAK-D Pro / DepthAI 3.x.
"""

from __future__ import annotations

import logging
from dataclasses import dataclass
from datetime import timedelta
from typing import Any, Optional

import numpy as np

try:
    import depthai as dai
except ImportError:
    dai = None


logger = logging.getLogger(__name__)


# ============================================================================
# DATA TYPES
# ============================================================================

@dataclass
class CameraFrame:
    """RGB camera frame."""

    timestamp_us: int
    frame: np.ndarray
    frame_id: int


@dataclass
class StereoDepth:
    """Depth frame aligned to RGB."""

    timestamp_us: int
    depth_map: np.ndarray
    frame_id: int


@dataclass
class RGBDFrame:
    """
    Synchronized RGB + depth frame.

    The depth image is aligned to the RGB image.
    Depth computed from stereo pair (left + right mono cameras).
    RGB provides visual context for spatial visualization overlay.
    """

    timestamp_us: int
    rgb: np.ndarray
    depth: np.ndarray
    frame_id: int


# ============================================================================
# OAK-D INTERFACE
# ============================================================================

class OAKDInterface:
    """
    OAK-D Pro interface for DepthAI 3.x - Stereo spatial with RGB overlay.

    Standard OAK-D camera layout:

        CAM_A -> RGB (visualization background)
        CAM_B -> LEFT mono (stereo baseline for depth)
        CAM_C -> RIGHT mono (stereo pair for depth)

    Pipeline:

        RGB
          \
           -> Sync -> RGBDFrame (RGB with RGB-aligned depth)
          /
        Stereo -> ImageAlign

    Spatial info from stereo depth, visual context from RGB.
    """

    def __init__(
        self,
        device_id: Optional[str] = None,
        config: Optional[dict[str, Any]] = None,
    ):
        self.device_id = device_id
        self.config = config or {}

        # DepthAI objects
        self.pipeline: Optional[Any] = None
        self.device: Optional[Any] = None

        # Output queue
        self.rgbd_queue = None

        # Calibration
        self.calibration = None

        # Device information
        self.device_info: Optional[dict[str, Any]] = None

        # Frame counters
        self.rgb_frame_id = 0
        self.depth_frame_id = 0
        self.rgbd_frame_id = 0

        # State
        self._initialized = False
        self._pipeline_started = False

    # ========================================================================
    # INITIALIZATION
    # ========================================================================

    def initialize(self) -> bool:
        """Initialize the OAK-D Pro."""

        if self._initialized:
            logger.warning("OAK-D is already initialized")
            return True

        try:
            if dai is None:
                logger.error(
                    "DepthAI is not installed in the current Python environment."
                )
                return False

            logger.info(
                "Initializing OAK-D Pro using DepthAI %s...",
                getattr(dai, "__version__", "unknown"),
            )

            if not self._detect_device():
                return False

            if not self._create_pipeline():
                self.shutdown()
                return False

            if not self._start_pipeline():
                self.shutdown()
                return False

            self._get_calibration()

            self._initialized = True

            logger.info(
                "OAK-D Pro initialized successfully"
            )

            return True

        except Exception:
            logger.exception(
                "OAK-D initialization failed"
            )
            self.shutdown()
            return False

    # ========================================================================
    # DEVICE DETECTION
    # ========================================================================

    def _detect_device(self) -> bool:
        """Detect connected OAK devices."""

        try:
            devices = dai.Device.getAllAvailableDevices()

            if not devices:
                logger.error(
                    "No OAK-D devices found."
                )
                return False

            logger.info(
                "Found %d OAK device(s)",
                len(devices),
            )

            selected_device = None

            # ---------------------------------------------------------------
            # Specific device requested
            # ---------------------------------------------------------------

            if self.device_id:
                for device in devices:
                    device_string = str(device)

                    if self.device_id in device_string:
                        selected_device = device
                        break

                if selected_device is None:
                    logger.error(
                        "Requested device '%s' was not found.",
                        self.device_id,
                    )
                    return False

            # ---------------------------------------------------------------
            # Otherwise use first device
            # ---------------------------------------------------------------

            else:
                selected_device = devices[0]

            self.device_info = {
                "device": str(selected_device),
                "device_count": len(devices),
            }

            logger.info(
                "Selected device: %s",
                selected_device,
            )

            return True

        except Exception:
            logger.exception(
                "Device detection failed"
            )
            return False

    # ========================================================================
    # PIPELINE CREATION
    # ========================================================================

    def _create_pipeline(self) -> bool:
        """Create the DepthAI 3.x pipeline."""

        try:
            logger.info(
                "Creating DepthAI v3 pipeline..."
            )

            self.pipeline = dai.Pipeline()

            # =================================================================
            # RGB CAMERA (visualization background with spatial overlay)
            # =================================================================

            rgb_camera = self.pipeline.create(
                dai.node.Camera
            )

            rgb_camera.build(
                dai.CameraBoardSocket.CAM_A
            )

            rgb_output = rgb_camera.requestOutput(
                size=(1280, 720),
                type=dai.ImgFrame.Type.BGR888p,
                resizeMode=dai.ImgResizeMode.CROP,
                fps=30,
                enableUndistortion=True,
            )

            # =================================================================
            # LEFT CAMERA (stereo baseline for depth)
            # =================================================================

            left_mono = self.pipeline.create(
                dai.node.Camera
            )

            left_mono.build(
                dai.CameraBoardSocket.CAM_B
            )

            left_output = left_mono.requestOutput(
                size=(640, 400),
                type=dai.ImgFrame.Type.GRAY8,
                resizeMode=dai.ImgResizeMode.CROP,
                fps=30,
            )

            # =================================================================
            # RIGHT CAMERA (stereo pair for depth)
            # =================================================================

            right_mono = self.pipeline.create(
                dai.node.Camera
            )

            right_mono.build(
                dai.CameraBoardSocket.CAM_C
            )

            right_output = right_mono.requestOutput(
                size=(640, 400),
                type=dai.ImgFrame.Type.GRAY8,
                resizeMode=dai.ImgResizeMode.CROP,
                fps=30,
            )

            # =================================================================
            # STEREO DEPTH (spatial detection)
            # =================================================================

            stereo = self.pipeline.create(
                dai.node.StereoDepth
            )

            # Use robotics preset if available.
            try:
                stereo.setDefaultProfilePreset(
                    dai.node.StereoDepth.PresetMode.ROBOTICS
                )

            except AttributeError:
                logger.warning(
                    "ROBOTICS stereo preset unavailable. "
                    "Using default stereo configuration."
                )

            # Left-right consistency check.
            try:
                stereo.setLeftRightCheck(True)
            except Exception:
                logger.debug(
                    "Left-right check unavailable."
                )

            # Extended disparity.
            extended_disparity = bool(
                self.config.get(
                    "extended_disparity",
                    False,
                )
            )

            try:
                stereo.setExtendedDisparity(
                    extended_disparity
                )
            except Exception:
                logger.debug(
                    "Extended disparity configuration unavailable."
                )

            # Subpixel depth.
            subpixel = bool(
                self.config.get(
                    "subpixel",
                    True,
                )
            )

            try:
                stereo.setSubpixel(
                    subpixel
                )
            except Exception:
                logger.debug(
                    "Subpixel configuration unavailable."
                )

            # Connect stereo cameras.
            left_output.link(
                stereo.left
            )

            right_output.link(
                stereo.right
            )

            # =================================================================
            # DEPTH ALIGNMENT
            # =================================================================

            image_align = self.pipeline.create(
                dai.node.ImageAlign
            )

            # Stereo depth becomes ImageAlign input.
            stereo.depth.link(
                image_align.input
            )

            # RGB camera determines target alignment.
            rgb_output.link(
                image_align.inputAlignTo
            )

            # =================================================================
            # RGB + DEPTH SYNCHRONIZATION
            # =================================================================

            sync = self.pipeline.create(
                dai.node.Sync
            )

            try:
                sync.setRunOnHost(True)
            except Exception:
                logger.debug(
                    "Sync.setRunOnHost unavailable."
                )

            try:
                sync.setSyncThreshold(
                    timedelta(milliseconds=50)
                )
            except Exception:
                logger.debug(
                    "Sync threshold configuration unavailable."
                )

            try:
                sync.setSyncAttempts(10)
            except Exception:
                logger.debug(
                    "Sync attempts configuration unavailable."
                )

            # RGB camera (visualization background)
            rgb_output.link(
                sync.inputs["rgb"]
            )

            # RGB-aligned depth (spatial information)
            image_align.outputAligned.link(
                sync.inputs["depth"]
            )

            # =================================================================
            # OUTPUT QUEUE
            # =================================================================

            self.rgbd_queue = (
                sync.out.createOutputQueue(
                    maxSize=4,
                    blocking=False,
                )
            )

            logger.info(
                "DepthAI pipeline created successfully."
            )

            return True

        except Exception:
            logger.exception(
                "Pipeline creation failed"
            )

            self.pipeline = None
            self.rgbd_queue = None

            return False

    # ========================================================================
    # START PIPELINE
    # ========================================================================

    def _start_pipeline(self) -> bool:
        """Start the DepthAI pipeline."""

        if self.pipeline is None:
            logger.error(
                "Cannot start pipeline: pipeline is None."
            )
            return False

        try:
            logger.info(
                "Starting OAK-D pipeline..."
            )

            self.pipeline.start()

            self._pipeline_started = True

            logger.info(
                "OAK-D pipeline started successfully."
            )

            return True

        except Exception:
            logger.exception(
                "Failed to start OAK-D pipeline"
            )

            self._pipeline_started = False

            return False

    # ========================================================================
    # CALIBRATION
    # ========================================================================

    def _get_calibration(self) -> None:
        """Retrieve calibration data with fallback to defaults."""

        try:
            # Try to retrieve from running device
            if self.device is None and self.pipeline is not None:
                device_fn = getattr(
                    self.pipeline,
                    "getDevice",
                    None,
                )

                if callable(device_fn):
                    try:
                        self.device = device_fn()
                    except Exception:
                        pass

            if self.device is not None:
                try:
                    self.calibration = (
                        self.device.readCalibration()
                    )

                    logger.info(
                        "Camera calibration retrieved from device."
                    )
                    return
                except Exception as e:
                    logger.debug(
                        "Could not read calibration from device: %s",
                        e,
                    )

            # Fallback: construct reasonable defaults
            logger.info(
                "Using default camera matrix (OAK-D Pro specs)."
            )

            self.calibration = (
                self._get_default_calibration()
            )

        except Exception:
            logger.exception(
                "Failed to retrieve camera calibration. "
                "Using defaults."
            )

            self.calibration = (
                self._get_default_calibration()
            )

    def _get_default_calibration(self) -> np.ndarray:
        """
        Return default camera intrinsic matrix for OAK-D Pro.

        Based on typical OAK-D Pro specs with 640x400 stereo resolution.
        Focal length ~1380 px (normalized to resolution).
        Principal point at image center.
        """

        # Output resolution from pipeline
        width = 1280
        height = 720

        # Approximate focal length in pixels
        # OAK-D Pro baseline ~75mm, typical depth range 0.2-5m
        fx = width * 1.08  # ~1382 pixels at 1280x720
        fy = height * 1.08

        # Principal point at center
        cx = width / 2.0
        cy = height / 2.0

        return np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1],
        ], dtype=np.float32)

    # ========================================================================
    # RGBD FRAME
    # ========================================================================

    def get_rgbd_frame(
        self,
        timeout_ms: int = 100,
    ) -> Optional[RGBDFrame]:
        """
        Return one synchronized RGB/depth frame.

        Returns:
            RGBDFrame or None if no complete synchronized frame
            is currently available.
        """

        if not self._initialized:
            return None

        if self.rgbd_queue is None:
            return None

        try:
            # Non-blocking queue.
            message_group = self.rgbd_queue.tryGet()

            if message_group is None:
                return None

            # ---------------------------------------------------------------
            # Extract RGB
            # ---------------------------------------------------------------

            try:
                rgb_message = message_group["rgb"]
            except Exception:
                rgb_message = None

            # ---------------------------------------------------------------
            # Extract depth
            # ---------------------------------------------------------------

            try:
                depth_message = message_group["depth"]
            except Exception:
                depth_message = None

            if rgb_message is None:
                logger.debug(
                    "Synchronized group has no RGB frame."
                )
                return None

            if depth_message is None:
                logger.debug(
                    "Synchronized group has no depth frame."
                )
                return None

            # ---------------------------------------------------------------
            # Convert frames
            # ---------------------------------------------------------------

            rgb = rgb_message.getCvFrame()

            depth = depth_message.getFrame()

            if rgb is None:
                return None

            if depth is None:
                return None

            # ---------------------------------------------------------------
            # Timestamp
            # ---------------------------------------------------------------

            timestamp_us = self._get_timestamp_us(
                rgb_message
            )

            self.rgbd_frame_id += 1

            return RGBDFrame(
                timestamp_us=timestamp_us,
                rgb=rgb,
                depth=depth,
                frame_id=self.rgbd_frame_id,
            )

        except Exception:
            logger.exception(
                "Error retrieving RGBD frame"
            )
            return None

    # ========================================================================
    # RGB FRAME
    # ========================================================================

    def get_rgb_frame(
        self,
    ) -> Optional[CameraFrame]:
        """
        Get the latest synchronized RGB frame.

        NOTE:
        This consumes one RGBD synchronization group.
        For applications needing both RGB and depth,
        use get_rgbd_frame() instead.
        """

        rgbd = self.get_rgbd_frame()

        if rgbd is None:
            return None

        self.rgb_frame_id += 1

        return CameraFrame(
            timestamp_us=rgbd.timestamp_us,
            frame=rgbd.rgb,
            frame_id=self.rgb_frame_id,
        )

    # ========================================================================
    # DEPTH FRAME
    # ========================================================================

    def get_depth_frame(
        self,
    ) -> Optional[StereoDepth]:
        """
        Get the latest synchronized depth frame.

        NOTE:
        This consumes one RGBD synchronization group.
        For applications needing both RGB and depth,
        use get_rgbd_frame() instead.
        """

        rgbd = self.get_rgbd_frame()

        if rgbd is None:
            return None

        self.depth_frame_id += 1

        return StereoDepth(
            timestamp_us=rgbd.timestamp_us,
            depth_map=rgbd.depth,
            frame_id=self.depth_frame_id,
        )

    # ========================================================================
    # TIMESTAMP
    # ========================================================================

    @staticmethod
    def _get_timestamp_us(
        frame: Any,
    ) -> int:
        """
        Convert DepthAI frame timestamp to microseconds.

        This is a device/monotonic timestamp.
        It is NOT Unix epoch time.
        """

        timestamp = frame.getTimestamp()

        return int(
            timestamp.total_seconds()
            * 1_000_000
        )

    # ========================================================================
    # CALIBRATION ACCESS
    # ========================================================================

    def get_calibration(self) -> Any:
        """Return camera calibration."""

        return self.calibration

    # ========================================================================
    # DEVICE INFORMATION
    # ========================================================================

    def get_device_info(
        self,
    ) -> Optional[dict[str, Any]]:
        """Return device information."""

        return self.device_info

    # ========================================================================
    # STATUS
    # ========================================================================

    def is_initialized(self) -> bool:
        """Return initialization state."""

        return self._initialized

    # ========================================================================
    # STATISTICS
    # ========================================================================

    def get_statistics(self) -> dict[str, Any]:
        """Return runtime statistics."""

        return {
            "initialized": self._initialized,
            "pipeline_started": self._pipeline_started,
            "rgb_frames": self.rgb_frame_id,
            "depth_frames": self.depth_frame_id,
            "rgbd_frames": self.rgbd_frame_id,
        }

    def diagnose(self) -> dict[str, Any]:
        """Return detailed diagnostic information."""

        diagnostics = {
            "initialized": self._initialized,
            "pipeline_started": self._pipeline_started,
            "pipeline": self.pipeline is not None,
            "device": self.device is not None,
            "rgbd_queue": self.rgbd_queue is not None,
            "calibration": self.calibration is not None,
            "device_info": self.device_info,
            "stats": self.get_statistics(),
        }

        # Try to check queue status
        if self.rgbd_queue is not None:
            try:
                msg = self.rgbd_queue.tryGet()
                diagnostics["queue_has_data"] = msg is not None
            except Exception as e:
                diagnostics["queue_error"] = str(e)

        return diagnostics

    # ========================================================================
    # SHUTDOWN
    # ========================================================================

    def shutdown(self) -> None:
        """Safely shut down the OAK-D."""

        try:
            if self.pipeline is not None:

                try:
                    self.pipeline.stop()

                except Exception:
                    logger.debug(
                        "Pipeline stop failed or "
                        "pipeline was already stopped.",
                        exc_info=True,
                    )

            self.rgbd_queue = None

            self.pipeline = None
            self.device = None

            self._pipeline_started = False
            self._initialized = False

            logger.info(
                "OAK-D interface shut down."
            )

        except Exception:
            logger.exception(
                "Error during OAK-D shutdown."
            )

            self.rgbd_queue = None
            self.pipeline = None
            self.device = None

            self._pipeline_started = False
            self._initialized = False

    # ========================================================================
    # CONTEXT MANAGER
    # ========================================================================

    def __enter__(self) -> "OAKDInterface":
        """Initialize when entering a context."""

        if not self.initialize():
            raise RuntimeError(
                "Failed to initialize OAK-D Pro."
            )

        return self

    def __exit__(
        self,
        exc_type,
        exc_value,
        traceback,
    ) -> None:
        """Shutdown when leaving a context."""

        self.shutdown()