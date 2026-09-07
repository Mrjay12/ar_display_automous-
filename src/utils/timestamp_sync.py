"""
Timestamp Synchronization

Responsibilities:
- Align multi-sensor frames by timestamp
- Detect synchronization issues
- Associate frames from different sensors
- Handle timestamp gaps

Input:
- Timestamped frames from multiple sensors

Output:
- Synchronized frame groups
- Synchronization quality metrics
"""

import logging
from typing import Optional, List, Tuple, Dict, Any
from dataclasses import dataclass

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class SynchronizedFrame:
    """Group of synchronized frames from different sensors."""
    timestamp_us: int
    rgb_frame: Optional[Any] = None
    depth_frame: Optional[Any] = None
    stereo_left_frame: Optional[Any] = None
    stereo_right_frame: Optional[Any] = None
    imu_samples: Optional[List[Any]] = None

    def get_timestamp_sec(self) -> float:
        """Return timestamp in seconds."""
        return self.timestamp_us / 1_000_000.0


class TimestampSynchronizer:
    """Synchronize timestamps across multiple sensor streams."""

    def __init__(self, max_frame_interval_ms: float = 100.0):
        """
        Initialize synchronizer.

        Args:
            max_frame_interval_ms: Maximum acceptable time between frames (ms)
        """
        self.max_frame_interval_us = max_frame_interval_ms * 1000  # Convert to microseconds

        # Buffers for each stream
        self.rgb_buffer = []
        self.depth_buffer = []
        self.stereo_left_buffer = []
        self.stereo_right_buffer = []
        self.imu_buffer = []

        # Statistics
        self.sync_errors = 0
        self.frames_synchronized = 0

    def add_rgb_frame(self, frame: Any) -> None:
        """Add RGB frame to synchronization buffer."""
        self.rgb_buffer.append(("rgb", frame.timestamp_us, frame))

    def add_depth_frame(self, frame: Any) -> None:
        """Add depth frame to synchronization buffer."""
        self.depth_buffer.append(("depth", frame.timestamp_us, frame))

    def add_imu_sample(self, sample: Any) -> None:
        """Add IMU sample to synchronization buffer."""
        self.imu_buffer.append(("imu", sample.timestamp_us, sample))

    def synchronize_by_rgb(self) -> Optional[SynchronizedFrame]:
        """
        Create synchronized frame group based on RGB frame timestamp.

        Returns:
            SynchronizedFrame if sufficient data available, None otherwise.
        """
        if not self.rgb_buffer:
            return None

        try:
            # Get oldest RGB frame
            _, rgb_timestamp, rgb_frame = self.rgb_buffer.pop(0)

            # Find closest depth frame
            depth_frame = self._find_closest_frame(self.depth_buffer, rgb_timestamp)

            # Find IMU samples within time window
            imu_samples = self._find_imu_samples(rgb_timestamp)

            synced = SynchronizedFrame(
                timestamp_us=rgb_timestamp,
                rgb_frame=rgb_frame,
                depth_frame=depth_frame,
                imu_samples=imu_samples,
            )

            self.frames_synchronized += 1
            return synced

        except Exception as e:
            logger.error(f"Synchronization error: {e}")
            self.sync_errors += 1
            return None

    def _find_closest_frame(self, buffer: List[Tuple[str, int, Any]],
                           target_timestamp_us: int) -> Optional[Any]:
        """Find frame closest to target timestamp."""
        if not buffer:
            return None

        closest_idx = 0
        min_diff = abs(buffer[0][1] - target_timestamp_us)

        for i, (_, timestamp_us, frame) in enumerate(buffer):
            diff = abs(timestamp_us - target_timestamp_us)

            if diff < min_diff:
                min_diff = diff
                closest_idx = i

            # If frame is significantly ahead, stop searching
            if timestamp_us > target_timestamp_us + self.max_frame_interval_us:
                break

        # Only return if close enough
        if min_diff <= self.max_frame_interval_us:
            return buffer[closest_idx][2]

        return None

    def _find_imu_samples(self, target_timestamp_us: int,
                         window_ms: float = 50.0) -> List[Any]:
        """Find IMU samples within time window of target timestamp."""
        window_us = int(window_ms * 1000)
        samples = []

        for _, timestamp_us, sample in self.imu_buffer:
            if abs(timestamp_us - target_timestamp_us) <= window_us:
                samples.append(sample)

        # Clean up old samples
        self.imu_buffer = [
            item for item in self.imu_buffer
            if item[1] >= target_timestamp_us - window_us
        ]

        return samples if samples else None

    def check_synchronization_quality(self) -> Dict[str, Any]:
        """Assess synchronization quality."""
        return {
            "frames_synchronized": self.frames_synchronized,
            "sync_errors": self.sync_errors,
            "error_rate": (self.sync_errors / (self.sync_errors + self.frames_synchronized)
                          if (self.sync_errors + self.frames_synchronized) > 0 else 0),
            "rgb_buffer_size": len(self.rgb_buffer),
            "depth_buffer_size": len(self.depth_buffer),
            "imu_buffer_size": len(self.imu_buffer),
        }

    def clear_buffers(self) -> None:
        """Clear all synchronization buffers."""
        self.rgb_buffer.clear()
        self.depth_buffer.clear()
        self.stereo_left_buffer.clear()
        self.stereo_right_buffer.clear()
        self.imu_buffer.clear()
        logger.debug("Synchronization buffers cleared")


class TimestampValidator:
    """Validate timestamp integrity across streams."""

    @staticmethod
    def validate_monotonic_increase(timestamps: List[int]) -> Tuple[bool, Optional[str]]:
        """
        Check if timestamps monotonically increase.

        Returns:
            (is_valid, error_message)
        """
        for i in range(1, len(timestamps)):
            if timestamps[i] <= timestamps[i-1]:
                return False, f"Timestamp went backward at index {i}: {timestamps[i-1]} -> {timestamps[i]}"

        return True, None

    @staticmethod
    def check_timestamp_gaps(timestamps: List[int],
                            max_gap_us: int = 100000) -> List[Tuple[int, int, int]]:
        """
        Find timestamp gaps larger than threshold.

        Returns:
            List of (index, gap_us, expected_gap_us)
        """
        gaps = []

        # Estimate expected gap from first few frames
        if len(timestamps) < 2:
            return gaps

        typical_gaps = []
        for i in range(1, min(100, len(timestamps))):
            typical_gaps.append(timestamps[i] - timestamps[i-1])

        expected_gap = int(np.median(typical_gaps)) if typical_gaps else 33333  # Default 33.3 ms

        for i in range(1, len(timestamps)):
            gap = timestamps[i] - timestamps[i-1]

            if gap > max_gap_us or gap < expected_gap / 2:
                gaps.append((i, gap, expected_gap))

        return gaps

    @staticmethod
    def detect_impossible_timestamps(timestamps: List[int]) -> List[Tuple[int, int]]:
        """Detect timestamps that are in the future or otherwise impossible."""
        import time
        current_time_us = int(time.time() * 1_000_000)

        impossible = []
        for i, ts in enumerate(timestamps):
            if ts < 0 or ts > current_time_us + 1_000_000:  # Allow 1 second future
                impossible.append((i, ts))

        return impossible
