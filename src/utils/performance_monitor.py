"""
Performance Monitoring

Responsibilities:
- Track CPU, memory, FPS
- Detect frame drops and timestamp gaps
- Log performance metrics
- Alert on threshold violations

Input:
- Live camera stream or replay
- Configuration thresholds

Output:
- Performance metrics
- Alerts for anomalies
"""

import logging
import time
from typing import Optional, Dict, Any
from collections import deque
from dataclasses import dataclass

import psutil

logger = logging.getLogger(__name__)


@dataclass
class PerformanceSnapshot:
    """Single performance measurement."""
    timestamp: float
    cpu_percent: float
    memory_mb: float
    fps: float
    frame_count: int
    dropped_frames: int


class PerformanceMonitor:
    """Monitor system performance during sensor operation."""

    def __init__(self, config: Optional[Dict[str, Any]] = None):
        """
        Initialize performance monitor.

        Args:
            config: Optional configuration with thresholds
        """
        self.config = config or {}

        # Thresholds
        self.max_memory_mb = self.config.get("alert_memory_mb", 500)
        self.max_cpu_percent = self.config.get("alert_cpu_percent", 80)
        self.max_frame_drop_rate = self.config.get("alert_frame_drop_rate", 0.01)

        # Metrics
        self.frame_count = 0
        self.dropped_frames = 0
        self.start_time = None
        self.last_frame_timestamp = None

        # History (last 60 seconds)
        self.snapshots = deque(maxlen=60)

    def start(self) -> None:
        """Start monitoring."""
        self.start_time = time.time()
        logger.info("Performance monitoring started")

    def record_frame(self, timestamp: Optional[float] = None) -> None:
        """Record frame arrival."""
        current_time = timestamp or time.time()

        # Check for frame timing anomalies
        if self.last_frame_timestamp is not None:
            interval = current_time - self.last_frame_timestamp

            # Expected interval at 30 FPS is ~33.3 ms
            if interval > 0.05:  # > 50 ms is worth noting
                logger.debug(f"Large frame interval: {interval*1000:.1f} ms")
                self.dropped_frames += 1

        self.last_frame_timestamp = current_time
        self.frame_count += 1

    def record_snapshot(self) -> PerformanceSnapshot:
        """Take a performance snapshot."""
        try:
            current_time = time.time()
            elapsed = current_time - self.start_time if self.start_time else 0

            # CPU and memory
            process = psutil.Process()
            cpu_percent = process.cpu_percent(interval=0.1)
            memory_info = process.memory_info()
            memory_mb = memory_info.rss / 1024 / 1024

            # FPS
            fps = self.frame_count / elapsed if elapsed > 0 else 0

            snapshot = PerformanceSnapshot(
                timestamp=current_time,
                cpu_percent=cpu_percent,
                memory_mb=memory_mb,
                fps=fps,
                frame_count=self.frame_count,
                dropped_frames=self.dropped_frames,
            )

            self.snapshots.append(snapshot)

            # Check thresholds
            self._check_thresholds(snapshot)

            return snapshot

        except Exception as e:
            logger.error(f"Error taking performance snapshot: {e}")
            return None

    def _check_thresholds(self, snapshot: PerformanceSnapshot) -> None:
        """Check performance against thresholds."""
        if snapshot.memory_mb > self.max_memory_mb:
            logger.warning(f"Memory usage high: {snapshot.memory_mb:.0f} MB "
                          f"(threshold: {self.max_memory_mb} MB)")

        if snapshot.cpu_percent > self.max_cpu_percent:
            logger.warning(f"CPU usage high: {snapshot.cpu_percent:.1f}% "
                          f"(threshold: {self.max_cpu_percent}%)")

        drop_rate = snapshot.dropped_frames / snapshot.frame_count if snapshot.frame_count > 0 else 0
        if drop_rate > self.max_frame_drop_rate:
            logger.warning(f"Frame drop rate high: {drop_rate*100:.2f}% "
                          f"(threshold: {self.max_frame_drop_rate*100:.2f}%)")

    def get_statistics(self) -> Dict[str, Any]:
        """Get performance statistics."""
        if not self.snapshots:
            return {}

        snapshots = list(self.snapshots)

        cpu_values = [s.cpu_percent for s in snapshots]
        memory_values = [s.memory_mb for s in snapshots]
        fps_values = [s.fps for s in snapshots]

        return {
            "elapsed_sec": time.time() - self.start_time if self.start_time else 0,
            "frame_count": self.frame_count,
            "dropped_frames": self.dropped_frames,
            "drop_rate_percent": (self.dropped_frames / self.frame_count * 100
                                 if self.frame_count > 0 else 0),
            "cpu": {
                "current": cpu_values[-1] if cpu_values else 0,
                "average": sum(cpu_values) / len(cpu_values) if cpu_values else 0,
                "max": max(cpu_values) if cpu_values else 0,
            },
            "memory_mb": {
                "current": memory_values[-1] if memory_values else 0,
                "average": sum(memory_values) / len(memory_values) if memory_values else 0,
                "max": max(memory_values) if memory_values else 0,
            },
            "fps": {
                "current": fps_values[-1] if fps_values else 0,
                "average": sum(fps_values) / len(fps_values) if fps_values else 0,
                "min": min(fps_values) if fps_values else 0,
                "max": max(fps_values) if fps_values else 0,
            },
        }

    def print_report(self) -> None:
        """Print performance report."""
        stats = self.get_statistics()

        if not stats:
            logger.info("No performance data available")
            return

        logger.info("\n" + "="*50)
        logger.info("Performance Report")
        logger.info("="*50)
        logger.info(f"Duration: {stats['elapsed_sec']:.1f} seconds")
        logger.info(f"Frames: {stats['frame_count']} (dropped: {stats['dropped_frames']})")
        logger.info(f"Drop rate: {stats['drop_rate_percent']:.2f}%")
        logger.info("")
        logger.info(f"CPU (%):")
        logger.info(f"  Current: {stats['cpu']['current']:.1f}%")
        logger.info(f"  Average: {stats['cpu']['average']:.1f}%")
        logger.info(f"  Max:     {stats['cpu']['max']:.1f}%")
        logger.info("")
        logger.info(f"Memory (MB):")
        logger.info(f"  Current: {stats['memory_mb']['current']:.1f}")
        logger.info(f"  Average: {stats['memory_mb']['average']:.1f}")
        logger.info(f"  Max:     {stats['memory_mb']['max']:.1f}")
        logger.info("")
        logger.info(f"FPS:")
        logger.info(f"  Current: {stats['fps']['current']:.1f}")
        logger.info(f"  Average: {stats['fps']['average']:.1f}")
        logger.info(f"  Range:   {stats['fps']['min']:.1f} - {stats['fps']['max']:.1f}")
        logger.info("="*50 + "\n")
