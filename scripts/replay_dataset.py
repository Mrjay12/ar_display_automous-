#!/usr/bin/env python3
"""
Replay Recorded Sensor Dataset

Usage:
    python scripts/replay_dataset.py --dataset data/recordings/my_dataset
"""

import logging
import argparse
from pathlib import Path

from src.camera.sensor_replay import SensorReplayer
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Replay dataset."""
    parser = argparse.ArgumentParser(description="Replay recorded sensor dataset")
    parser.add_argument("--dataset", type=Path, required=True, help="Dataset directory")
    parser.add_argument("--output-mode", choices=["display", "info"], default="info",
                       help="Output mode")

    args = parser.parse_args()

    # Setup logging
    setup_logging()

    logger.info("="*60)
    logger.info("Replaying Sensor Dataset")
    logger.info("="*60)

    if not args.dataset.exists():
        logger.error(f"Dataset not found: {args.dataset}")
        return 1

    # Create replayer
    replayer = SensorReplayer(args.dataset)

    # Load dataset info
    dataset_info = replayer.get_dataset_info()

    if dataset_info:
        logger.info(f"Dataset info:")
        recording_info = dataset_info.get("recording_info", {})
        logger.info(f"  Start time: {recording_info.get('start_time', 'unknown')}")
        logger.info(f"  Duration: {recording_info.get('duration_sec', 'unknown')} seconds")

        sensor_stats = dataset_info.get("sensor_stats", {})
        logger.info(f"  RGB frames: {sensor_stats.get('rgb_frames', 'unknown')}")
        logger.info(f"  IMU samples: {sensor_stats.get('imu_samples', 'unknown')}")

    # Verify calibration
    calibration = replayer.get_calibration()
    if calibration:
        logger.info(f"✓ Calibration available")
        logger.info(f"  Baseline: {calibration.baseline_mm:.1f} mm")
    else:
        logger.warning("✗ Calibration not found in dataset")

    # Replay
    frame_count = 0
    imu_count = 0

    try:
        logger.info("")
        logger.info("Replaying...")

        while not replayer.is_complete():
            # Get frames
            rgb_frame = replayer.get_rgb_frame()
            if rgb_frame:
                frame_count += 1

            depth_frame = replayer.get_depth_frame()

            imu_data = replayer.get_imu_data()
            if imu_data:
                imu_count += len(imu_data)

            # Log progress
            if frame_count % 300 == 0:
                logger.info(f"  {frame_count} RGB frames, {imu_count} IMU samples")

        logger.info("")
        logger.info("Replay Summary:")
        logger.info(f"  RGB frames: {frame_count}")
        logger.info(f"  IMU samples: {imu_count}")
        logger.info("✓ Replay completed successfully")
        return 0

    except Exception as e:
        logger.error(f"✗ Replay error: {e}", exc_info=True)
        return 1


if __name__ == "__main__":
    exit(main())
