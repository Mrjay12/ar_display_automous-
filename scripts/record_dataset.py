#!/usr/bin/env python3
"""
Record Synchronized Sensor Dataset

Usage:
    python scripts/record_dataset.py --duration 300 --output data/recordings/my_dataset
"""

import logging
import argparse
from pathlib import Path

from src.camera.oak_d_interface import OAKDInterface
from src.camera.sensor_recording import SensorRecorder
from src.config.config_loader import ConfigLoader
from src.utils.logging_config import setup_logging

logger = logging.getLogger(__name__)


def main():
    """Record dataset."""
    parser = argparse.ArgumentParser(description="Record synchronized sensor dataset")
    parser.add_argument("--duration", type=float, default=300, help="Recording duration (seconds)")
    parser.add_argument("--output", type=Path, default=Path("data/recordings/dataset"),
                       help="Output directory")
    parser.add_argument("--device-id", type=str, default=None, help="OAK-D device ID")

    args = parser.parse_args()

    # Setup logging
    config_loader = ConfigLoader()
    logging_config = config_loader.config.get("logging", {})
    setup_logging(logging_config)

    logger.info("="*60)
    logger.info("Recording Synchronized Sensor Dataset")
    logger.info("="*60)

    # Load configuration
    config_loader.load_all()
    camera_config = config_loader.camera

    logger.info(f"Output directory: {args.output}")
    logger.info(f"Duration: {args.duration} seconds")

    # Initialize OAK-D
    oak_d = OAKDInterface(device_id=args.device_id, config=camera_config)

    if not oak_d.initialize():
        logger.error("Failed to initialize OAK-D Pro")
        return 1

    device_info = oak_d.get_device_info()
    logger.info(f"Device: {device_info['name']} ({device_info['mxId']})")

    # Create recorder
    args.output.mkdir(parents=True, exist_ok=True)
    recorder = SensorRecorder(args.output, oak_d, config=camera_config)

    # Record
    try:
        success = recorder.record(args.duration)

        if success:
            stats = recorder.get_statistics()
            logger.info("")
            logger.info("Recording Summary:")
            logger.info(f"  Duration: {stats['elapsed_sec']:.1f} seconds")
            logger.info(f"  RGB frames: {stats['rgb_frames']}")
            logger.info(f"  IMU samples: {stats['imu_samples']}")
            logger.info(f"  Output: {stats['output_dir']}")
            logger.info("✓ Recording completed successfully")
            return 0
        else:
            logger.error("✗ Recording failed")
            return 1

    except KeyboardInterrupt:
        logger.info("Recording interrupted by user")
        recorder.stop()
        return 0

    finally:
        oak_d.shutdown()


if __name__ == "__main__":
    exit(main())
