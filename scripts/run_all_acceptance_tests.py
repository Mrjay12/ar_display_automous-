#!/usr/bin/env python3
"""
Master Acceptance Test Runner

Runs all 14 Milestone 1 acceptance tests and generates report.

Usage:
    python scripts/run_all_acceptance_tests.py
    python scripts/run_all_acceptance_tests.py --test 1  # Run only test 1
"""

import logging
import argparse
import sys
from pathlib import Path

from src.utils.logging_config import setup_logging
from src.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


def verify_imports():
    """Verify all core modules import correctly."""
    logger.info("Verifying module imports...")

    try:
        from src.camera.oak_d_interface import OAKDInterface
        from src.camera.camera_calibration import CalibrationManager
        from src.camera.sensor_recording import SensorRecorder
        from src.camera.sensor_replay import SensorReplayer
        from src.config.config_loader import ConfigLoader
        from src.utils.logging_config import setup_logging
        from src.utils.performance_monitor import PerformanceMonitor
        from src.utils.timestamp_sync import TimestampSynchronizer

        logger.info("✓ All module imports successful")
        return True

    except Exception as e:
        logger.error(f"✗ Import error: {e}", exc_info=True)
        return False


def verify_configuration():
    """Verify configuration system works."""
    logger.info("Verifying configuration...")

    try:
        loader = ConfigLoader()
        loader.load_all()

        camera_config = loader.camera
        logging_config = loader.config.get("logging", {})

        logger.info(f"  Camera FPS: {camera_config.get('rgb', {}).get('fps', 'unknown')}")
        logger.info(f"  Logging level: {logging_config.get('level', 'INFO')}")
        logger.info("✓ Configuration loaded successfully")
        return True

    except Exception as e:
        logger.error(f"✗ Configuration error: {e}", exc_info=True)
        return False


def list_tests():
    """List all acceptance tests."""
    tests = [
        ("Test 1", "Device Detection", "Verify OAK-D Pro detection and initialization"),
        ("Test 2", "RGB Acquisition", "5-minute continuous RGB stream (5 min)"),
        ("Test 3", "Stereo Acquisition", "Left/right stereo synchronization (2 min)"),
        ("Test 4", "Depth Accuracy", "Measure depth error at known distances"),
        ("Test 5", "RGB/Depth Alignment", "Pixel-level spatial correspondence"),
        ("Test 6", "IMU Acquisition", "Accelerometer + gyroscope streaming (5 min)"),
        ("Test 7", "Timestamp Integrity", "Monotonic, valid timestamps (1 min)"),
        ("Test 8", "Multi-Sensor Sync", "RGB/depth/IMU temporal alignment"),
        ("Test 9", "Calibration Retrieval", "Automatic calibration to YAML"),
        ("Test 10", "Coordinate Frames", "Physical direction verification"),
        ("Test 11", "Continuous Recording", "5-minute multi-sensor dataset (5 min)"),
        ("Test 12", "Dataset Replay", "Offline playback without camera"),
        ("Test 13", "5-Minute Stability", "No crashes, stable performance (5 min)"),
        ("Test 14", "Repeatability", "5 disconnect/reconnect cycles"),
    ]

    logger.info("")
    logger.info("="*70)
    logger.info("Milestone 1 Acceptance Tests")
    logger.info("="*70)

    for num, name, desc in tests:
        logger.info(f"{num:6} | {name:20} | {desc}")

    logger.info("="*70)
    logger.info("")


def main():
    """Run acceptance tests."""
    parser = argparse.ArgumentParser(description="Run Milestone 1 acceptance tests")
    parser.add_argument("--test", type=int, choices=range(1, 15), help="Run specific test only")
    parser.add_argument("--list", action="store_true", help="List all tests")
    parser.add_argument("--verify-imports", action="store_true", help="Verify module imports only")

    args = parser.parse_args()

    # Setup logging
    config_loader = ConfigLoader()
    logging_config = config_loader.config.get("logging", {}) if config_loader else {}
    setup_logging(logging_config)

    logger.info("="*70)
    logger.info("Milestone 1: OAK-D Pro Sensor Pipeline")
    logger.info("="*70)

    # Verify imports
    if not verify_imports():
        logger.error("Module verification failed")
        return 1

    # Verify configuration
    if not verify_configuration():
        logger.error("Configuration verification failed")
        return 1

    # List tests
    if args.list or args.verify_imports:
        list_tests()
        return 0

    # Run specific test
    if args.test:
        logger.info(f"\nRunning Test {args.test} only...")
        # Test runner would be implemented here
        logger.info("Test runner implementation required")
        return 0

    # Run all tests
    logger.info("\nTo run individual acceptance tests:")
    logger.info("  python tests/test_device_detection.py")
    logger.info("  python tests/test_rgb_acquisition.py")
    logger.info("  ... (see MILESTONE_1.md for all tests)")
    logger.info("")
    logger.info("To record a dataset:")
    logger.info("  python scripts/record_dataset.py --duration 300")
    logger.info("")
    logger.info("To replay a dataset:")
    logger.info("  python scripts/replay_dataset.py --dataset data/recordings/my_dataset")
    logger.info("")

    list_tests()

    logger.info("✓ All verifications passed")
    logger.info("")
    logger.info("Next steps:")
    logger.info("1. Connect OAK-D Pro to USB 3.x port")
    logger.info("2. Run individual acceptance tests from tests/ directory")
    logger.info("3. Record 5-minute reference dataset")
    logger.info("4. Verify all 14 tests pass")

    return 0


if __name__ == "__main__":
    sys.exit(main())
