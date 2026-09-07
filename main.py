#!/usr/bin/env python3
"""
Unified entry point for GPS-Denied Visual Localization & AR Mapping System.

Single command to run all device detection, recording, replay, and testing.
No need to remember multiple script names or arguments.

Usage:
    python main.py --help              # Show all options
    python main.py                     # Auto-detect and run full pipeline
    python main.py --test-only         # Run device detection test only
    python main.py --record 300        # Record 300-second dataset only
    python main.py --replay            # Replay last recorded dataset
    python main.py --all               # Full pipeline (same as default)
"""

import argparse
import sys
import time
from pathlib import Path
from datetime import datetime

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logging_config import setup_logging, get_logger
from config.config_loader import ConfigLoader


def main():
    parser = argparse.ArgumentParser(
        description="GPS-Denied Visual Localization System - Unified Entry Point",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                    # Full pipeline (detect → record → replay)
  python main.py --test-only        # Device detection only
  python main.py --record 300       # Record 5-minute dataset
  python main.py --replay           # Replay last dataset
  python main.py --all              # Explicit full pipeline
        """,
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run device detection test only",
    )
    parser.add_argument(
        "--record",
        type=int,
        metavar="SECONDS",
        help="Record dataset for N seconds (requires camera)",
    )
    parser.add_argument(
        "--replay",
        action="store_true",
        help="Replay last recorded dataset (no camera needed)",
    )
    parser.add_argument(
        "--all",
        action="store_true",
        help="Run full pipeline: detect → record → replay",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=10,
        metavar="SECONDS",
        help="Default duration for recording (default: 10 seconds)",
    )
    parser.add_argument(
        "--output",
        type=str,
        metavar="PATH",
        help="Output directory for recordings (default: data/recordings/auto-TIMESTAMP)",
    )
    parser.add_argument(
        "--dataset",
        type=str,
        metavar="PATH",
        help="Dataset path for replay (default: latest in data/recordings/)",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--list-tests",
        action="store_true",
        help="List all 14 acceptance tests and exit",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(config={"level": log_level, "console_output": True, "file_output": True})
    logger = get_logger(__name__)

    logger.info("=" * 70)
    logger.info("GPS-Denied Visual Localization & AR Mapping System")
    logger.info("=" * 70)

    # List acceptance tests if requested
    if args.list_tests:
        list_acceptance_tests()
        return 0

    # Determine what to run
    run_test = args.test_only or args.all or (not args.record and not args.replay)
    run_record = args.record or args.all or (not args.test_only and not args.replay)
    run_replay = args.replay or args.all or (not args.test_only and not args.record)

    # Default recording duration if not specified
    record_duration = args.record if args.record else args.duration

    try:
        # Step 1: Device Detection Test
        if run_test:
            logger.info("\n[STEP 1/3] Device Detection Test")
            logger.info("-" * 70)
            success = run_device_detection_test()
            if not success:
                logger.warning(
                    "Device detection failed. Camera may not be connected."
                )
                if not run_replay:
                    logger.error("Cannot proceed without camera. Exiting.")
                    return 1
                logger.info("Proceeding to replay existing dataset...")
            else:
                logger.info("✓ Device detection successful")

        # Step 2: Record Dataset
        if run_record and not args.replay:
            logger.info(f"\n[STEP 2/3] Recording Dataset ({record_duration} seconds)")
            logger.info("-" * 70)
            output_path = args.output or generate_output_path()
            success = run_recording(output_path, record_duration)
            if not success:
                logger.error("Recording failed. Cannot proceed to replay.")
                return 1
            logger.info(f"✓ Dataset recorded to: {output_path}")

        # Step 3: Replay Dataset
        if run_replay:
            logger.info("\n[STEP 3/3] Replaying Dataset")
            logger.info("-" * 70)
            dataset_path = args.dataset or find_latest_dataset()
            if not dataset_path:
                logger.error("No dataset found to replay.")
                return 1
            success = run_replay_dataset(dataset_path)
            if not success:
                logger.error("Replay failed.")
                return 1
            logger.info(f"✓ Dataset replayed from: {dataset_path}")

        logger.info("\n" + "=" * 70)
        logger.info("✓ Pipeline Complete!")
        logger.info("=" * 70)
        return 0

    except KeyboardInterrupt:
        logger.warning("\n\nInterrupted by user.")
        return 130
    except Exception as e:
        logger.exception(f"Unexpected error: {e}")
        return 1


def run_device_detection_test():
    """Run device detection test (Acceptance Test 1)."""
    logger = get_logger(__name__)
    try:
        from camera.oak_d_interface import OAKDInterface

        logger.info("Attempting to detect OAK-D Pro device...")
        start_time = time.time()

        device = OAKDInterface()
        elapsed = time.time() - start_time

        logger.info(f"✓ Device detected in {elapsed:.2f} seconds")
        logger.info(f"  Device Info: {device.get_device_info()}")

        device.shutdown()
        logger.info("✓ Device initialized and shut down successfully")
        return True

    except Exception as e:
        logger.error(f"Device detection failed: {e}")
        return False


def run_recording(output_path, duration):
    """Record a dataset."""
    logger = get_logger(__name__)
    try:
        from camera.oak_d_interface import OAKDInterface
        from camera.sensor_recording import SensorRecorder

        logger.info(f"Initializing camera for {duration}-second recording...")
        device = OAKDInterface()

        recorder = SensorRecorder(device, output_path=output_path)
        logger.info(f"Starting recording to: {output_path}")

        recorder.start()
        start_time = time.time()

        while time.time() - start_time < duration:
            elapsed = time.time() - start_time
            frames_recorded = recorder.get_frame_count()
            logger.info(
                f"  Recording: {elapsed:.1f}s / {duration}s - {frames_recorded} frames"
            )
            time.sleep(1)

        logger.info("Stopping recording...")
        recorder.stop()
        logger.info(f"✓ Recording complete: {recorder.get_frame_count()} frames")

        device.shutdown()
        return True

    except Exception as e:
        logger.error(f"Recording failed: {e}")
        return False


def run_replay_dataset(dataset_path):
    """Replay a recorded dataset."""
    logger = get_logger(__name__)
    try:
        from camera.sensor_replay import SensorReplayer

        dataset_path = Path(dataset_path)
        if not dataset_path.exists():
            logger.error(f"Dataset not found: {dataset_path}")
            return False

        logger.info(f"Loading dataset from: {dataset_path}")
        replayer = SensorReplayer(dataset_path)

        logger.info(f"Dataset info:")
        logger.info(f"  Total frames: {replayer.get_frame_count()}")
        logger.info(f"  Duration: {replayer.get_duration():.2f} seconds")
        logger.info(f"  RGB resolution: {replayer.get_rgb_resolution()}")
        logger.info(f"  Depth resolution: {replayer.get_depth_resolution()}")

        # Iterate through some frames to verify replay works
        logger.info("Verifying dataset playback...")
        frame_count = 0
        for _ in range(min(10, replayer.get_frame_count())):
            rgb = replayer.get_rgb_frame()
            depth = replayer.get_depth_frame()
            imu = replayer.get_imu_data()
            if rgb is not None:
                frame_count += 1
            time.sleep(0.033)  # 30 FPS

        logger.info(f"✓ Replay verified: {frame_count} frames played back successfully")
        return True

    except Exception as e:
        logger.error(f"Replay failed: {e}")
        return False


def find_latest_dataset():
    """Find the most recently created dataset."""
    logger = get_logger(__name__)
    recordings_dir = Path("data/recordings")

    if not recordings_dir.exists():
        logger.error(f"Recordings directory not found: {recordings_dir}")
        return None

    datasets = sorted(
        [d for d in recordings_dir.iterdir() if d.is_dir()],
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )

    if not datasets:
        logger.error("No datasets found in data/recordings/")
        return None

    latest = datasets[0]
    logger.info(f"Found latest dataset: {latest.name}")
    return str(latest)


def generate_output_path():
    """Generate output path with timestamp."""
    recordings_dir = Path("data/recordings")
    recordings_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    output_path = recordings_dir / f"recording_{timestamp}"
    return str(output_path)


def list_acceptance_tests():
    """List all 14 acceptance tests."""
    tests = [
        "Test 1:  Device Detection",
        "Test 2:  RGB Acquisition (5 min, <1% drops)",
        "Test 3:  Stereo Acquisition (L/R sync within 10ms)",
        "Test 4:  Depth Accuracy (0.5m, 1m, 2m, 5m)",
        "Test 5:  RGB/Depth Alignment (<5 pixel offset)",
        "Test 6:  IMU Acquisition (200Hz accel + gyro)",
        "Test 7:  Timestamp Integrity (monotonic, no NaT)",
        "Test 8:  Multi-Sensor Sync (temporal order verification)",
        "Test 9:  Calibration Retrieval (auto to YAML)",
        "Test 10: Coordinate Frames (physical direction verification)",
        "Test 11: Continuous Recording (5 min dataset)",
        "Test 12: Dataset Replay (no camera needed)",
        "Test 13: 5-Minute Stability (no crashes, bounded memory)",
        "Test 14: Repeatability (5 disconnect/reconnect cycles)",
    ]

    print("\nAcceptance Tests for Milestone 1 (OAK-D Sensor Pipeline):")
    print("=" * 70)
    for test in tests:
        print(f"  {test}")
    print("=" * 70)
    print("\nFor detailed specs, see: MILESTONE_1.md")


if __name__ == "__main__":
    sys.exit(main())
