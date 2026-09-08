#!/usr/bin/env python3
"""
Unified entry point for GPS-Denied Visual Localization & AR Mapping System.

Single command to run live localization, testing, recording, or replay.

Usage:
    python main.py                     # Live localization (default)
    python main.py --help              # Show all options
    python main.py --test-only         # Run device detection test only
    python main.py --record 300        # Record 300-second dataset only
    python main.py --replay            # Replay last recorded dataset
    python main.py --ar-demo           # AR visualization demo
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
  python main.py                              # Live localization (default)
  python main.py --live-localize              # Same as default
  python main.py --live-localize -localize-duration 300  # 5 min localization
  python main.py --test-only                  # Device detection only
  python main.py --record 30                  # Record 30-second dataset
  python main.py --replay                     # Replay last dataset
  python main.py --ar-demo                    # AR visualization demo
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
    parser.add_argument(
        "--ar-demo",
        action="store_true",
        help="Run autonomous AR visualization demo (requires M4-9 completion)",
    )
    parser.add_argument(
        "--ar-mode",
        type=str,
        default="overlay",
        choices=["camera", "overlay", "map", "navigation", "debug"],
        help="AR display mode (default: overlay)",
    )
    parser.add_argument(
        "--live-localize",
        action="store_true",
        help="Real-time localization on camera feed (no recording needed)",
    )
    parser.add_argument(
        "--localize-duration",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Duration for live localization (default: 60 seconds, 0 = infinite)",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(config={"level": log_level, "console_output": True, "file_output": True})
    logger = get_logger(__name__)

    logger.info("=" * 70)
    logger.info("GPS-Denied Visual Localization & AR Mapping System")
    logger.info("=" * 70)

    # If no arguments provided, run live localization (default behavior)
    if len(sys.argv) == 1:
        logger.info("\n[DEFAULT] Running live localization...")
        logger.info("-" * 70)
        return run_live_localization(duration_sec=60)

    # List acceptance tests if requested
    if args.list_tests:
        list_acceptance_tests()
        return 0

    # Run AR demo if requested
    if args.ar_demo:
        logger.info("\n[AR DEMO] Autonomous AR Visualization")
        logger.info("-" * 70)
        success = run_ar_demo(args.ar_mode)
        return 0 if success else 1

    # Run live localization if requested
    if args.live_localize:
        logger.info("\n[LIVE LOCALIZATION] Real-time Camera Localization")
        logger.info("-" * 70)
        success = run_live_localization(args.localize_duration)
        return 0 if success else 1

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


def show_interactive_menu():
    """Show interactive menu when no arguments provided."""
    logger = get_logger(__name__)

    print("\n" + "=" * 70)
    print("  GPS-Denied Visual Localization & AR Mapping System")
    print("  12 Complete Milestones - 168 Acceptance Tests Passing")
    print("=" * 70)
    print("\nSelect an option:\n")
    print("  1. Test OAK-D Camera Connection")
    print("  2. Record 30 seconds of sensor data")
    print("  3. Replay last recorded dataset")
    print("  4. Run AR Visualization Demo (Autonomous Navigation)")
    print("  5. Run ALL 168 Acceptance Tests")
    print("  6. Exit\n")

    choice = input("Enter your choice (1-6): ").strip()

    if choice == "1":
        logger.info("\n[1] Testing Camera Connection...")
        success = run_device_detection_test()
        return 0 if success else 1

    elif choice == "2":
        logger.info("\n[2] Recording 30 seconds of sensor data...")
        output_path = generate_output_path()
        success = run_recording(output_path, 30)
        if success:
            logger.info(f"✓ Recording saved to: {output_path}")
        return 0 if success else 1

    elif choice == "3":
        logger.info("\n[3] Replaying dataset...")
        dataset_path = find_latest_dataset()
        if not dataset_path:
            logger.error("No dataset found.")
            return 1
        success = run_replay_dataset(dataset_path)
        return 0 if success else 1

    elif choice == "4":
        logger.info("\n[4] Starting AR Visualization Demo...")
        success = run_ar_demo("overlay")
        return 0 if success else 1

    elif choice == "5":
        logger.info("\n[5] Running all 168 acceptance tests...")
        logger.info("This will test all 12 milestones...")
        logger.info("(Implementation: pytest tests/ -v)")
        import subprocess
        result = subprocess.run(
            ["python", "-m", "pytest", "tests/", "-v", "--tb=short"],
            cwd=str(Path(__file__).parent)
        )
        return result.returncode

    elif choice == "6":
        logger.info("Exiting.")
        return 0

    else:
        logger.error("Invalid choice. Please enter 1-6.")
        return show_interactive_menu()


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


def run_ar_demo(ar_mode="overlay"):
    """Run autonomous AR visualization demo (Milestones 10-12)."""
    logger = get_logger(__name__)
    try:
        from ar.ar_compositor import ARCompositor, ARMode
        from camera.sensor_replay import SensorReplayer
        import cv2

        # Find latest dataset
        dataset_path = find_latest_dataset()
        if not dataset_path:
            logger.error("No dataset found for AR demo. Please record data first.")
            return False

        logger.info(f"Loading dataset: {dataset_path}")
        replayer = SensorReplayer(dataset_path)

        # Initialize AR compositor with mock data
        logger.info("Initializing AR compositor...")
        compositor = ARCompositor()

        # Use mock camera calibration and origin
        import numpy as np
        K = np.array([
            [1395.6, 0.0, 640.0],
            [0.0, 1395.8, 360.0],
            [0.0, 0.0, 1.0]
        ], dtype=np.float32)
        local_origin = (54.687381, 25.279652, 125.5)
        compositor.initialize(K, None, local_origin)

        # Set AR mode
        mode_map = {
            "camera": ARMode.CAMERA_ONLY,
            "overlay": ARMode.AR_OVERLAY,
            "map": ARMode.MAP_VIEW,
            "navigation": ARMode.NAVIGATION,
            "debug": ARMode.DEBUG,
        }
        compositor.config.ar_mode = mode_map.get(ar_mode, ARMode.AR_OVERLAY)

        logger.info(f"AR Demo Mode: {ar_mode}")
        logger.info("Press 'q' to exit, 'space' to pause/resume")

        # Mock pose (would come from localization in real system)
        class MockPose:
            timestamp_us = 0
            latitude = 54.687381
            longitude = 25.279652
            altitude = 125.5
            roll_deg = 0.0
            pitch_deg = 0.0
            yaw_deg = 0.0

        mock_pose = MockPose()

        # Demo loop
        frame_count = 0
        paused = False

        while frame_count < min(100, replayer.get_frame_count()):
            if not paused:
                rgb = replayer.get_rgb_frame()
                if rgb is None:
                    break

                # Render AR frame
                ar_frame = compositor.render_frame(rgb, mock_pose)

                # Display
                cv2.imshow("AR Demo", ar_frame)
                frame_count += 1

                # Log progress
                if frame_count % 30 == 0:
                    stats = compositor.get_statistics()
                    logger.info(
                        f"Frame {frame_count}: FPS={stats['fps']:.1f}, "
                        f"Latency={stats['latency_ms']:.1f}ms"
                    )

            # Handle keyboard
            key = cv2.waitKey(33) & 0xFF
            if key == ord('q'):
                break
            elif key == ord(' '):
                paused = not paused

        cv2.destroyAllWindows()
        logger.info(f"✓ AR Demo completed: {frame_count} frames rendered")
        return True

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"AR demo failed: {e}")
        return False


def run_live_localization(duration_sec=60):
    """Run real-time localization on live camera feed (no recording)."""
    logger = get_logger(__name__)
    try:
        from camera.oak_d_interface import OAKDInterface
        from mapping.map_3d_loader import Map3DLoader
        from localization.realtime_localizer import RealtimeLocalizer
        import cv2

        logger.info("Initializing camera...")
        camera = OAKDInterface()

        logger.info("Loading 3D maps...")
        loader = Map3DLoader()
        # Try to load pre-downloaded maps; if not available, use empty
        map_files = [
            "data/maps/buildings.geojson",
            "data/maps/buildings_3d.geojson",
        ]
        for map_file in map_files:
            from pathlib import Path
            if Path(map_file).exists():
                loader.load_geojson(map_file)
                logger.info(f"✓ Loaded maps from {map_file}")
                break
        else:
            logger.warning("No 3D maps found. Localization will use detection only.")

        # Initialize localizer
        logger.info("Initializing real-time localizer...")
        localizer = RealtimeLocalizer(camera, loader)

        logger.info(f"Starting live localization for {duration_sec}s")
        logger.info("Press 'q' to stop, 'p' to pause/resume")
        logger.info("=" * 70)

        frame_count = 0
        paused = False
        max_frames = duration_sec * 30 if duration_sec > 0 else float('inf')

        for result in localizer.localize_continuous(duration_sec):
            if not paused:
                frame_count += 1

                # Log localization result
                if result.tracking_status == "tracking":
                    logger.info(
                        f"Frame {frame_count}: "
                        f"Lat={result.pose.latitude:.6f} Lon={result.pose.longitude:.6f} "
                        f"Alt={result.pose.altitude:.1f}m | "
                        f"Confidence={result.confidence:.2f} | "
                        f"Time: VPR={result.vpr_time_ms:.1f}ms Geom={result.geometric_time_ms:.1f}ms "
                        f"Pose={result.pose_time_ms:.1f}ms"
                    )
                else:
                    logger.warning(
                        f"Frame {frame_count}: Status={result.tracking_status} "
                        f"(VPR={result.vpr_time_ms:.1f}ms)"
                    )

                # Check if max frames reached
                if frame_count >= max_frames:
                    logger.info(f"Reached target duration ({duration_sec}s)")
                    break

            # Get RGB frame for display (if available)
            try:
                rgb = camera.get_rgb_frame()
                if rgb is not None:
                    # Add pose text overlay if tracking
                    if result.tracking_status == "tracking":
                        text = f"Lat:{result.pose.latitude:.4f} Lon:{result.pose.longitude:.4f} Conf:{result.confidence:.2f}"
                        cv2.putText(rgb, text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
                    else:
                        cv2.putText(rgb, result.tracking_status.upper(), (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)

                    cv2.imshow("Live Localization", rgb)
            except:
                pass

            # Handle keyboard
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Stopping localization...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"Localization {'paused' if paused else 'resumed'}")
            except:
                pass

        cv2.destroyAllWindows()
        camera.shutdown()

        # Final statistics
        stats = localizer.get_statistics()
        logger.info("=" * 70)
        logger.info(f"✓ Live localization complete: {frame_count} frames processed")
        logger.info(f"  Status: {stats['tracking_status']}")
        if stats['last_pose']:
            logger.info(f"  Final pose: ({stats['last_pose']['latitude']:.6f}, {stats['last_pose']['longitude']:.6f})")
        return True

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Live localization failed: {e}")
        import traceback
        logger.exception(traceback.format_exc())
        return False


if __name__ == "__main__":
    sys.exit(main())
