#!/usr/bin/env python3
"""
Spatial Visualization System - Real-time 3D object detection from stereo depth.

Simple entry point for live spatial visualization using OAK-D stereo camera.

Usage:
    python main.py                     # Live spatial visualization (default, 60s)
    python main.py --help              # Show all options
    python main.py --test-only         # Run device detection test only
    python main.py --duration 120      # Run for 120 seconds
"""

import argparse
import sys
import time
from pathlib import Path

# Add src directory to path
sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logging_config import setup_logging, get_logger


def main():
    parser = argparse.ArgumentParser(
        description="Spatial Visualization System - Stereo Depth-based 3D Object Detection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python main.py                              # Live spatial visualization (default, 60s)
  python main.py --test-only                  # Device detection test only
  python main.py --duration 120               # Run for 120 seconds
        """,
    )

    parser.add_argument(
        "--test-only",
        action="store_true",
        help="Run device detection test only",
    )
    parser.add_argument(
        "--verbose",
        action="store_true",
        help="Enable verbose logging",
    )
    parser.add_argument(
        "--duration",
        type=int,
        default=60,
        metavar="SECONDS",
        help="Duration for spatial visualization (default: 60 seconds, 0 = infinite)",
    )
    parser.add_argument(
        "--diagnose",
        action="store_true",
        help="Show camera diagnostic information",
    )

    args = parser.parse_args()

    # Setup logging
    log_level = "DEBUG" if args.verbose else "INFO"
    setup_logging(config={"level": log_level, "console_output": True, "file_output": True})
    logger = get_logger(__name__)

    logger.info("=" * 70)
    logger.info("Spatial Visualization System - Stereo Depth-based 3D Object Detection")
    logger.info("=" * 70)

    # If no arguments provided, run spatial visualization (default behavior)
    if len(sys.argv) == 1:
        logger.info("\n[DEFAULT] Running spatial visualization for 60 seconds...")
        logger.info("-" * 70)
        return run_spatial_visualization(duration_sec=60)

    # Run device detection test if requested
    if args.test_only:
        logger.info("\n[TEST] Device Detection Test")
        logger.info("-" * 70)
        return 0 if run_device_detection_test() else 1

    # Show diagnostics if requested
    if args.diagnose:
        logger.info("\n[DIAGNOSE] Camera Diagnostic Information")
        logger.info("-" * 70)
        return 0 if run_camera_diagnostics() else 1

    # If --duration is specified, run spatial visualization
    logger.info(f"\n[VISUALIZATION] Running spatial visualization for {args.duration} seconds...")
    logger.info("-" * 70)
    return 0 if run_spatial_visualization(duration_sec=args.duration) else 1


def run_device_detection_test():
    """Run device detection test."""
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


def run_camera_diagnostics():
    """Run camera diagnostics to see what's happening."""
    logger = get_logger(__name__)
    try:
        from camera.oak_d_interface import OAKDInterface
        import json

        logger.info("Initializing camera for diagnostics...")
        camera = OAKDInterface()
        if not camera.initialize():
            logger.error("Failed to initialize camera")
            return False

        logger.info("Camera initialized. Checking diagnostics...")
        diag = camera.diagnose()

        logger.info("\n=== CAMERA DIAGNOSTICS ===")
        logger.info(json.dumps(diag, indent=2, default=str))

        logger.info("\n=== FRAME CAPTURE TEST ===")
        logger.info("Attempting to capture 10 frames...")

        frame_count = 0
        for i in range(10):
            rgbd = camera.get_rgbd_frame(timeout_ms=500)
            if rgbd is not None:
                frame_count += 1
                logger.info(f"  Frame {frame_count}: RGB={rgbd.rgb.shape}, Depth={rgbd.depth.shape}")
            else:
                logger.warning(f"  Attempt {i+1}: No frame available")
            time.sleep(0.1)

        logger.info(f"\n✓ Successfully captured {frame_count}/10 frames")

        final_diag = camera.diagnose()
        logger.info("\n=== FINAL DIAGNOSTICS ===")
        logger.info(json.dumps(final_diag, indent=2, default=str))

        camera.shutdown()
        logger.info("\n✓ Camera diagnostics complete")
        return True

    except Exception as e:
        logger.exception(f"Camera diagnostics failed: {e}")
        return False


def run_spatial_visualization(duration_sec=60):
    """Run spatial visualization with real-time depth-based object detection."""
    logger = get_logger(__name__)
    try:
        from camera.oak_d_interface import OAKDInterface
        from visualization.spatial_visualizer import SpatialVisualizer
        import cv2

        logger.info("Initializing camera...")
        camera = OAKDInterface()
        if not camera.initialize():
            logger.error("Failed to initialize camera")
            return False

        # Get camera calibration (always available with defaults)
        calibration = camera.get_calibration()
        if calibration is None:
            logger.warning("Camera calibration not available, using defaults")
            # Create fallback calibration
            import numpy as np
            width, height = 1280, 720
            fx = width * 1.08
            fy = height * 1.08
            calibration = np.array([
                [fx, 0, width/2],
                [0, fy, height/2],
                [0, 0, 1],
            ], dtype=np.float32)

        # Initialize spatial visualizer
        logger.info("Initializing spatial visualizer...")
        visualizer = SpatialVisualizer(
            camera_matrix=calibration,
            image_width=640,
            image_height=360
        )

        logger.info(f"Starting spatial visualization for {duration_sec}s")
        logger.info("Press 'q' to stop, 'p' to pause/resume, 'g' to toggle grid")
        logger.info("=" * 70)

        frame_count = 0
        paused = False
        start_time = time.time()

        while True:
            # Check duration
            if duration_sec > 0 and time.time() - start_time > duration_sec:
                logger.info(f"Reached target duration ({duration_sec}s)")
                break

            if not paused:
                # Get frames from camera
                rgb_frame = camera.get_rgb_frame()
                depth_frame = camera.get_depth_frame()

                if rgb_frame is None or depth_frame is None:
                    logger.warning("Failed to get frames from camera")
                    continue

                frame_count += 1

                # Convert depth frame to array
                if hasattr(depth_frame, 'depth_map'):
                    depth_array = depth_frame.depth_map
                else:
                    depth_array = depth_frame

                # Process depth and detect objects
                objects = visualizer.process_depth_frame(depth_array)

                # Render visualization
                if hasattr(rgb_frame, 'frame'):
                    rgb_data = rgb_frame.frame
                else:
                    rgb_data = rgb_frame

                visualized = visualizer.render_frame(rgb_data, depth_array, objects)

                # Display
                cv2.imshow("Spatial Visualization", visualized)

                # Log progress every 30 frames
                if frame_count % 30 == 0:
                    logger.info(
                        f"Frame {frame_count}: {len(objects)} objects detected | "
                        f"Processing time: {time.time() - start_time:.1f}s"
                    )

            # Handle keyboard
            try:
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    logger.info("Stopping visualization...")
                    break
                elif key == ord('p'):
                    paused = not paused
                    logger.info(f"Visualization {'paused' if paused else 'resumed'}")
                elif key == ord('g'):
                    visualizer.show_grid = not visualizer.show_grid
                    logger.info(f"Grid {'enabled' if visualizer.show_grid else 'disabled'}")
                elif key == ord('d'):
                    visualizer.show_debug = not visualizer.show_debug
                    logger.info(f"Debug {'enabled' if visualizer.show_debug else 'disabled'}")
            except:
                pass

        cv2.destroyAllWindows()
        camera.shutdown()

        logger.info("=" * 70)
        logger.info(f"✓ Spatial visualization complete: {frame_count} frames processed")
        return True

    except Exception as e:
        logger = get_logger(__name__)
        logger.error(f"Spatial visualization failed: {e}")
        import traceback
        logger.exception(traceback.format_exc())
        return False


if __name__ == "__main__":
    sys.exit(main())
