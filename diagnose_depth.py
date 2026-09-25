#!/usr/bin/env python3
"""
Comprehensive depth diagnostic tool.
Run this to understand what your camera is seeing and where the visualization is rendering.
"""

import sys
from pathlib import Path
import numpy as np
import cv2
import time

sys.path.insert(0, str(Path(__file__).parent / "src"))

from utils.logging_config import setup_logging, get_logger
from camera.oak_d_interface import OAKDInterface
from visualization.spatial_visualizer import SpatialVisualizer

setup_logging(config={"level": "INFO", "console_output": True, "file_output": True})
logger = get_logger(__name__)

def diagnose():
    """Run comprehensive diagnosis."""
    logger.info("=" * 70)
    logger.info("DEPTH DIAGNOSTIC TOOL")
    logger.info("=" * 70)

    try:
        # Initialize camera
        logger.info("\n[1/4] Initializing camera...")
        camera = OAKDInterface()
        if not camera.initialize():
            logger.error("Failed to initialize camera!")
            return False

        calibration = camera.get_calibration()
        if calibration is None:
            logger.warning("Camera calibration not available, using defaults")
            width, height = 1280, 720
            fx = width * 1.08
            fy = height * 1.08
            calibration = np.array([
                [fx, 0, width/2],
                [0, fy, height/2],
                [0, 0, 1],
            ], dtype=np.float32)

        logger.info(f"✓ Camera initialized")
        logger.info(f"  Camera matrix:\n{calibration}")

        # Get some frames and analyze
        logger.info("\n[2/4] Capturing test frames (10 frames)...")
        visualizer = SpatialVisualizer(calibration, 640, 360)

        depth_stats = []
        frame_count = 0

        for i in range(10):
            rgbd = camera.get_rgbd_frame()
            if rgbd is None:
                logger.warning(f"  Frame {i+1}: Failed to get frame")
                continue

            frame_count += 1
            depth = rgbd.depth

            # Analyze depth
            valid_mask = (depth > 0.1) & (depth < 10.0) & np.isfinite(depth)
            valid_count = np.count_nonzero(valid_mask)

            if valid_count > 0:
                valid_depths = depth[valid_mask]
                stats = {
                    'frame': frame_count,
                    'valid_pixels': valid_count,
                    'min_depth': float(np.min(valid_depths)),
                    'max_depth': float(np.max(valid_depths)),
                    'mean_depth': float(np.mean(valid_depths)),
                    'std_depth': float(np.std(valid_depths)),
                }
                depth_stats.append(stats)
                logger.info(
                    f"  Frame {frame_count}: {valid_count:,} valid pixels | "
                    f"Depth: {stats['min_depth']:.2f}m - {stats['max_depth']:.2f}m | "
                    f"Mean: {stats['mean_depth']:.2f}m"
                )
            else:
                logger.warning(f"  Frame {frame_count}: NO valid depth pixels!")

            time.sleep(0.1)

        # Summary
        logger.info("\n[3/4] Depth Analysis Summary")
        logger.info("-" * 70)

        if depth_stats:
            avg_valid = np.mean([s['valid_pixels'] for s in depth_stats])
            logger.info(f"Average valid pixels per frame: {avg_valid:,.0f}")
            logger.info(f"Average mean depth: {np.mean([s['mean_depth'] for s in depth_stats]):.2f}m")
            logger.info(f"✓ Depth data IS being captured")
        else:
            logger.error("✗ NO depth data captured!")
            logger.error("  This means the stereo depth pipeline may not be working.")
            logger.error("  Check camera cable connections and try --test-only flag.")
            camera.shutdown()
            return False

        # Test visualization
        logger.info("\n[4/4] Testing Point Cloud Visualization")
        logger.info("-" * 70)

        # Get one frame and render it
        rgbd = camera.get_rgbd_frame()
        if rgbd is not None:
            objects = visualizer.process_depth_frame(rgbd.depth)
            rendered = visualizer.render_frame(rgbd.rgb, rgbd.depth, objects)

            # Save output
            output_path = "/tmp/diagnostic_visualization.png"
            cv2.imwrite(output_path, rendered)
            logger.info(f"✓ Visualization saved to {output_path}")
            logger.info(f"  Objects detected: {len(objects)}")
            logger.info(f"  Point cloud size: {visualizer._last_point_cloud_count:,} points")

            # Show what's in the visualization
            non_black = np.count_nonzero(rendered)
            logger.info(f"  Non-black pixels in output: {non_black:,}")

            # Analyze colors
            h, w = rendered.shape[:2]
            hsv = cv2.cvtColor(rendered, cv2.COLOR_BGR2HSV)

            # Count colors
            blue_mask = cv2.inRange(hsv, (100, 50, 50), (130, 255, 255))
            green_mask = cv2.inRange(hsv, (35, 50, 50), (90, 255, 255))
            red_mask = cv2.inRange(hsv, (0, 50, 50), (30, 255, 255))

            logger.info(f"  Color distribution:")
            logger.info(f"    Blue points: {np.count_nonzero(blue_mask):,}")
            logger.info(f"    Green points: {np.count_nonzero(green_mask):,}")
            logger.info(f"    Red points: {np.count_nonzero(red_mask):,}")

            logger.info(f"\n✓ DIAGNOSTIC COMPLETE")
            logger.info(f"  Expected behavior:")
            logger.info(f"    - Should show colored point cloud across image")
            logger.info(f"    - Colors range from blue (near) to red (far)")
            logger.info(f"    - Ground grid should be visible")
            logger.info(f"    - Detected object contours should be outlined")

        camera.shutdown()
        return True

    except Exception as e:
        logger.exception(f"Diagnostic failed: {e}")
        return False

if __name__ == "__main__":
    success = diagnose()
    sys.exit(0 if success else 1)
