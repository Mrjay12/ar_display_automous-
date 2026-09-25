#!/usr/bin/env python3
"""Quick test to verify point cloud visualization."""

import sys
from pathlib import Path
import numpy as np
import cv2

sys.path.insert(0, str(Path(__file__).parent / "src"))

from visualization.spatial_visualizer import SpatialVisualizer

# Create a simple depth frame with known content
width, height = 640, 360

# Create a synthetic depth frame
depth_frame = np.zeros((height, width), dtype=np.float32)

# Add a few "objects" at different distances
# Near object (2m away)
depth_frame[100:150, 200:250] = 2.0

# Mid object (5m away)
depth_frame[150:200, 300:400] = 5.0

# Far object (8m away)
depth_frame[200:250, 100:200] = 8.0

# Create RGB frame (dark)
rgb_frame = np.zeros((height, width, 3), dtype=np.uint8)

# Create camera matrix (default OAK-D specs)
fx = width * 1.08
fy = height * 1.08
camera_matrix = np.array([
    [fx, 0, width/2],
    [0, fy, height/2],
    [0, 0, 1],
], dtype=np.float32)

print(f"Creating visualizer with camera matrix:\n{camera_matrix}")
visualizer = SpatialVisualizer(camera_matrix, image_width=width, image_height=height)

# Process and render
objects = visualizer.process_depth_frame(depth_frame)
print(f"Detected {len(objects)} objects")

rendered = visualizer.render_frame(rgb_frame, depth_frame, objects)

# Check where points are being drawn
print(f"\nRendered frame shape: {rendered.shape}")
print(f"Non-zero pixels (colored points): {np.count_nonzero(rendered)}")

# Save output for inspection
output_path = "/tmp/test_visualization.png"
cv2.imwrite(output_path, rendered)
print(f"Saved test visualization to {output_path}")

# Show stats about point cloud rendering
valid_mask = (depth_frame > 0.1) & (depth_frame < 10.0) & np.isfinite(depth_frame)
valid_count = np.count_nonzero(valid_mask)
print(f"\nDepth frame stats:")
print(f"  Valid depth pixels: {valid_count}")
print(f"  Downsampled (every 2nd): ~{valid_count // 4} points rendered")
print(f"  Point radius: 3px")
print(f"  Color mapping: height-based rainbow (blue=low, red=high)")
