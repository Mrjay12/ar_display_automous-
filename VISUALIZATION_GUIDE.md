# Spatial Visualization Guide

## Overview

The system renders a **depth-based point cloud visualization** overlaid on the RGB camera feed. This shows all detected obstacles in 3D space using an Octomap-style point cloud.

## What You Should See

When running the visualization, you will see:

### 1. **Point Cloud** (Colored Individual Points)
- **Location**: Scattered across the entire image wherever the depth camera detects valid depth
- **What it shows**: 3D positions of obstacles rendered as colored dots
- **Colors**: Rainbow gradient based on height (3D Y-axis):
  - 🔵 **Blue**: Low positions (below camera)
  - 🟢 **Green**: Middle positions (ground level)
  - 🔴 **Red**: High positions (above camera)

### 2. **Ground Plane Grid** (Green Lines)
- **Location**: Base of the image, receding into the distance
- **What it shows**: 3D perspective grid showing ground plane at 0.5m below camera
- **Spacing**: 0.5m between grid lines for scale reference

### 3. **Detected Object Contours** (Green Outlines)
- **Location**: Regions where objects are detected
- **What it shows**: Boundaries of detected obstacles
- **Labels**: Distance and confidence for each object at ground plane

### 4. **Info Overlay** (Top-left corner)
- **Objects**: Number of spatial objects detected
- **Point Cloud**: Total number of 3D points rendered
- **Range**: Detection range limits
- **Mean Depth**: Average distance to detected surfaces

## Point Cloud Details

### How Points Are Positioned
- Each **colored point** represents one depth pixel from the stereo camera
- Points are projected onto the image using the camera's 3D perspective
- A point at pixel `(u, v)` with depth `z` appears at that exact pixel location

### Color Mapping

The system uses a **two-tier coloring approach**:

**Primary (Height-based):** If points have varied heights:
```
Min Height     Mid Height     Max Height
    ↓              ↓              ↓
  Blue ——→ Cyan ——→ Green ——→ Yellow ——→ Red
```

**Fallback (Depth-based):** If all points at same height:
```
Near Distance    Mid Distance    Far Distance
      ↓              ↓               ↓
    Blue ——→ Cyan ——→ Green ——→ Yellow ——→ Red
```

### Point Density
- **Current**: Rendering ALL valid depth pixels (downsample = 1)
- **Total points**: Typically 1,000 - 25,000+ depending on camera view
- **Point size**: 4 pixels (radius) for clear visibility

## Interpreting the Visualization

### If you see a RAINBOW CLOUD OF DOTS:
✓ This is correct! It means:
- Stereo depth camera is working
- Multiple 3D obstacles are visible
- Height (or distance) variation exists

### If you see a SINGLE COLOR cloud:
✓ Also correct! It means:
- All obstacles at similar height/distance
- Depth camera is working
- System using fallback depth-based coloring

### If you see NO colored points:
✗ Something needs fixing:
1. Stereo depth pipeline may not be initialized
2. Camera may not have valid depth data
3. All depth pixels may be invalid/zero

**→ Run `python diagnose_depth.py` to identify the issue**

## Common Issues & Solutions

### Problem: "I don't see any colored points"

**Check 1:** Run the diagnostic
```bash
python diagnose_depth.py
```
Look for: "X valid pixels per frame"

**Check 2:** Verify stereo camera cable connections
- Ensure CAM_B (left mono) and CAM_C (right mono) are connected
- Check for loose connectors

**Check 3:** Verify USB 3.1 connection
- Stereo depth requires high bandwidth
- Use USB 3.1 ports, not USB 2.0

### Problem: "Points look washed out / all same color"

**Solution:** This is usually correct behavior when:
- All obstacles at similar distance (fallback to depth coloring)
- Scene has uniform depth distribution
- You're looking at a flat wall/surface

### Problem: "Point cloud looks very sparse"

**Note:** This is expected! The visualization shows *every* valid depth pixel, but:
- Some regions have no valid depth (reflections, transparent objects)
- Downsampling may have reduced density (current: downsample=1 = no reduction)
- To increase density: Modify `downsample` in `spatial_visualizer.py` line 254

## Running the System

### Default (60 second visualization):
```bash
python main.py
```

### Specific duration:
```bash
python main.py --duration 120
```

### Diagnostic mode:
```bash
python diagnose_depth.py
```

### Device test:
```bash
python main.py --test-only
```

## Keyboard Controls

While visualization is running:

- **`q`**: Quit
- **`p`**: Pause/Resume
- **`g`**: Toggle grid on/off
- **`d`**: Toggle debug info

## Technical Details

### Coordinate System
- **X-axis**: Left-Right (camera's perspective)
- **Y-axis**: Up-Down (camera's vertical, negative = below)
- **Z-axis**: Forward-Backward (depth into scene)

### Camera Specifications
- **Device**: OAK-D Pro Stereo
- **Resolution**: 640×360 (for visualization)
- **Stereo cameras**: CAM_B (left), CAM_C (right)
- **Focal length**: ~691 pixels (X), ~389 pixels (Y)
- **Principal point**: (320, 180) at 640×360 resolution

### Pipeline
```
Stereo Depth → 3D Points → Height-based Coloring → Rendered Overlay
             ↓                                ↓
          Point Cloud                   RGB Display
             ↓                                ↓
        Obstacle Detection            Ground Grid
```

## Performance

- **Point rendering**: ~3000-10000 points per frame
- **FPS**: ~20-30 fps (real-time visualization)
- **Latency**: <100ms (synchronized RGB+Depth)

## Next Steps

If visualization is working:
- ✓ System is successfully detecting obstacles in 3D space
- ✓ Point cloud overlay is working correctly
- ✓ Ready for AR applications or autonomous navigation

If visualization is not working:
1. Run `python diagnose_depth.py` to identify the issue
2. Check camera connections and USB cable
3. Verify OAK-D device is detected: `python main.py --test-only`
4. Review logs in `logs/` directory for detailed error messages
