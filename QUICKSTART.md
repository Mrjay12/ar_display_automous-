# Quick Start - GPS-Denied Visual Localization System

## Single Command to Run Everything

You no longer need to remember multiple script names. Use `main.py` for everything.

### On Windows

```bash
# First time: Install dependencies
pip install -r requirements.txt

# Then run everything with one command
python main.py
```

That's it! This will:
1. ✓ Detect your OAK-D Pro camera
2. ✓ Record a 10-second dataset (or custom duration)
3. ✓ Replay the dataset to verify it works

### Examples

```bash
# Full pipeline (detect + record + replay) - default
python main.py

# Show all options
python main.py --help

# Test camera only
python main.py --test-only

# Record for 5 minutes (300 seconds)
python main.py --record 300

# Record with custom output location
python main.py --record 300 --output my_recording

# Replay a specific dataset
python main.py --replay --dataset data/recordings/recording_20260907_120000

# Replay the latest dataset
python main.py --replay

# Verbose output for debugging
python main.py --verbose

# List all 14 acceptance tests
python main.py --list-tests
```

### Hardware Setup

1. **Connect OAK-D Pro** to USB 3.x port on your Windows machine
2. **Verify connection:**
   ```bash
   python -c "import depthai; print(depthai.Device.getAllAvailableDevices())"
   ```

If you see device info, you're ready to run `python main.py`.

### What Happens

When you run `python main.py` without arguments:

**Step 1: Device Detection**
- Searches for connected OAK-D Pro camera
- Reads device information and calibration
- Should complete in <2 seconds

**Step 2: Recording** (requires camera)
- Records synchronized RGB, depth, and IMU data
- Default: 10 seconds
- Saves to: `data/recordings/recording_TIMESTAMP/`
- Shows frame count and progress

**Step 3: Replay** (no camera needed)
- Plays back recorded dataset
- Verifies all frames loaded correctly
- Shows dataset statistics

### Troubleshooting

**Error: "Device detection failed"**
- Check USB connection to OAK-D Pro
- Ensure camera is powered (light should be on)
- Try a different USB 3.x port
- Verify drivers: `python -c "import depthai; print(depthai.Device.getAllAvailableDevices())"`

**Error: "No dataset found to replay"**
- Make sure you ran `python main.py --record` first
- Check if `data/recordings/` directory exists
- If you interrupted a recording, try `--replay --dataset <path>`

**Python: "No module named 'depthai'"**
- Run: `pip install -r requirements.txt`
- Wait for dependencies to install

**Windows: "python command not found"**
- Use `python3 main.py` instead
- Or add Python to PATH in Windows Environment Variables

### Configuration

Edit these files to customize behavior:

```
src/config/camera_config.yaml        # RGB, depth, IMU settings
src/config/localization_config.yaml  # Algorithm parameters
```

Default values work for most use cases.

### Next Steps

1. ✓ Run `python main.py` and verify everything works
2. ✓ Read `MILESTONE_1.md` for detailed acceptance tests
3. ✓ Check `ARCHITECTURE.md` for system design
4. ✓ See `COORDINATE_FRAMES.md` for frame definitions

### Testing All 14 Acceptance Tests

Once you confirm `python main.py` works, run individual tests:

```bash
# Device detection (included in main.py)
python tests/test_device_detection.py

# RGB acquisition (runs in background)
python tests/test_rgb_acquisition.py --duration 300

# Stereo and depth
python tests/test_stereo_acquisition.py --duration 120
python tests/test_depth_accuracy.py
```

See `MILESTONE_1.md` for complete testing specifications.

### File Structure

```
ar_display_autonomous/
├── main.py                  ← NEW: Run everything from here!
├── README.md                (Full documentation)
├── QUICKSTART.md           (This file - Windows users start here)
├── MILESTONE_1.md          (Acceptance test specifications)
├── ARCHITECTURE.md         (System design)
│
├── src/
│   ├── camera/             (Hardware interface)
│   ├── config/             (Configuration files)
│   └── utils/              (Utilities)
│
├── scripts/
│   ├── record_dataset.py    (Record single dataset)
│   ├── replay_dataset.py    (Replay single dataset)
│   └── run_all_acceptance_tests.py  (Run all tests)
│
├── tests/                  (Individual acceptance tests)
├── data/                   (Datasets, calibration)
└── requirements.txt        (Python dependencies)
```

### Performance (Milestone 1 Targets)

- **RGB:** 30 FPS, zero drops
- **Depth:** 30 FPS, < 2% error at 5m
- **IMU:** 200 Hz, continuous
- **Stability:** 5+ minutes without crash
- **CPU:** 15-25% on laptop

### Support

For detailed information:
- Hardware setup: See `README.md` section "Hardware Setup"
- Architecture: See `ARCHITECTURE.md`
- Coordinate frames: See `COORDINATE_FRAMES.md`
- Sensor timing: See `SENSOR_TIMING.md`
- Acceptance tests: See `MILESTONE_1.md`

---

**That's it!** No more juggling multiple commands. Just `python main.py` and go.
