# Unified Command Usage Guide

## The Problem (What You Requested)

**Your request:** "no i need one code to run everything i hate multiple starting commands"

Previously, you needed to run multiple separate commands:
```bash
python tests/test_device_detection.py
python scripts/record_dataset.py --duration 300
python scripts/replay_dataset.py --dataset data/recordings/my_dataset
```

This was tedious and hard to remember.

## The Solution: `python main.py`

Now there's **ONE command** to run everything:

```bash
python main.py
```

That's it! No need to remember multiple script names or arguments.

## What It Does

When you run `python main.py` (without arguments), it automatically:

1. **Detects** your OAK-D Pro camera
2. **Records** a 10-second synchronized dataset (RGB + depth + IMU)
3. **Replays** the dataset to verify everything works

## Common Usage Patterns

### Full Pipeline (Default)
```bash
# Runs: detect → record → replay
python main.py
```

### Just Check if Camera Works
```bash
# Only runs device detection test
python main.py --test-only
```

### Just Record Data
```bash
# Records for 5 minutes (300 seconds)
python main.py --record 300

# With custom output location
python main.py --record 300 --output my_recording
```

### Just Replay Data
```bash
# Replays the latest dataset (no camera needed!)
python main.py --replay

# Replays a specific dataset
python main.py --replay --dataset data/recordings/recording_20260907_120000
```

### Get Help
```bash
# Show all options
python main.py --help

# List all 14 acceptance tests
python main.py --list-tests
```

### Debug Mode
```bash
# Verbose output for troubleshooting
python main.py --verbose
```

## Architecture

The unified entry point (`main.py`) delegates to existing modules:
- `src/camera/oak_d_interface.py` — Hardware interface
- `src/camera/sensor_recording.py` — Dataset recording
- `src/camera/sensor_replay.py` — Dataset playback
- `src/utils/logging_config.py` — Logging configuration

## Why This Is Better

| Before | Now |
|--------|-----|
| Remember: `test_device_detection.py` | Remember: `python main.py --test-only` |
| Remember: `record_dataset.py` | Remember: `python main.py --record 300` |
| Remember: `replay_dataset.py` | Remember: `python main.py --replay` |
| 3 different commands | 1 command with options |
| Google for docs each time | Run `python main.py --help` |

## Backward Compatibility

Don't worry—the old individual scripts still work:
```bash
python tests/test_device_detection.py           # Still works
python scripts/record_dataset.py --duration 300 # Still works
python scripts/replay_dataset.py --dataset path # Still works
```

But you don't need to use them anymore.

## Windows-Specific Notes

See `QUICKSTART.md` for Windows-specific examples and troubleshooting.

Key points:
- Works with forward slashes in paths (e.g., `data/recordings/`)
- Works with Windows command prompt and PowerShell
- Use `python` or `python3` depending on your setup

## Next Steps

1. Install dependencies: `pip install -r requirements.txt`
2. Connect OAK-D Pro to USB 3.x port
3. Run: `python main.py`
4. Check the output for any errors or issues

## For Developers

The unified entry point is implemented as a clean wrapper:
- File: `main.py`
- ~335 lines of Python
- Uses argparse for CLI
- Integrates with existing logging system
- Platform-agnostic (works on Windows/Linux/macOS)

The CLI is designed to be:
- **Intuitive:** Sensible defaults, minimal required arguments
- **Helpful:** Clear error messages and suggestions
- **Composable:** Combine flags for different workflows
- **Transparent:** Shows progress and status at each step

## Examples Summary

```bash
# Default: everything
python main.py

# Component-specific
python main.py --test-only
python main.py --record 300
python main.py --replay

# Advanced
python main.py --record 300 --output my_data --verbose
python main.py --replay --dataset data/recordings/recording_20260907_120000
python main.py --list-tests

# Getting help
python main.py --help
```

---

**That's it!** One command. Everything. No juggling scripts.
