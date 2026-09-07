# Sensor Timing & Synchronization

## Overview

This document specifies how multiple sensor streams are timestamped and synchronized in the GPS-Denied Visual Localization System.

---

## Timestamp Format

### Microsecond Precision

All timestamps use **microsecond (µs)** precision:

```
timestamp_us = integer microseconds since UNIX epoch (1970-01-01 00:00:00 UTC)
timestamp_sec = timestamp_us / 1,000,000

Example:
  timestamp_us = 1694095200000000
  timestamp_sec = 1694095200.0 (seconds)
  datetime = 2023-09-07 12:00:00 UTC
```

### Availability

Every sensor measurement MUST have a valid timestamp:
- RGB frames: timestamp_us
- Depth frames: timestamp_us
- Stereo left/right frames: timestamp_us (synchronized)
- IMU samples: timestamp_us

---

## Sensor Acquisition Rates

| Sensor | Rate | Interval | Notes |
|--------|------|----------|-------|
| RGB | 30 FPS | 33.33 ms | Configurable: 15, 20, 24, 25, 30 |
| Stereo (L/R) | 30 FPS | 33.33 ms | Must match RGB for sync |
| Depth | 30 FPS | 33.33 ms | Generated from stereo |
| IMU | 200 Hz | 5 ms | Accelerometer + Gyroscope |

### Frame Intervals (Target)

```
RGB Frame 0:        timestamp = 0 µs
RGB Frame 1:        timestamp ≈ 33,333 µs (30 ms + precision jitter)
RGB Frame 2:        timestamp ≈ 66,666 µs
...
RGB Frame 9000:     timestamp ≈ 300,000,000 µs (5 minutes)

Expected: 9,000 frames in 5 minutes
Actual:   8,997-9,003 frames acceptable (±0.03%)
```

### Maximum Acceptable Jitter

```
RGB Frame Interval:
  Target:           33.33 ms (30 FPS)
  Tolerance:        ±10% (30-37 ms)
  Alert threshold:  >50 ms (indicates frame drop)

IMU Sample Interval:
  Target:           5.00 ms (200 Hz)
  Tolerance:        ±5% (4.75-5.25 ms)
  Alert threshold:  >10 ms
```

---

## Multi-Sensor Synchronization

### Strategy

The system uses **timestamp-based association** to synchronize frames:

1. **Each frame is tagged** with its hardware-generated timestamp
2. **Frames are buffered** separately by sensor
3. **Synchronization** associates frames by proximity in time
4. **Association window** is ±50 ms (half-frame interval)

### Association Algorithm

```
For each RGB frame at time T:
  1. Find depth frame closest to T (within ±50 ms)
  2. Collect IMU samples in range [T-25ms, T+25ms]
  3. Create synchronized frame group (RGB, depth, IMU)
  4. Log timestamp skew for analysis
```

### Timestamp Skew

**Definition:** Maximum time difference between any sensors in a synchronized group.

**Acceptance Criteria:**
- Target: < 10 ms skew
- Acceptable: < 50 ms skew
- Failure: > 100 ms skew (indicates desync)

**Measurement:**
```python
def measure_skew(rgb_timestamp, depth_timestamp, imu_timestamps):
    """Calculate synchronization skew."""
    all_timestamps = [rgb_timestamp, depth_timestamp] + imu_timestamps
    skew = max(all_timestamps) - min(all_timestamps)
    return skew  # microseconds
```

---

## Recording & Replay

### Recorded Format

Timestamps are preserved exactly as acquired from hardware:

```
Recording Directory:
├── metadata.yaml           # Recording parameters
├── calibration.yaml        # Camera calibration at record time
├── rgb/
│   ├── 000000.png         # First RGB frame
│   ├── 000001.png
│   └── index.yaml         # Frame index with timestamps
├── depth/
│   ├── 0000000000000000.npz  # Timestamped depth file
│   └── index.yaml
└── imu/
    └── imu_log.csv        # CSV: timestamp_us, ax, ay, az, gx, gy, gz
```

### Index File Format

```yaml
rgb:
  - frame_id: 0
    timestamp_us: 1694095200000000
    filename: 000000.png
  - frame_id: 1
    timestamp_us: 1694095200033333
    filename: 000001.png
  ...

depth:
  - timestamp_us: 1694095200000000
    filename: 0000000000000000.npz
  ...

imu:
  - timestamp_us: 1694095200000000
    sample_count: 1
  ...
```

### Playback

During replay, timestamps are delivered in **original order** but playback speed can be adjusted:

```python
replayer = SensorReplayer(dataset_dir)
replayer.playback_rate = 1.0    # Real-time (1x)
replayer.playback_rate = 0.5    # Half-speed (0.5x)
replayer.playback_rate = 2.0    # Double-speed (2x)

# Frames delivered with original timestamps preserved
rgb_frame = replayer.get_rgb_frame()
assert rgb_frame.timestamp_us == expected_timestamp
```

---

## Timestamp Validation

### Monotonic Increase

Timestamps MUST strictly increase in time:

```
Frame 0: timestamp_us = 1694095200000000
Frame 1: timestamp_us = 1694095200033333 ✓ (increased)
Frame 2: timestamp_us = 1694095200033332 ✗ (BACKWARD JUMP - ERROR)
```

**Test:**
```python
def test_monotonic_timestamps():
    """Verify timestamps never decrease."""
    prev_ts = 0
    for frame in frames:
        assert frame.timestamp_us > prev_ts, "Timestamp went backward"
        prev_ts = frame.timestamp_us
```

### Gap Detection

Large gaps indicate missing frames:

```
Frame 100: timestamp_us = 1694095203333000  (100*33333)
Frame 101: timestamp_us = 1694095203400000  (gap: 67ms, ~2 frames)
Frame 102: timestamp_us = 1694095203433333  (normal interval)
```

**Alert Logic:**
```python
def detect_gaps(timestamps, expected_interval_us):
    """Find frame intervals larger than expected."""
    gaps = []
    for i in range(1, len(timestamps)):
        actual_interval = timestamps[i] - timestamps[i-1]
        if actual_interval > expected_interval_us * 1.5:  # >50% larger
            gaps.append((i, actual_interval, expected_interval_us))
    return gaps
```

### Impossible Timestamps

Timestamps must be within reasonable bounds:

```python
def validate_timestamp_sanity(timestamp_us):
    """Check timestamp is physically possible."""
    import time
    current_time_us = int(time.time() * 1_000_000)
    
    # Timestamp must not be in future (allow 1 second margin)
    if timestamp_us > current_time_us + 1_000_000:
        return False, "Timestamp in future"
    
    # Timestamp must be reasonable (after device started)
    # Typically within last 24 hours
    if timestamp_us < current_time_us - 86400_000_000:
        return False, "Timestamp too old"
    
    return True, None
```

---

## Performance Requirements

### Deadline Compliance

- RGB frame acquisition: < 33 ms from sensor to queue
- Depth generation: < 33 ms from stereo images
- IMU reading: < 5 ms per sample
- **Overall latency:** < 50 ms end-to-end

### Frame Delivery Guarantees

- **RGB:** Continuous stream, <1% drops acceptable
- **Depth:** <1% drops acceptable
- **IMU:** Continuous, no gaps > 10 ms

### Timestamp Precision

- Hardware resolution: Microsecond level (OAK-D Pro)
- Software tracking: Microsecond precision maintained
- No rounding to milliseconds

---

## Synchronization Quality Metrics

### Acceptance Criteria

| Metric | Target | Acceptable | Failure |
|--------|--------|-----------|---------|
| Timestamp skew | <10 ms | <50 ms | >100 ms |
| Frame drop rate | <0.1% | <1.0% | >5.0% |
| Timestamp gaps | None | <0.1% | >1% |
| Monotonic increase | 100% | 100% | <99% |

### Test Sequence

```python
def test_synchronization_quality():
    """Comprehensive sync quality test."""
    
    # Record 5-minute dataset
    recorder.record(300)
    
    # Verify timestamps
    assert test_monotonic_increase(rgb_timestamps)
    assert test_monotonic_increase(depth_timestamps)
    assert test_monotonic_increase(imu_timestamps)
    
    # Verify gaps
    gaps = detect_timestamp_gaps(rgb_timestamps, expected_33ms)
    assert len(gaps) < 30  # <0.1% of 9000 frames
    
    # Verify skew
    skew_values = measure_sync_skew_per_frame()
    assert np.percentile(skew_values, 95) < 50_000  # 95% <50ms
    
    # Verify frame rates
    assert 8997 <= rgb_frame_count <= 9003
    assert 59700 <= imu_sample_count <= 60300
```

---

## Debugging Timestamp Issues

### Common Problems & Solutions

**Problem:** Frames appear to jump backward in time

**Diagnosis:**
```python
# Check for monotonic violations
prev_ts = 0
for i, ts in enumerate(timestamps):
    if ts <= prev_ts:
        print(f"Frame {i}: timestamp {ts} <= previous {prev_ts}")
    prev_ts = ts
```

**Solution:** Likely hardware issue or USB bandwidth problem. Reconnect camera.

---

**Problem:** Unusual gaps in frame sequence

**Diagnosis:**
```python
# Plot frame interval distribution
intervals = np.diff(timestamps)
plt.hist(intervals, bins=50)
plt.xlabel("Frame Interval (µs)")
plt.ylabel("Count")
plt.title("Frame Interval Distribution (should be ~33,333 µs)")
```

**Solution:** Reduce resolution or FPS. Check USB bandwidth.

---

**Problem:** IMU samples not synchronized with RGB

**Diagnosis:**
```python
# Check timestamp correlation
rgb_ts = rgb_timestamps[::10]  # Every 10th RGB frame
imu_ts = imu_timestamps[::100]  # Every 100th IMU sample (roughly 5ms)

# Should be roughly overlapping in time
print(f"RGB range: {rgb_ts[0]} to {rgb_ts[-1]}")
print(f"IMU range: {imu_ts[0]} to {imu_ts[-1]}")
```

**Solution:** Verify IMU is enabled in pipeline. Check device configuration.

---

## Reference Implementation

### TimestampValidator Class

See `src/utils/timestamp_sync.py` for:
- `validate_monotonic_increase(timestamps)`
- `check_timestamp_gaps(timestamps)`
- `detect_impossible_timestamps(timestamps)`

### Recording Format

See `src/camera/sensor_recording.py` for:
- Frame indexing with timestamps
- Synchronized multi-sensor buffering
- Index file generation

### Replay

See `src/camera/sensor_replay.py` for:
- Timestamp-accurate playback
- Frame retrieval by timestamp
- Dataset information retrieval

---

This specification ensures that all sensor data is timestamped consistently and can be replayed with full temporal accuracy offline.
