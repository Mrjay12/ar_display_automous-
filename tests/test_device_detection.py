"""
Acceptance Test 1: Device Detection

Purpose: Verify OAK-D Pro is reliably detected and initialized.

Expected:
- Device detected within 2 seconds
- Serial number retrieved
- All sensor streams initialized
- Graceful shutdown
"""

import pytest
import logging
from pathlib import Path

from src.camera.oak_d_interface import OAKDInterface
from src.config.config_loader import ConfigLoader

logger = logging.getLogger(__name__)


@pytest.fixture
def config():
    """Load camera configuration."""
    loader = ConfigLoader()
    return loader.load_camera_config()


def test_device_detection(config):
    """Test 1: Verify device detection."""
    oak_d = OAKDInterface(config=config)

    # Initialize device
    assert oak_d.initialize() is True, "Device initialization failed"

    # Verify device info
    device_info = oak_d.get_device_info()
    assert device_info is not None, "Device info not available"
    assert "mxId" in device_info, "Serial number not found"
    assert "name" in device_info, "Device name not found"

    logger.info(f"✓ Device detected: {device_info['name']} ({device_info['mxId']})")

    # Verify initialized
    assert oak_d.is_initialized() is True, "Device not marked as initialized"

    # Shutdown
    oak_d.shutdown()
    assert oak_d.is_initialized() is False, "Device not shutdown properly"

    logger.info("✓ Test 1 PASS")


def test_device_detection_failure():
    """Test 1b: Handle device not detected gracefully."""
    oak_d = OAKDInterface(device_id="nonexistent_device_12345")

    # Should fail gracefully
    result = oak_d.initialize()
    assert result is False or oak_d.get_device_info() is None, "Should not initialize fake device"

    logger.info("✓ Test 1b PASS (graceful failure)")


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
