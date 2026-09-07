"""
Logging Configuration

Responsibilities:
- Configure logging for entire system
- Provide structured logging format
- Route logs to console and files
- Set appropriate levels per module

Input:
- Logging config from camera_config.yaml

Output:
- Configured logger for all modules
"""

import logging
import logging.handlers
from pathlib import Path
from typing import Optional, Dict, Any

def setup_logging(
    config: Optional[Dict[str, Any]] = None,
    log_dir: Optional[Path] = None
) -> None:
    """
    Configure logging for the application.

    Args:
        config: Logging configuration dict (from YAML)
        log_dir: Directory to store log files
    """
    if config is None:
        config = {}

    if log_dir is None:
        log_dir = Path("./logs")

    log_dir.mkdir(parents=True, exist_ok=True)

    # Get logging settings
    level_str = config.get("level", "INFO")
    level = getattr(logging, level_str, logging.INFO)

    console_output = config.get("console_output", True)
    file_output = config.get("file_output", True)
    log_prefix = config.get("log_file_prefix", "app_")

    # Create formatter
    formatter = logging.Formatter(
        fmt='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
        datefmt='%Y-%m-%d %H:%M:%S'
    )

    # Root logger
    root_logger = logging.getLogger()
    root_logger.setLevel(level)

    # Clear existing handlers
    root_logger.handlers.clear()

    # Console handler
    if console_output:
        console_handler = logging.StreamHandler()
        console_handler.setLevel(level)
        console_handler.setFormatter(formatter)
        root_logger.addHandler(console_handler)

    # File handler
    if file_output:
        from datetime import datetime
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        log_file = log_dir / f"{log_prefix}{timestamp}.log"

        file_handler = logging.FileHandler(log_file)
        file_handler.setLevel(level)
        file_handler.setFormatter(formatter)
        root_logger.addHandler(file_handler)

    # Module-specific levels (optional)
    logging.getLogger("depthai").setLevel(logging.WARNING)  # DepthAI can be verbose
    logging.getLogger("PIL").setLevel(logging.WARNING)  # PIL debug output


def get_logger(name: str) -> logging.Logger:
    """
    Get a logger for a module.

    Args:
        name: Module name (typically __name__)

    Returns:
        Configured logger instance
    """
    return logging.getLogger(name)
