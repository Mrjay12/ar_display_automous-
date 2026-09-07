"""
Configuration Loader

Responsibilities:
- Load configuration from YAML files
- Merge configurations
- Provide configuration access
- Validate configuration values

Input:
- YAML configuration files

Output:
- Merged configuration dictionary
- Access interface for configuration values
"""

import logging
from pathlib import Path
from typing import Optional, Dict, Any

import yaml

logger = logging.getLogger(__name__)


class ConfigLoader:
    """Load and manage configuration from YAML files."""

    def __init__(self, config_dir: Optional[Path] = None):
        """
        Initialize configuration loader.

        Args:
            config_dir: Directory containing YAML config files.
                       Defaults to ./src/config
        """
        self.config_dir = config_dir or Path("./src/config")
        self.config: Dict[str, Any] = {}

    def load_camera_config(self) -> Dict[str, Any]:
        """Load camera configuration."""
        return self._load_file("camera_config.yaml", "camera")

    def load_localization_config(self) -> Dict[str, Any]:
        """Load localization algorithm configuration."""
        return self._load_file("localization_config.yaml", "localization")

    def load_map_config(self) -> Dict[str, Any]:
        """Load map configuration."""
        return self._load_file("map_config.yaml", "map")

    def _load_file(self, filename: str, config_key: str) -> Dict[str, Any]:
        """
        Load a configuration file.

        Args:
            filename: Name of YAML file
            config_key: Key to store in config dict

        Returns:
            Configuration dictionary for this file
        """
        try:
            filepath = self.config_dir / filename

            if not filepath.exists():
                logger.warning(f"Configuration file not found: {filepath}")
                return {}

            with open(filepath, "r") as f:
                file_config = yaml.safe_load(f) or {}

            self.config[config_key] = file_config

            logger.info(f"Loaded configuration from {filename}")
            return file_config

        except Exception as e:
            logger.error(f"Error loading {filename}: {e}", exc_info=True)
            return {}

    def get(self, key: str, default: Any = None) -> Any:
        """
        Get configuration value.

        Args:
            key: Configuration key (dot-separated for nested access)
            default: Default value if key not found

        Returns:
            Configuration value or default
        """
        keys = key.split(".")
        value = self.config

        for k in keys:
            if isinstance(value, dict):
                value = value.get(k)
                if value is None:
                    return default
            else:
                return default

        return value if value is not None else default

    def load_all(self) -> None:
        """Load all available configuration files."""
        self.load_camera_config()
        self.load_localization_config()
        self.load_map_config()
        logger.info(f"All configurations loaded from {self.config_dir}")

    @property
    def camera(self) -> Dict[str, Any]:
        """Get camera configuration."""
        return self.config.get("camera", {})

    @property
    def localization(self) -> Dict[str, Any]:
        """Get localization configuration."""
        return self.config.get("localization", {})

    @property
    def map(self) -> Dict[str, Any]:
        """Get map configuration."""
        return self.config.get("map", {})
