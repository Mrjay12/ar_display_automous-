"""
Mapping Module - Geographic data management

Submodules:
  - osm_loader: Load OpenStreetMap building and road geometries
  - map_manager: Manage map data in memory with spatial indexing
"""

from .osm_loader import (
    OSMLoader,
    Building,
    Road,
)
from .map_manager import (
    MapManager,
    MapTile,
)

__all__ = [
    'OSMLoader',
    'Building',
    'Road',
    'MapManager',
    'MapTile',
]
