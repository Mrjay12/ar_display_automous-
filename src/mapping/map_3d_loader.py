"""
3D Map Loader - Load geographic 3D geometry from various sources.

Supports:
- GeoJSON with 3D building footprints
- KML with altitude data
- GLTF/GLB 3D models
- Custom JSON formats from 2GIS and other sources

Provides geographic region queries and coordinate transformations.
"""

import logging
from typing import Optional, List, Dict, Tuple
from dataclasses import dataclass
from pathlib import Path
import json

import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Building3D:
    """3D building geometry and metadata."""
    building_id: str
    latitude: float
    longitude: float
    altitude: float  # Base altitude in meters
    height: float  # Building height in meters
    footprint: List[Tuple[float, float]]  # (lat, lon) vertices
    centroid: Tuple[float, float, float]  # (lat, lon, alt) center point
    geometry_type: str  # "polygon", "prism", etc.
    properties: Dict = None


@dataclass
class Road3D:
    """3D road geometry."""
    road_id: str
    path: List[Tuple[float, float, float]]  # (lat, lon, alt) points
    width_m: float
    properties: Dict = None


class Map3DLoader:
    """Load and query 3D geographic maps."""

    def __init__(self):
        self.buildings: List[Building3D] = []
        self.roads: List[Road3D] = []
        self._location_index = {}  # Quick lookup by region
        logger.info("Map3DLoader initialized")

    def load_geojson(self, file_path: str) -> bool:
        """Load GeoJSON file with building geometry."""
        try:
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"GeoJSON file not found: {file_path}")
                return False

            with open(path, 'r') as f:
                data = json.load(f)

            feature_count = len(data.get('features', []))
            logger.info(f"Loading {feature_count} features from {file_path}")

            for feature in data.get('features', []):
                self._parse_geojson_feature(feature)

            logger.info(f"✓ Loaded {len(self.buildings)} buildings from GeoJSON")
            return True

        except Exception as e:
            logger.error(f"GeoJSON load error: {e}")
            return False

    def load_gltf(self, file_path: str) -> bool:
        """Load GLTF/GLB 3D model with geographic metadata."""
        try:
            logger.info(f"GLTF loading not yet implemented: {file_path}")
            return False
        except Exception as e:
            logger.error(f"GLTF load error: {e}")
            return False

    def load_kml(self, file_path: str) -> bool:
        """Load KML file with placemarks and altitude."""
        try:
            import xml.etree.ElementTree as ET
            path = Path(file_path)
            if not path.exists():
                logger.warning(f"KML file not found: {file_path}")
                return False

            tree = ET.parse(path)
            root = tree.getroot()

            ns = {'kml': 'http://www.opengis.net/kml/2.2'}
            placemarks = root.findall('.//kml:Placemark', ns)

            logger.info(f"Loading {len(placemarks)} placemarks from KML")

            for placemark in placemarks:
                self._parse_kml_placemark(placemark, ns)

            logger.info(f"✓ Loaded KML: {len(self.buildings)} buildings, {len(self.roads)} roads")
            return True

        except Exception as e:
            logger.error(f"KML load error: {e}")
            return False

    def get_buildings_in_region(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float = 500.0
    ) -> List[Building3D]:
        """Get all buildings within radius of center point."""
        from geographiclib.geodesic import Geodesic

        results = []
        geod = Geodesic.WGS84

        for building in self.buildings:
            g = geod.Inverse(center_lat, center_lon, building.latitude, building.longitude)
            distance_m = g['s12']

            if distance_m <= radius_m:
                results.append(building)

        return sorted(results, key=lambda b: geod.Inverse(
            center_lat, center_lon, b.latitude, b.longitude)['s12'])

    def get_all_buildings(self) -> List[Building3D]:
        """Return all loaded buildings."""
        return self.buildings

    def get_building_by_id(self, building_id: str) -> Optional[Building3D]:
        """Get specific building by ID."""
        for b in self.buildings:
            if b.building_id == building_id:
                return b
        return None

    def _parse_geojson_feature(self, feature: Dict):
        """Parse single GeoJSON feature into Building3D."""
        try:
            props = feature.get('properties', {})
            geom = feature.get('geometry', {})

            building_id = props.get('id', str(len(self.buildings)))
            lat = props.get('lat', 0.0)
            lon = props.get('lon', 0.0)
            alt = float(props.get('altitude', 0.0))
            height = float(props.get('height', 10.0))

            # Parse footprint from coordinates
            footprint = []
            if geom.get('type') == 'Polygon':
                coords = geom.get('coordinates', [[]])[0]
                # Convert [lon, lat] to (lat, lon)
                footprint = [(c[1], c[0]) for c in coords]

            if footprint:
                building = Building3D(
                    building_id=building_id,
                    latitude=lat,
                    longitude=lon,
                    altitude=alt,
                    height=height,
                    footprint=footprint,
                    centroid=(lat, lon, alt + height / 2),
                    geometry_type="polygon",
                    properties=props
                )
                self.buildings.append(building)

        except Exception as e:
            logger.warning(f"Could not parse GeoJSON feature: {e}")

    def _parse_kml_placemark(self, placemark, ns):
        """Parse single KML Placemark."""
        try:
            name = placemark.findtext('kml:name', '', ns) or "unknown"
            alt_elem = placemark.find('.//kml:altitude', ns)
            altitude = float(alt_elem.text) if alt_elem is not None else 0.0

            # Parse polygon coordinates
            coords_elem = placemark.find('.//kml:coordinates', ns)
            if coords_elem is not None and coords_elem.text:
                coords = [tuple(map(float, c.split(',')))
                         for c in coords_elem.text.strip().split()]
                # coords are (lon, lat, alt), convert to (lat, lon)
                footprint = [(c[1], c[0]) for c in coords]

                if footprint:
                    lat, lon = footprint[0]
                    building = Building3D(
                        building_id=name,
                        latitude=lat,
                        longitude=lon,
                        altitude=altitude,
                        height=10.0,
                        footprint=footprint,
                        centroid=(lat, lon, altitude + 5.0),
                        geometry_type="polygon"
                    )
                    self.buildings.append(building)

        except Exception as e:
            logger.warning(f"Could not parse KML placemark: {e}")

    def __repr__(self):
        return f"Map3DLoader(buildings={len(self.buildings)}, roads={len(self.roads)})"
