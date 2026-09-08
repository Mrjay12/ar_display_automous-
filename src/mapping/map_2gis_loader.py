"""
2GIS Map Loader - Fetch and load 3D maps from 2GIS

2GIS (2ГИС) is a Russian/Central Asian mapping service with extensive 3D building data.

This module handles:
- Converting 2GIS map data to internal 3D geometry format
- Caching downloaded tiles locally
- Querying 2GIS API (if key available) or loading pre-downloaded data
- Merging multiple map sources

Modes:
1. Pre-downloaded: User manually downloads from 2GIS and saves as GeoJSON
2. API mode: Use 2GIS API key for dynamic queries (requires account)
3. Hybrid: Mix 2GIS data with OSM fallback for coverage

Example:
    >>> loader = Map2GISLoader()
    >>> buildings = loader.load_from_2gis_file("data/2gis_export.geojson")
    >>> # or
    >>> buildings = loader.query_region(lat=53.9, lon=27.5, api_key="your_key")
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Any, Tuple
from dataclasses import dataclass
import time

logger = logging.getLogger(__name__)


class Map2GISLoader:
    """Load 3D maps from 2GIS or 2GIS-compatible data sources."""

    def __init__(self, cache_dir: str = "data/2gis_cache"):
        """
        Initialize 2GIS loader.

        Args:
            cache_dir: Directory for caching downloaded tiles
        """
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)

        # Import base loader
        try:
            from map_3d_loader import Map3DLoader
            self.base_loader = Map3DLoader(cache_dir=cache_dir)
        except ImportError:
            logger.error("map_3d_loader not found - run from src/mapping/")
            self.base_loader = None

    def load_from_2gis_file(self, filepath: str) -> Dict[str, Any]:
        """
        Load 2GIS data from locally saved file.

        2GIS exports GeoJSON with building/road geometry and metadata.
        This method converts 2GIS GeoJSON to internal format.

        Args:
            filepath: Path to 2GIS GeoJSON export file

        Returns:
            Dict with 'buildings' and 'roads' lists
        """
        if not self.base_loader:
            logger.error("Base loader not initialized")
            return {"buildings": [], "roads": []}

        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"2GIS file not found: {filepath}")
            return {"buildings": [], "roads": []}

        logger.info(f"Loading 2GIS map from {filepath.name}")

        # Load as GeoJSON using base loader
        result = self.base_loader.load_geojson(str(filepath))
        logger.info(f"Loaded {len(result['buildings'])} buildings, "
                   f"{len(result['roads'])} roads from 2GIS")

        return result

    def load_from_kml(self, filepath: str) -> Dict[str, Any]:
        """
        Load 3D maps from KML file (2GIS KML export).

        KML format used by some 2GIS exports includes:
        - <Placemark> for buildings with height in <description>
        - <coordinates> with elevation: lon,lat,alt

        Args:
            filepath: Path to KML file

        Returns:
            Dict with 'buildings' and 'roads' lists
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"KML file not found: {filepath}")
            return {"buildings": [], "roads": []}

        try:
            import xml.etree.ElementTree as ET
        except ImportError:
            logger.error("xml module not available")
            return {"buildings": [], "roads": []}

        try:
            tree = ET.parse(filepath)
            root = tree.getroot()
        except Exception as e:
            logger.error(f"Failed to parse KML: {e}")
            return {"buildings": [], "roads": []}

        buildings = []
        roads = []

        # KML namespace handling
        ns = {'kml': 'http://www.opengis.net/kml/2.2'}

        for placemark in root.findall('.//kml:Placemark', ns):
            try:
                name = placemark.find('kml:name', ns)
                name = name.text if name is not None else None

                # Try to get geometry
                polygon = placemark.find('.//kml:Polygon', ns)
                linestring = placemark.find('.//kml:LineString', ns)

                if polygon is not None:
                    outer = polygon.find('.//kml:outerBoundaryIs/kml:LinearRing/kml:coordinates', ns)
                    if outer is not None:
                        vertices = self._parse_kml_coords(outer.text)
                        if vertices:
                            buildings.append({
                                "name": name,
                                "vertices": vertices,
                                "height": 15.0  # Default if not in description
                            })

                elif linestring is not None:
                    coords_elem = linestring.find('kml:coordinates', ns)
                    if coords_elem is not None:
                        vertices = self._parse_kml_coords(coords_elem.text)
                        if vertices:
                            roads.append({
                                "name": name,
                                "vertices": vertices,
                            })

            except Exception as e:
                logger.debug(f"Skipped KML placemark: {e}")
                continue

        logger.info(f"Loaded {len(buildings)} buildings, {len(roads)} roads from KML")
        return {"buildings": buildings, "roads": roads}

    def _parse_kml_coords(self, coords_str: str) -> Optional[List[Tuple[float, float, float]]]:
        """Parse KML coordinate string (lon,lat,alt format)."""
        if not coords_str:
            return None

        vertices = []
        for coord in coords_str.strip().split():
            parts = coord.split(',')
            if len(parts) >= 2:
                try:
                    lon = float(parts[0])
                    lat = float(parts[1])
                    alt = float(parts[2]) if len(parts) >= 3 else 0.0
                    vertices.append((lat, lon, alt))
                except ValueError:
                    continue

        return vertices if vertices else None

    def query_region(
        self,
        lat: float,
        lon: float,
        radius_m: float = 500,
        api_key: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Query 3D map data for a geographic region.

        Two modes:
        1. With API key: Query 2GIS API directly (requires account)
        2. Without API key: Show download instructions

        Args:
            lat: Center latitude
            lon: Center longitude
            radius_m: Search radius in meters
            api_key: Optional 2GIS API key

        Returns:
            Dict with 'buildings' and 'roads' lists
        """
        if api_key is None:
            logger.warning("No 2GIS API key provided. Use local files instead:")
            logger.warning("1. Download from: https://maps.2gis.com/")
            logger.warning("2. Export as GeoJSON")
            logger.warning("3. Load with: loader.load_from_2gis_file('path.geojson')")
            return {"buildings": [], "roads": []}

        # API query would go here (2GIS API structure)
        logger.info(f"Would query 2GIS API for region ({lat}, {lon}) with radius {radius_m}m")
        logger.warning("2GIS API integration not yet implemented. Use pre-downloaded data.")

        return {"buildings": [], "roads": []}

    def merge_maps(self, *loaders) -> Dict[str, Any]:
        """
        Merge multiple 3D maps into one dataset.

        Useful for combining 2GIS data with OSM fallback for complete coverage.

        Args:
            *loaders: Multiple Map3DLoader or Map2GISLoader instances

        Returns:
            Merged buildings and roads
        """
        all_buildings = []
        all_roads = []

        for loader in loaders:
            if hasattr(loader, 'buildings'):
                all_buildings.extend(loader.buildings)
            if hasattr(loader, 'roads'):
                all_roads.extend(loader.roads)

        # Remove duplicates by OSM ID
        seen_buildings = set()
        unique_buildings = []
        for building in all_buildings:
            if building.osm_id not in seen_buildings:
                seen_buildings.add(building.osm_id)
                unique_buildings.append(building)

        seen_roads = set()
        unique_roads = []
        for road in all_roads:
            if road.osm_id not in seen_roads:
                seen_roads.add(road.osm_id)
                unique_roads.append(road)

        logger.info(f"Merged maps: {len(unique_buildings)} buildings, {len(unique_roads)} roads")
        return {"buildings": unique_buildings, "roads": unique_roads}

    def export_for_offline_use(self, output_file: str):
        """
        Export loaded maps to file for offline use.

        Useful for preprocessing maps before deployment.

        Args:
            output_file: Path to save export (GeoJSON format)
        """
        if not self.base_loader or not self.base_loader.buildings:
            logger.warning("No buildings loaded to export")
            return

        output_path = Path(output_file)
        output_path.parent.mkdir(parents=True, exist_ok=True)

        features = []

        # Export buildings as polygons
        for building in self.base_loader.buildings:
            coords = [[lon, lat, alt] for lat, lon, alt in building.footprint_vertices]
            coords.append(coords[0])  # Close ring

            feature = {
                "type": "Feature",
                "properties": {
                    "id": building.osm_id,
                    "name": building.name,
                    "height": building.height_m,
                    "building_type": building.building_type,
                },
                "geometry": {
                    "type": "Polygon",
                    "coordinates": [coords]
                }
            }
            features.append(feature)

        # Export roads as linestrings
        for road in self.base_loader.roads:
            coords = [[lon, lat, alt] for lat, lon, alt in road.path_vertices]

            feature = {
                "type": "Feature",
                "properties": {
                    "id": road.osm_id,
                    "name": road.name,
                    "road_type": road.road_type,
                    "width_m": road.width_m,
                },
                "geometry": {
                    "type": "LineString",
                    "coordinates": coords
                }
            }
            features.append(feature)

        geojson = {
            "type": "FeatureCollection",
            "features": features
        }

        try:
            with open(output_path, 'w', encoding='utf-8') as f:
                json.dump(geojson, f, indent=2)
            logger.info(f"Exported {len(features)} features to {output_path}")
        except Exception as e:
            logger.error(f"Export failed: {e}")


# Quick reference for users
def print_2gis_guide():
    """Print user guide for getting 2GIS data."""
    print("""
    ╔══════════════════════════════════════════════════════════════╗
    ║         HOW TO GET 3D MAPS FROM 2GIS                         ║
    ╚══════════════════════════════════════════════════════════════╝

    Option 1: Download from 2GIS Web (Free)
    ─────────────────────────────────────────
    1. Go to https://2gis.com/ or https://2gis.ae/ (regional versions)
    2. Navigate to your area of interest
    3. Look for export/download options
    4. Save as GeoJSON or KML
    5. Load with:
       loader = Map2GISLoader()
       buildings = loader.load_from_2gis_file("your_export.geojson")

    Option 2: Use 2GIS API (Requires Account)
    ───────────────────────────────────────────
    1. Get API key from https://dev.2gis.com/
    2. Use loader.query_region(lat, lon, api_key="your_key")
    3. (Implementation provided once API key obtained)

    Option 3: Use Overpass API / OSM 3D Data (Free)
    ────────────────────────────────────────────────
    1. Visit https://overpass-turbo.eu/
    2. Query buildings in your area:
       [bbox:...];
       (node["building"];way["building"];relation["building"];);
       out geom;
    3. Export as GeoJSON
    4. Load with loader.load_from_2gis_file()

    Option 4: Use Online Converters
    ────────────────────────────────
    1. Download from 2GIS (KML or other format)
    2. Convert to GeoJSON online:
       https://mapbox.github.io/togeojson/ (KML→GeoJSON)
    3. Load the converted file

    Integration:
    ────────────
    Once you have the GeoJSON file:

        from mapping.map_2gis_loader import Map2GISLoader
        loader = Map2GISLoader()
        data = loader.load_from_2gis_file("maps/2gis_buildings.geojson")

    The system will automatically use 3D building heights and geometry
    for improved localization and AR visualization.
    """)


if __name__ == "__main__":
    print_2gis_guide()
