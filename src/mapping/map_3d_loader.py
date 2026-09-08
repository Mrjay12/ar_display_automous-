"""
3D Map Loader - Load 3D building models from various sources

Supports:
- GeoJSON with 3D coordinates (2GIS, Mapbox, OSM-3D)
- glTF/GLB 3D model files
- OSM building data with elevation estimates
- Custom 3D map sources (extensible)

Provides unified interface for:
- Building 3D geometry (vertices, edges, faces)
- Road networks with elevation
- Texture/material data
- Semantic labels (building name, height, type)

Example:
    >>> loader = Map3DLoader()
    >>> buildings = loader.load_geojson("maps/buildings.geojson")
    >>> roads = loader.load_geojson("maps/roads.geojson")
    >>> model = loader.load_gltf("models/city.glb")
"""

import json
import logging
from pathlib import Path
from typing import Optional, List, Dict, Tuple, Any
from dataclasses import dataclass
import numpy as np

logger = logging.getLogger(__name__)


@dataclass
class Building3D:
    """3D building geometry and metadata."""
    osm_id: int
    name: Optional[str]
    building_type: Optional[str]
    height_m: float

    # Geometry: vertices in [lat, lon, altitude]
    footprint_vertices: List[Tuple[float, float, float]]  # Base footprint
    roof_vertices: Optional[List[Tuple[float, float, float]]]  # Roof outline
    edges: List[Tuple[int, int]]  # Vertex index pairs

    # Optional 3D model
    model_path: Optional[str] = None
    model_format: Optional[str] = None  # 'gltf', 'glb', 'obj'

    # Texture/material
    color_rgb: Optional[Tuple[int, int, int]] = None
    has_texture: bool = False


@dataclass
class Road3D:
    """3D road geometry."""
    osm_id: int
    name: Optional[str]
    road_type: Optional[str]  # primary, secondary, residential, etc.
    width_m: float

    # Path vertices in [lat, lon, altitude]
    path_vertices: List[Tuple[float, float, float]]

    # Optional additional attributes
    speed_limit_kmh: Optional[int] = None
    surface_type: Optional[str] = None


class Map3DLoader:
    """Load 3D map data from various sources."""

    def __init__(self, cache_dir: Optional[str] = None):
        """
        Initialize 3D map loader.

        Args:
            cache_dir: Directory for caching downloaded map tiles
        """
        self.cache_dir = Path(cache_dir) if cache_dir else Path("data/map_cache")
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.buildings: List[Building3D] = []
        self.roads: List[Road3D] = []

    def load_geojson(self, filepath: str) -> Dict[str, Any]:
        """
        Load 3D geometries from GeoJSON file.

        Expects features with:
        - geometry.type: "Point", "LineString", "Polygon", "MultiPolygon"
        - geometry.coordinates: [lon, lat] or [lon, lat, altitude]
        - properties: {name, height, building_type, etc.}

        Args:
            filepath: Path to GeoJSON file

        Returns:
            Dict with 'buildings' and 'roads' lists
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"GeoJSON file not found: {filepath}")
            return {"buildings": [], "roads": []}

        try:
            with open(filepath, 'r', encoding='utf-8') as f:
                geojson_data = json.load(f)
        except Exception as e:
            logger.error(f"Failed to load GeoJSON: {e}")
            return {"buildings": [], "roads": []}

        buildings = []
        roads = []

        if "features" not in geojson_data:
            logger.warning("GeoJSON has no 'features' key")
            return {"buildings": [], "roads": []}

        for feature in geojson_data["features"]:
            try:
                geom = feature.get("geometry", {})
                props = feature.get("properties", {})

                # Determine feature type
                feature_type = props.get("type") or props.get("building") or props.get("highway")

                if feature_type in ["building", "yes", "house", "residential"] or \
                   (geom.get("type") == "Polygon" and props.get("height")):
                    building = self._parse_building_geojson(feature)
                    if building:
                        buildings.append(building)
                elif feature_type in ["road", "primary", "secondary", "residential", "highway"] or \
                     geom.get("type") in ["LineString", "MultiLineString"]:
                    road = self._parse_road_geojson(feature)
                    if road:
                        roads.append(road)
            except Exception as e:
                logger.debug(f"Skipped feature: {e}")
                continue

        logger.info(f"Loaded {len(buildings)} buildings, {len(roads)} roads from {filepath.name}")
        self.buildings.extend(buildings)
        self.roads.extend(roads)

        return {"buildings": buildings, "roads": roads}

    def _parse_building_geojson(self, feature: Dict) -> Optional[Building3D]:
        """Parse a single building feature from GeoJSON."""
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})

        if geom.get("type") not in ["Polygon", "MultiPolygon"]:
            return None

        coords = geom.get("coordinates", [])
        if not coords:
            return None

        # Extract vertices
        if geom.get("type") == "Polygon":
            vertices = self._extract_vertices_from_coords(coords[0])
        else:  # MultiPolygon - use largest polygon
            polygons = [self._extract_vertices_from_coords(p[0]) for p in coords]
            vertices = max(polygons, key=len) if polygons else []

        if not vertices:
            return None

        # Parse properties
        height = float(props.get("height", 15.0))
        name = props.get("name")
        building_type = props.get("building_type") or props.get("building")
        osm_id = int(props.get("id", hash(str(vertices)) & 0x7fffffff))

        # Create roof vertices (simple extrusion)
        roof_vertices = [(lat, lon, alt + height) for lat, lon, alt in vertices]

        # Create edges (footprint + vertical + roof)
        n = len(vertices)
        edges = []
        # Footprint edges
        for i in range(n):
            edges.append((i, (i + 1) % n))
        # Vertical edges
        for i in range(n):
            edges.append((i, i + n))
        # Roof edges
        for i in range(n):
            edges.append((i + n, ((i + 1) % n) + n))

        return Building3D(
            osm_id=osm_id,
            name=name,
            building_type=building_type,
            height_m=height,
            footprint_vertices=vertices,
            roof_vertices=roof_vertices,
            edges=edges,
            color_rgb=self._get_building_color(building_type)
        )

    def _parse_road_geojson(self, feature: Dict) -> Optional[Road3D]:
        """Parse a single road feature from GeoJSON."""
        geom = feature.get("geometry", {})
        props = feature.get("properties", {})

        if geom.get("type") not in ["LineString", "MultiLineString"]:
            return None

        coords = geom.get("coordinates", [])
        if not coords:
            return None

        # Extract vertices
        if geom.get("type") == "LineString":
            vertices = self._extract_vertices_from_coords(coords)
        else:  # MultiLineString - use longest segment
            segments = [self._extract_vertices_from_coords(seg) for seg in coords]
            vertices = max(segments, key=len) if segments else []

        if not vertices:
            return None

        # Parse properties
        name = props.get("name")
        road_type = props.get("highway") or props.get("road_type")
        width = float(props.get("width", 10.0))
        speed_limit = props.get("maxspeed")
        if speed_limit and isinstance(speed_limit, str):
            try:
                speed_limit = int(speed_limit.split()[0])
            except:
                speed_limit = None

        osm_id = int(props.get("id", hash(str(vertices)) & 0x7fffffff))

        return Road3D(
            osm_id=osm_id,
            name=name,
            road_type=road_type,
            width_m=width,
            path_vertices=vertices,
            speed_limit_kmh=speed_limit
        )

    def _extract_vertices_from_coords(self, coords: List) -> List[Tuple[float, float, float]]:
        """
        Extract 3D vertices from coordinate list.

        Handles both [lon, lat] and [lon, lat, alt] formats.
        Converts to [lat, lon, alt] (geographic order) for internal use.
        """
        vertices = []
        for coord in coords:
            if len(coord) >= 2:
                lon, lat = coord[0], coord[1]
                alt = float(coord[2]) if len(coord) >= 3 else 0.0
                vertices.append((lat, lon, alt))
        return vertices

    def _get_building_color(self, building_type: Optional[str]) -> Optional[Tuple[int, int, int]]:
        """Get color for building type."""
        color_map = {
            "house": (200, 180, 150),
            "apartment": (180, 160, 140),
            "commercial": (150, 150, 200),
            "industrial": (120, 120, 120),
            "retail": (220, 100, 100),
            "office": (100, 150, 220),
        }
        return color_map.get(building_type) if building_type else (170, 170, 170)

    def load_gltf(self, filepath: str) -> Optional[Dict[str, Any]]:
        """
        Load 3D model from glTF/GLB file.

        Returns model metadata and path for rendering.
        Note: Actual model rendering deferred to geometry_renderer.

        Args:
            filepath: Path to .glb or .gltf file

        Returns:
            Model metadata dict or None
        """
        filepath = Path(filepath)
        if not filepath.exists():
            logger.error(f"Model file not found: {filepath}")
            return None

        if filepath.suffix.lower() not in ['.glb', '.gltf']:
            logger.error(f"Unsupported model format: {filepath.suffix}")
            return None

        try:
            logger.info(f"Loaded 3D model: {filepath.name}")
            return {
                "path": str(filepath),
                "format": "glb" if filepath.suffix.lower() == ".glb" else "gltf",
                "name": filepath.stem,
                "size_bytes": filepath.stat().st_size
            }
        except Exception as e:
            logger.error(f"Failed to load 3D model: {e}")
            return None

    def get_buildings_in_region(
        self,
        center_lat: float,
        center_lon: float,
        radius_m: float = 500
    ) -> List[Building3D]:
        """
        Get buildings within radius of center point.

        Args:
            center_lat: Center latitude
            center_lon: Center longitude
            radius_m: Search radius in meters

        Returns:
            List of Building3D objects
        """
        # Simple distance check (approx)
        deg_per_meter = 1.0 / 111320.0  # At equator
        radius_deg = radius_m * deg_per_meter

        nearby = []
        for building in self.buildings:
            for lat, lon, _ in building.footprint_vertices:
                if abs(lat - center_lat) < radius_deg and abs(lon - center_lon) < radius_deg:
                    nearby.append(building)
                    break

        return nearby

    def clear_cache(self):
        """Clear cached map data."""
        import shutil
        if self.cache_dir.exists():
            shutil.rmtree(self.cache_dir)
            self.cache_dir.mkdir(parents=True, exist_ok=True)
        logger.info("Map cache cleared")

    def export_statistics(self) -> Dict[str, Any]:
        """Export loading statistics."""
        return {
            "buildings_loaded": len(self.buildings),
            "roads_loaded": len(self.roads),
            "cache_dir": str(self.cache_dir),
            "avg_building_height_m": np.mean([b.height_m for b in self.buildings]) if self.buildings else 0,
        }
