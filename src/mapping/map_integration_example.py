"""
Example: How to use 3D maps (2GIS, GeoJSON, etc.) with the AR localization system

This shows how to:
1. Download/prepare 3D map data
2. Load into the system
3. Use for visual localization and AR rendering
"""

from pathlib import Path
from map_3d_loader import Map3DLoader
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


def example_load_geojson_maps():
    """
    Example 1: Load 3D maps from GeoJSON files.

    GeoJSON format (expected structure):
    ```json
    {
      "type": "FeatureCollection",
      "features": [
        {
          "type": "Feature",
          "properties": {
            "name": "Building Name",
            "height": 25.0,
            "building_type": "apartment",
            "id": 12345
          },
          "geometry": {
            "type": "Polygon",
            "coordinates": [
              [
                [27.5123, 54.6789, 125.5],  // [lon, lat, altitude]
                [27.5125, 54.6789, 125.5],
                [27.5125, 54.6791, 125.5],
                [27.5123, 54.6791, 125.5],
                [27.5123, 54.6789, 125.5]
              ]
            ]
          }
        }
      ]
    }
    ```

    Usage:
    """
    loader = Map3DLoader()

    # Load buildings from 3D map data
    buildings_result = loader.load_geojson("data/maps/buildings_3d.geojson")
    roads_result = loader.load_geojson("data/maps/roads_3d.geojson")

    logger.info(f"Buildings loaded: {len(buildings_result['buildings'])}")
    logger.info(f"Roads loaded: {len(roads_result['roads'])}")

    # Access loaded data
    for building in loader.buildings[:5]:  # First 5 buildings
        logger.info(f"  {building.name} - Height: {building.height_m}m, "
                   f"Vertices: {len(building.footprint_vertices)}")

    return loader


def example_query_nearby_buildings():
    """
    Example 2: Query buildings near camera position.

    Useful for localization - only project buildings near where
    the camera currently is.
    """
    loader = example_load_geojson_maps()

    # Camera position (example: Minsk, Belarus)
    camera_lat = 53.9045
    camera_lon = 27.5615
    search_radius_m = 200  # 200m radius

    nearby = loader.get_buildings_in_region(camera_lat, camera_lon, search_radius_m)
    logger.info(f"Buildings within {search_radius_m}m: {len(nearby)}")


def example_use_with_projection_engine():
    """
    Example 3: Project 3D buildings into camera view.

    This is how the AR system uses the loaded maps.
    """
    from ar.projection_engine import ProjectionEngine
    import numpy as np

    loader = example_load_geojson_maps()

    # Initialize projection engine with camera calibration
    K = np.array([
        [1395.6, 0.0, 640.0],
        [0.0, 1395.8, 360.0],
        [0.0, 0.0, 1.0]
    ], dtype=np.float32)

    projector = ProjectionEngine(camera_intrinsics=K)

    # Example pose estimate (where camera is in world)
    class MockPose:
        latitude = 53.9045
        longitude = 27.5615
        altitude = 125.5
        roll_deg = 0.0
        pitch_deg = -15.0  # Looking slightly down
        yaw_deg = 0.0

    pose = MockPose()
    origin = (pose.latitude, pose.longitude, pose.altitude)

    projector.set_pose(pose, origin)

    # Project buildings to 2D image
    buildings_2d = projector.project_buildings(loader.buildings, origin)

    logger.info(f"Projected buildings: {len(buildings_2d)}")
    for building_2d in buildings_2d[:3]:
        if building_2d.is_visible:
            logger.info(f"  {building_2d.name}: visible at pixel "
                       f"({building_2d.centroid_pixel[0]:.0f}, "
                       f"{building_2d.centroid_pixel[1]:.0f})")


def example_use_with_geometric_verifier():
    """
    Example 4: Use 3D buildings for geometric verification.

    This helps verify that detected features actually match
    the 3D map geometry.
    """
    from localization.geometric_verifier import GeometricVerifier

    loader = example_load_geojson_maps()

    verifier = GeometricVerifier()

    # Example depth map (from camera)
    import numpy as np
    depth_map = np.random.rand(480, 640) * 50  # 0-50m depth

    # Check if buildings match the observed depth
    for building in loader.buildings[:5]:
        # This is pseudo-code - actual implementation depends on
        # how GeometricVerifier processes 3D building data
        consistency = verifier.verify_building_consistency(
            building, depth_map, camera_pose=(53.9045, 27.5615, 125.5)
        )
        logger.info(f"  {building.name}: consistency = {consistency:.2f}")


def example_download_2gis_maps():
    """
    Example 5: How to download 3D maps from 2GIS.

    2GIS provides API access to map data.

    Note: Requires API key from https://dev.2gis.com/
    """
    logger.info("""
    To use 2GIS maps:

    1. Get API key from https://dev.2gis.com/
    2. Use 2GIS API to fetch map data (buildings, roads)
    3. Export as GeoJSON
    4. Load with: loader.load_geojson("path/to/2gis_export.geojson")

    Alternative 2GIS approaches:
    - Download pre-made map tiles from their web interface
    - Use 2GIS export tools (if available in your region)
    - Use OverPass API (OSM alternative): https://overpass-turbo.eu/

    For open data without API keys, consider:
    - OpenStreetMap (OSM) with 3D building heights: load_geojson()
    - Mapbox 3D building layer: export as GeoJSON
    - Cesium 3D Tiles: would need custom loader
    """)


def main():
    """Run all examples."""
    logger.info("=== 3D Map Loading Examples ===\n")

    logger.info("Example 1: Load GeoJSON maps")
    example_load_geojson_maps()

    logger.info("\nExample 2: Query nearby buildings")
    example_query_nearby_buildings()

    # Examples 3-4 require actual map files, so just show the code
    logger.info("\nExample 3: Project to camera view (see code)")
    logger.info("Example 4: Geometric verification (see code)")
    logger.info("Example 5: Download 2GIS maps")
    example_download_2gis_maps()

    logger.info("\n=== Integration with Main System ===")
    logger.info("""
    To integrate 3D maps into main.py:

    1. In main.py, replace OSM loader:
        # from mapping.osm_loader import load_buildings, load_roads
        from mapping.map_3d_loader import Map3DLoader

        loader = Map3DLoader()
        buildings = loader.load_geojson("data/maps/buildings_3d.geojson")
        roads = loader.load_geojson("data/maps/roads_3d.geojson")

    2. Pass to localization system:
        localization.set_map_geometries(loader.buildings, loader.roads)

    3. AR rendering will automatically use 3D building geometry
       with proper height and wireframe rendering

    4. Geometric verification will use 3D coordinates for
       better depth consistency checks
    """)


if __name__ == "__main__":
    main()
