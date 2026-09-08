# 3D Maps Integration - Quick Start

Use downloaded 3D maps (2GIS, OpenStreetMap, etc.) to match rover vision against real buildings and roads.

## Quick Start (5 minutes)

### 1. Get a 3D Map File

**Option A: Free - Use Overpass API**
```bash
# Visit https://overpass-turbo.eu/
# Paste this query:
[bbox:27.48,53.88,27.60,53.92];  # Minsk example - change coordinates
(node["building"];way["building"];relation["building"];);
out geom;

# Click Run → Export → GeoJSON
# Save as data/maps/buildings.geojson
```

**Option B: 2GIS (Russian regions)**
```
1. Go to https://2gis.com or 2gis.ae
2. Find your city
3. Export → Download as GeoJSON or KML
4. Save to data/maps/2gis_buildings.geojson
```

**Option C: Mapbox 3D Building Data**
```
# Sign up free at mapbox.com
# Use their 3D building tileset
# Export as GeoJSON
```

### 2. Load Maps in Your Code

```python
from mapping.map_3d_loader import Map3DLoader

loader = Map3DLoader()
buildings = loader.load_geojson("data/maps/buildings.geojson")
roads = loader.load_geojson("data/maps/roads.geojson")

print(f"Loaded {len(loader.buildings)} buildings")
print(f"Loaded {len(loader.roads)} roads")
```

### 3. Use in Main System

Add to `main.py` before AR demo:

```python
from mapping.map_3d_loader import Map3DLoader

def run_ar_demo_with_3d_maps(ar_mode="overlay"):
    """AR demo using 3D maps instead of OSM."""
    
    # Load 3D maps
    loader = Map3DLoader()
    buildings = loader.load_geojson("data/maps/buildings.geojson")
    
    # Rest of AR demo logic...
    # Projection engine automatically uses 3D building geometry
```

## Supported Formats

### GeoJSON (Recommended)

```json
{
  "type": "FeatureCollection",
  "features": [
    {
      "type": "Feature",
      "properties": {
        "name": "Building Name",
        "height": 25.5,
        "building_type": "apartment",
        "id": 12345
      },
      "geometry": {
        "type": "Polygon",
        "coordinates": [[
          [27.5123, 54.6789, 125.5],  // [lon, lat, altitude]
          [27.5125, 54.6789, 125.5],
          [27.5125, 54.6791, 125.5],
          [27.5123, 54.6791, 125.5],
          [27.5123, 54.6789, 125.5]
        ]]
      }
    }
  ]
}
```

### KML Format

```xml
<?xml version="1.0" encoding="UTF-8"?>
<kml xmlns="http://www.opengis.net/kml/2.2">
  <Document>
    <Placemark>
      <name>Building Name</name>
      <description>Height: 25m</description>
      <Polygon>
        <outerBoundaryIs>
          <LinearRing>
            <coordinates>
              27.5123,54.6789,125.5
              27.5125,54.6789,125.5
              27.5125,54.6791,125.5
              27.5123,54.6791,125.5
              27.5123,54.6789,125.5
            </coordinates>
          </LinearRing>
        </outerBoundaryIs>
      </Polygon>
    </Placemark>
  </Document>
</kml>
```

## How It Works

### Before (OSM only)
```
Camera Feed → Feature Detection → VPR on OSM 2D footprints
                                 → Limited 3D geometry (estimated height)
                                 → Wireframe AR overlay
```

### After (3D Maps)
```
Camera Feed → Feature Detection → VPR on actual 3D building geometry
                                 → Accurate height + structure
                                 → Better depth verification
                                 → Realistic AR overlay
```

## API Reference

### Load GeoJSON
```python
loader = Map3DLoader()
result = loader.load_geojson("buildings.geojson")
# result = {"buildings": [...], "roads": [...]}
```

### Query Nearby Buildings
```python
# Get buildings within 500m of location
nearby = loader.get_buildings_in_region(
    center_lat=53.9045,
    center_lon=27.5615,
    radius_m=500
)
```

### Access Loaded Data
```python
for building in loader.buildings:
    print(f"{building.name}")
    print(f"  Height: {building.height_m}m")
    print(f"  Type: {building.building_type}")
    print(f"  Vertices: {len(building.footprint_vertices)}")
```

### Integration with Projection Engine
```python
from ar.projection_engine import ProjectionEngine

projector = ProjectionEngine(camera_intrinsics=K)
projector.set_pose(camera_pose, origin)

# Project 3D buildings to 2D image
buildings_2d = projector.project_buildings(
    loader.buildings,
    origin=(lat, lon, alt)
)

# Render wireframes
for b in buildings_2d:
    if b.is_visible:
        cv2.polylines(frame, [footprint], True, (0, 255, 0), 2)
```

## Common Issues

### "Buildings loaded: 0"
- Check file exists: `ls data/maps/buildings.geojson`
- Verify GeoJSON format is valid
- Check coordinates are [lon, lat] not [lat, lon]

### "Buildings in image, but not rendering"
- Verify camera pose is set correctly
- Check depth map is valid
- Ensure building height is reasonable (>0)

### Performance Slow
- Use `loader.get_buildings_in_region()` to load only nearby buildings
- Limit to buildings within 1km of camera
- Pre-process and cache maps

## Performance Tips

1. **Load once, use many times:**
   ```python
   loader = Map3DLoader()  # Once
   buildings = loader.load_geojson("maps.geojson")  # Once
   
   # Then in loop
   while camera_running:
       nearby = loader.get_buildings_in_region(...)  # Fast
   ```

2. **Cache preprocessed data:**
   ```python
   loader.export_statistics()
   # Use get_buildings_in_region() with small radius
   ```

3. **Skip distant buildings:**
   ```python
   nearby = loader.get_buildings_in_region(
       center_lat, center_lon,
       radius_m=200  # Closer range = fewer buildings
   )
   ```

## Next Steps

1. Download a 3D map for your test location
2. Load with `Map3DLoader().load_geojson()`
3. Run AR demo: `python main.py --ar-demo`
4. Verify buildings render with correct heights
5. Test localization matching against 3D geometry

## Advanced: 2GIS Integration

If you have 2GIS exports in your region:

```python
from mapping.map_2gis_loader import Map2GISLoader

loader = Map2GISLoader()
data = loader.load_from_2gis_file("2gis_export.geojson")

# Merge with OSM fallback
from mapping.osm_loader import load_buildings as load_osm
osm_buildings = load_osm()

merged = loader.merge_maps(loader, osm_loader)
```

---

**File Structure:**
```
src/mapping/
  ├── map_3d_loader.py          # Generic 3D geometry loader
  ├── map_2gis_loader.py         # 2GIS-specific integration
  ├── map_integration_example.py # Usage examples
  └── osm_loader.py              # OpenStreetMap (existing)

data/maps/
  ├── buildings.geojson          # Your 3D building data
  ├── roads.geojson              # Road network
  └── ...
```

**Test Your Setup:**
```bash
cd src/mapping
python -c "from map_integration_example import main; main()"
```

This will show loading examples and statistics.
