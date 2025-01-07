import numpy as np
import metpy.calc as mpcalc
from metpy.units import units

# Example latitude and longitude arrays (in degrees)

lats = np.array([
    [40.0, 40.0, 40.0],
    [41.0, 41.0, 41.0],
    [42.0, 42.0, 42.0]
])

lons = np.array([
    [-105.0, -104.0, -103.0],
    [-105.0, -104.0, -103.0],
    [-105.0, -104.0, -103.0]
])

# Convert latitude and longitude to units of degrees
lats = lats * units.degrees
lons = lons * units.degrees

# Calculate dx and dy
dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)

# Print results
print("dx (meters):")
print(dx)
print("\ndy (meters):")
print(dy)