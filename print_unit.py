import xarray as xr

# Open the NetCDF file
file_name = "2014013100_4HD.nc"
ds = xr.open_dataset(f"data/unzipdata/1402_KAJIKI/nwp/{file_name}")

# List all variables and their units
for var in ds.data_vars:
    variable = ds[var]
    # Check if the 'units' attribute exists and print it
    if 'units' in variable.attrs:
        print(f"Variable: {var}, Units: {variable.attrs['units']}")
    else:
        print(f"Variable: {var}, Units: Not specified")