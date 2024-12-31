import metpy.calc as mpcalc
from metpy.units import units
import xarray as xr
import datetime
import tqdm
import numpy as np
import os 
import shutil
# import tqd

saved_data_dir = "data/added_features_data"

for folder_name in tqdm.tqdm(os.listdir("data/unzipdata2")):
    file_list = os.listdir(f"data/unzipdata2/{folder_name}/nwp")
    os.makedirs(f"{saved_data_dir}/{folder_name}/nwp/", exist_ok=True)
    
    for file_name in file_list:
        
        shutil.copy2(f"data/unzipdata2/{folder_name}/nwp/{file_name}", f"{saved_data_dir}/{folder_name}/nwp/{file_name}")
        ds = xr.open_dataset(f"data/unzipdata2/{folder_name}/nwp/{file_name}").metpy.parse_cf()

        datetime_str = file_name[:10]  # Lấy chuỗi "2014013100"
        year = int(datetime_str[:4])
        month = int(datetime_str[4:6])
        day = int(datetime_str[6:8])
        hour = int(datetime_str[8:10])

        time_origin = datetime.datetime(year, month, day, hour)
        time_data = ds.time.values
        time_converted = [time_origin + datetime.timedelta(hours=int(t)) for t in time_data]
        time_strings = [t.strftime("%Y-%m-%d %H:%M:%S") for t in time_converted]
        lats = ds.lat.data
        lons = ds.lon.data


        u10m = ds['u10m'].data
        v10m = ds['v10m'].data
        t2m = ds['t2m'].data
        td2m = ds['td2m'].data
        ps = ds['ps'].data
        pmsl = ds['pmsl'].data
        skt = ds['skt'].data
        total_cloud = ds['total_cloud'].data
        rain = ds['rain'].data
        features = ['',]
        lev = ds['lev'].values

        h = ds['h'].values
        u = ds['u'].values
        v = ds['v'].values
        t = ds['t'].values
        q = ds['q'].values
        w = ds['w'].values

        Albedo = ds['Albedo'].values
        Z = ds['Z'].values
        terrain = ds['terrain'].values


        list_vorticity = []
        list_speed = []
        list_div = []
        for level in tqdm.tqdm(lev):
            
            level = level * units.hPa  #lev = 1000, 950, 925, 850, 700, 500, 300, 200 ;

            uwnd_level = ds['u'].metpy.sel(vertical=level).squeeze()
            vwnd_level = ds['v'].metpy.sel(vertical=level).squeeze()
            dx, dy = mpcalc.lat_lon_grid_deltas(lons, lats)
            

            avor_level = mpcalc.absolute_vorticity(uwnd_level, vwnd_level)
            speed_level = mpcalc.wind_speed(uwnd_level, vwnd_level)
            div_level = mpcalc.divergence(uwnd_level, vwnd_level)
            list_vorticity.append(avor_level.values)
            list_speed.append(speed_level.values)
            list_div.append(div_level.values)

        vorticity = np.stack(list_vorticity,1)
        wind_speed = np.stack(list_speed,1)
        divergence = np.stack(list_div,1)

        # os.makedirs(f"{saved_data_dir}/{folder_name}/arr", exist_ok=True)
        # np.savez(f"{saved_data_dir}/{folder_name}/arr/{file_name.split('.')[0]}.npz",lon=lons, lat=lats, lev_soil=lev, u10m=u10m, v10m=v10m, t2m = t2m, td2m=td2m, ps=ps, total_cloud=total_cloud, h=h,u=u, v=v, t=t, q=q, w=w, Albedo=Albedo, Z=Z, terrain=terrain, time_strings=time_strings, skt=skt, rain=rain, pmsl=pmsl, vorticity=vorticity, wind_speed=wind_speed, divergence=divergence)