import ee, requests
import numpy as np
# import rasterio
# from rasterio.io import MemoryFile
import geemap
import netCDF4 as nc
import ipywidgets as widgets
from ipyleaflet import WidgetControl
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon, shape
from shapely.prepared import prep
from rtree import index
from tqdm import tqdm   # optional, nice progress bar
from datetime import datetime


#######################################################################################################################
#                           This script loads in gridded temperature, precipitation and GDP data                      # 
#                           Temporal resolution: 1990:2022                                                            #       
#                           Spatial Resolution: [-180, 180] lon, [-90, 90] lat, 0.5 degree grid                       #
#                           Data Source: CRU TS4.08 (temperature and precipitation), Google Earth Engine (GDP)        #
#######################################################################################################################       

# Authenticate Earth Engine
ee.Authenticate() 

# Initialize the Earth Engine module.
ee.Initialize(project='ee-justmarius98')

def get_region_geometries():
    'defines geometries for different continents'
    regions={
        
        'europe': ee.Geometry.Rectangle([-10, 35, 40, 70]),
        'africa': ee.Geometry.Rectangle([-20, -35, 55, 35]),
        'asia': ee.Geometry.Rectangle(60, -10, 150, 50),
        'northAmerica': ee.Geometry.Rectangle([-170, 25, -50, 75]),
        'southAmerica': ee.Geometry.Rectangle([-90, -60, -30, 15])
    }
    return regions

#load in color palettes for visualization
def get_palettes():
    palettes = {
    "mako": [
        "#0d0887", "#3d049b", "#6300a7", "#8b0aa5",
        "#b12ba0", "#cf4c92", "#e9717f", "#f8956b",
        "#ffa95e", "#ffbd4c", "#ffd940", "#fcfdbf"
    ],
    "viridis": [
        "#440154", "#482878", "#3e4989", "#31688e",
        "#26828e", "#1f9e89", "#35b779", "#6ece58",
        "#b5de2b", "#fde725"
    ],
    "inferno": [
        "#000004", "#1b0c41", "#4a0c6b", "#781c6d",
        "#a52c60", "#cf4446", "#ed6925", "#fb9b06",
        "#f7d13d", "#fcffa4"
    ],
    "plasma": [
        "#0d0887", "#41049d", "#6a00a8", "#8f0da4",
        "#b12a90", "#cb4679", "#e16462", "#f1834c",
        "#fca834", "#fcce25", "#f0f921"
    ],
    "turbo": [
        "#30123b", "#4145ab", "#4679f1", "#249cff",
        "#00b5ff", "#00cbff", "#2bd9df", "#53e2bc",
        "#78e89e", "#98ed84", "#b7f171", "#d4f661",
        "#effb56", "#fafb50", "#f2ea4c", "#e1c649",
        "#cd9f47", "#b27645", "#945043", "#73353f",
        "#511e33",
    ] ,
    "ornblue":[
    "#001219",
    "#005f73",
    "#0a9396",
    "#94d2bd",
    "#e9d8a6",
    "#ee9b00",
    "#ca6702",
    "#bb3e03",
    "#ae2012",
    "#9b2226"
] 
    }
    return palettes

palettes = get_palettes()

# relevant years
years=range(1990,2023)

cutoff_year=1990

#raw_data
gdp_adm1_polygon=ee.FeatureCollection("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/poly_adm1_gdp_perCapita_1990_2022")
gdp_adm1=ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/adm1_gdp_perCapita_1990_2022")
gdp_adm0=ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/adm0_gdp_perCapita_1990_2022")
temp= nc.Dataset("RawData/cru_ts4.08.1901.2023.tmp.dat.nc")
precip= nc.Dataset("RawData/cru_ts4.08.1901.2023.pre.dat.nc")
density= nc.Dataset("RawData/gpw_v4_population_density_rev11_30_min.nc")

raster=density.variables['raster']

raster[0]
raster[1]

data_density=density.variables['Population Density, v4.11 (2000, 2005, 2010, 2015, 2020): 30 arc-minutes'][0,:,:].filled(np.nan)

longitudes = {float(lon): i for i, lon in enumerate(temp.variables['lon'][:])}
latitudes  = {float(lat): i for i, lat in enumerate(temp.variables['lat'][:])}

time_= temp.variables['time']
cuttoff_1990= (datetime(1990,1,1) - datetime(1900,1,1)).days
time_mask=np.where(temp.variables['time'][:] >= cuttoff_1990)[0]

temp=temp.variables['tmp'][time_mask,:,:].filled(np.nan)
precip=precip.variables['pre'][time_mask,:,:].filled(np.nan)

n_regions=gdp_adm1_polygon.size().getInfo()


#######################################################################################################################
#                                                  Data illustration                                                  # 
#######################################################################################################################


def illustrate_ee_map(var, vis_params=None):
    'function to illustrate an ee.Image on a map'
    Map = geemap.Map(center=[0, 0], zoom=2, basemap='SATELLITE')
    vis_params = {
    'min': 1000, 'max': 80000,
    'palette':  palettes['ornblue'],
  
    }
    custom_ticks = [1000, 2000, 3000, 4000]
    Map.addLayer(var, vis_params, 'GDP per capita')
    colors = vis_params["palette"]
    vmin = vis_params["min"]
    vmax = vis_params["max"]

    Map.add_colorbar_branca(colors=colors, vmin=vmin, vmax=vmax, layer_name="GDP per capita", label="GDP per capita (USD)", max_labels=4, custom_ticks=custom_ticks)  


    return Map

Map=illustrate_ee_map(gdp_adm1.select('PPP_1990'))

Map.to_html("gdp_cap_1990.html")


# Get band names
band_names = gdp_adm1.bandNames().getInfo()  # ['gdp_1990', 'gdp_1991', ..., 'gdp_2022']

# Create an ImageCollection from bands
frames = []
for band in band_names:
    gdp_vis = gdp_adm1.select(band).visualize(min=0, max=100000, palette=palettes['ornblue'])
    frames.append(gdp_vis)

Movie = ee.ImageCollection(frames)

# Define arguments for animation function parameters.
videoArgs = {
    "dimensions": 768,
    "region": europe,
    "framesPerSecond": 10,
    "crs": "EPSG:3857",
    
}

# Add time text to the GIF
text = [year for year in range(1990, 2023)]

gif_path = "europe.gif"
geemap.download_ee_video(Movie, videoArgs, gif_path)

#add text to gif
saved_gif = gif_path

geemap.add_text_to_gif(
    saved_gif,
    saved_gif,
    xy=("3%", "5%"),
    text_sequence=text,
    font_size=30,
    font_color="#ffffff",
)

geemap.add_text_to_gif(
    saved_gif,
    saved_gif,
    xy=("3%", "10%"),
    text_sequence="GDP per capita (USD)",
    font_color="white",
)


#find the index corresponding to this date in the netCDF time able. make into a dictionary for easy lookup later. The key is the index in the netCDF time able, the value is the coordinate value


#######################################################################################################################
#                                                  Metadata                                                           # 
#######################################################################################################################


instance=gdp_adm1_polygon.getInfo().get('features')

rows=[]
for k in range(n_regions):
        rows.append({
            'fid': instance[k].get('properties').get('fid'),
            'Country': instance[k].get('properties').get('Country'),
            'iso3': instance[k].get('properties').get('iso3'),
            'region': instance[k].get('properties').get('Subnat'),
            'coordinates': instance[k].get('geometry').get('coordinates'),
            'polygon':shape({"type": "Polygon", "coordinates": instance[k].get('geometry').get('coordinates')}) if isinstance(instance[k].get('geometry').get('coordinates')[0][0][0], (float, int)) else shape({"type":"MultiPolygon", "coordinates": instance[k].get('geometry').get('coordinates')})
        
        })
        
info = pd.DataFrame(rows)


lon_grid, lat_grid = np.meshgrid(list(longitudes), list(latitudes))
grid_points = np.column_stack([lon_grid.ravel(), lat_grid.ravel()])



def update_with_gridpoints(metadata, grid_points):
    
    metadata=info    
    n_points = grid_points.shape[0] 
    # ---------- 1) Build R-tree of points ----------
    pt_idx = index.Index()
    for pi, (lon, lat) in enumerate(grid_points):
        pt_idx.insert(pi, (lon, lat, lon, lat))

    # ---------- 2) Prepare assigned mask and outputs ----------
    assigned = np.zeros(n_points, dtype=bool)      # True when a point is already mapped
    assignments = np.full(n_points, -1, dtype=int) # polygon index for each point, -1 = unassigned
    region_to_points = {}                          # fid -> list of (lon, lat)

    
    # ---------- 3) Iterate polygons (recommended) ----------
    for poly_i, row in tqdm(metadata.reset_index(drop=True).iterrows(), total=len(metadata)):
        poly = row['polygon']
        if poly.is_empty:
            continue
        prepared = prep(poly)                       # speed up repeated tests
        fid = row['fid']
        iso3 = row['iso3']
        region_name = row['region']

        # candidate point indices within polygon bounding box
        cand_ids = list(pt_idx.intersection(poly.bounds))

        # filter out already assigned candidates (faster to check mask in Python)
        # iterate remaining candidate ids and test precise containment
        assigned_list = []
        for pi in cand_ids:
            if assigned[pi]:
                continue
            lon, lat = grid_points[pi]
            # use prepared.covers for point-in-polygon test (covers includes boundary)
            if prepared.covers(Point(lon, lat)):
                assigned[pi] = True
                assignments[pi] = poly_i
                region_to_points.setdefault(fid, []).append((float(lon), float(lat)))
                assigned_list.append(pi)
        # finished polygon


    rows=[]
    for k in range(n_regions):
            fid_key = instance[k].get('properties').get('fid')
            rows.append({
                'fid': fid_key,
                'Country': instance[k].get('properties').get('Country'),
                'iso3': instance[k].get('properties').get('iso3'),
                'region': instance[k].get('properties').get('Subnat'),
                'coordinates': instance[k].get('geometry').get('coordinates'),
                'polygon':shape({"type": "Polygon", "coordinates": instance[k].get('geometry').get('coordinates')}) if isinstance(instance[k].get('geometry').get('coordinates')[0][0][0], (float, int)) else shape({"type":"MultiPolygon", "coordinates": instance[k].get('geometry').get('coordinates')}),
                'lon_lat_points': region_to_points.get(fid_key, []),
            })
        
    return pd.DataFrame(rows)
    
    


metadata=update_with_gridpoints(info, grid_points)




#######################################################################################################################
#                                             GDP, Temp, Precipitation                                                # 
#######################################################################################################################

def calculate_climate(var, lon_lat_points, time_keys, tot_density):
    'gets a temperature or precipitation value for a given year and list of lon-lat points'
    'looks in the NC files loaded above and averages over the lon-lat points provided, the shape of the temp and precip arrays is (time, lat, lon)' 
    
    if var =='tmp':
        
        temp_density_weighted = np.nansum([temp[time_index, lat, lon]*data_density[360-lat-1, lon] for lon, lat in lon_lat_points for time_index in time_keys])
        average_monthly_temperature_weighted = temp_density_weighted/ tot_density/12
        return average_monthly_temperature_weighted
    
    elif var =='pre':
        average_value = np.nansum([precip[time_index, lat, lon]*data_density[360-lat-1, lon] for lon, lat in lon_lat_points for time_index in time_keys])
        average_yearly_precipitation_weighted = average_value/ tot_density
        return average_yearly_precipitation_weighted




def observations_in_year(year):

    'calculates the index value of the netcdf corresponding to the given year. time_dt is a datetime array loaded above from the netcdf time able. The function returns the index in time_dt where the year matches the input year'
    valid_indexes=[np.where(np.array([dt.year for dt in time_dt]) == year)[0]]
    
    return valid_indexes[0]

def convert_lon_lat_to_indexes(array_of_lon_lat):

    'converts a list of (lon, lat) points to indexes in the netcdf files'
    
    indexes=[]
    for lon, lat in array_of_lon_lat:
        lon_index=longitudes[lon]
        lat_index=latitudes[lat]
        indexes.append((lon_index, lat_index))
        
        
    return indexes


#calculate an array of an array, where the first entry is 1-12, the second is 13-24, etc. up to 396 (for 1990-2022)

months_since_1990 = {year: list(range((year - 1990) * 12+1, (year - 1990) * 12 + 13)) for year in years }


rows=[]
for k in range(n_regions):
    for year, months in months_since_1990.items():

        lon_lat_indexes=convert_lon_lat_to_indexes(metadata.loc[k, 'lon_lat_points'])
   
        #density indexes are reversed in latitude
        tot_density = np.nansum([data_density[360-lat_idx-1, lon_idx] for lon_idx, lat_idx in lon_lat_indexes])
        
        rows.append({
            'fid': instance[k].get('properties').get('fid'),
            'iso3': instance[k].get('properties').get('iso3'),
            'year': year,
            'GDP_per_capita': instance[k].get('properties').get(str(year)),
            'temperature': calculate_climate('tmp', lon_lat_points=lon_lat_indexes,  time_keys=months, tot_density= tot_density ),
            'precipitation': calculate_climate('pre', lon_lat_points=lon_lat_indexes,  time_keys=months, tot_density= tot_density),
            'population_density': tot_density
        })
        print(f"Processed year {year} for region {k+1} of {n_regions}")
    print(f"Finished processing region {k+1} of {n_regions}")

        
        
fact_data = pd.DataFrame(rows)

#now we merge fact_data with metadata to get lon_lat points in the final dataframe
metadata_reduced = metadata[['fid', 'iso3', 'region', 'Country']]

#merge on fid and drop missing values for temperature and precipitation
full_data = fact_data.merge(metadata_reduced, on=['fid', 'iso3'], how='left').replace({'temperature': {0: np.nan}, 'precipitation': {0: np.nan}}).dropna(subset=['GDP_per_capita', 'temperature', 'precipitation'])

#calculate the log difference of GDP per capita compared to previous year
full_data = full_data.sort_values(by=['fid', 'year']).reset_index(drop=True)
full_data['log_gdp_per_capita'] = np.log(full_data['GDP_per_capita'])
full_data['growth'] = full_data.groupby('fid')['log_gdp_per_capita'].diff()

#we dont need the log gdp_per_capita column anymore
full_data = full_data.drop(columns=['log_gdp_per_capita'])

#write out the full data to a csv file
full_data.to_csv("ee_data.csv", index=False)


##Illustrate the grid points for a specific country

afg_data=full_data[full_data['iso3']=='AFG']
#select the row where iso3 equals DZA
dza_data = metadata[metadata['iso3'] == 'DZA']

len(dza_data['lon_lat_points'][0])


#illustrate the lon_lat for DZA on a map, note that the polygon is accesible in dza_data['polygon'].values[0]
import folium

m = folium.Map(location=[28, 3], zoom_start=5)
for lon, lat in dza_data['lon_lat_points'].values[0]:
    folium.CircleMarker(location=[lat, lon], radius=2, color='blue', fill=True).add_to(m)
m.save("dza_lon_lat_points.html")

