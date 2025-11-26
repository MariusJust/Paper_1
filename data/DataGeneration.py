s
import ee
import numpy as np
import datetime
import pandas as pd
import geemap

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

time=range(1990,2023)

#calculate the days passing from 1900 to 1990, we exclude all data before 1990

# latitudes= range(-90,90,0.5)
# longitudes= range(-180,180,0.5)





temp= nc.Dataset("RawData/cru_ts4.08.1901.2023.tmp.dat.nc")
precip= nc.Dataset("RawData/cru_ts4.08.1901.2023.pre.dat.nc")
gdp=ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/total_gdp_perCapita_1990_2022_30arcmin")
gdp_adm1=ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/adm1_gdp_perCapita_1990_2022")


gdp_adm1_1990=gdp_adm1.select('PPP_1990')



# define full-world rectangle matching the image extent (band CRS indicates EPSG:4326, bounds -180,-90,180,90)
region = ee.Geometry.Rectangle([-180, -90, 180, 90])

print("Requesting sampleRectangle().getInfo() ...")
info = gdp_adm1_1990.sampleRectangle(region=region).getInfo()


print("Requesting sampleRectangle() - this may take a few seconds...")
sample_info = gdp.sampleRectangle(region=region)



gdp_1990=gdp.select('PPP_1990')
gdp_1990.getInfo()
sample_1990 = gdp_1990.sampleRectangle(region=region).getInfo()

def ee_array_to_df(arr, list_of_bands):
    """Transforms client-side ee.Image.getRegion array to pandas.DataFrame."""
    df = pd.DataFrame(arr)

    # Rearrange the header.
    headers = df.iloc[0]
    df = pd.DataFrame(df.values[1:], columns=headers)

    # Remove rows without data inside.
    df = df[['longitude', 'latitude', 'time', *list_of_bands]].dropna()

    # Convert the data to numeric values.
    for band in list_of_bands:
        df[band] = pd.to_numeric(df[band], errors='coerce')

    # Convert the time field into a datetime.
    df['datetime'] = pd.to_datetime(df['time'], unit='ms')

    # Keep the columns of interest.
    df = df[['time','datetime',  *list_of_bands]]

    return df

df = pd.DataFrame(gdp)











cuttoff_1990= (datetime.datetime(1990,1,1) - datetime.datetime(1900,1,1)).days
#find the index corresponding to this date in the netCDF time variable

time_mask=np.where(temp.variables['time'][:] >= cuttoff_1990)[0]

temp_1990=temp.variables['tmp'][time_mask,:,:].filled(np.nan)
precip_1990=precip.variables['pre'][time_mask,:,:].filled(np.nan)

#now change the time variable to year instead of days since 1900-01-01
years= np.array([1990 + int(t/12) for t in range(temp_1990.shape[0])])


time_dt = nc.num2date(temp.variables['time'][:], units=temp.variables['time'].units, calendar=temp.variables['time'].calendar)

years_for_each_t = np.array([dt.year for dt in time_dt[time_mask]])

years = np.arange(1990, 2023)   # 1990..2022
n_years = len(years)            # 33
lat_size = temp_1990.shape[1]
lon_size = temp_1990.shape[2]

#we need one dimension for temperature, precipitation and GDP

data_array = np.full((n_years, lat_size, lon_size, 2), np.nan, dtype=np.float32)

data_array.shape

for i, year in enumerate(years):
        mask = (years_for_each_t == year)
        
        data_array[i, :, :, 0] = np.nanmean(temp_1990[mask, :, :], axis=0)
        data_array[i, :, :, 1] = np.nanmean(precip_1990[mask, :, :], axis=0)
        data_array[i, :, :, 2] = np.nan
        print(f"Processed year: {year}")

        
        









#function to illustrate in a map

# def map_image(var):
   
#     vis_params = { 'min': 0, # lower bound (adjust if your GDP is in PPP per-capita) 
#                    'max': 70000,  # upper bound — adjust to taste 
#                    'palette': ['ffffe5','ffd59b','ff8c3b','d73027','7f0000'] }
#     m = geemap.Map(center=(0, 0), zoom=2)
#     m.addLayer(var, vis_params, 'GDP per capita 2010') 
#     m.addLayerControl()
#     return m

# map=map_image(gdp_adm1_1990)