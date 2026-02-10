import ee, requests
import numpy as np
import geemap
import netCDF4 as nc
import ipywidgets as widgets
from ipyleaflet import WidgetControl
import pandas as pd
from shapely.geometry import Point, Polygon, MultiPolygon, shape, mapping, box, GeometryCollection, LineString, MultiLineString
from shapely.prepared import prep
from rtree import index
from tqdm import tqdm   # optional, nice progress bar
from datetime import datetime
from io import BytesIO
from PIL import Image
import matplotlib.pyplot as plt
import matplotlib.patheffects as path_effects
import img2pdf
from pathlib import Path
import re
import folium
import geopandas as gpd


#######################################################################################################################
#                           This script loads in gridded temperature, precipitation and GDP data                      # 
#                           Temporal resolution: 1990:2022                                                            #       
#                           Spatial Resolution: [-180, 180] lon, [-90, 90] lat, 0.5 degree grid                       #
#                           Data Source: CRU TS4.08 (temperature and precipitation), Google Earth Engine (GDP)        #
#######################################################################################################################       

#######################################################################################################################
#                                         Earth engine setup + raw data loading                                       # 
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
gdp_adm0_polygon=ee.FeatureCollection("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/poly_adm0_gdp_perCapita_1990_2022")
gdp_adm1=ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/adm1_gdp_perCapita_1990_2022")
gdp_adm0=ee.Image("projects/sat-io/open-datasets/GRIDDED_HDI_GDP/adm0_gdp_perCapita_1990_2022")
temp= nc.Dataset("RawData/cru_ts4.08.1901.2023.tmp.dat.nc")
precip= nc.Dataset("RawData/cru_ts4.08.1901.2023.pre.dat.nc")
density= nc.Dataset("RawData/gpw_v4_population_density_rev11_30_min.nc")
stations="RawData/tmp_stations.dtb"

raster=density.variables['raster']

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


##################################################### Create map ######################################################

def _palette_to_colormap(palette):
    """Convert list of hex colors to matplotlib colormap."""
    # palette may be comma-separated string or list
    if isinstance(palette, str):
        colors = palette.split(',')
    else:
        colors = list(palette)
    # convert hex -> RGB normalized
    rgb = [tuple(int(h.lstrip('#')[i:i+2], 16)/255. for i in (0, 2, 4)) for h in colors]
    cmap = plt.cm.colors.ListedColormap(rgb)
    return cmap

def save_ee_map(
    ee_image,
    palette,
    out_png='map_with_legend.png',
    out_pdf=None,
    region=None,
    width=3000,
    legend_title='GDP per Capita (2015 USD)',
    legend_fontsize=20,
    ticks=None,
    left_label=None,
    right_label=None
):
    """
    Create downloadable PNG (and optional PDF) with map image and a neat legend.

    - ee_image: ee.Image (should be an RGB visualize() result or a raw image plus vis_params)
    - vis_params: dict with 'min','max','palette' (palette = list of hex or comma-separated)
    - region: Earth Engine geometry or bbox [xmin,ymin,xmax,ymax]; defaults to world
    - width: target image width in pixels (thumbnail width)
    - out_png: output path for combined PNG
    - out_pdf: if given, will also save a PDF at this path
    - ticks: list of tick values to show under colorbar
    - left_label/right_label: strings for small end labels (e.g. '$1K', '$80K+')
    """
    
    # calculate growth between 1990 and 2022

    
    # Earth Engine expects comma-separated palette for thumbnail params
    viz_img = ee_image.visualize(
        bands=['PPP_1990'], 
        min=1000,
        max=80000,
        palette=palette
    )
        
    # Prepare region and thumbnail parameters
    if region is None:
        region = ee.Geometry.Rectangle([-180, -90, 180, 90])
    elif isinstance(region, (list, tuple)) and len(region) == 4:
        region = ee.Geometry.Rectangle(region)

    thumb_params = {
        'region': region.getInfo()['coordinates'] if isinstance(region, ee.Geometry) else region,
        'dimensions': width,
        'format': 'png',
    }

    # request a thumbnail url
    thumb_url = viz_img.getThumbURL(thumb_params)

    # download the thumbnail PNG
    resp = requests.get(thumb_url, stream=True)
    resp.raise_for_status()
    img = Image.open(BytesIO(resp.content)).convert('RGBA')

    # Create a legend image with matplotlib
    cmap = _palette_to_colormap(palette)
    vmin = 1000
    vmax = 80000

    ####################################### MAke legend and combine with map #########################################
   
    # Legend size parameters (keep your values)
    legend_width_px = int(img.width * 0.15)
    legend_height_px = max(80, int(img.height * 0.05))

    # Create figure - slightly taller to accomodate larger labels/title
    fig, ax = plt.subplots(
        figsize=(legend_width_px / 100, legend_height_px / 100 + 0.12),
        dpi=200,  # increase dpi for crisper text
        facecolor='white'  # change to desired legend background
    )
    fig.subplots_adjust(left=0.05, right=0.95, top=0.78, bottom=0.18)

    # Create gradient image for legend and plot with extent so ticks map 0..1
    nb = 256
    gradient = np.linspace(0, 1, nb).reshape(1, nb)
    im = ax.imshow(gradient, aspect='auto', cmap=cmap, extent=[0, 1, 0, 1])
    ax.set_yticks([])

    # Tick values and labels
    if ticks is None:
        ticks = [vmin, (vmin + vmax) / 2, vmax]
    tick_labs = [f'{round(int(lab)/1000)}K' for lab in ticks]

    
    ax.set_xticks([0.0, 0.25, 1.0])
    
    
    txts = ax.set_xticklabels(tick_labs, fontsize=12, fontweight='bold', color='black')
    for t in txts:
        t.set_path_effects([path_effects.withStroke(linewidth=2, foreground='white')])

    # Add small tick marks (optional)
    ax.tick_params(axis='x', which='major', length=6, pad=6)

    # Title: larger, bold, and with outline for readability over any background
    ax.set_title(legend_title, fontsize=legend_fontsize, fontweight='bold', pad=8, color='black')
  
    # Save legend to buffer and preserve facecolor
    buf = BytesIO()
    plt.savefig(buf, format='png', bbox_inches='tight', dpi=200, facecolor=fig.get_facecolor())
    plt.close(fig)
    buf.seek(0)
    legend_img = Image.open(buf).convert('RGBA')

    # Resize legend to desired width
    legend_img = legend_img.resize(
        (legend_width_px, int(legend_img.height * (legend_width_px / legend_img.width))),
        Image.LANCZOS
    )

    # Compose final image: place legend in the bottom left corner
    padding = -2800
    total_w = img.width 
    total_h = img.height # some top space
    combined = Image.new('RGBA', (total_w, total_h), (255,255,255,255))

    # Paste map
    combined.paste(img, (0, 0))

    # Paste legend (centered vertically)
    legend_y = int((combined.height - legend_img.height)/1.5)
    combined.paste(legend_img, (img.width + padding, legend_y), legend_img)
  
    # Save combined PNG
    combined.save(out_png, dpi=(300,300))
    print(f"Saved combined PNG: {out_png}")

    # Optionally save PDF
    if out_pdf:
       with open("gdp1990_with_legend.png", "rb") as f_png, open("gdp1990_with_legend.pdf", "wb") as f_pdf:
            f_pdf.write(img2pdf.convert(f_png))
    return out_png, out_pdf


save_ee_map(
    ee_image=gdp_adm0.select('PPP_1990'),
    out_png='gdp1990_with_legend.png',
    out_pdf='gdp1990_with_legend.pdf',
    region=[-180, -90, 180, 90],
    width=3000,
    legend_title='GDP per Capita 1990 (PPP)',
    ticks=[1000, 20000, 80000],
    left_label='$1K',
    right_label='$80K+',
    legend_fontsize=20,
    palette=palettes['ornblue'],
)

# # 1) Derive available years from band names (server-side)
# bands = gdp_adm0.bandNames()
# first_year = ee.Number.parse(ee.String(bands.get(0)).split('_').get(1))
# last_year = ee.Number.parse(ee.String(bands.get(-1)).split('_').get(1))

# # sequence of valid growth years: first_year+1 .. last_year
# years = ee.List.sequence(first_year.add(1), last_year)

# # 2) Growth image maker: returns ONE-BAND float image named "growth"
# def growth_image(y):
#     y = ee.Number(y).toInt()
#     band_t = ee.String('PPP_').cat(y.format())
#     band_tm1 = ee.String('PPP_').cat(y.subtract(1).format())
#     g_t = gdp_adm0.select(band_t).toFloat()
#     g_tm1 = gdp_adm0.select(band_tm1).toFloat()
#     # avoid division-by-zero: mask pixels where previous year is zero or masked
#     valid = g_tm1.neq(0).And(g_tm1.mask().reduce(ee.Reducer.min()))  # ensures not masked and not zero
#     growth = g_t.subtract(g_tm1).divide(g_tm1).multiply(100).updateMask(valid)
#     return growth.rename('growth')

# # Map years -> list of single-band images (server-side)
# growth_images_list = years.map(lambda y: growth_image(y))

# # Build a homogeneous ImageCollection (each image has band 'growth')
# growth_ic = ee.ImageCollection(growth_images_list)

# # 3) Compute per-pixel average growth across years (single-band image)
# avg_growth_img = growth_ic.mean().rename('avg_growth')  # single band

# # 4) Aggregate to countries (FeatureCollection of country means)
# average_growth_fc = avg_growth_img.reduceRegions(
#     collection=gdp_adm0_polygon,
#     reducer=ee.Reducer.mean(),
#     scale=10000,           # adjust scale as needed
#     crs='EPSG:4326'        # optional: set CRS
# )

# # average_growth_fc now has a property 'mean' containing the country average.
# # Convert the FC back to a raster (one band) so you can add it as a choropleth
# average_growth_full_sample = average_growth_fc.reduceToImage(
#     properties=['mean'],
#     reducer=ee.Reducer.first()
# ).rename('avg_growth')

# # 5) Display on map
# m = geemap.Map(center=[20, 0], zoom=2)

# vis = {
#     'min': -0.5,   # choose sensible range for your data
#     'max': 0.5,
#     'palette': ['darkred', 'white', 'darkgreen']
# }

# m.addLayer(country_raster, vis, 'Avg yearly GDP growth (%)')
# # show country outlines
# m.addLayer(gdp_adm0_polygon.style(**{'fillColor': '00000000', 'color': '000000', 'width': 1}),
#            {}, 'Country borders')
# m



##################################################### Create gif ######################################################

# Get band names
band_names = gdp_adm0.bandNames().getInfo()  
# Create an ImageCollection from bands
frames = []
for band in band_names:
    gdp_vis = gdp_adm0.select(band).visualize(min=0, max=100000, palette=palettes['ornblue'])
    frames.append(gdp_vis)

Movie = ee.ImageCollection(frames)

# Define arguments for animation function parameters.
videoArgs = {
    "dimensions": 768,
    "region": get_region_geometries().get('europe'),
    "framesPerSecond": 10,
    "crs": "EPSG:3857",
    
}

# Add time text to the GIF
text = [year for year in range(1990, 2023)]

gif_path = "europe_adm0.gif"
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
# full_data.to_csv("ee_data.csv", index=False)



##################################################### Visualise grid points ######################################################

def parse_stations_file(path_or_text):
    """
    Parse a CRU .clean.dtb observation file into a pandas.DataFrame of stations with metadata.
    Returns DataFrame with columns: ['wmo_packed','lat','lon','alt','name','country','start','end'].
    """

    with open(path_or_text, 'r', encoding='latin1') as f:
            lines = f.readlines()


    rows = []
    i = 0
    n = len(lines)
    while i < n:
        header = lines[i].rstrip("\n")
        i += 1
        #check if the header contains a countryname or a combination of integers
        if header.strip() == "" or not header[26:46].strip() or not header[26:46].strip()[0].isalpha(): 
            continue
        # Header format (from CRU doc):
        # (i7,1x,i5,1x,i6,1x,i4,1x,a20,1x,a13,2(1x,i4))
        # Example given in docs:
        # 0305900 5750 -420 4 INVERNESS UK 1781 1994 ...
        # We'll parse by fixed-width positions to be robust.
        try:
            wmo = header[0:7].strip()                    # world meterol
            lat100 = int(header[8:13].strip())           # i5  lat*100
            lon100 = int(header[14:20].strip())          # i6  lon*100  (note sign is included)
            alt = int(header[21:25].strip())             # i4
            name = header[26:46].strip()                 # a20
            country = header[47:60].strip()              # a13
            start = int(header[61:65].strip())           # i4
            end = int(header[66:70].strip())             # i4
        except Exception:
            # fallback: whitespace split (less reliable if name contains spaces)
            parts = re.split(r'\s+', header.strip())
            # try to map
            if len(parts) >= 8:
                wmo = parts[0]
                lat100 = int(parts[1])
                lon100 = int(parts[2])
                alt = int(parts[3])
                # name+country might be merged; best effort:
                name = parts[4]
                country = parts[5] if len(parts) > 5 else ""
                start = int(parts[-2])
                end = int(parts[-1])
            else:
                # if unrecoverable, skip
                continue

        # normals line (always present) -> skip
        if i >= n:
            break
        i += 1

        # number of years of data = end - start + 1
        n_years = end - start + 1
        # skip n_years lines of monthly data (they are compact - no spaces)
        for _ in range(n_years):
            if i < n:
                _ = lines[i]
                i += 1
            else:
                break

        # convert lat/lon from hundredths of degrees to decimal degrees
        lat = lat100 / 100.0
        lon = lon100 / 100.0
        rows.append({
            "wmo_packed": wmo,
            "lat": lat,
            "lon": lon,
            "alt_m": alt,
            "name": name,
            "country": country,
            "start": start,
            "end": end
        })

    df = pd.DataFrame(rows)
    return df


stations_df = parse_stations_file(stations)

#get stations that were active in the year 1990 


stations_1990 = stations_df[(stations_df['start'] <= 1990) & (stations_df['end'] >= 1990) ].reset_index(drop=True)   



attr = (
    'Tiles &copy; Esri &mdash; Esri, DeLorme, NAVTEQ'
)
tiles = "https://server.arcgisonline.com/ArcGIS/rest/services/Canvas/World_Light_Gray_Base/MapServer/tile/{z}/{y}/{x}"

Map = folium.Map(
    zoom_start=5,
    location=[10,0],
    tiles=tiles,
    attr=attr
    
)

# Add each point as an individual CircleMarker (no clustering)
for _, row in stations_1990.iterrows():
    # folium uses (lat, lon) order
    folium.CircleMarker(
        location=(row['lat'], row['lon']),
        radius=1,               # small dot; tweak to taste
        weight=0,               # no outline
        fill=True,
        fill_opacity=0.3,
        fill_color='blue',      # change color or compute from row
        min_zoom = 3
    ).add_to(Map)

    
Map.save("stations_1990.html")


    path = os.path.abspath('stations_1990.html')
    converter.convert(f'file:///{path}', 'sample.pdf')
    

# # --- create mask: world box minus Austria geometry ---
# url = "https://naciscdn.org/naturalearth/110m/cultural/ne_110m_admin_0_countries.zip"

# world = gpd.read_file(url)
# austria = world[world['SOVEREIGNT'] == 'Austria'].to_crs(epsg=4326)
# austria_geom = austria.geometry.iloc[0] # union (handles MultiPolygons)

# add station markers in 1990
Map = geemap.Map(center=(0, 0), zoom=2)
Map.add_basemap("SATELLITE")
Map.add_points_from_xy(stations_1990, x="lon", y="lat", marker_type="marker", popup_property="name")

html_file = ( "world_stations.html")

Map.to_html(filename=html_file, title="My Map", width="100%", height="880px")


# big world box (lon/lat)
world_box = box(-180, -90, 180, 90)


# mask = everything except Austria
mask = world_box.difference(austria_geom)

# convert to GeoJSON-like mapping
mask_geo = {"type": "Feature", "properties": {}, "geometry": mapping(mask)}

# add the mask *now* so it lies above tiles but we will add grid/markers after (so they are visible)
folium.GeoJson(
    mask_geo,
    name="mask_outside_austria",
    style_function=lambda feat: {
        "fillColor": "white",
        "color": "white",
        "fillOpacity": 1.0,
        "weight": 0
    }
).add_to(m)


# #adding 0.5*0.5 degree grid cells for austria

# INTERVAL = 0.5  # 1 km to degrees, gives ~0.009

# LINE_STYLE = {"color": "black", "weight": 0.5, "opacity": 0.5}

# minx, miny, maxx, maxy = austria_geom.bounds

# #only draw gridlines within austria


# def add_clipped_gridlines(m, polygon, interval=INTERVAL, line_style=LINE_STYLE):
#     # Create vertical (constant lon) gridlines and clip to polygon
#     for lon in np.arange(minx - 1e-9, maxx + interval, interval):
#         candidate = LineString([(lon, miny - 0.1), (lon, maxy + 0.1)])  # slightly larger extents
#         inter = polygon.intersection(candidate)
#         if inter.is_empty:
#             continue
#         # Handle possible geometry types returned by intersection
#         geoms = []
#         if isinstance(inter, (LineString, MultiLineString)):
#             if isinstance(inter, LineString):
#                 geoms = [inter]
#             else:
#                 geoms = list(inter.geoms)
#         elif isinstance(inter, GeometryCollection):
#             geoms = [g for g in inter.geoms if isinstance(g, LineString)]
#         # add each clipped segment to folium (remember folium expects [lat, lon])
#         for seg in geoms:
#             coords = [(lat, lon) for lon, lat in seg.coords]  # flip order
#             folium.PolyLine(coords, **line_style).add_to(m)

#     # Create horizontal (constant lat) gridlines and clip to polygon
#     for lat in np.arange(miny - 1e-9, maxy + interval, interval):
#         candidate = LineString([(minx - 0.1, lat), (maxx + 0.1, lat)])
#         inter = polygon.intersection(candidate)
#         if inter.is_empty:
#             continue
#         geoms = []
#         if isinstance(inter, (LineString, MultiLineString)):
#             if isinstance(inter, LineString):
#                 geoms = [inter]
#             else:
#                 geoms = list(inter.geoms)
#         elif isinstance(inter, GeometryCollection):
#             geoms = [g for g in inter.geoms if isinstance(g, LineString)]
#         for seg in geoms:
#             coords = [(lat, lon) for lon, lat in seg.coords]
#             folium.PolyLine(coords, **line_style).add_to(m)

# #compute min and max for austria


#     for _, row in stations_1990.iterrows():
#         folium.Marker(
#             location=[row['lat'], row['lon']],
#             icon=folium.Icon(icon='cloud')).add_to(m)
        
# add_clipped_gridlines(m, austria_geom, interval=0.5, line_style=LINE_STYLE)

# m.save("austria_stations_1990.html")

