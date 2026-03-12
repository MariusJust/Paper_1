import numpy as np
import pandas as pd
# from panelsplit.cross_validation import PanelSplit



def Prepare(data, data_source=None):
       #the growth data should contain the following columns: year, county, and GrowthWDI
    
    time_periods = len(data['Year'].unique())

    if data_source.lower()=='wb':
            growth=data[['CountryCode', 'RegionCode', 'Year', 'GrowthWDI']]

            #precipitation data
            precip=data[['CountryCode', 'RegionCode', 'Year', 'PrecipPopWeight']]

            #temperature data
            temp=data[['CountryCode', 'RegionCode', 'Year', 'TempPopWeight']]
            
    elif data_source.lower()=='ee':
            growth=data[['iso3', 'fid', 'RegionCode', 'Year', 'growth (gdp per capita)']].rename(columns={'iso3':'CountryCode', 'growth (gdp per capita)':'GrowthWDI'})

            #precipitation data
            precip=data[['iso3', 'fid', 'RegionCode', 'Year', 'precipitation (mm)']].rename(columns={'iso3':'CountryCode', 'precipitation (mm)':'PrecipPopWeight'})

            #temperature data
            temp=data[['iso3', 'fid', 'RegionCode', 'Year', 'temperature (celsius)']].rename(columns={'iso3':'CountryCode','Year':'Year', 'temperature (celsius)':'TempPopWeight'})
            
    else:
            raise ValueError("data_source must be either 'WB' or 'ee'")
        
        
    growth_dict={}
    precip_dict={}
    temp_dict={}
    
    dict_and_vars = [(growth_dict, growth),
        (precip_dict, precip),
        (temp_dict, temp)]

    
    # dictionary that holds the region code and the reference country in each region
    
    if data_source.lower()=='wb':
        regions = {'Asia': [142, "CHN"], 'Europe': [150, 'DEU'], 'Africa': [2, 'ZAF'], 'Americas': [19, 'USA'], 'Oceania': [9, 'AUS']}
    else:
        # the reference regions are Asia-Beijing=408, Europe-Stockholm=2364, Africa-Northern Cape Town=2854,USA-New York City=2709, Oceania-Victoria(Melbourne)=AUS
        regions = {'Asia': [142, 408], 'Europe': [150, 2364], 'Africa': [2, 2854], 'Americas': [19, 2709], 'Oceania': [9, 196]}
    
    #Now I will loop through the regions and create a dataframe for each region
    for region, value in regions.items():
        #assign the region code and the reference country to variables
        regionCode, referenceCountry = value
        for dict, var in dict_and_vars:
            #get the region specific data
            region_data = var[var['RegionCode'] == regionCode]
            
            
            # Pivot the data so that the years are the index and the countries are the columns
            if data_source == 'ee':
                pivot_data = region_data.pivot(index='Year', columns='fid', values=region_data.columns[-1]).iloc[1:time_periods, :]
            else:
                pivot_data = region_data.pivot(index='Year', columns='CountryCode', values=region_data.columns[-1])
            
            #Reorder the columns so that the reference country is the first column
            cols = pivot_data.columns.tolist()  
            cols.insert(0, cols.pop(cols.index(referenceCountry)))  
            pivot_data = pivot_data[cols] 
            
            # we do not standardise the growth data
            if var is growth:
                dict[region] = pivot_data
            else:
                #standardise the data
                mean = np.nanmean(pivot_data.values)
                std = np.nanstd(pivot_data.values)
                pivot_data = (pivot_data - mean) / std
                dict[region] = pivot_data

    return growth_dict, precip_dict, temp_dict
    
    
    

    
def load_data(model_selection, n_splits=None, growth=None, data_source=None):
    
    if model_selection == 'IC':
        if data_source.lower()=='wb':
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp = Prepare(data, data_source=data_source)
            return growth, precip, temp
        elif data_source.lower()=='ee':
            data = pd.read_csv("data/ee_data.csv", sep=";")
            #delete row if growthWDI is missing 
            growth, precip, temp= Prepare(data, data_source=data_source)
            return growth, precip, temp
        
    elif model_selection == 'CV':
        if growth is None:
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp = Prepare(data)
            panel_split = PanelSplit(periods=growth['global'].index, n_splits=n_splits, gap=0, test_size=1)

            return growth, precip, temp, panel_split
        else: #mc case
            
            # Create a PanelSplit object for cross-validation
            growth_global = growth['global'].reset_index()
            growth_global['Year'] = pd.to_datetime(growth_global['Year'], format='%Y')

            panel_split = PanelSplit(periods=growth_global['Year'], n_splits=n_splits, gap=0, test_size=1)
            
            return panel_split
       
    else:
        raise ValueError("Invalid model_selection argument. Use 'IC' or 'CV'.")


