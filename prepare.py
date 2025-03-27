import numpy as np
import pandas as pd

def Prepare(data):
       #the growth data should contain the following columns: year, county, and GrowthWDI
    
    data=pd.read_excel('data/MainData.xlsx')


    growth=data[['CountryCode', 'RegionCode', 'Year', 'GrowthWDI']]

    #precipitation data
    precip=data[['CountryCode', 'RegionCode', 'Year', 'PrecipPopWeight']]

    #temperature data
    temp=data[['CountryCode', 'RegionCode', 'Year', 'TempPopWeight']]

    #Now I make dictionaries, to capture the region dependent variables

    growth_dict={}
    precip_dict={}
    temp_dict={}

 
    dict_and_dfs = [
                     (precip_dict, precip),
                        (temp_dict, temp)]
        # return the global data for the growth, precipitation, and temperature
    for dict, var in dict_and_dfs:
       
        pivot_data = var.pivot(index='Year', columns='CountryCode', values=var.columns[-1])
        mean = np.nanmean(pivot_data.values)       
        std = np.nanstd(pivot_data.values)
        standardised_data = (pivot_data - mean) / std
        dict['global'] = standardised_data
        # dict['global'] = pivot_data
        
    growth_dict['global'] = growth.pivot(index='Year', columns='CountryCode', values=growth.columns[-1])

    return growth_dict, precip_dict, temp_dict
    

