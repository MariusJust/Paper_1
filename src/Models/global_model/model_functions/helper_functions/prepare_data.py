import numpy as np
import pandas as pd
from panelsplit.cross_validation import PanelSplit
from panelsplit.plot import plot_splits


def Prepare(data,  n_countries, time_periods):
       #the growth data should contain the following columns: year, county, and GrowthWDI
  

    growth=data[['CountryCode', 'RegionCode', 'Year', 'GrowthWDI']]

    #precipitation data
    precip=data[['CountryCode', 'RegionCode', 'Year', 'PrecipPopWeight']]

    #temperature data
    temp=data[['CountryCode', 'RegionCode', 'Year', 'TempPopWeight']]

    #Now I make dictionaries, to capture the region dependent variables

    growth_dict={}
    precip_dict={}
    temp_dict={}
    
    dict_and_vars = [(growth_dict, growth),
        (precip_dict, precip),
        (temp_dict, temp)]


    for dict, var in dict_and_vars:
    
        pivot_data = var.pivot(index='Year', columns='CountryCode', values=var.columns[-1]).iloc[:time_periods, :n_countries]

        mean = np.nanmean(pivot_data.values)
    
        std = np.nanstd(pivot_data.values)
        
        #we do not standardise the growth data 
        if var is growth:
            dict['global'] = pivot_data
        else:
            standardised_data = (pivot_data - mean) / std
            dict['global'] = standardised_data
    
    return growth_dict, precip_dict, temp_dict

    
def load_data(model_selection, n_countries, time_periods, n_splits=None, growth=None):
    
    if model_selection == 'IC':
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp = Prepare(data, n_countries, time_periods)
            return growth, precip, temp
        
    elif model_selection == 'CV':
        if growth is None:
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp = Prepare(data, n_countries, time_periods)
            return growth, precip, temp, panel_split
        else: #mc case
            # Create a PanelSplit object for cross-validation
            growth_global = growth['global'].reset_index()
            growth_global['Year'] = pd.to_datetime(growth_global['Year'], format='%Y')
            panel_split = PanelSplit(periods=growth_global['Year'], n_splits=n_splits, gap=0, test_size=1)
            
            return panel_split
       
    else:
        raise ValueError("Invalid model_selection argument. Use 'IC' or 'CV'.")


