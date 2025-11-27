import numpy as np
import pandas as pd
from panelsplit.cross_validation import PanelSplit
from panelsplit.plot import plot_splits


def Prepare(data, data_source='WB'):
       #the growth data should contain the following columns: year, county, and GrowthWDI
  

    if data_source=='WB':
        growth=data[['CountryCode', 'RegionCode', 'Year', 'GrowthWDI']]

        #precipitation data
        precip=data[['CountryCode', 'RegionCode', 'Year', 'PrecipPopWeight']]

        #temperature data
        temp=data[['CountryCode', 'RegionCode', 'Year', 'TempPopWeight']]
        n_countries = len(growth['CountryCode'].unique())
        time_periods = len(growth['Year'].unique())
        
    elif data_source=='ee':
        growth=data[['iso3', 'fid', 'year', 'growth']].rename(columns={'iso3':'CountryCode', 'year':'Year', 'growth':'GrowthWDI'})

        #precipitation data
        precip=data[['iso3', 'fid', 'year', 'precipitation']].rename(columns={'iso3':'CountryCode', 'year':'Year', 'precipitation':'PrecipPopWeight'})

        #temperature data
        temp=data[['iso3', 'fid', 'year', 'temperature']].rename(columns={'iso3':'CountryCode','year':'Year', 'temperature':'TempPopWeight'})
        
        #countries here are really regions identified by fid but we keep the same naming convention
        n_countries = len(growth['fid'].unique())
        time_periods = len(growth['Year'].unique())
        
    else:
        raise ValueError("data_source must be either 'WB' or 'ee'")
    #Now I make dictionaries, to capture the region dependent variables

    growth_dict={}
    precip_dict={}
    temp_dict={}
    
    dict_and_vars = [(growth_dict, growth),
        (precip_dict, precip),
        (temp_dict, temp)]


    for dict, var in dict_and_vars:
        
        if data_source=='WB':
                pivot_data = var.pivot(index='Year', columns='CountryCode', values=var.columns[-1]).iloc[:time_periods, :n_countries]
        elif data_source=='ee':
            #we now pivot based on fid and drop the year 1990 as we don't have growth data for that year
                pivot_data = var.pivot(index='Year', columns='fid', values=var.columns[-1]).iloc[1:time_periods, :n_countries]
                
        mean = np.nanmean(pivot_data.values)
    
        std = np.nanstd(pivot_data.values)
        
#we do not standardise the growth data 
        if var is growth:
            dict['global'] = pivot_data
        else:
            standardised_data = (pivot_data - mean) / std
            dict['global'] = standardised_data
    
    return growth_dict, precip_dict, temp_dict, n_countries, time_periods

def load_data(model_selection, data_source, n_splits=None, growth=None):
    
    if model_selection == 'IC':
        if data_source.lower()=='wb':
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp, n_countries, time_periods = Prepare(data, data_source=data_source)
            return growth, precip, temp, n_countries, time_periods
        elif data_source.lower()=='ee':
            data = pd.read_csv("data/ee_data.csv")
            growth, precip, temp, n_countries, time_periods = Prepare(data, data_source=data_source)
            return growth, precip, temp, n_countries, time_periods
        
    elif model_selection == 'CV':
        if growth is None:
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp, n_countries, time_periods = Prepare(data, data_source=data_source)
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


