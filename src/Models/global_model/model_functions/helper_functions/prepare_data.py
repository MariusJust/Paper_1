import numpy as np
import pandas as pd
# from panelsplit.cross_validation import PanelSplit
# from panelsplit.plot import plot_splits


def Prepare(data, data_source='WDI'):
       #the growth data should contain the following columns: year, county, and GrowthWDI
  
    time_periods = len(data['Year'].unique())-1
    
    if data_source == 'WDI':
        growth=data[['CountryCode', 'RegionCode', 'Year', 'GrowthWDI']]

        #precipitation data
        precip=data[['CountryCode', 'RegionCode', 'Year', 'PrecipPopWeight']]

        #temperature data
        temp=data[['CountryCode', 'RegionCode', 'Year', 'TempPopWeight']]
        n_countries = len(data['ISO'].unique())
        unit='CountryCode'
        
    elif data_source == 'ee':
        #fid is the unique identifier for each region, iso3 is the country code, and the last column is the growth variable
        growth=data[['fid', 'iso3', 'Year',  data.columns[-1]]]

        #precipitation data
        precip=data[['fid', 'iso3', 'Year',  data.columns[5]]]

        #temperature data
        temp=data[['fid', 'iso3', 'Year',  data.columns[4]]]
        
        n_countries = len(data['fid'].unique())
        unit='fid'
    else:
        raise ValueError("Invalid data_source argument. Use 'WDI' or 'ee'.")

    #Now I make dictionaries, to capture the region dependent variables

    growth_dict={}
    precip_dict={}
    temp_dict={}
    
    dict_and_vars = [(growth_dict, growth),
        (precip_dict, precip),
        (temp_dict, temp)]

    stats={}
    for dict, var in dict_and_vars:

        pivot_data = var.pivot(index='Year', columns=unit, values=var.columns[-1]).iloc[:time_periods, :n_countries]

        mean = np.nanmean(pivot_data.values)
    
        std = np.nanstd(pivot_data.values)
        
        if var is growth:
            dict['global'] = pivot_data
        else:
            standardised_data = (pivot_data - mean) / std
            
            if var is precip:
                stats[f"mean_precip"] = mean
                stats[f"std_precip"] = std
            else:
                stats[f"mean_temp"] = mean
                stats[f"std_temp"] = std
                
                
            dict['global'] = standardised_data
    
    return growth_dict, precip_dict, temp_dict, stats

    
def load_data(model_selection, datasource="", n_splits=None, growth=None):
    
    if model_selection == 'IC':
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp, stats = Prepare(data, data_source=datasource)
            return growth, precip, temp, stats
        
    elif model_selection == 'CV':
        if growth is None:
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp, stats = Prepare(data, data_source=datasource)
            panel_split = PanelSplit(periods=growth['global'].index, n_splits=n_splits, gap=0, test_size=1)

            return growth, precip, temp, stats, panel_split
        else: #mc case
            
            # Create a PanelSplit object for cross-validation
            growth_global = growth['global'].reset_index()
            growth_global['Year'] = pd.to_datetime(growth_global['Year'], format='%Y')

            panel_split = PanelSplit(periods=growth_global['Year'], n_splits=n_splits, gap=0, test_size=1)
            
            return panel_split
       
    else:
        raise ValueError("Invalid model_selection argument. Use 'IC' or 'CV'.")


