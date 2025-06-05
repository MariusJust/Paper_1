import numpy as np
import pandas as pd
from panelsplit.cross_validation import PanelSplit
from panelsplit.plot import plot_splits


def Prepare(data, formulation, n_countries, time_periods):
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

    if formulation == 'regional':

        # dictionary that holds the region code and the reference country in each region
        regions = {'Asia': [142, "CHN"], 'Europe': [150, 'DEU'], 'Africa': [2, 'ZAF'], 'Americas': [19, 'USA'], 'Oceania': [9, 'AUS']}

        
        #Now I will loop through the regions and create a dataframe for each region
        for region, value in regions.items():
            #assign the region code and the reference country to variables
            regionCode, referenceCountry = value
            for dict, var in dict_and_vars:
                #get the region specific data
                region_data = var[var['RegionCode'] == regionCode]
                
                # Pivot the data so that the years are the index and the countries are the columns
                pivot_data = region_data.pivot(index='Year', columns='CountryCode', values=region_data.columns[-1])
                
                #Reorder the columns so that the reference country is the first column
                cols = pivot_data.columns.tolist()  # Get current list of columns
                cols.insert(0, cols.pop(cols.index(referenceCountry)))  # Move reference country to the first column
                pivot_data = pivot_data[cols]  # Reorder columns in the dataframe
                
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
                
    elif formulation == 'global':
       
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
    else:
        raise ValueError("Invalid formulation. Use 'regional' or 'global'.")

    
def load_data(model_selection, n_splits, formulation, n_countries, time_periods):
    if model_selection == 'IC':
            data = pd.read_excel('data/MainData.xlsx')
            growth, precip, temp = Prepare(data, formulation, n_countries, time_periods)
            return growth, precip, temp
    elif model_selection == 'CV':
        data = pd.read_excel('data/MainData.xlsx')
        growth, precip, temp = Prepare(data, formulation, n_countries, time_periods)
        
        # Create a PanelSplit object for cross-validation
        growth_global = growth['global'].reset_index()
        growth_global['Year'] = pd.to_datetime(growth_global['Year'], format='%Y')
        panel_split = PanelSplit(periods=growth_global['Year'], n_splits=n_splits, gap=0, test_size=1)
        
        return growth, precip, temp, panel_split
    else:
        raise ValueError("Invalid model_selection argument. Use 'IC' or 'CV'.")


