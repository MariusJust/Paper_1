# this script prepares the data for the analysis
import pandas as pd
import numpy as np

def prepare(data):



    #the growth data should contain the following columns: year, county, and GrowthWDI
    growth=data[['CountryCode', 'RegionCode', 'Year', 'GrowthWDI']]

    #precipitation data
    precip=data[['CountryCode', 'RegionCode', 'Year', 'PrecipPopWeight']]

    #temperature data
    temp=data[['CountryCode', 'RegionCode', 'Year', 'TempPopWeight']]

    #Now I make dictionaries, to capture the region dependent variables
    #make dictionary to capture each region 

    #find the data that corresponds to germany in th growth data
    germany_growth=growth[growth['CountryCode']=='DEU']

    growth_dict=dict()
    precip_dict=dict()
    temp_dict=dict()

    # dictionary that holds the region code and the reference country in each region
    regions = {'Asia': [142, "CHN"], 'Europe': [150, 'DEU'], 'Africa': [2, 'ZAF'], 'Americas': [19, 'USA'], 'Oceania': [9, 'AUS']}

    dict_and_dfs = [(growth_dict, growth),
        (precip_dict, precip),
        (temp_dict, temp)]

    #Now I will loop through the regions and create a dataframe for each region
    for region, value in regions.items():
        #assign the region code and the reference country to variables
        regionCode, referenceCountry = value
        for region_dict, df in dict_and_dfs:
            #create a dictionary for each region, containing the dataframes for each country in the region
            region_data = df[df['RegionCode'] == regionCode]
            
            # Separate the reference country's data
            reference_country_data = region_data[region_data['CountryCode'] == referenceCountry]

            # Separate the other countries' data (exclude the reference country)
            other_countries_data = region_data[region_data['CountryCode'] != referenceCountry]
            
            # Concatenate the reference country data on top
            region_dict[region] = pd.concat([reference_country_data, other_countries_data])

    return growth_dict, precip_dict, temp_dict
   
        
        
        
    
   
    
