import numpy as np

def Prepare(data):
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

    # dictionary that holds the region code and the reference country in each region
    # regions = {'Asia': [142, "CHN"], 'Europe': [150, 'DEU'], 'Africa': [2, 'ZAF'], 'Americas': [19, 'USA'], 'Oceania': [9, 'AUS']}

    # dict_and_dfs = [(growth_dict, growth),
    #     (precip_dict, precip),
    #     (temp_dict, temp)]

    # #Now I will loop through the regions and create a dataframe for each region
    # for region, value in regions.items():
    #     #assign the region code and the reference country to variables
    #     regionCode, referenceCountry = value
    #     for region_dict, df in dict_and_dfs:
    #         #get the region specific data
    #         region_data = df[df['RegionCode'] == regionCode]
            
    #         # Pivot the data so that the years are the index and the countries are the columns
    #         pivot_data = region_data.pivot(index='Year', columns='CountryCode', values=region_data.columns[-1])
            
            
    #         #Reorder the columns so that the reference country is the first column
    #         cols = pivot_data.columns.tolist()  # Get current list of columns
    #         cols.insert(0, cols.pop(cols.index(referenceCountry)))  # Move reference country to the first column
    #         pivot_data = pivot_data[cols]  # Reorder columns in the dataframe
            
    #         #standardising the data
    #         mean = np.nanmean(pivot_data.values)
    #         std = np.nanstd(pivot_data.values)
    #         pivot_data = (pivot_data - mean) / std
            
    #         # Concatenate the reference country data on top
    #         region_dict[region] = pivot_data


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
    

