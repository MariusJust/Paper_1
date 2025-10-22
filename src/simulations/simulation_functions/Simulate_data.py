import os

import numpy as np
import tensorflow as tf
import plotly.graph_objects as go
from utils.miscelaneous import Find_data_file


def simulate(seed, n_countries, n_years, specification, add_noise, sample_data, dynamic):
    import pandas as pd
    """
    Simulate a synthetic panel dataset.

    For each country and time period:
      - Country fixed effect drawn from N(0, 0.1)
      - A cubic time trend
      - precipitation and temperature values as inputs
      - change in logGDP as the target variable
    """
    # 1. Reproducibility
    np.random.seed(seed)

    # 2. Build time & country indices
    years      = pd.date_range(start='1961', periods=n_years, freq='YE')
    countries  = [f'Country_{i}' for i in range(n_countries)]

    output = []
    
    if sample_data: 
        #if sample data is true, we use the real data from our analysis, ie we take the true values of precipitation and temperature from the dataset and use that to simulate the growth. 
        data_path = Find_data_file('MainData.xlsx')
        
        data=pd.read_excel(data_path)
        temperature=data['TempPopWeight']
        precipitation=data['PrecipPopWeight']/1000
        year=data['Year']

        country_effect = np.random.normal(0, 0.025, size=len(temperature))
        
        time_idx =   -0.00013665 *(year - 1961)
        time_idx_sq=0.00001*time_idx**2
       
    
        
        growth = calculate_growth(specification, temperature, precipitation, country_effect, time_idx, time_idx_sq, add_noise, dynamic=dynamic, year=year)

        
        final_dataset = pd.DataFrame({
            'CountryCode': data['CountryCode'],
            'Year': year,
            'delta_logGDP': growth,
            'precipitation': precipitation,
            'temperature': temperature
        })
    else:
        if dynamic: 
            temperature = np.random.uniform(0,30, size=(n_countries * n_years))
            precipitation = np.random.uniform(0.012, 5.435, size=(n_countries * n_years))
            years = np.tile(np.array([year.year for year in years]),n_countries)
            country_effect = np.random.normal(0, 0.025, size=len(temperature))
            growth = calculate_growth(specification, temperature, precipitation, country_effect, time_idx=None, time_idx_sq=None, add_noise=add_noise, dynamic=dynamic, year=years)
            final_dataset = pd.DataFrame({
                'CountryCode': np.repeat(countries, n_years),
                'Year': years,
                'delta_logGDP': growth,
                'precipitation': precipitation,
                'temperature': temperature
            })
        else:
            for country in countries:
                country_effect = np.random.normal(0, 0.025)

                for year in years:
                    # 6. Draw inputs
                    
                    time_idx =   -0.00013665 *(year.year - 1961)
                    time_idx_sq=0.00001*time_idx**2

                    # 7. Centered inputs
                    temp = np.random.uniform(0,30)
                    precip = np.random.uniform(0.012, 5.435)


                    #cast into dataframe
                    output.append({
                        'CountryCode': country,
                        'Year': year,
                        'delta_logGDP': calculate_growth(specification, temp, precip, country_effect, time_idx, time_idx_sq, add_noise, dynamic=dynamic, year=year),
                        'precipitation': precip,
                        'temperature': temp
                    })

            final_dataset=pd.DataFrame(output)     
    
    return final_dataset

def calculate_growth(specification, temp, precip, country_effect, time_idx, time_idx_sq, add_noise, dynamic, year):
     
            if dynamic: 
                time_periods=year.max()-year.min()+1
                t=year - year.min() + 1  # t goes from 1 to time_periods'
            
                if specification=='Burke':
                    true_y = (
                          (time_periods - t + 1) / time_periods * 0.0127 * temp
                        + (time_periods - t + 1) / time_periods * 0.145 * precip
                        - (t - 1) / time_periods * 0.0005 * temp**2
                        - (t - 1) / time_periods * 0.047 * precip**2
                        + country_effect
                    )
                    
                    
                elif specification=='Leirvik':
                        true_y = (
                            (time_periods - t + 1) / time_periods * 0.0127 * temp
                            + (time_periods - t + 1) / time_periods * 0.145 * precip
                            - (t - 1) / time_periods * 0.0005 * temp**2
                            - (t - 1) / time_periods * 0.047 * precip**2
                            - (time_periods - t + 1)/time_periods * 0.0125 * temp * precip
                            + (t-1)/time_periods * 0.00029 * precip * temp**2
                            + (t-1)/time_periods * 0.007 * temp * precip**2
                            - (t-1)/time_periods * 0.00013 * temp**2 * precip**2
                            + country_effect
                        )
                      

                else:
                    raise ValueError(f"Unknown specification: {specification}")
             
                    

            else:
                if specification == 'linear':
                    true_y = (
                        0.0008 * temp
                    + 0.007 * precip
                    + country_effect
                    + time_idx
                    )           

                elif specification == 'Burke':
                    true_y = (
                        0.0127 * temp
                    + 0.145 * precip
                    -0.0005 * temp**2
                    -0.047* precip**2
                    + country_effect
                    + time_idx
                    + time_idx_sq
                    )
                    
                
                elif specification == 'Leirvik':
                    true_y = (
                        0.01 * temp
                    + 0.105 * precip
                    -0.00048 * temp**2
                    -0.07* precip**2
                    -0.0125*temp*precip
                    +0.00029*precip*temp**2
                    +0.007*temp*precip**2
                    -0.00013*temp**2*precip**2
                    + country_effect
                    + time_idx
                    + time_idx_sq
                    
                    )
            
                elif specification == 'Trig':
                # periodic structure in precip and mild modulation by temp
                # If precip is in meters across 0..5, use period ~5 to get one cycle across range:
                # sin(2*pi*precip/5). If precip was standardized, change the frequency accordingly.
                    true_y = (
                        0.001     * temp
                    + 0.0025    * precip
                    + 0.02      * np.sin(2 * np.pi * precip / 5.0)           # wave across precipitation
                    + 0.01      * np.cos(2 * np.pi * temp / 15.0)            # gentle seasonal-like temp cycle
                    - 0.005     * temp * np.sin(2 * np.pi * precip / 5.0)   # interaction: wave amplitude depends on temp
                    + country_effect
                    + time_idx
                    + time_idx_sq
                    )
                    

                else:
                    raise ValueError(f"Unknown specification: {specification}")

            if add_noise:
                # 9. Add noise
                noise = np.random.normal(0, 0.001)
                y     = true_y + noise
            else:
                y = true_y
                
            return y
        


def Pivot(data):
    growth=data[['CountryCode', 'Year', 'delta_logGDP']]
    precip=data[['CountryCode', 'Year', 'precipitation']]
    temp=data[['CountryCode', 'Year', 'temperature']]

    
    #make dictionaries
    
    growth_dict={}
    precip_dict={}
    temp_dict={}

    dict_and_vars = [(growth_dict, growth),
    (precip_dict, precip),
    (temp_dict, temp)]

    
    for dict, var in dict_and_vars:

        pivot_data = var.pivot(index='Year', columns='CountryCode', values=var.columns[-1])

        mean = np.nanmean(pivot_data.values)
    
        std = np.nanstd(pivot_data.values)
       
        #we do not standardise the growth data 
        if var is growth:
            dict['global'] = pivot_data
        else:
            standardised_data = (pivot_data - mean) / std
            dict['global'] = standardised_data
    
    return growth_dict, precip_dict, temp_dict



def Surface(temp, precip, specification, dynamic, time_periods):
    """ Calculate the surface of growth based on temperature and precipitation.

    Args:
        temp (np.ndarray): 1D array of temperature values.
        precip (np.ndarray): 1D array of precipitation values.
        specification (str): Specification for the surface calculation ('linear', 'q_Leirvik', or 'interaction').
    Returns:
        np.ndarray: 1D array of growth values based on the specified surface.
    """
    if dynamic: 
        if specification=='Burke':
                  for t in range (1, time_periods+1):
                      
                    true_y= (
                          (time_periods - t + 1) / time_periods * 0.0127 * temp
                        + (time_periods - t + 1) / time_periods * 0.145 * precip
                        - (t - 1) / time_periods * 0.0005 * temp**2
                        - (t - 1) / time_periods * 0.047 * precip**2
                    
                    )
                    
                    return true_y
                    
        elif specification=='Leirvik':
                   true_y = (
                         (time_periods - t + 1) / time_periods * 0.0127 * temp
                        + (time_periods - t + 1) / time_periods * 0.145 * precip
                        - (t - 1) / time_periods * 0.0005 * temp**2
                        - (t - 1) / time_periods * 0.047 * precip**2
                        - (time_periods - t + 1)/time_periods * 0.0125 * temp * precip
                        + (t-1)/time_periods * 0.00029 * precip * temp**2
                        + (t-1)/time_periods * 0.007 * temp * precip**2
                        - (t-1)/time_periods * 0.00013 * temp**2 * precip**2
                    )
                   return true_y

        
    else: 
        if specification == 'linear':
            return 0.0008 * temp + 0.007 * precip
        
        elif specification == 'Burke':
                    return (
                        0.0127 * temp
                    + 0.145 * precip
                    -0.0005 * temp**2
                    -0.047* precip**2
                    )
                
                
        elif specification == 'Leirvik':
                    return (
                        0.01 * temp
                    + 0.105 * precip
                    -0.00048 * temp**2
                    -0.07* precip**2
                    -0.0125*temp*precip
                    +0.00029*precip*temp**2
                    +0.007*temp*precip**2
                    -0.00013*temp**2*precip**2
                    )

        elif specification == 'Trig':
            # periodic structure in precip and mild modulation by temp
            # If precip is in meters across 0..5, use period ~5 to get one cycle across range:
            # sin(2*pi*precip/5). If precip was standardized, change the frequency accordingly.
                return (
                    0.001     * temp
                + 0.0025    * precip
                + 0.02      * np.sin(2 * np.pi * precip / 5.0)           # wave across precipitation
                + 0.01      * np.cos(2 * np.pi * temp / 15.0)            # gentle seasonal-like temp cycle
                - 0.005     * temp * np.sin(2 * np.pi * precip / 5.0)   # interaction: wave amplitude depends on temp
                )

def dynamic_surface(spec, temp_grid, precip_grid, T_total):

    # --- build frames ---
    for t in range(1, T_total + 1):
        # growth formula from your function (broadcasting over the 2D grids)
        if spec=='Leirvik':
            growth = (
                (T_total - t + 1) / T_total * 0.0127 * temp_grid
                + (T_total - t + 1) / T_total * 0.145 * precip_grid
                - (T_total - t + 1)/T_total * 0.0125 * temp_grid * precip_grid
                + (t-1)/T_total * 0.00029 * precip_grid * temp_grid**2
                + (t-1)/T_total * 0.007 * temp_grid * precip_grid**2
                - (t-1)/T_total * 0.00013 * temp_grid**2 * precip_grid**2
                - (t - 1) / T_total * 0.0005 * temp_grid**2
                - (t - 1) / T_total * 0.047 * precip_grid**2
            )
        else:
            growth =(
                 (T_total - t + 1) / T_total * 0.0127 * temp_grid
                + (T_total - t + 1) / T_total * 0.145 * precip_grid
                - (t - 1) / T_total * 0.0005 * temp_grid**2
                - (t - 1) / T_total * 0.047 * precip_grid**2
            )
        z_arr[:, :, t-1] = growth
        
    

    return z_arr

                
def illustrate_synthetic_data(x,y,z):
    
    fig = go.Figure(
    data=go.Scatter3d(
    x=x,
    y=y,
    z=z,
    mode='markers',
    marker=dict(size=2, opacity=0.6)
    )
    )

    fig.update_layout(
    scene=dict(
        xaxis_title='Temperature',
        yaxis_title='precipitation',
        zaxis_title='Δ logGDP')
    )

    return fig

def illustate_surface(temp, precip, growth):
    """ Illustrate the 3d surface of growth based on temperature and precipitation.
    Args:
        temp (np.ndarray): 1D array of temperature values.
        precip (np.ndarray): 1D array of precipitation values.
        growth (np.ndarray): 1D array of growth values.
    """
    plot_data= go.Surface(
        x=temp, y=precip, z=growth.reshape(temp.shape),
        colorscale='Cividis',
        opacity=0.85,
        showscale=False,
        name='mean_surface'
        )


    fig = go.Figure(data=[plot_data])
    fig.update_layout(
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Precipitation (m)',
            zaxis_title='Δ ln(Growth)',
            camera=dict(eye=dict(x=2.11, y=0.12, z=0.38))
        ),
        legend=dict(
            bgcolor='rgba(255,255,255,0.7)',
            bordercolor='black',
            borderwidth=1
        )
    )

    fig.show()    
