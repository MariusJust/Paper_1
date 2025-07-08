import os
import pandas as pd
import numpy as np
import tensorflow as tf
import plotly.graph_objects as go


def simulate(seed, n_countries, n_years, specification, add_noise):
       
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

    data = []
    for country in countries:
        country_effect = np.random.normal(0, 0.025)

        for year in years:
            # 6. Draw inputs
            
            time_idx =   -0.00013665 *(year.year - 1961)
            time_idx_sq=0.00001*time_idx**2

            # 7. Centered inputs
            temp = np.random.uniform(0,30)
            precip = np.random.uniform(0.012, 5.435)

            # 8. Compute true_y by specification
            if specification == 'linear':
                true_y = (
                    0.0008 * temp
                  + 0.007 * precip
                  + country_effect
                  + time_idx
                )         

            elif specification == 'q_Leirvik':
                true_y = (
                    0.01 * temp
                  + 0.105 * precip
                  -0.00048 * temp**2
                  -0.065* precip**2
                  -0.0125*temp*precip
                  +0.00029*precip*temp**2
                  +0.006*temp*precip**2
                  -0.00013*temp**2*precip**2
                  + country_effect
                  + time_idx
                  + time_idx_sq
                )
                
            
            elif specification == 'interaction':
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
                
            else:
                raise ValueError(f"Unknown specification: {specification}")

            if add_noise:
                # 9. Add noise
                noise = np.random.normal(0, 0.001)
                y     = true_y + noise
            else:
                y = true_y

            #cast into dataframe
            data.append({
                'CountryCode': country,
                'Year': year,
                'delta_logGDP': y,
                'precipitation': precip,
                'temperature': temp
            })
    
    data=pd.DataFrame(data)     
 
    return data
  
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
          

def surface(temp, precip, specification):
    """ Calculate the surface of growth based on temperature and precipitation.

    Args:
        temp (np.ndarray): 1D array of temperature values.
        precip (np.ndarray): 1D array of precipitation values.
        specification (str): Specification for the surface calculation ('linear', 'q_Leirvik', or 'interaction').
    Returns:
        np.ndarray: 1D array of growth values based on the specified surface.
    """
    
    if specification == 'linear':
        return 0.0008 * temp + 0.007 * precip
    
    elif specification == 'q_Leirvik':
                return (
                    0.01 * temp
                  + 0.105 * precip
                  -0.00048 * temp**2
                  -0.065* precip**2
                  -0.0125*temp*precip
                  +0.00029*precip*temp**2
                  +0.006*temp*precip**2
                  -0.00013*temp**2*precip**2
              
                )
                
            
    elif specification == 'interaction':
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

    fig.show()
    return None

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
