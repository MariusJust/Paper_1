

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import plotly.graph_objects as go
import plotly.io as pio


##############################################  Create the prediction input  ########################################################



def create_pred_input(mc, mean_T, std_T, mean_P, std_P, time_periods=None):
   
    """
    Create the input for predictions by standardizing temperature and precipitation.
    
    Parameters:
    - mc: bool either true or false 
    - mean_T: mean of temperature
    - std_T: standard deviation of temperature 
    - mean_p: mean of precipitation
    - std_p: standard deviation of precipitation
    Returns:
    - pred_input: Standardized input for predictions.
    """
    
    temp_vals = np.linspace(0, 30, 90) 
   
    if mc: #we use meters in mc
        precip_vals=np.linspace(0.012,5.435,90)
    else:
        precip_vals= np.linspace(12.03731002, 5435.30011, 90)
   

    if time_periods is not None:
        
        T, P, time = np.meshgrid(temp_vals, precip_vals, np.arange(0,time_periods+1))  
        P_std=(P-mean_P)/std_P  
        T_std=(T - mean_T) / std_T
    
        flat_T_std = T_std.ravel()  
        flat_P_std = P_std.ravel()  
        time_input=time.ravel()
        
        pred_input = np.stack([flat_T_std, flat_P_std, time_input], axis=-1)  # shape (900, 3)
        return pred_input.reshape((1, 1, -1, 3)), T, P
    else:
        
        T, P = np.meshgrid(temp_vals, precip_vals)  
        P_std=(P-mean_P)/std_P  
        T_std=(T - mean_T) / std_T
    
        flat_T_std = T_std.ravel()  
        flat_P_std = P_std.ravel()  
        

        pred_input = np.stack([flat_T_std, flat_P_std], axis=-1)  # shape (900, 2)
        return pred_input.reshape((1, 1, -1, 2)), T, P




##############################################  Compare fixed effects with benchmark models  ########################################################

                                  
def compare_fixed_effects(model):
    """
    Compare fixed effects between the model and benchmark data.
    
    Parameters:
    - model: The trained model instance.

    Returns:
    - None
    """
    
    time_fe_bench=pd.read_csv('data/Benchmark/time_fixed_effects_Burke.csv')
    country_fe_bench=pd.read_csv('data/Benchmark/country_fixed_effects_Burke.csv')

    country_fe_diff = np.abs(model.alpha.values.flatten() - country_fe_bench['x'].values.flatten())

    # Create a DataFrame for plotting
    diff_df = pd.DataFrame({
        'Country_Index': np.arange(1, len(country_fe_diff) + 1),
        'FE_Diff': country_fe_diff
    })

    # Sort by difference for nicer visualization (optional)
    diff_df = diff_df.sort_values(by='FE_Diff', ascending=False)

    # Plot country fixed effects against the benchmark data 
    plt.figure(figsize=(12, 6))
    plt.bar(diff_df['Country_Index'], diff_df['FE_Diff'], color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel('Country')
    plt.ylabel('Fixed Effect Difference')
    plt.title('Country Fixed Effect Differences')
    plt.tight_layout()
    plt.show()
        
    #plot the time fixed effects against the benchmark data 
    time_fe_data = pd.DataFrame({
            'Time': model.time_periods[model.time_periods_not_na['global']][0:62],
            'Model': model.beta.values.flatten(),
            'Benchmark': pd.array(time_fe_bench['x'])[:-1]

        })  
    
    plt.plot(time_fe_data['Time'], time_fe_data['Model'], label='Model', color='red', linewidth=2)
    plt.plot(time_fe_data['Time'], time_fe_data['Benchmark'], label='Benchmark', color='blue', linewidth=2)
    plt.scatter(time_fe_data['Time'], time_fe_data['Model'], color='red', marker='o', s=10)
    plt.scatter(time_fe_data['Time'], time_fe_data['Benchmark'], color='blue', marker='o', s=10)
    plt.xlabel('Time')
    plt.ylabel('Time Fixed Effects')
    plt.title('Time Fixed Effects: Model vs. Benchmark')
    plt.legend()
    plt.show()




##############################################  plotting comparison of 3d models ########################################################




def compare_models_3d(model1, model2, pred_input, save_as_html, node1, node2):
    """
    Compare two models by plotting their predictions in 3D.
    
    Parameters:
    - model1: The first model instance.
    - model2: The second model instance.
    - save_as_html: Boolean to save the plot as an HTML file.
    - pred_input: Input data for predictions (standardized temperature and precipitation).
    - node1: Configuration of the first model (e.g., (4,)).
    - node2: Configuration of the second model (e.g., (8,2,2)).
    Returns:
    - None
    """
    
 

     #predictions   
    growth_pred_flat = model1.model_pred.predict(pred_input)
    growth_pred_flat = np.reshape(growth_pred_flat, (-1,))   # shape (900,)
    # Reshape predictions back to (30, 30) for surface plotting
    Growth1 = growth_pred_flat.reshape(T.shape)  # shape (30, 30)

    #predictions
    growth_pred_flat = model2.model_pred.predict(pred_input)
    growth_pred_flat = np.reshape(growth_pred_flat, (-1,))   # shape (900,)
    # Reshape predictions back to (30, 30) for surface plotting
    Growth2 = growth_pred_flat.reshape(T.shape)  # shape (30, 30)
    
    
    #plot the models
    fig = go.Figure(data=[go.Surface(z=Growth1, x=T, y=P, colorscale='Viridis', opacity=0.8, name='First model', showscale=True),
                        go.Surface(z=Growth2, x=T, y=P, colorscale='Cividis', opacity=0.8, name='Second model', showscale=False)])
    # Update layout with labels, title, and other settings
    fig.update_layout(
        title=f'3D Surface Plot: comparison of model {node1} and {node2}',
        # showlegend=False,
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Precipitation (mm)',
            zaxis_title='Growth',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust the camera angle
            )
        )
    )
    # Show the plot
    fig.show()
    if save_as_html: 
         # Save the figure as an HTML file
        pio.write_html(fig, file=f'images/interactive_3d_plot_comparison_{node1}_{selection_method}_and_{node2}_{selection_method2}.html', auto_open=False)

    return None



##############################################  Makes confidence plot of top n models ########################################################

## Load the results from the file


def confidence_plot(ref_model, results, n_models, save_as_html, pred_input, model, x_train, growth):
    

    # Function to create a confidence plot for the model predictions.   
    """
    Function to create a confidence plot for the model predictions.
    calculates the ensemble mean and standard deviation of the top n models.
    The plot shows the chosen model's prediction surface colored by the ensemble standard deviation.
    
    Parameters:
    - ref_model: The reference model configuration (e.g., (8,2,2)).
    - results: Dictionary containing model configurations and their performance metrics.
    - n_models: Number of top models to consider for the ensemble.
    - save_as_html: Boolean to save the plot as an HTML file.
    - pred_input: Input data for predictions (standardized temperature and precipitation).
    - model: The model class 

    Returns:
    - None
    """
    # Sort the keys (node configurations) by performance metric and select the top 10.
    top_models = sorted(results, key=lambda node: results[node])[:n_models]
    
    
    # Initialize a list to store the prediction surfaces for each model.
    prediction_surfaces = []

    for node1 in top_models:
    
        # Instantiate your model with the given node configuration.

        model_instance = Model(nodes=node1, x_train=x_train, y_train=growth, dropout=0.2)
        

    # Convert the list into a NumPy array: shape (10, 30, 30)
    prediction_surfaces = np.array(prediction_surfaces)

    # Calculate the pointwise mean and standard deviation across the 10 models.
    ensemble_mean = np.mean(prediction_surfaces, axis=0)
    ensemble_std = np.std(prediction_surfaces, axis=0)

    # Select the Chosen Model (e.g. configuration (8,2,2))
    chosen_index = top_models.index(ref_model)
    chosen_surface = prediction_surfaces[chosen_index]



def model_instance_pred(model, date_of_run, node, T, pred_input):
    weight_file = f'../results/Model Parameters/global/cv/{date_of_run}/{str(node)}.weights.h5'
    model.load_params(weight_file)  # Load the model weights
    
    # Use the model to predict on the standardized (T,P) grid.
    growth_pred_flat = model.model_pred.predict(pred_input)
    growth_pred_flat = np.reshape(growth_pred_flat, (-1,))   # shape: (900,)
    Growth = growth_pred_flat.reshape(T.shape)  # reshape to (30, 30)
    
    return Growth

def model_confidence_plot(node, chosen_surface, ensemble_std, T, P, save_as_html):
      
    """
    Function to create a 3D surface plot of the chosen model's predictions colored by ensemble standard deviation.
    """
    fig = go.Figure()
    fig.add_trace(
        go.Surface(
            z=chosen_surface, 
            x=T, 
            y=P, 
            surfacecolor=ensemble_std,
            colorscale='Viridis',
            opacity=0.8,
            name=f'Chosen Model {node}',
            colorbar=dict(title='Ensemble Std')
        )
    )

    fig.update_layout(
        title=f"Model Confidence Plot: {node} vs. Ensemble Agreement",
        scene=dict(
            xaxis_title="Temperature (°C)",
            yaxis_title="Precipitation (mm)",
            zaxis_title="Growth",
            camera=dict(eye=dict(x=1.5, y=1.5, z=1.5))
        ),
        showlegend=False
    )

# Show the plot and write to an HTML file for interactive exploration.
    fig.show()
    if save_as_html:
        # Save the figure as an HTML file
        pio.write_html(fig, file=f'images/model_confidence_plot_{node}.html', auto_open=False)
        

def plotly_theme_bw(fig, text_size=14):
    fig.update_layout(
        font=dict(
            size=text_size,
            color='black'
        ),
        scene=dict(
            xaxis=dict(
                title_font=dict(size=text_size, color='black'),
                tickfont=dict(size=text_size, color='black'),
                showgrid=True,
                zeroline=False,
                showline=True,
                linecolor='black',
                linewidth=0.5,
                ticks='outside',
                tickwidth=1,
                tickcolor='black',
                ticklen=2,
            ),
            yaxis=dict(
                title_font=dict(size=text_size, color='black'),
                tickfont=dict(size=text_size, color='black'),
              
                showgrid=True,
       
                zeroline=False,
                showline=True,
                linecolor='black',
                linewidth=0.5,
                ticks='outside',
                tickwidth=1,
                tickcolor='black',
                ticklen=5,
            ),
            zaxis=dict(
                title_font=dict(size=text_size, color='black'),
                tickfont=dict(size=text_size, color='black'),
              
                showgrid=True,
       
                zeroline=False,
                showline=True,
                linecolor='black',
                linewidth=0.5,
                ticks='outside',
                tickwidth=1,
                tickcolor='black',
                ticklen=5,
            ),
        
        ),
        margin=dict(l=10, r=10, t=20, b=0.4),
   
        legend=dict(
            font=dict(size=text_size),
            orientation='h',
            yanchor='bottom',
            y=-0.,
            xanchor='center',
            x=0.5
        ),
      
    )

    return fig
