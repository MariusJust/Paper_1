

import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import plotly.graph_objects as go
import plotly.io as pio


##############################################  Create the prediction input  ########################################################

def create_pred_input(mc):
    """
    Create the input for predictions by standardizing temperature and precipitation.
    
    Parameters:
    - None
    Returns:
    - pred_input: Standardized input for predictions.
    """
    # Standardize temperature and precipitation
    temp_vals = np.linspace(0, 30, 30)  # 30 points from 0 to 30
    precip_vals = np.linspace(12.03731002, 5435.30011, 30)  #30 points between the min and max of the precipitation data

    # 2. Create a meshgrid of temperature (T) and precipitation (P)
    T, P = np.meshgrid(temp_vals, precip_vals)  # Each shape: (30, 30)

    # 3. Standardize T and P according to your known mean/std
    #    (adjust to the same approach you used in your code for standardization).
    if mc:
       
        P=P/1000
        P_std=(P-1094.316/1000)/678.313/1000
    else:
         P_std = (P - 1094.316) / 678.313  # example standardization if that matches your data

    T_std=(T - 18.05) / 7.10
   
    # 4. Reshape for model input
    flat_T_std = T_std.ravel()  # shape (30*30,)
    flat_P_std = P_std.ravel()  # shape (30*30,)

    # Concatenate temperature & precipitation along last dimension
    pred_input = np.stack([flat_T_std, flat_P_std], axis=-1)  # shape (900, 2)
    

    # We might need (1, 1, 900, 2):
    pred_input = pred_input.reshape((1, 1, -1, 2))  # shape (1,1,900,2)
    
   

    return pred_input, T, P



##############################################  Compare fixed effects with benchmark models  ########################################################

                                  
def compare_fixed_effects(model):
    """
    Compare fixed effects between the model and benchmark data.
    
    Parameters:
    - model: The trained model instance.
    - country_fe_bench: Benchmark country fixed effects.
    - time_fe_bench: Benchmark time fixed effects.
    
    Returns:
    - None
    """
    time_fe_bench=pd.read_csv('data/Benchmark/time_fixed_effects_Burke.csv')

    country_fe_bench=pd.read_csv('data/Benchmark/country_fixed_effects_Burke.csv')

    # Calculate the difference between the model and the benchmark data
    country_fe_diff = np.abs(model.alpha.values.flatten() - country_fe_bench['x'].values.flatten())
    
    country_fe_diff = np.abs(model.alpha.values.flatten() - country_fe_bench['x'].values.flatten())
    country_names = model.individuals['global'][1:196]
    # Create a DataFrame for plotting
    diff_df = pd.DataFrame({
        'Country_Index': np.arange(1, len(country_fe_diff) + 1),
        'FE_Diff': country_fe_diff
    })

    # Sort by difference for nicer visualization (optional)
    diff_df = diff_df.sort_values(by='FE_Diff', ascending=False)

    # Plot
    plt.figure(figsize=(12, 6))
    plt.bar(diff_df['Country_Index'], diff_df['FE_Diff'], color='skyblue')
    plt.xticks(rotation=90)
    plt.xlabel('Country')
    plt.ylabel('Fixed Effect Difference')
    plt.title('Country Fixed Effect Differences')
    plt.tight_layout()
    plt.show()
        
    
    #time fixed effects 
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


##############################################  Compare model predictions with benchmark data  ########################################################
def compare_with_benchmark(model):
    """
    Compare model predictions with benchmark data.
    
    Parameters:
    - model: The trained model instance.
    - benchmark_Burke: Benchmark data from Burke et al. (2015).
    - benchmark_Leirvik: Benchmark data from Leirvik et al. (2023).
    - temperature_dict: Dictionary mapping temperature to index.
    - growth_data: Growth data for the model.
    
    Returns:
    - None
    """
    
    benchmark_Burke = pd.read_csv('data/Benchmark/PredictedGrowthBurke2015.csv')
    benchmark_Leirvik = pd.read_csv('data/Benchmark/Leirvik2023.csv')
    
    #mean is zero, since we standardized the data
    mean_precip = 0
    mean_precip_array = np.full((35, 1), mean_precip).reshape((1, 1, -1, 1))
    
    temperature_array = np.linspace(0, 35, 35)  # e.g. 30 temperature points

    #make a dict with each temperature value and the corresponding standardised value

    temperature_dict = {temp: (temp - 18.05) / 7.10 for temp in temperature_array}


    #make the temperature array from the dict values 
    temperature_array = np.array([temperature_dict[temp] for temp in temperature_array])

    temperature_array = temperature_array.reshape((1, 1, -1, 1))  # shape (30,1)
        
    
    # Predict for mean precipitation
    pred_vector_mean = tf.concat([temperature_array, mean_precip_array], axis=3)
    
    pred_mean = np.reshape(model.model_pred.predict(pred_vector_mean), (-1, 1), order='F')

    # Create a DataFrame for plotting
    plot_data = pd.DataFrame({
        'Temperature': np.linspace(0, 3, 30),
        'Mean Growth': pred_mean.flatten()[5:35],
        'Burke': benchmark_Burke['avg_prediction'][5:35],
        'Leirvik': benchmark_Leirvik['avg_prediction.4'][5:35],
        
    })
    
    # Plotting
    plt.plot(plot_data['Temperature'], plot_data['Mean Growth'], label='Mean Growth', color='red', linewidth=2)
    plt.plot(plot_data['Temperature'], plot_data['Burke'], label='Burke2023', color='blue', linewidth=2)
    plt.plot(plot_data['Temperature'], plot_data['Leirvik'], label='Leirvik2023', color='purple', linewidth=2)
  
    plt.scatter(plot_data['Temperature'], plot_data['Mean Growth'], color='red', marker='o', s=10)
    plt.scatter(plot_data['Temperature'], plot_data['Burke'], color='blue', marker='o', s=10)
    plt.scatter(plot_data['Temperature'], plot_data['Leirvik'], color='purple', marker='o', s=10)
    
    plt.xlabel('Temperature')
    plt.ylabel('Growth')
    plt.title('Comparison of Model Predictions with Benchmark Data')
    plt.legend()

    
##############################################  plotting 3D  ########################################################

def plotting_3d(node, model,  pred_input, save_as_html):
    """
    Plotting the 3D surface for the given model configuration.
    
    Parameters:
    - node: Model configuration (e.g., (8,2,2)).
    - growth: Growth data.
    - x_train: Input features (temperature and precipitation).
    - pred_input: Standardized input for predictions.
    Returns:
    - None
    """


    # 5. Run prediction
    growth_pred_flat = model.model_pred.predict(pred_input)  
    growth_pred_flat = np.reshape(growth_pred_flat, (-1,))   # shape (900,)

    # 6. Reshape predictions back to (30, 30) for surface plotting
    Growth = growth_pred_flat.reshape(T.shape)  # shape (30, 30)


    # Create a surface plot
    fig = go.Figure(data=[go.Surface(z=Growth, x=T, y=P, colorscale='Viridis', opacity=0.8)])

    # Update layout with labels, title, and other settings
    fig.update_layout(
        title=f'3D Surface Plot: Growth vs. Temp & Precip {node}',
        scene=dict(
            xaxis_title='Temperature (°C)',
            yaxis_title='Precipitation (mm)',
            zaxis_title='Growth',
            camera=dict(
                eye=dict(x=1.5, y=1.5, z=1.5)  # Adjust the camera angle
            )
        ),
        coloraxis_colorbar=dict(title="Growth")
    )

    # Show the plot
    fig.show()
    if save_as_html:
        pio.write_html(fig, file=f'images/interactive_3d_plot_{node}_{selection_method}_{date_of_run}.html', auto_open=False)
    return None

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
                showbackground=False,
                showgrid=True,
                gridcolor='lightgray',
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
                showbackground=False,
                showgrid=True,
                gridcolor='lightgray',
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
                showbackground=False,
                showgrid=True,
                gridcolor='lightgray',
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
        paper_bgcolor='white',
        plot_bgcolor='white',
        legend=dict(
            font=dict(size=text_size),
            orientation='h',
            yanchor='bottom',
            y=-0.,
            xanchor='center',
            x=0.5,
            bgcolor='rgba(0,0,0,0)',
        ),
      
    )

    return fig
