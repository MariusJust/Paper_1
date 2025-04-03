
import numpy as np
import tensorflow as tf
from Model.ModelFunctions import Prepare
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  # needed for 3D plots in Matplotlib
import plotly.graph_objects as go
import plotly.io as pio
import os

os.environ['PYTHONHASHSEED'] = str(0)

from Model import multivariate_model as Model

                                  
lr = 0.001                        # initial learning rate for the Adam optimizer
min_delta = 1e-6                  # tolerance to be used for optimization
patience = 100                    # patience to be used for optimization
verbose = 2                   # verbosity mode for optimization



data=pd.read_excel('data/MainData.xlsx')

growth, precip, temp = Prepare(data)



results=dict(np.load('results/03042025/results.npy', allow_pickle=True).item())

#find the best 10 models 
min_nodes = sorted(results, key=lambda node: results[node][1])[:10]

min_node=(4,)


x_train=[temp, precip]

temperature_array = np.linspace(0, 35, 35)  # e.g. 30 temperature points

#make a dict with each temperature value and the corresponding standardised value

temperature_dict = {temp: (temp - 18.05) / 7.10 for temp in temperature_array}


#make the temperature array from the dict values 
temperature_array = np.array([temperature_dict[temp] for temp in temperature_array])

temperature_array = temperature_array.reshape((1, 1, -1, 1))  # shape (30,1)

# precipitation at its means 
mean_precip=0

lower_quantile_precip=(578.274-1094.316)/678.313

upper_quantile_precip=(1488.75-1094.316)/678.313


# Broadcast this single scalar for 30 time steps
mean_precip_array = np.full((35, 1), mean_precip).reshape((1, 1, -1, 1))
lower_precip_array = np.full((35, 1), lower_quantile_precip).reshape((1, 1, -1, 1))
upper_precip_array = np.full((35, 1), upper_quantile_precip).reshape((1, 1, -1, 1))

# Predict for mean precipitation
pred_vector_mean = tf.concat([temperature_array, mean_precip_array], axis=3)
pred_vector_upper = tf.concat([temperature_array, upper_precip_array], axis=3)
pred_vector_lower = tf.concat([temperature_array, lower_precip_array], axis=3)

model=Model(min_node, x_train=x_train, y_train=growth)

#loadig the best model weights for reproducability
model.load_params('Model Parameters/BIC/03042025/' +  str(min_node)+'.weights.h5')


model.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)


#country fixed effects
model.alpha  

#time fixed effects
model.beta


#read the country and time fixed effects from the benchmark data 

time_fe_bench=pd.read_csv('data/Benchmark/time_fixed_effects_Burke.csv')

country_fe_bench=pd.read_csv('data/Benchmark/country_fixed_effects_Burke.csv')

#calculate the difference between the model and the benchmark data
country_fe_diff = np.abs(- )


# Plot the comparison
plt.figure(figsize=(10, 6))
plt.scatter(model.alpha.values.flatten() , country_fe_bench['x'].values.flatten(), label='Model vs Benchmark', color='blue', alpha=0.7)

# Label each point with the corresponding country name
# for i, country in enumerate(countries):
#     plt.text(model_alpha_values[i], benchmark_values[i], country, fontsize=9, ha='right')

# Add a reference line (y = x) for comparison
plt.plot([-1, 1], [-1, 1], linestyle='--', color='red', label='y = x (Perfect Match)')

# Labels and title
plt.xlabel('Model Alpha')
plt.ylabel('Benchmark Fixed Effects')
plt.title('Comparison of Model Alpha and Benchmark Country Fixed Effects')
plt.legend()
plt.grid(True)

# Show the plot
plt.show()



country_fe_data = pd.DataFrame({
        'Country': model.individuals['global'][1:196],
        'Model': model.alpha.values.flatten(),
        'Benchmark': country_fe_bench['x'],
        'diff': abs(model.alpha.values.flatten()-country_fe_bench['x'])
    })


country_fe_data['diff'] 


country_fe_data = country_fe_data.sort_values(by='diff', ascending=False)

# Set figure size
plt.figure(figsize=(14, 6))

# Define x-axis positions
x = np.arange(len(country_fe_data))

# Bar width
width = 0.4

# Plot Model values
plt.bar(x - width/2, country_fe_data['Model'], width=width, color='blue', label='Model Alpha', alpha=0.7)

# Plot Benchmark values
plt.bar(x + width/2, country_fe_data['Benchmark'], width=width, color='orange', label='Benchmark', alpha=0.7)

# Set labels and title
plt.xlabel('Country')
plt.ylabel('Fixed Effects Estimate')
plt.title('Comparison of Model and Benchmark Fixed Effects by Country')

# Set x-axis tick labels (rotate for readability)
plt.xticks(x, country_fe_data['Country'], rotation=90, fontsize=8)

# Add legend
plt.legend()

# Show the plot
plt.tight_layout()
plt.show()

countries = model.individuals['global'][1:196]

plt.figure(figsize=(10, 6))

plt.hist(model.alpha.values.flatten(), bins=20, color='blue', edgecolor='black', alpha=0.5, label='Model Alpha')
plt.hist(country_fe_bench['x'].values.flatten(), bins=20, color='orange', edgecolor='black', alpha=0.5, label='Benchmark')

# Labels and title
plt.xlabel('Fixed Effects Estimates')
plt.ylabel('Frequency')
plt.title('Histogram of Model vs Benchmark Fixed Effects')
plt.legend()

# Show grid for better readability
plt.grid(axis='y', linestyle='--', alpha=0.7)

# Show the plot
plt.show()

#plot the time fixed effects against the benchmark data 
time_fe_data = pd.DataFrame({
        'Time': model.time_periods[model.time_periods_not_na['global']][1:63],
        'Model': model.beta.values.flatten(),
        'Benchmark': time_fe_bench['x']
        
    })  
    

#plot the time fixed effects against the benchmark data
plt.plot(time_fe_data['Time'], time_fe_data['Model'], label='Model', color='red', linewidth=2)
plt.plot(time_fe_data['Time'], time_fe_data['Benchmark'], label='Benchmark', color='blue', linewidth=2)
plt.scatter(time_fe_data['Time'], time_fe_data['Model'], color='red', marker='o', s=10)
plt.scatter(time_fe_data['Time'], time_fe_data['Benchmark'], color='blue', marker='o', s=10)
plt.xlabel('Time')
plt.ylabel('Time Fixed Effects')
plt.title('Time Fixed Effects: Model vs. Benchmark')
plt.legend()
plt.show()


#make a table to compare the country fixed effects from the model and the benchmark data
country_fe_bench=pd.read_csv('data/Benchmark/country_fixed_effects_Burke.csv')

country_fe_data = pd.DataFrame({
        'Country': model.individuals['global'][1:196],
        'Model': model.alpha.values.flatten(),
        'Benchmark': country_fe_bench['x'],
        'diff': model.alpha.values.flatten()-country_fe_bench['x']  
    })  



model.in_sample_predictions()

pred_mean = np.reshape(model.model_pred.predict(pred_vector_mean), (-1, 1), order='F')
pred_upper = np.reshape(model.model_pred.predict(pred_vector_upper), (-1, 1), order='F')
pred_lower = np.reshape(model.model_pred.predict(pred_vector_lower), (-1, 1), order='F')



##############################################  plotting 2D  ########################################################

# Load the benchmark predictions

benchmark_Burke=pd.read_csv('data/Benchmark/PredictedGrowthBurke2015.csv')
benchmark_Leirvik=pd.read_csv('data/Benchmark/Leirvik2023.csv')


plot_data = pd.DataFrame({
    'Temperature': temperature_dict.keys(),
    'Mean Growth': pred_mean.flatten(),
    'Lower Growth': pred_lower.flatten(),
    'Upper Growth': pred_upper.flatten(),
    # 'Burke': benchmark_Burke['avg_prediction'][5:35:],
    # 'Leirvik': benchmark_Leirvik['avg_prediction.4'][5:35:],
    # 'data': avg_growth_per_bin.values
})






# Plotting

plt.plot(plot_data['Temperature'], plot_data['Mean Growth'], label='Mean Growth', color='red', linewidth=2)
# plt.plot(plot_data['Temperature'], plot_data['Burke'], label='Burke2023', color='blue', linewidth=2)
# plt.plot(plot_data['Temperature'], plot_data['Leirvik'], label='Leirvik2023', color='purple', linewidth=2)
# plt.plot(plot_data['Temperature'], plot_data['data'], label='Observed Growth', color='green', linewidth=2)
plt.plot(plot_data['Temperature'], plot_data['Lower Growth'], label='Lower Quantile Growth', color='blue', linestyle='--')
plt.plot(plot_data['Temperature'], plot_data['Upper Growth'], label='Upper Quantile Growth', color='green', linestyle='--')

plt.scatter(plot_data['Temperature'], plot_data['Mean Growth'], color='red', marker='o', s=10)
# plt.scatter(plot_data['Temperature'], plot_data['Burke'], color='blue', marker='o', s=10)
# plt.scatter(plot_data['Temperature'], plot_data['Leirvik'], color='purple', marker='o', s=10)
# plt.scatter(plot_data['Temperature'], plot_data['data'], color='green', marker='o', s=10)
plt.scatter(plot_data['Temperature'], plot_data['Lower Growth'], color='blue', marker='x', s=10)
plt.scatter(plot_data['Temperature'], plot_data['Upper Growth'], color='green', marker='x', s=10)
# plt.scatter(scatter_data["TempPopWeight"], scatter_data["GrowthWDI" ], label='Observed Growth', color='green', marker='o', s=10)

plt.xlabel('Temperature')
plt.ylabel('Growth')
plt.title('Comparison of fractiles')
plt.legend()
plt.show()


# Assuming plot_data is already created with the specified columns
plt.figure(figsize=(8, 6))

# Plot 'Temperature' against 'Mean Growth', 'Burke', and 'Leirvik'
plt.plot(plot_data['Temperature'], plot_data['Mean Growth'], label='Mean Growth', color='red', linewidth=2)
plt.plot(plot_data['Temperature'], plot_data['Burke'], label='Burke2023', color='blue', linewidth=2)
plt.plot(plot_data['Temperature'], plot_data['Leirvik'], label='Leirvik2023', color='purple', linewidth=2)

# Add scatter points for better visibility
plt.scatter(plot_data['Temperature'], plot_data['Mean Growth'], color='red', marker='o', s=10)
plt.scatter(plot_data['Temperature'], plot_data['Burke'], color='blue', marker='o', s=10)
plt.scatter(plot_data['Temperature'], plot_data['Leirvik'], color='purple', marker='o', s=10)

# Add labels, title, and legend
plt.xlabel('Temperature')
plt.ylabel('Growth')
plt.title('Growth Predictions vs. Temperature')
plt.legend()

# Show the plot
plt.show()


##############################################  plotting 3D  ########################################################

log_dir = "logs/fit"
min_node=(4,)

model=Model(min_node, x_train=x_train, y_train=growth)

#loadig the best model weights for reproducability
model.load_params('Model Parameters/BIC/03042025/' +  str(min_node)+'.weights.h5')


model.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose, log_dir=log_dir)

model.in_sample_predictions()

model.params

# 1. Create your temperature and precipitation ranges
#    Adjust these ranges as needed to match your data domain.
temp_vals = np.linspace(0, 30, 30)  # 30 points from 0 to 30
precip_vals = np.linspace(12.03731002, 5435.30011, 30)  

# 2. Create a meshgrid of temperature (T) and precipitation (P)
T, P = np.meshgrid(temp_vals, precip_vals)  # Each shape: (30, 30)

# 3. Standardize T and P according to your known mean/std
#    (adjust to the same approach you used in your code for standardization).
T_std = (T - 18.05) / 7.10
P_std = (P - 1094.316) / 678.313  # example standardization if that matches your data

# 4. Reshape for model input

flat_T_std = T_std.ravel()  # shape (30*30,)
flat_P_std = P_std.ravel()  # shape (30*30,)

# Concatenate temperature & precipitation along last dimension
pred_input = np.stack([flat_T_std, flat_P_std], axis=-1)  # shape (900, 2)

# We might need (1, 1, 900, 2):
pred_input = pred_input.reshape((1, 1, -1, 2))  # shape (1,1,900,2)

# 5. Run prediction
growth_pred_flat = model.model_pred.predict(pred_input)  
growth_pred_flat = np.reshape(growth_pred_flat, (-1,))   # shape (900,)

# 6. Reshape predictions back to (30, 30) for surface plotting
Growth = growth_pred_flat.reshape(T.shape)  # shape (30, 30)


# Create a surface plot
fig = go.Figure(data=[go.Surface(z=Growth, x=T, y=P, colorscale='Viridis', opacity=0.8)])

# Update layout with labels, title, and other settings
fig.update_layout(
    title='3D Surface Plot: Growth vs. Temp & Precip (Model 4,)',
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

pio.write_html(fig, file='images/interactive_3d_plot(4,)_dropout.html', auto_open=False)

##############################################  plotting comparison of 10 best models  ########################################################
#make a dataframe to store the results
results_df=pd.DataFrame(columns=['node', 'predicted_growth'])
#run model and predict growth for each of the best 10 models
for idx, min_node in enumerate(min_nodes):
    
    print(idx)




    temp_df = pd.DataFrame({'node': [min_node], 'predicted_growth': [pred_mean]})
    
    results_df = pd.concat([results_df, temp_df], ignore_index=True)



#plotting the predicted growth for the best 10 models, results are in results_df

benchmark_Burke=pd.read_csv('data/Benchmark/PredictedGrowthBurke2015.csv')
benchmark_Leirvik=pd.read_csv('data/Benchmark/Leirvik2023.csv')


plt.figure(figsize=(10, 6))

# Iterate over the rows of the DataFrame
for idx, row in results_df[:5].iterrows():
    # Extract the predicted growth values (flatten to 1D array)
    predicted_growth = np.array(row['predicted_growth']).flatten()
    if idx==2:
        continue
    else:

        plt.plot(range(len(predicted_growth)), predicted_growth, label=f'Model {idx}, Node {row["node"]}')

#add the benchmark predictions
plt.plot(benchmark_Burke['temp'][5:35], benchmark_Burke['avg_prediction'][5:35], label='Burke2023', color='blue', linewidth=2)

# Add labels and title
plt.xlabel('X-axis (0, 1, 2, ..., 30)')
plt.ylabel('Predicted Growth')
plt.title('Comparison of Predicted Growth Across Models')

# Add a legend
plt.legend()

# Show the plot
plt.show()