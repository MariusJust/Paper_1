
import numpy as np
import tensorflow as tf
from Static.ModelFunctions_global import Prepare
import pandas as pd
import matplotlib.pyplot as plt
import os
import random
from Static import multivariate_model as Model

                                  
lr = 0.001                        # initial learning rate for the Adam optimizer
min_delta = 1e-6                  # tolerance to be used for optimization
patience = 100                    # patience to be used for optimization
verbose = 2                   # verbosity mode for optimization

# # Setting optional choice parameters
# seed_value = 0                                                
# lr = 0.001                        # initial learning rate for the Adam optimizer
# min_delta = 0                  # tolerance to be used for optimization
# patience =100                     # patience to be used for optimization
# verbose = False                   # verbosity mode for optimization





# specification='Multivariate'    

# if specification=='Univariate': 
#     from Static import Multivariate as Model
    

#     scatter_data=data[['GrowthWDI', 'TempPopWeight']]

#     #take temperature values that are less than 5 out of the model
#     results = np.load('BIC/(8, 2).npy', allow_pickle=True)
#     #open results.npy 
#     results = dict(np.load('Model Parameters/11022025/results.npy', allow_pickle=True).item())

#     results1=dict(np.load('Model Parameters/08022025/results.npy', allow_pickle=True).item())


#     min_node=min(results, key=results.get)

#     #loading weight for the best model 
#     tf.keras.backend.clear_session()

#     model=Model(min_node, x_train=temp, y_train=growth, formulation='Global')

#     #loadig the best model weights for reproducability
#     model.load_params('11022025/Model Parameters/' +  str(min_node)+'.weights.h5')



#     model.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)

#     loss, acc = model.evaluate( verbose=2)

#     len(model.epochs) 

#     #plotting the loss function
#     plt.plot(model.epochs, model.losses['loss'], label='Training Loss')

#     plt(np(model.losses))

#     pred=model.in_sample_predictions()

#     pred_flat=np.reshape(pred,(-1,1))

#     pred_flat=np.reshape(pred,(-1,1))


#     # Create a linear range between min and max, reshaped to fit the model's expected input shape
#     x_test = np.linspace(0, 30, 90).reshape(1, 1,90, 1)  # Reshape to (1, 1, 10000, 1)

#     # Make predictions using the model
#     y_test = model.predict(x_test)

#     x_test = np.reshape(x_test,(-1,1))


#     benchmark = pd.read_csv('Benchmark/PredictedGrowthBurke2015.csv')

    
#     #plotting the figure

#     plt.figure(figsize=(8, 6))

#     # Plot the benchmark data
#     plt.plot(benchmark['temp'][5:35:], benchmark['avg_prediction'][5:35:], label='Benchmark Growth', color='blue', linewidth=2)

#     # Plot the model's predicted data
#     plt.plot(x_test, y_test, label='Predicted Growth', color='red', linewidth=2)
#     plt.scatter(scatter_data["TempPopWeight"], scatter_data["GrowthWDI" ], label='Observed Growth', color='green', marker='o', s=10)

#     # Add labels, title, and legend
#     plt.xlabel('Temperature')
#     plt.ylabel('Growth')
#     plt.title('Model vs Benchmark Growth Predictions')
#     plt.legend()

#     # Display the plot
#     plt.show()

    
# else: 
#     pass 


data=pd.read_excel('data/MainData.xlsx')

growth, precip, temp = Prepare(data)

results=dict(np.load('results.npy', allow_pickle=True).item())

min_node=min(results, key=results.get)

min_node=(32,16)
x_train=[temp, precip]


model=Model(min_node, x_train=x_train, y_train=growth)

#loadig the best model weights for reproducability
model.load_params('Model Parameters/14032025/' +  str(min_node)+'.weights.h5')


model.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)



temperature_array = np.linspace(0, 30, 30)  # e.g. 30 temperature points

#make a dict with each temperature value and the corresponding standardised value

temperature_dict = {temp: (temp - 18.05) / 7.10 for temp in temperature_array}


#make the temperature array from the dict values 
temperature_array = np.array([temperature_dict[temp] for temp in temperature_array])

temperature_array = temperature_array.reshape((1, 1, -1, 1))  # shape (30,1)

# precipitation at its means 
mean_precip=0.1

lower_quantile_precip=(578.274-1094.316)/678.313

upper_quantile_precip=(1488.75-1094.316)/678.313


# Broadcast this single scalar for 30 time steps
mean_precip_array = np.full((30, 1), mean_precip).reshape((1, 1, -1, 1))
lower_precip_array = np.full((30, 1), lower_quantile_precip).reshape((1, 1, -1, 1))
upper_precip_array = np.full((30, 1), upper_quantile_precip).reshape((1, 1, -1, 1))

# Predict for mean precipitation
pred_vector_mean = tf.concat([temperature_array, mean_precip_array], axis=3)
pred_mean = np.reshape(model.model_pred.predict(pred_vector_mean), (-1, 1), order='F')

# Predict for lower quantile precipitation
pred_vector_lower = tf.concat([temperature_array, lower_precip_array], axis=3)
pred_lower = np.reshape(model.model_pred.predict(pred_vector_lower), (-1, 1), order='F')

# Predict for upper quantile precipitation
pred_vector_upper = tf.concat([temperature_array, upper_precip_array], axis=3)
pred_upper = np.reshape(model.model_pred.predict(pred_vector_upper), (-1, 1), order='F')

# Create the plot data for each case


benchmark_Burke=pd.read_csv('data/Benchmark/PredictedGrowthBurke2015.csv')
benchmark_Leirvik=pd.read_csv('data/Benchmark/Leirvik2023.csv')



#add scatterplot data from growth 
scatter_data=data[['GrowthWDI', 'TempPopWeight']]

bins = pd.interval_range(start=0, end=30, freq=1)
scatter_data['TempBin'] = pd.cut(scatter_data['TempPopWeight'], bins)

# Group by the bins and calculate the average GrowthWDI for each bin
avg_growth_per_bin = scatter_data.groupby('TempBin')['GrowthWDI'].mean()



plot_data = pd.DataFrame({
    'Temperature': temperature_dict.keys(),
    'Mean Growth': pred_mean.flatten(),
    'Lower Growth': pred_lower.flatten(),
    'Upper Growth': pred_upper.flatten(),
    'Burke': benchmark_Burke['avg_prediction'][5:35:],
    'Leirvik': benchmark_Leirvik['avg_prediction.4'][5:35:],
    'data': avg_growth_per_bin.values
})



# Plotting
plt.figure(figsize=(8, 6))
plt.plot(plot_data['Temperature'], plot_data['Mean Growth'], label='Mean Growth', color='red', linewidth=2)
plt.plot(plot_data['Temperature'], plot_data['Burke'], label='Burke2023', color='blue', linewidth=2)
plt.plot(plot_data['Temperature'], plot_data['Leirvik'], label='Leirvik2023', color='purple', linewidth=2)
plt.plot(plot_data['Temperature'], plot_data['data'], label='Observed Growth', color='green', linewidth=2)
# plt.plot(plot_data['Temperature'], plot_data['Lower Growth'], label='Lower Quantile Growth', color='blue', linestyle='--')
# plt.plot(plot_data['Temperature'], plot_data['Upper Growth'], label='Upper Quantile Growth', color='green', linestyle='--')

plt.scatter(plot_data['Temperature'], plot_data['Mean Growth'], color='red', marker='o', s=10)
plt.scatter(plot_data['Temperature'], plot_data['Burke'], color='blue', marker='o', s=10)
plt.scatter(plot_data['Temperature'], plot_data['Leirvik'], color='purple', marker='o', s=10)
plt.scatter(plot_data['Temperature'], plot_data['data'], color='green', marker='o', s=10)
# plt.scatter(plot_data['Temperature'], plot_data['Lower Growth'], color='blue', marker='x', s=10)
# plt.scatter(plot_data['Temperature'], plot_data['Upper Growth'], color='green', marker='x', s=10)
# plt.scatter(scatter_data["TempPopWeight"], scatter_data["GrowthWDI" ], label='Observed Growth', color='green', marker='o', s=10)

plt.xlabel('Temperature')
plt.ylabel('Growth')
plt.title('Comparison of Growth Predictions')
plt.legend()
plt.show()
