from tensorflow.keras.layers import Input, Add, concatenate
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from models.global_model.model_functions.helper_functions import Dummies, create_fixed_effects, Vectorize, Count_params, Matrixize, create_hidden_layers, create_output_layer, Visual_model, prediction_model, create_country_trends

def SetupGlobalModel(self):
    
    """
    Setting up the global model.
    """
    
    
    # Creating input layers
    # The input shape is (T, N), where T is the time period and N is the number of countries
    input_precip =Input(shape=(self.T, int(self.N['global'])))
    input_temp = Input(shape=(self.T, int(self.N['global'])))


    self.input_data_temp = tf.reshape(tf.convert_to_tensor(self.x_train_transf[0]['global']), (1, self.T, self.N['global']))
    self.input_data_precip=tf.reshape(tf.convert_to_tensor(self.x_train_transf[1]['global']), (1, self.T, self.N['global']))
    #creates a target tensor of dimension (1, T, N) where the first dimension is the batch size, the second dimension is the time period, and the third dimension is the country. variable is the growth rate
    self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf['global']), (1, self.T, self.N['global']))
   
    self.Mask = tf.reshape(
    tf.convert_to_tensor(self.mask['global']),
                        (1, self.T, self.N['global']) )
    
    self.time_periods = np.arange(1, self.T+1, 1)
    
                            
    # If country trends
    if self.country_trends:
       dummies_layer = Dummies(self.N['global'], self.T, self.time_periods_na['global'], country_trends=True)
       Delta1, Delta2, linear_trend, quadratic_trend = dummies_layer(input_temp)
       
       linear_country_trend, quadratic_country_trend, self.linear_trend_layer, self.quadratic_trend_layer = create_country_trends(self, linear_trend, quadratic_trend)

    else: 
       dummies_layer = Dummies(self.N['global'], self.T, self.time_periods_na['global'], country_trends=False)
       Delta1, Delta2 = dummies_layer(input_temp)

    # Creating fixed effects
    country_FE, time_FE, self.country_FE_layer, self.time_FE_layer= create_fixed_effects(self, Delta1, Delta2)
    
      # Vectorize the inputs
    temp_input=Vectorize(self.N, 'temp')(input_temp)

    precip_input= Vectorize(self.N, 'precip')(input_precip)

    if self.dynamic_model:
      time_input=Vectorize(self.N, 'time', time_periods=self.time_periods)(input_temp)
      input_first= concatenate([temp_input, precip_input, time_input], axis=2)
    else:
      input_first= concatenate([temp_input, precip_input], axis=2)
    
           
    # neural network model
    input_last=create_hidden_layers(self, input_first)
 
    # Creating temporary output layer, without fixed effects
    output_tmp = create_output_layer(self, input_last)
    
    if self.dynamic_model: 
      output = Add()([country_FE, output_tmp])
    else:
      if self.country_trends:
    # Adding fixed effects
       output = Add()([time_FE, country_FE, linear_country_trend, quadratic_country_trend, output_tmp])
      else:
       output = Add()([time_FE, country_FE, output_tmp])

    # Creating t he final output matrix with the correct dimensions
    output_matrix = Matrixize(N=self.N['global'], T=self.T, noObs=self.noObs['global'], mask=self.Mask)(output)

    # Compiling the model
    self.model = Model(inputs=[input_temp, input_precip], outputs=output_matrix)

  

    # Counting number of parameters

    self.m = Count_params(self)

    #setting up the visualisation of the model, and the prediction model
    
    self.model_pred=prediction_model.pred_model(self)
    self.model_visual=Visual_model(self)
        
    