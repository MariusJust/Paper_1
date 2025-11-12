from tensorflow.keras.layers import Input, Add, concatenate
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from models.global_model.model_functions.helper_functions import Dummies, create_fixed_effects, Vectorize, Count_params, Matrixize, create_hidden_layers, create_output_layer, Visual_model, prediction_model, create_country_trends

class RegionalModel:
  def __init__(self):
    for key, value in vars(self.cfg).items():
              setattr(self, key, value)
  
  def CreateRegion(self):
    
      # The input shape is (T, N), where T is the time period and N is the number of countries
      input_temp = Input(shape=(self.T, self.N_region), name=f"temp_in_{self.region}")
      input_precip = Input(shape=(self.T, self.N_region), name=f"precip_in_{self.region}")
      self.input_data_temp = tf.reshape(tf.convert_to_tensor(self.x_train_transf[0][self.region]), (1, self.T, self.N_region))
      self.input_data_precip = tf.reshape(tf.convert_to_tensor(self.x_train_transf[1][self.region]), (1, self.T, self.N_region))    
      self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf[self.region]), (1, self.T, self.N_region)) 
      

      self.Mask = tf.reshape(
      tf.convert_to_tensor(self.mask[self.region]),
                          (1, self.input_data_temp.shape[1], self.N_region) )
      

      dummies_layer = Dummies(self.N_region, self.T, self.time_periods_na, country_trends=False)
      Delta1, Delta2 = dummies_layer(input_temp)

        # Creating fixed effects
      country_FE, time_FE, self.country_FE_layer, self.time_FE_layer = create_fixed_effects(self, Delta1, Delta2)
        
        # Vectorize the inputs
      temp_input=Vectorize(self.N_region, 'temp')(input_temp)

      precip_input= Vectorize(self.N_region, 'precip')(input_precip)

      if self.dynamic_model:
        time_input=Vectorize(self.N_region, 'time', time_periods=self.time_periods)(input_temp)
        input_first= concatenate([temp_input, precip_input, time_input], axis=2)
      else:
        input_first= concatenate([temp_input, precip_input], axis=2)
    
      # neural network model
      input_last=create_hidden_layers(self, input_first)
  
      # Creating temporary output layer, without fixed effects
      output_tmp = create_output_layer(self, input_last)
      
      #when we use the within transformation, we do not add fixed effects
      if self.add_fe:
          if self.dynamic_model: 
            output = Add()([country_FE, output_tmp])
          else:
            output = Add()([time_FE, country_FE, output_tmp])
      else:
        output = output_tmp
      
      
      # Creating the final output matrix with the correct dimensions
      self.output_matrix = Matrixize(N=self.N_region, T=self.T, mask=self.Mask, holdout=self.holdout, n_obs_train=self.noObs["train"])(output)

  def SetupRegionalModel(self):
      """
      Setting up the regional model.
      """
      
      for region in self.regions:
              #note we have 2 observations for each region (temp and precip)
              region_builder=self.CreateRegion()
              self.region_builders.append(region_builder)
      
      #extract model lists: 
      self.inputs_temp =   [b.input_temp for b in self.region_builders]
      self.inputs_precip = [b.input_precip for b in self.region_builders]
      self.output_matrix = [b.output_matrix for b in self.region_builders]
          
      self.model = Model(inputs=[self.inputs_temp, self.inputs_precip], outputs=self.output_matrix)  

      
      # compute the visual model for each region
      self.model_visual={}
      for region in self.regions:
          self.model_visual[region]=Visual_model(self, region)
      