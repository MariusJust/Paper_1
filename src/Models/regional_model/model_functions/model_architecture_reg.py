from tensorflow.keras.layers import Input, Add, concatenate
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from models.global_model.model_functions.helper_functions import Dummies, create_fixed_effects, Vectorize, Count_params, Matrixize, create_hidden_layers, create_output_layer, Visual_model, prediction_model, create_country_trends


class Regions:
    def __init__(self, parent, regions):
       self.regions=regions
      
      #unpack parent attributes
       for key, value in vars(parent).items():
            setattr(self, key, value)
            
       self.region_builders = []

      
    def SetupRegionalModel(self):
      
        for region in self.regions:
            self.region_builders.append(self.BuildRegion(region))

        #extract model lists: 
        self.extract_lists()
        
        self.model = Model(inputs=[self.inputs_temp, self.inputs_precip], outputs=self.output_matrix)  

        # compute the visual model for each region
        self.model_visual={}
        for region in self.regions:
            self.model_visual[region]=Visual_model(self, region)
            
            
    def extract_lists(self):
        self.inputs_temp =   [b.input_temp for b in self.region_builders]
        self.inputs_precip = [b.input_precip for b in self.region_builders]
        self.output_matrix = [b.output_matrix for b in self.region_builders]
        
        self.input_data_temp = [b.input_data_temp for b in self.region_builders]
        self.input_data_precip = [b.input_data_precip for b in self.region_builders]
        self.targets = [b.targets for b in self.region_builders]
        self.masks = [b.mask for b in self.region_builders]
        self.country_FE_layer=[b.country_FE_layer for b in self.region_builders]
        self.time_FE_layer=[b.time_FE_layer for b in self.region_builders]
        
    
    def BuildRegion(self, region):
      self.region=region
      # The input shape is (T, N), where T is the time period and N is the number of countries
      self.input_temp = Input(shape=(self.T, self.N[self.region]), name=f"temp_in_{self.region}")
      self.input_precip = Input(shape=(self.T, self.N[self.region]), name=f"precip_in_{self.region}")
      
      
      self.input_data_temp = tf.reshape(tf.convert_to_tensor(self.x_train_transf[0][self.region]), (1, self.T, self.N[self.region]))
      self.input_data_precip = tf.reshape(tf.convert_to_tensor(self.x_train_transf[1][self.region]), (1, self.T, self.N[self.region]))    
      self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf[self.region]), (1, self.T, self.N[self.region])) 
      

      self.Mask = tf.reshape(
      tf.convert_to_tensor(self.mask[self.region]),
                          (1, self.input_data_temp.shape[1], self.N[self.region]) )
      

      dummies_layer = Dummies(self.N[self.region], self.T, self.time_periods_na[self.region], country_trends=False)
      Delta1, Delta2 = dummies_layer(self.input_temp)

        # Creating fixed effects
      country_FE, time_FE, self.country_FE_layer, self.time_FE_layer = create_fixed_effects(self, Delta1, Delta2)
        
        # Vectorize the inputs
      temp_input=Vectorize(self.N[self.region], 'temp')(self.input_temp)

      precip_input= Vectorize(self.N[self.region], 'precip')(self.input_precip)

      if self.dynamic_model:
        time_input=Vectorize(self.N[self.region], 'time', time_periods=self.time_periods)(self.input_temp)
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
      self.output_matrix = Matrixize(N=self.N[self.region], T=self.T, mask=self.Mask, holdout=self.holdout, n_obs_train=self.noObs[self.region])(output)

        
        
        