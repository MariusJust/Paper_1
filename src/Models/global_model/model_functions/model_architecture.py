from tensorflow.keras.layers import Input, Add, concatenate
from tensorflow.keras import Model
import tensorflow as tf
import numpy as np
from models.global_model.model_functions.helper_functions import Dummies, create_fixed_effects, Vectorize, Count_params, Matrixize, create_hidden_layers, create_output_layer, Visual_model, prediction_model, create_country_trends

def SetupGlobalModel(self):
    
    """
    Setting up the global model.
    """
    
    
    # The input shape is (T, N), where T is the time period and N is the number of countries
    input_precip = Input(shape=(None, int(self.N['global'])), name='precip_input')
    input_temp   = Input(shape=(None, int(self.N['global'])), name='temp_input')

    if self.x_val is not None:
      
        #training period targets and inputs for holdout
        self.input_data_temp_train_val= tf.reshape(tf.convert_to_tensor(self.x_train_val_transf[0]['global']), (1, self.T, self.N['global']))
        self.input_data_precip_train_val = tf.reshape(tf.convert_to_tensor(self.x_train_val_transf[1]['global']), (1, self.T, self.N['global']))
        self.targets_train_val = tf.reshape(tf.convert_to_tensor(self.y_train_val_transf['global']), (1, self.T, self.N['global']))
        
        #validation input and targets for holdout
        self.input_data_temp_val = tf.reshape(tf.convert_to_tensor(self.x_val_transf[0]['global']), (1, self.holdout, self.N['global']))
        self.input_data_precip_val = tf.reshape(tf.convert_to_tensor(self.x_val_transf[1]['global']), (1, self.holdout, self.N['global']))
        self.targets_val = tf.reshape(tf.convert_to_tensor(self.y_val_transf['global']), (1, self.holdout, self.N['global']))

        #full sample
        self.input_data_temp = tf.reshape(tf.convert_to_tensor(self.x_train_transf[0]['global']), (1, self.T + self.holdout, self.N['global']))
        self.input_data_precip = tf.reshape(tf.convert_to_tensor(self.x_train_transf[1]['global']), (1, self.T + self.holdout, self.N['global']))
        self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf['global']), (1, self.T + self.holdout, self.N['global']))
    else:
        self.noObs['train'] = self.noObs['global']
        self.input_data_temp = tf.reshape(tf.convert_to_tensor(self.x_train_transf[0]['global']), (1, self.T, self.N['global']))
        self.input_data_precip = tf.reshape(tf.convert_to_tensor(self.x_train_transf[1]['global']), (1, self.T, self.N['global']))    
        self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf['global']), (1, self.T, self.N['global'])) 
        

    self.Mask = tf.reshape(
    tf.convert_to_tensor(self.mask['global']),
                        (1, self.input_data_temp.shape[1], self.N['global']) )
    


    self.time_periods = np.arange(1, self.T+1, 1)

    #when we are apllying within transformation, we do not include country trends, instead we use the P matrix
    if not self.within_transform:
      dummies_layer = Dummies(self.N['global'], self.T, self.time_periods_na['global'], country_trends=False)
      Delta1, Delta2 = dummies_layer(input_temp)

      # Creating fixed effects
      country_FE, time_FE, self.country_FE_layer, self.time_FE_layer = create_fixed_effects(self, Delta1, Delta2)
      
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
    
    #when we use the within transformation, we do not add fixed effects
    if self.add_fe:
      output = output_tmp
    else:
      if self.dynamic_model: 
        output = Add()([country_FE, output_tmp])
      else:
        output = Add()([time_FE, country_FE, output_tmp])
      
      
     
    # tf.print(">>> Setting up Matrixize layer with T:", self.T, " and noObs['train']:", self.noObs["train"]  )
    # Creating the final output matrix with the correct dimensions
    output_matrix = Matrixize(N=self.N['global'], T=self.T, mask=self.Mask, holdout=self.holdout, n_obs_train=self.noObs["train"])(output)

    # Compiling the model
    self.model = Model(inputs=[input_temp, input_precip], outputs=output_matrix)

  

    # Counting number of parameters

    self.m = Count_params(self)

    #setting up the visualisation of the model, and the prediction model
    
    self.model_pred=prediction_model.pred_model(self)
    self.model_visual=Visual_model(self)
        
    