import numpy as np
import tensorflow as tf
import pandas as pd 


from models.model_functions.helper_functions import initialize_parameters, Preprocess, Vectorize, Create_dummies, create_fixed_effects, create_hidden_layer, create_output_layer, Count_params, Matrixize, individual_loss
from tensorflow.keras.layers import Input, Add, concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from .model_architecture import SetupGlobalModel

class MultivariateModel:
    """
    Class implementing the static neural network model.
    """

    def __init__(self, nodes, x_train, y_train, dropout, formulation, penalty):
        """
        Instantiating class.

        ARGUMENT
            * nodes:          tuple defining the model architecture.
            * x_train:        array of dicts of TxN_r dataframes of input data (aligned) with a key for each region.
            * y_train:        dict of TxN_r dataframes of target data (aligned) with a key for each region.
            * formulation:    str determining the formulation of the model. Must be one of 'global' or 'regional' or 'national'.

        NB: regions are inferred from the    keys of x_train and y_train.
        """
        
        self.nodes = nodes
        self.x_train = x_train
        self.y_train = y_train
        self.dropout = dropout
        self.formulation = formulation
        self.penalty = penalty
        self._cache = {}
        


    def _model_definition(self):
        
        # Initializing parameters
        initialize_parameters(self)
        
        # # Preprocessing data - for both precipitation and temperature        
        Preprocess(self)
        
        #note we have 2 observations for each country, one for precipitation and one for temperature, therefore the input is of dimension (T, N, 2)
        SetupGlobalModel(self)
        
    def get_model(self):
        key = tuple(self.nodes)
        if key not in self._cache:
            self._model_definition()
            self._cache[key] = self
        return self._cache[key]
      
    def fit(self, lr, min_delta, patience, verbose):
        """
        Fitting the model.

        ARGUMENTS
            * lr:            initial learning rate of the Adam optimizer.
            * min_delta:     tolerance to be used for optimization.
            * patience:      patience to be used for optimization.
            * verbose:       verbosity mode for optimization.
        """
      
       
        self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask))

  
        
        callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience=patience,
                                   restore_best_weights=True, verbose=verbose)
                
                     ]
        
        
        self.model.fit([self.input_data_temp, self.input_data_precip], self.targets, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)
        
        #metrics
        self.losses = self.model.history.history
        self.epochs = self.model.history.epoch
        self.params = self.model.get_weights()

        #fixed effects
        self.alpha = pd.DataFrame(self.country_FE_layer.weights[0].numpy().T)
        self.alpha.columns = self.individuals['global'][1:196]
        self.beta = pd.DataFrame(self.time_FE_layer.weights[0].numpy())
        self.beta.set_index(self.time_periods[self.time_periods_not_na['global']][1:196], inplace=True)
        
    def load_params(self, filepath):
        """
        Loading model parameters.

         ARGUMENTS
            * filepath: string containing path/name of saved file.
        """

        self.model.load_weights(filepath)
        self.params = self.model.get_weights()
       
        # loading fixed effects estimates
        self.alpha = pd.DataFrame(self.country_FE_layer.weights[0].numpy().T)
        self.alpha.columns = self.individuals['global'][1:self.N['global']]
        self.beta = pd.DataFrame(self.time_FE_layer.weights[0].numpy())
        self.beta.set_index(self.time_periods[self.time_periods_not_na['global']][1:self.N['global']], inplace=True)

    def save_params(self, filepath):
        """
        Saving model parameters.

         ARGUMENTS
            * filepath: string containing path/name of file to be saved.
        """

        self.model.save_weights(filepath)
        

    def in_sample_predictions(self):
        """
        Making in-sample predictions.

        """
        # Generate in-sample predictions using the model
        in_sample_preds = self.model([self.input_data_temp, self.input_data_precip])

        # Initialize aggregation variable
        MSE = 0
        

    
        # Copy the structure of the observed global data
        self.in_sample_pred['global'] = self.y_train['global'].copy()

        # Replace the copied data with the in-sample predictions
        self.in_sample_pred['global'].iloc[:, :] = np.array(in_sample_preds[0, :, 0:self.N['global']])

        # Flatten the prediction and training data to vectors
        pred_vector = np.reshape(np.array(self.in_sample_pred['global']), (-1))
        train_vector = np.reshape(np.array(self.y_train['global']), (-1))

        # Store the global predictions and actuals for comparison
        in_sample_pred_global = pred_vector
        in_sample_global = train_vector
        mean_growth = np.nanmean(np.reshape(np.array(in_sample_global), (-1)))
    
        SST = np.nansum((in_sample_global- mean_growth) ** 2)
        SSR = np.nansum((in_sample_pred_global - mean_growth) ** 2)
        SSE = np.nansum((in_sample_global - in_sample_pred_global) ** 2)

        MSE = SSE / self.noObs['global']
    
        self.BIC = (np.log(MSE))*self.noObs['global'] + self.m * np.log(self.noObs['global']) 
        self.AIC= (np.log(MSE))*self.noObs['global'] + 2 * self.m

        return in_sample_preds


        
    def predict(self, temperature_array, precip_array, idx=False):
        """
        Making predictions.

        ARGUMENTS
            * x_test:  (-1,1) array of input data.
            * idx:     Name identifying the country/region to be used for making predictions (if national or regional formulation).


        RETURNS
            * pred_df: Dataframe containing predictions.
        """
        
        pred_vector=tf.concat([temperature_array, precip_array], axis=3)

        predictions=self.model_pred.predict(pred_vector)

        return predictions