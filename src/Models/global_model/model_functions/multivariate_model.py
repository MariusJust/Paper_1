import numpy as np
import tensorflow as tf
import pandas as pd 

from .helper_functions import initialize_parameters, Preprocess, individual_loss, HoldoutMonitor, WithinHelper
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from .model_architecture import SetupGlobalModel

class MultivariateModel:
    """
    Class implementing the static neural network model.
    """

    def __init__(self, node, x_train, y_train, dropout, penalty, country_trends, dynamic_model, within_transform, y_val=None, x_val=None, holdout=0):
        """
        Instantiating class.

        ARGUMENT
            * node:          tuple defining the model architecture.
            * x_train:        array of dicts of TxN_r dataframes of input data (aligned) with a key for each region.
            * y_train:        dict of TxN_r dataframes of target data (aligned) with a key for each region.
            * formulation:    str determining the formulation of the model. Must be one of 'global' or 'regional' or 'national'.

        NB: regions are inferred from the    keys of x_train and y_train.
        """
        
        self.node = node
        self.x_train = x_train
        self.y_train = y_train
        self.dropout = dropout
        self.penalty = penalty
        self._cache = {}
        self.country_trends = country_trends
        self.dynamic_model = dynamic_model
        self.y_val = y_val
        self.x_val = x_val
        self.holdout = holdout
        self.within_transform = within_transform


    def _model_definition(self):
        
        # Initializing parameters
        initialize_parameters(self)
        
        # # Preprocessing data - for both precipitation and temperature        
        Preprocess(self)
        
        #note we have 2 observations for each country, one for precipitation and one for temperature, therefore the input is of dimension (T, N, 2)
        SetupGlobalModel(self)
        
    def get_model(self):
        key = tuple(self.node)
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
      
        # if self.within_transform:
        #     self.P_matrix = WithinHelper(self.input_data_temp).calculate_P_matrix()
        #     self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask, P_matrix=self.P_matrix))
        # else:   
            
            
        self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask))


        
        callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience=patience,
                                   restore_best_weights=True, verbose=verbose)
                
                     ]
  
        if self.holdout>0 and self.within_transform==True:
            if self.x_val is not None and self.y_val is not None:
                
                x_train = [self.input_data_temp, self.input_data_precip]

                P_train = WithinHelper(self.input_data_temp).calculate_P_matrix()
                P_val = WithinHelper(self.input_data_temp_val).calculate_P_matrix()
                
                # preprojecting the validation and training targets to save memory during training
                self.y_val_transf = tf.matmul(P_val, self.targets_val)

                holdout_callback=HoldoutMonitor(self, patience=patience, min_delta=min_delta, verbose=verbose, P_matrix_train=P_train, P_matrix_val=P_val)
                callbacks = [holdout_callback]
                
                
                self.model.fit(x_train, self.targets, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)
                # self.holdout_loss = np.min(self.model.history.history['holdout_mse'])
            else:
                x_train = [self.input_data_temp, self.input_data_precip]
                self.model.fit(x_train, self.targets, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)
                
        else:
            x_train = [self.input_data_temp, self.input_data_precip]
            self.model.fit(x_train, self.targets, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)
           
        #metrics
        self.in_sample_loss = self.model.history.history['loss']
        
        
        self.best_weights = self.model.get_weights()
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