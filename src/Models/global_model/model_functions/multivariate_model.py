import numpy as np
import tensorflow as tf
import pandas as pd 
import os 

from .helper_functions import initialize_parameters, Preprocess, individual_loss,  WithinHelper
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from .model_architecture import SetupGlobalModel

class MultivariateModel:
    """
    Class implementing the static neural network model.
    """

    def __init__(self, node, x_train, y_train, dropout, country_trends, dynamic_model, within_transform, x_train_val=None, y_train_val=None, y_val=None, x_val=None, holdout=0, add_fe=True):
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
        self.x_train_val = x_train_val
        self.y_train_val = y_train_val
        self.y_val = y_val
        self.x_val = x_val
        self.dropout = dropout
        self._cache = {}
        self.country_trends = country_trends
        self.dynamic_model = dynamic_model
        self.holdout = holdout
        self.within_transform = within_transform
        self.add_fe = add_fe  #whether to add fixed effects or not


    def _model_definition(self):
        
        # Initializing parameters
        initialize_parameters(self)
        
        # # Preprocessing data - for both precipitation and temperature        
        Preprocess(self)
        
        #note we have 2 observations for each country, one for precipitation and one for temperature, therefore the input is of dimension (T, N, 2)
        SetupGlobalModel(self)
        
    def get_model(self):
        tf.keras.backend.clear_session()
    
        # Build model fresh
        self._model_definition()
        return self
        
        # key = tuple(self.node)
        # if key not in self._cache:
        #     self._model_definition()
        #     self._cache[key] = self
        # return self._cache[key]
      
    def fit(self, lr, min_delta, patience, verbose):
        """
        Fitting the model.

        ARGUMENTS
            * lr:            initial learning rate of the Adam optimizer.
            * min_delta:     tolerance to be used for optimization.
            * patience:      patience to be used for optimization.
            * verbose:       verbosity mode for optimization.
        """
      
      
            
    
        if self.within_transform==True:
            #compute p matrix on whole data
            P_helper_full=WithinHelper(self.input_data_temp)
            P_matrix_full=P_helper_full.calculate_P_matrix()
            p_tensor=tf.convert_to_tensor(P_matrix_full, dtype=tf.float32)


            y_true_target_train=tf.matmul(p_tensor, tf.cast(tf.reshape(self.targets[~self.Mask], (1, -1, 1)), dtype=tf.float32))[:, :self.noObs['train'], :]
            
            if self.holdout>0:
            
                y_true_target_val=tf.matmul(p_tensor, tf.cast(tf.reshape(self.targets[~self.Mask], (1, -1, 1)), dtype=tf.float32))[:, self.noObs['train']:, :]

                n_obs_holdout= self.noObs['global'] - self.noObs['train']
            
                self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask, p_matrix=p_tensor, n_holdout=n_obs_holdout))
                
   
                # if n_obs_holdout>0:
                callbacks = [EarlyStopping(monitor='val_loss', mode='min', min_delta=min_delta, patience=patience,
                                    restore_best_weights=True, verbose=verbose)
                ]
                #validation data preprocessing
                x_train_val = [self.input_data_temp_train_val, self.input_data_precip_train_val]
                x_val = [self.input_data_temp_val, self.input_data_precip_val]
                

                
                self.model.fit(x_train_val, y_true_target_train, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False, validation_data=(x_val, y_true_target_val))
                self.holdout_loss = np.min(self.model.history.history['val_loss'])
            # else:
                   
        
            #     self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask, p_matrix=p_tensor, n_holdout=0))


            #     callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience=patience,
            #                         restore_best_weights=True, verbose=verbose)
                    
            #             ]

            #     x_train = [self.input_data_temp, self.input_data_precip]
            #     self.model.fit(x_train, y_true_target_train, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)
            
            #drop p matrix to save memory
            del p_tensor
        else:
           
            y_true_target_train= tf.reshape(self.targets[~self.Mask], (1, -1, 1))
            self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask, p_matrix=None, n_holdout=0))


            callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience=patience,
                                   restore_best_weights=True, verbose=verbose)
                
                     ]

            x_train = [self.input_data_temp, self.input_data_precip]
            self.model.fit(x_train, y_true_target_train, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)
 
        #metrics
        self.in_sample_loss = self.model.history.history['loss']
    
        self.best_weights = self.model.get_weights()
        self.epochs = self.model.history.epoch
        

        #fixed effects
        # self.alpha = pd.DataFrame(self.country_FE_layer.weights[0].numpy().T)
        # self.alpha.columns = self.individuals['global'][1:196]
        # self.beta = pd.DataFrame(self.time_FE_layer.weights[0].numpy())
        # self.beta.set_index(self.time_periods[self.time_periods_not_na['global']][1:196], inplace=True)
        
    def load_params(self, filepath):
        """
        Loading model parameters.

         ARGUMENTS
            * filepath: string containing path/name of saved file.
        """

        self.model.load_weights(filepath)
        self.params = self.model.get_weights()
        if self.within_transform:
            pass
        else:
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