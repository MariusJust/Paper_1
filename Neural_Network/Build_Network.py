import numpy as np
import tensorflow as tf
import pandas as pd 
import logging

from .Models.ModelFunctions import initialize_parameters, Preprocess, Vectorize, Create_dummies, create_fixed_effects, create_hidden_layer, create_output_layer, Count_params, Matrixize, individual_loss, model_with_dropout, model_without_dropout, prediction_model_with_dropout, prediction_model_without_dropout
from tensorflow.keras.layers import Input, Add, concatenate
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
from .Models.setup_regional_model import Regions

class multivariate_model:
    """
    Class implementing the static neural network model.
    """

    def __init__(self, nodes, x_train, y_train, dropout, formulation):
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
        self.Depth = len(self.nodes)
        self.x_train = x_train
        self.y_train = y_train
        self.dropout = dropout
        self.formulation = formulation
        
        
        # Initializing parameters
        initialize_parameters(self)
        
        # # Preprocessing data - for both precipitation and temperature        
        Preprocess(self)
        
    #     # Model definition
        self._model_definition()


    def _model_definition(self):
        
        #note we have 2 observations for each country, one for precipitation and one for temperature, therefore the input is of dimension (T, N, 2)
        
        if self.formulation == 'global':
            setup_global_model(self)
        elif self.formulation == 'regional':
            Regions.Setup_regional_model(self)
        else:
            raise ValueError("Invalid formulation. Use 'global' or 'regional'.")
        
     

    def fit(self, lr, min_delta, patience, verbose):
        """
        Fitting the model.

        ARGUMENTS
            * lr:            initial learning rate of the Adam optimizer.
            * min_delta:     tolerance to be used for optimization.
            * patience:      patience to be used for optimization.
            * verbose:       verbosity mode for optimization.
        """
      
        if self.formulation == 'global':
            self.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=self.Mask))

        elif self.formulation == 'regional':
             self.model.compile(optimizer=Adam(lr), loss=self.loss_list, loss_weights=[1 / self.no_regions] * self.no_regions)
            
        
        callbacks = [EarlyStopping(monitor='loss', mode='min', min_delta=min_delta, patience=patience,
                                   restore_best_weights=True, verbose=verbose)
                    #  ,tensorboard_callback
                     ]
        
        
        # for b in self.region_builders:
        #     logging.DEBUG(f"[{b.region}] T={b.T}, N_region={b.N_region}, noObs={b.noObs}")
           
        self.model.fit([self.input_data_temp, self.input_data_precip], self.targets, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)
        
        self.losses = self.model.history.history
       
        self.epochs = self.model.history.epoch

        self.params = self.model.get_weights()


        if self.formulation == 'global':
            # Saving fixed effects estimates
            self.alpha = pd.DataFrame(self.country_FE_layer.weights[0].numpy().T)
            self.alpha.columns = self.individuals['global'][1:196]

            self.beta = pd.DataFrame(self.time_FE_layer.weights[0].numpy())
            self.beta.set_index(self.time_periods[self.time_periods_not_na['global']][1:196], inplace=True)
            
        elif self.formulation == 'regional':
            for idx, region in enumerate(self.regions): 
                self.alpha[region] = pd.DataFrame(self.country_FE_layer[idx].weights[0].numpy().T)
                self.alpha[region].columns = self.individuals[region][1:]

                self.beta[region] = pd.DataFrame(self.time_FE_layer[idx].weights[0].numpy())
                self.beta[region].set_index(self.time_periods[self.time_periods_na[region]+1:], inplace=True)
    
    def load_params(self, filepath):
        """
        Loading model parameters.

         ARGUMENTS
            * filepath: string containing path/name of saved file.
        """
        self.model.load_weights(filepath)
        self.params = self.model.get_weights()
        if self.formulation == 'regional':
            for i, region in enumerate(self.regions):
                self.alpha[self.regions[i]] = pd.DataFrame(self.country_FE_layer[i].weights[0].numpy().T)
                self.alpha[self.regions[i]].columns = self.individuals[self.regions[i]][1:]

                self.beta[self.regions[i]] = pd.DataFrame(self.time_FE_layer[i].weights[0].numpy())
                self.beta[self.regions[i]].set_index(self.time_periods[self.time_periods_na[self.regions[i]] + 1:], inplace=True)

        else:
    
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
        sigma2_tmp = 0
        noObs_tmp = 0
        N_agg = 0
        country_counter = 0
        MSE = 0
        
        if self.formulation=='global':
        
            # Copy the structure of the observed global data
            self.in_sample_pred['global'] = self.y_train['global'].copy()

            # Replace the copied data with the in-sample predictions
            self.in_sample_pred['global'].iloc[:, :] = np.array(in_sample_preds[0, :, 0:self.N['global']])

            # Flatten the prediction and training data to vectors
            pred_vector = np.reshape(np.array(self.in_sample_pred['global']), (-1))
            train_vector = np.reshape(np.array(self.y_train_df['global']), (-1))

            # Store the global predictions and actuals for comparison
            in_sample_pred_global = pred_vector
            in_sample_global = train_vector
            mean_growth = np.nanmean(np.reshape(np.array(in_sample_global), (-1)))
        
            SST = np.nansum((in_sample_global- mean_growth) ** 2)
            SSR = np.nansum((in_sample_pred_global - mean_growth) ** 2)
            SSE = np.nansum((in_sample_global - in_sample_pred_global) ** 2)

            self.R2[region] = SSR / SST
            MSE = SSE / self.noObs['global']
            
        elif self.formulation == 'regional':
            for region in self.regions:
                self.in_sample_pred[region] = self.y_train[region].copy()
                self.in_sample_pred[region].iloc[:, :] = np.array(in_sample_preds[self.regions.index(region)][0, :, :])

                if self.regions.index(region) == 0:
                    in_sample_pred_global = self.in_sample_pred[region]
                    in_sample_global = self.y_train_df[region]
                else:
                    in_sample_pred_global = pd.concat([in_sample_pred_global, self.in_sample_pred[region]], axis=1)
                    in_sample_global = pd.concat([in_sample_global, self.y_train_df[region]], axis=1)
                    
                noObs_tmp = noObs_tmp + self.noObs[region]
                mean_growth = np.nanmean(np.reshape(np.array(in_sample_global), (-1)))
        
                SST = np.nansum((in_sample_global- mean_growth) ** 2)
                SSR = np.nansum((in_sample_pred_global - mean_growth) ** 2)
                SSE = np.nansum((in_sample_global - in_sample_pred_global) ** 2)

                self.R2[region] = SSR / SST
                MSE = MSE + SSE / self.noObs[region]

        if self.formulation == 'global':
            self.BIC = (np.log(MSE))*self.noObs['global'] + self.m * np.log(self.noObs['global']) 
            self.AIC= (np.log(MSE))*self.noObs['global'] + 2 * self.m
        elif self.formulation == 'regional':    
            self.BIC = np.log(MSE)*noObs_tmp + self.m * np.log(noObs_tmp)
            self.AIC = np.log(MSE)*noObs_tmp + 2 * self.m
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