import numpy as np
import tensorflow as tf
import pandas as pd 
from .ModelFunctions_global import Create_dummies, create_fixed_effects, create_hidden_layer, create_output_layer, Count_params, individual_loss
from .ModelFunctions_Univariate import Preprocess, Matrixize, Vectorize, setup_prediction_model, initialize_parameters


from tensorflow.keras.layers import Input, Add
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping



class univariate_model:
    """
    Class implementing the univariate neural network model.
    """

    def __init__(self, nodes, x_train, y_train, formulation):
        """
        Instantiating class.

        ARGUMENT
            * nodes:          tuple defining the model architecture.
            * x_train:        array of dicts of TxN_r dataframes of input data (aligned) with a key for each region.
            * y_train:        dict of TxN_r dataframes of target data (aligned) with a key for each region.
            * formulation:    str determining the formulation of the model. Must be one of 'global' or 'regional' or 'national'.

        NB: regions are inferred from the keys of x_train and y_train.
        """
        
        self.nodes = nodes
        self.Depth = len(self.nodes)
        self.x_train = x_train[0]
        self.y_train = y_train
        self.formulation = formulation

        # Initializing parameters
        initialize_parameters(self)
        
        # Preprocessing data - for both precipitation and temperature
        Preprocess(self)
        
        # Model definition
        self._model_definition()


    def _model_definition(self):
        
        #note we have 2 observations for each country, one for precipitation and one for temperature, therefore the input is of dimension (T, N, 2)
        input_x = Input(shape=(self.T, int(self.N['global'])))
    
        # creating an input tensor of dimension (1, T, N, 2), where the first dimension is the batch size, the second dimension is the time period, the third dimension is the country, and the fourth dimension is the variable (precipitation or temperature)
        self.inputs = tf.reshape(tf.convert_to_tensor(self.x_train_transf['global']), (1, self.T, self.N['global']))
        self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf['global']), (1, self.T, self.N['global']))
        #creates a mask with dimension (1, T, N) where the mask is set to true if the data is missing
        self.Mask = tf.reshape(tf.convert_to_tensor(self.mask['global']), (1, self.T, self.N['global']))

        # Creating dummies
        Delta1, Delta2 = Create_dummies(self, input_x)
        
        # Creating fixed effects
        country_FE, time_FE = create_fixed_effects(self, Delta1, Delta2)
        
    
            
        # Creating the forward pass
    
        input_first = Vectorize()(input_x)

        # First hidden layer
        self.hidden_1 = create_hidden_layer(self, self.nodes[0])
        hidden_1 = self.hidden_1(input_first)

        # Handle depth and subsequent layers
        if self.Depth > 1:
            self.hidden_2 = create_hidden_layer(self, self.nodes[1])
            hidden_2 = self.hidden_2(hidden_1)
            
            if self.Depth > 2:
                
                self.hidden_3 = create_hidden_layer(self, self.nodes[2])
                hidden_3 = self.hidden_3(hidden_2)
                input_last =  hidden_3
            else:
                input_last =  hidden_2
        else:
            input_last =  hidden_1


        # Creating temporary output layer, without fixed effects
        output_tmp = create_output_layer(self, input_last)

        # Adding fixed effects
        output = Add()([time_FE, country_FE, output_tmp])

        # Creating the final output matrix with the correct dimensions
        output_matrix = Matrixize(N=self.N['global'], T=self.T, noObs=self.noObs['global'], mask=self.Mask)(output)

        # Compiling the model
        self.model = Model(inputs=input_x, outputs=output_matrix)
        
        
        # Counting number of parameters
        
        self.m = Count_params(self)

        #setting up the prediction model
        input_x_pred = Input(shape=(1, None, 1))
        self.model_pred=setup_prediction_model(self, input_x_pred)
        




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
                                   restore_best_weights=True, verbose=verbose)]
   
        
        self.model.fit(self.inputs, self.targets, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False)
        
        self.losses = self.model.history.history
       
        self.epochs = self.model.history.epoch

        self.params = self.model.get_weights()

        # Saving fixed effects estimates

        self.alpha = pd.DataFrame(self.country_FE_layer.weights[0].numpy().T)
        self.alpha.columns = self.individuals['global'][1:]

        self.beta = pd.DataFrame(self.time_FE_layer.weights[0].numpy())
        self.beta.set_index(self.time_periods[self.time_periods_not_na['global']][1:], inplace=True)

    def load_params(self, filepath):
        """
        Loading model parameters.

         ARGUMENTS
            * filepath: string containing path/name of saved file.
        """

        self.model.load_weights(filepath)
        self.params = self.model.get_weights()

        self.alpha = pd.DataFrame(self.country_FE_layer.weights[0].numpy().T)
        self.alpha.columns = self.individuals['global'][1:]

        self.beta = pd.DataFrame(self.time_FE_layer.weights[0].numpy())
        self.beta.set_index(self.time_periods[self.time_periods_not_na['global']][1:], inplace=True)

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

        in_sample_preds = self.model(self.inputs)
        N_agg = 0
        
        for region in self.regions:
            #copying the structure of the observed data 
            self.in_sample_pred[region] =  self.y_train[region].copy()
            #replacing the just copied data (via the iloc function) with the in-sample predictions
            self.in_sample_pred[region].iloc[:, :] = np.array(in_sample_preds[0, :, N_agg:N_agg+self.N[region]])
            
            #updates the total count of countries
            N_agg = N_agg + self.N[region]
            
            pred_vector = np.reshape(np.array(self.in_sample_pred[region]), (-1))
            train_vector = np.reshape(np.array(self.y_train_df[region]), (-1))

            if self.regions.index(region) == 0:
                in_sample_pred_global = pred_vector
                in_sample_global = train_vector
            else:
                in_sample_pred_global = np.concatenate([in_sample_pred_global, pred_vector], axis=0)
                in_sample_global = np.concatenate([in_sample_global, train_vector], axis=0)

        
        mean_growth = np.nanmean(np.reshape(np.array(in_sample_global), (-1)))
        
        SST = np.nansum((in_sample_global- mean_growth) ** 2)
        SSR = np.nansum((in_sample_pred_global - mean_growth) ** 2)
        SSE = np.nansum((in_sample_global - in_sample_pred_global) ** 2)
        
        self.R2['global'] = SSR / SST
        self.MSE['global'] = SSE / self.noObs['global']

        self.BIC = np.log(SSE) - np.log(self.noObs['global']) + self.m * np.log(self.noObs['global']) / self.noObs['global']
        return in_sample_preds


        
    def predict(self, x_test, idx=False):
        """
        Making predictions.

        ARGUMENTS
            * x_test:  (-1,1) array of input data.
            * idx:     Name identifying the country/region to be used for making predictions (if national or regional formulation).


        RETURNS
            * pred_df: Dataframe containing predictions.
        """
        
        model.x_train
        x_test_tf = tf.convert_to_tensor(x_test)
        
        pred_np = np.reshape(self.model_pred.predict(x_test_tf), (-1, 1), order='F')

        pred_df = pd.DataFrame(pred_np)
        pred_df.set_index(np.reshape(x_test, (-1,)), inplace=True)

        return