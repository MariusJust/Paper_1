import numpy as np
import tensorflow as tf
import pandas as pd 
from util import Dummies, individual_loss, Extend, Vectorize, Matrixize

from tensorflow.keras.layers import Input, Dense, Add
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal, Zeros
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.generic_utils import get_custom_objects
from tensorflow.python.keras.backend import count_params



class static_model:
    """
    Class implementing the static neural network model.
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
        self.x_train = x_train
        self.y_train = y_train
        self.formulation = formulation

        # Initializing parameters
        self._initialize_parameters()
        
        # Preprocessing data
        self._preprocess_data()
        
        # Model definition
        self._model_definition()


    def _model_definition(self):
        
        input_x = Input(shape=(self.T, int(self.N['global']))) 
        
        #reshapes the predictors (input) and dependent variable (targets) data to a tensor with dimension (1, T, N)
        self.inputs = tf.reshape(tf.convert_to_tensor(self.x_train_transf['global']), (1, self.T, self.N['global']))
        self.targets = tf.reshape(tf.convert_to_tensor(self.y_train_transf['global']), (1, self.T, self.N['global']))
        #creates a mask with dimension (1, T, N) where the mask is set to true if the data is missing
        self.Mask = tf.reshape(tf.convert_to_tensor(self.mask['global']), (1, self.T, self.N['global']))

        # Creating dummies
        Delta1, Delta2 = self._create_dummies(input_x)
        
        # Creating fixed effects
        country_FE, time_FE = self._create_fixed_effects(Delta1, Delta2)
        
    
            
        # Creating the forward pass
    
        input_first = Vectorize()(input_x)

        # First hidden layer
        self.hidden_1 = self._create_hidden_layer(self.nodes[0])
        hidden_1 = self.hidden_1(input_first)

        # Handle depth and subsequent layers
        if self.Depth > 1:
            self.hidden_2 = self._create_hidden_layer(self.nodes[1])
            hidden_2 = self.hidden_2(hidden_1)
            
            if self.Depth > 2:
                
                self.hidden_3 = self._create_hidden_layer(self.nodes[2])
                hidden_3 = self.hidden_3(hidden_2)
                input_last =  hidden_3
            else:
                input_last =  hidden_2
        else:
            input_last =  hidden_1


        # Creating temporary output layer, without fixed effects
        output_tmp = self._create_output_layer(input_last)

        # Adding fixed effects
        output = Add()([time_FE, country_FE, output_tmp])

        # Creating the final output matrix with the correct dimensions
        output_matrix = Matrixize(N=self.N['global'], T=self.T, noObs=self.noObs['global'], mask=self.Mask)(output)

        # Compiling the model
        self.model = Model(inputs=input_x, outputs=output_matrix)
        
        
        # Counting number of parameters
        
        self.m = sum(count_params(w) for w in self.model.trainable_weights)

        #setting up the prediction model
        input_x_pred = Input(shape=(1, None, 1))
        self.model_pred=self.setup_prediction_model(input_x_pred)
        
    


    def _create_dummies(self, input_x):
        """ Create and apply the dummies layer to input_x. """
        my_layer = Dummies(N=int(self.N['global']), T=self.T, time_periods_na=self.time_periods_na['global'])
        return my_layer(input_x)
    
    def _create_fixed_effects(self, Delta1, Delta2):
        """ Create country and time fixed effect layers. """
        self.country_FE_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
        self.time_FE_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=Zeros())
        country_FE = self.country_FE_layer(Delta1)
        time_FE = self.time_FE_layer(Delta2)
        return country_FE, time_FE
    
    def _create_hidden_layer(self, node):
        """ Create a hidden layer with the specified input and layer number. """
        kernel_initializer = he_normal()
        bias_initializer = Zeros()
        hidden_layer = Dense(node, activation='swish', use_bias=True,
                             kernel_initializer=kernel_initializer, bias_initializer=bias_initializer)
        return hidden_layer

    def _initialize_parameters(self):
      
        

        self.individuals = {}
        self.N = {}
        self.noObs = {}

        self.time_periods_na = {}
        self.time_periods_not_na = {}

        self.in_sample_pred = {}
        self.R2 = {}
        self.MSE = {}

        self.Min = {}
        self.Max = {}
        
        self.x_train_np = {}
        self.y_train_df = {}

        self.x_train_transf = {}
        self.y_train_transf = {}
        self.pop_train_transf = {}

        self.mask = {}

        self.losses = None
        self.epochs = None
        self.params = None
        self.BIC = None

        self.model_pred = None

        # Preparing data - getting the keys from the dictionaries, region names
        self.regions = list(self.x_train.keys())
        
        #number of regions
        self.no_regions = len(self.regions)

        #number of time periods, and the specific time periods
        self.T = self.x_train[self.regions[0]].shape[0]
        self.time_periods = self.x_train[self.regions[0]].index.values

    def _preprocess_data(self):
          #for each of the regions in the data
        for region in self.regions:
            #get the columns of the data 
            self.individuals[region] = self.x_train[region].columns
            
            #initialize the time periods not na and time periods na. This tells us, whether there is missing data in each period - MAY be deleted
            self.time_periods_not_na[region] = np.sum(~np.isnan(self.x_train[region]), axis=1) > 0
            self.time_periods_na[region] = np.sum(~self.time_periods_not_na[region])
            
            #number of countries in each
            self.N[region] = len(self.individuals[region])

            #casting data from dataframes to numpy arrays
            self.x_train_np[region] = np.array(self.x_train[region])
            self.y_train_df[region] = np.array(self.y_train[region])
            
            # Total number  of observations
            self.noObs[region] = self.N[region] * self.T - np.isnan(self.x_train_np[region]).sum()

            #Get the min, max and quantiles of the data - WHY used? 
            self.Min[region] = np.nanmin(self.x_train_np[region])
            self.Max[region] = np.nanmax(self.x_train_np[region])
           

            for individual in self.individuals[region]:
                
                #get the position of the individual in the data. IE China is the first country in the ASIA region so pos=0 
                pos = np.where(self.individuals[region] == individual)[0][0]

                #get the min, max and quantiles of the data for each individual country
                self.Min[individual] = np.nanmin(self.x_train_np[region][:, pos])
                self.Max[individual] = np.nanmax(self.x_train_np[region][:, pos])


        # Making data transformation. Start by copying the original data and cast it to numpy arrays
            self.x_train_transf[region] = np.array(self.x_train[region].copy())
            self.y_train_transf[region] = np.array(self.y_train[region].copy())
        
            #define the missing data for each region. If the data is missing, the mask is set to true
            self.mask[region] = np.isnan(self.x_train_transf[region])

            # createas a nig numpy array of the data, where data from each region is added to the data from the previous region - COULD BE IMPROVED
            if region == self.regions[0]:
                self.individuals['global'] = list(self.individuals[region])
                self.time_periods_not_na['global'] = np.sum(~np.isnan(self.x_train[region]), axis=1) > 0

                self.x_train_transf['global'] = self.x_train[self.regions[0]].copy()
                self.y_train_transf['global'] = self.y_train[self.regions[0]].copy()
        

            else:
                self.individuals['global'] = self.individuals['global'] + list(self.individuals[region])
                self.time_periods_not_na['global'] = self.time_periods_not_na['global'] | (np.sum(~np.isnan(self.x_train[region]), axis=1) > 0)

                self.x_train_transf['global'] = pd.concat([self.x_train_transf['global'], self.x_train[region]], axis=1)
                self.y_train_transf['global'] = pd.concat([self.y_train_transf['global'], self.y_train[region]], axis=1)
                

### Doing exactly the same as what have been done for the regions, but now for the global data
        self.time_periods_na['global'] = np.sum(~self.time_periods_not_na['global'])
        self.N['global'] = np.sum(list(self.N.values()))

        self.x_train_np['global'] = np.array(self.x_train_transf['global'])
        self.noObs['global'] = self.N['global'] * self.T - np.isnan(self.x_train_np['global']).sum()

        self.Min['global'] = np.nanmin(self.x_train_np['global'])
        self.Max['global'] = np.nanmax(self.x_train_np['global'])
   
        self.x_train_transf['global'] = np.array(self.x_train_transf['global'])
        self.y_train_transf['global'] = np.array(self.y_train_transf['global'] )

        self.mask['global'] = np.isnan(self.x_train_transf['global'])

    def _create_output_layer(self, input_tensor):
        """ Create the output layer with the specified input tensor. """
        kernel_initializer = he_normal()
        self.output_layer = Dense(1, activation='linear', use_bias=False, kernel_initializer=kernel_initializer)
        return self.output_layer(input_tensor)

    def count_params(self):
        
        
        ### needs to be updated count params is deprecated
        """
        Counting the number of parameters in the model.

        RETURNS
            * m: number of parameters in the model.
        """

        m = self.hidden_1.count_params()

        if self.Depth > 1:
            m = m + self.hidden_2.count_params()

            if self.Depth > 2:
                m = m + self.hidden_3.count_params()

        m = m + self.output_layer.count_params()

        return m
   
    def setup_prediction_model(self, input_x_pred):
        """
        Setting up the prediction model.

        ARGUMENTS
            * input_x_pred: input tensor for the prediction model.

        RETURNS
            * model_pred: prediction model.
        """

        hidden_1_pred = self.hidden_1(input_x_pred)

        if self.Depth > 1:
            hidden_2_pred = self.hidden_2(hidden_1_pred)

            if self.Depth > 2:
                hidden_3_pred = self.hidden_3(hidden_2_pred)
                input_last_pred = hidden_3_pred

            else:
                input_last_pred = hidden_2_pred

        else:
            input_last_pred = hidden_1_pred

        output_pred = self.output_layer(input_last_pred)

        model_pred = Model(inputs=input_x_pred, outputs=output_pred)

        return model_pred

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
        sigma2_tmp = 0
        N_agg = 0
        


# mean_observed = np.nanmean(observed)
#     SST = np.sum((observed - mean_observed) ** 2)
#     pred = pred[~np.isnan(pred)]
#     SSR = np.sum((pred - mean_observed) ** 2)
#     R2 = SSR / SST
    
    
        
    
        for region in self.regions:
            self.in_sample_pred[region] = self.y_train[region].copy()
            self.in_sample_pred[region].iloc[:, :] = np.array(in_sample_preds[0, :, N_agg:N_agg+self.N[region]])
            N_agg = N_agg + self.N[region]

            if self.regions.index(region) == 0:
                in_sample_pred_global = self.in_sample_pred[region]
                in_sample_global = self.y_train_df[region]
            else:
                in_sample_pred_global = pd.concat([in_sample_pred_global, self.in_sample_pred[region]], axis=1)
                in_sample_global = np.concatenate([in_sample_global, self.y_train_df[region]], axis=1)

            mean_tmp = np.nanmean(np.reshape(np.array(self.y_train_df[region]), (-1)))

            SSR = np.sum(np.sum((self.y_train_df[region] - self.in_sample_pred[region]) ** 2))
            SST = np.sum(np.sum((self.y_train_df[region] - mean_tmp) ** 2))
            
            
            
            
            self.R2[region] = 1 - SSR / SST
            self.MSE[region] = SSR / self.noObs[region]
            sigma2_tmp = sigma2_tmp + SSR

        mean_tmp = np.nanmean(np.reshape(np.array(in_sample_global), (-1)))

        SSR = np.sum(np.sum((in_sample_global - in_sample_pred_global)**2))
        SST = np.sum(np.sum((in_sample_global - mean_tmp)**2))
        self.R2['global'] = 1 - SSR / SST
        self.MSE['global'] = SSR / self.noObs['global']

        self.BIC = np.log(sigma2_tmp) - np.log(self.noObs['global']) + self.m * np.log(self.noObs['global']) / self.noObs['global']
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
        
        x_test_tf = tf.convert_to_tensor(x_test)
        
        pred_np = np.reshape(self.model_pred.predict(x_test_tf), (-1, 1), order='F')

        pred_df = pd.DataFrame(pred_np)
        pred_df.set_index(np.reshape(x_test, (-1,)), inplace=True)

        return pred_df