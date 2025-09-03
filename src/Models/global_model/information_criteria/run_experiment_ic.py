import os
import numpy as np
import tensorflow as tf
import random
from utils.miscelaneous.warnings import turn_off_warnings
from models.global_model.model_functions.helper_functions import load_data
from models import MultivariateModelGlobal as Model
import datetime

turn_off_warnings()

class MainLoop:
    def __init__(self, node, no_inits, seed_value, lr, min_delta, patience, verbose, dropout, n_countries, time_periods, penalty, country_trends, dynamic_model, data=None):
        self.node = node
        self.no_inits = no_inits
        self.seed_value = seed_value
        self.lr = lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.dropout = dropout
        self.penalty = penalty
        self.data = data
        self.models_tmp = np.zeros(no_inits, dtype=object)
        self.BIC_list = np.zeros(no_inits)
        self.AIC_list = np.zeros(no_inits)
        self.country_trends = country_trends
        self.dynamic_model = dynamic_model
        
        #build a factory for the model, so we don't have to re-initialize the model each time
        self.factory = Model(
            node=None, 
            x_train=None,     
            y_train=None,
            dropout=self.dropout,
            penalty=self.penalty,
            country_trends=self.country_trends,
            dynamic_model=self.dynamic_model
        )
        
        
        # Load data
        if data is not None: #ie we are running a Monte Carlo experiment
            from simulations.simulation_functions import Pivot
            self.growth, self.precip, self.temp = Pivot(data)
        else:   
            self.growth, self.precip, self.temp = load_data('IC', n_countries, time_periods)
   
   
    def run_experiment(self):   
        #pass model inputs to the factory
        self.factory.x_train = {0:self.temp, 1:self.precip}
        self.factory.y_train = self.growth
        self.factory.node = self.node
        
        #loop over initializations
        for j in range(self.no_inits):
        
            current_seed = self.seed_value + j  # update seed
            tf.random.set_seed(current_seed)
            np.random.default_rng(current_seed)
            random.seed(current_seed)

            
            model_instance=self.factory.get_model()
            model_instance.fit(lr=self.lr, min_delta=self.min_delta, patience=self.patience, verbose=self.verbose)
            model_instance.in_sample_predictions()
            self.models_tmp[j] = model_instance

            #saves the average AIC and BIC values for each model configuration
            self.BIC_list[j] = model_instance.BIC
            self.AIC_list[j] = model_instance.AIC

            print(f"Process {os.getpid()} completed initialization {j+1}/{self.no_inits} (IC mode) for node {self.node}", flush=True)

        # Select the best initialization based on BIC (or AIC)
        best_idx_BIC = int(np.argmin(self.BIC_list))
        best_idx_AIC = int(np.argmin(self.AIC_list))

        #only save the model parameters if the data is the real data, and not simulated data
        if self.data is None:
            self.models_tmp[best_idx_BIC].save_params('results/Model Parameters/IC/' + datetime.today().strftime('%Y-%m-%d') + '/' + str(self.node) + '.weights.h5')
            return self.BIC_list[best_idx_BIC], self.AIC_list[best_idx_AIC], self.node
        else: #Monte carlo simulation
            best_surface=self.models_tmp[best_idx_BIC].model_visual
            country_FE = self.models_tmp[best_idx_BIC].alpha
            return self.BIC_list[best_idx_BIC], self.AIC_list[best_idx_AIC], self.node, best_surface, country_FE 


