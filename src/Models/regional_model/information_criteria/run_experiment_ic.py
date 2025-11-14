import os
import numpy as np
import tensorflow as tf
import random
from utils.miscelaneous.warnings import turn_off_warnings
from models.regional_model.model_functions.helper_functions import load_data
from models import MultivariateModelRegional as Model
from datetime import datetime

turn_off_warnings()

class MainLoop:
    def __init__(self, parent, node):
        
        self.cfg=parent.cfg
        self.data=parent.data
        self.node= node
        
        #build a factory for the model, so we don't have to re-initialize the model each time
        self.factory = Model(
            node=None, 
            cfg=self.cfg,
            x_train=None,     
            y_train=None,
            x_train_val=None,
            y_train_val=None,
            x_val=None,
            y_val=None
        )

        # Load data
        if self.data is not None: #ie we are running a Monte Carlo experiment
            from simulations.simulation_functions import Pivot
            self.growth, self.precip, self.temp = Pivot(self.data)
        else:   
            self.growth, self.precip, self.temp = load_data('IC', self.cfg.n_countries, self.cfg.time_periods)
   
   
    def run_experiment(self):  
         
        self.setup_model_params()
        
        #loop over initializations
        for j in range(self.cfg.no_inits):
        
            current_seed = self.cfg.seed_value + j  # update seed
            tf.random.set_seed(current_seed)
            np.random.default_rng(current_seed)
            random.seed(current_seed)

            
            model_instance=self.factory.get_model()
            model_instance.fit(lr=self.cfg.lr, min_delta=self.cfg.min_delta, patience=self.cfg.patience, verbose=self.cfg.verbose)
         
            if self.cfg.holdout>0:
                self.models_tmp[j] = model_instance
                self.holdout_MSE[j] = model_instance.holdout_loss
            else:
                model_instance.in_sample_predictions()
                self.models_tmp[j] = model_instance

                #saves the information criteria
                self.BIC_list[j] = model_instance.BIC
                self.AIC_list[j] = model_instance.AIC
                    
        # Select the best initialization based on BIC (or AIC)
        best_idx_BIC = int(np.argmin(self.BIC_list))
        best_idx_AIC = int(np.argmin(self.AIC_list))
        best_idx_holdout = int(np.argmin(self.holdout_MSE))
        
        
        # # retrain the best model on the full data (train + val)
        # if self.holdout > 0:
            
        #     if hasattr(self.factory, '_cache'):
        #         try:
        #             self.factory._cache.clear()
        #         except Exception:
        #             self.factory._cache = {}
                
        #     self.factory.x_train = {0: self.temp, 1: self.precip}
        #     self.factory.y_train = self.growth
        #     self.factory.x_val = None
        #     self.factory.y_val = None
        #     self.factory.node = self.node
        #     self.factory.holdout=0
            
        #     tf.random.set_seed(self.seed_value + best_idx_holdout)
        #     np.random.default_rng(self.seed_value + best_idx_holdout)
        #     random.seed(self.seed_value + best_idx_holdout)

         
                
        #     best_model=self.factory.get_model()

        #     best_model.fit(lr=self.lr, min_delta=self.min_delta, patience=self.patience, verbose=self.verbose)
    
        #     self.models_tmp[best_idx_holdout] = best_model
        
        
        #only save the model parameters if the data is the real data, and not simulated data
        if self.data is None:
            
            # Create directory if it doesn't exist
            path=f"results/Model Parameters/IC/{datetime.today().strftime('%Y-%m-%d')}/{self.node}.weights.h5"
            dir_path = os.path.dirname(path)
            os.makedirs(dir_path, exist_ok=True)

            self.models_tmp[best_idx_holdout].save_params(path)
            return self.holdout_MSE[best_idx_holdout], self.BIC_list[best_idx_BIC], self.AIC_list[best_idx_AIC], self.node
        else: #Monte carlo simulation
            best_surface=self.models_tmp[best_idx_BIC].model_visual
            country_FE = self.models_tmp[best_idx_BIC].alpha
            return self.holdout_MSE[best_idx_holdout], self.BIC_list[best_idx_BIC], self.AIC_list[best_idx_AIC], self.node, best_surface, country_FE 

    def setup_model_params(self):
         #inistialize params
        if self.cfg.holdout > 0:
            self.factory.x_train = {0: self.temp, 1: self.precip}
            self.factory.y_train = self.growth
            
            temp_train_val = {key: df.iloc[:-self.cfg.holdout, :] for key, df in self.temp.items()}
            temp_val = {key: df.iloc[-self.cfg.holdout:, :] for key, df in self.temp.items()}
            precip_train_val = {key: df.iloc[:-self.cfg.holdout, :] for key, df in self.precip.items()}
            precip_val = {key: df.iloc[-self.cfg.holdout:, :] for key, df in self.precip.items()}
            growth_train_val = {key: df.iloc[:-self.cfg.holdout, :] for key, df in self.growth.items()}
            growth_val = {key: df.iloc[-self.cfg.holdout:, :] for key, df in self.growth.items()}
            
            self.factory.x_train_val = {0: temp_train_val, 1: precip_train_val}
            self.factory.y_train_val = growth_train_val
            self.factory.x_val = {0: temp_val, 1: precip_val}
            self.factory.y_val = growth_val
            self.factory.add_fe = False
            
        else:
            self.factory.x_train = {0: self.temp, 1: self.precip}
            self.factory.y_train = self.growth
            
        self.factory.node = self.node
