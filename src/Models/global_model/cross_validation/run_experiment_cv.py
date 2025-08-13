import numpy as np
import tensorflow as tf
import random
from utils.miscelaneous.warnings import turn_off_warnings
from models.global_model.model_functions.helper_functions import load_data
from models import MultivariateModelGlobal as Model
from models.global_model.cross_validation import train_itteration, test_itteration, train_on_full_sample



class MainLoop:
    """
    Class implementing the cross-validation loop for model training and evaluation.
    """

    def __init__(self, node, no_inits, seed_value, lr, min_delta, patience, verbose, dropout, n_splits, cv_approach, penalty, n_countries, time_periods, data=None):
         
        turn_off_warnings()

        if data is not None: #ie we are running a Monte Carlo experiment
            from simulations.simulation_functions.simulate_data import Pivot
            growth, precip, temp = Pivot(data)
            panel_split=load_data('CV', n_countries, time_periods, n_splits=n_splits, growth=growth)
        else:   
            growth, precip, temp, panel_split = load_data('CV', n_countries, time_periods, n_splits=n_splits)
   
   
    
        self.models_tmp =  np.zeros(no_inits, dtype=object)
        self.cv_errors_inits = np.zeros(no_inits)
        self.node= node
        self.no_inits = no_inits
        self.seed_value = seed_value
        self.lr = lr
        self.min_delta = min_delta
        self.patience = patience
        self.verbose = verbose
        self.dropout = dropout
        self.n_splits = n_splits
        self.growth = growth
        self.temp = temp
        self.precip = precip
        self.panel_split = panel_split
        self.cv_approach = cv_approach
        self.penalty = penalty
        self.data = data
    
       # make a factory object to store pycache and avoid re-initializing the model each time
        self.factory = Model(
                node=None,     
                x_train=None,   
                y_train=None,
                dropout=self.dropout,
                penalty=self.penalty
            )
     
      
    def run_experiment(self):   
        """
        Main loop for running the cross-validation experiment. 
        Runs either "full" cross validation or "last period" cross validation based on the specified approach 
        returns the best MSE based on the cross-validation error
        """
        for j in range(self.no_inits):
            self.current_seed = self.seed_value + j  
            
            if self.cv_approach=='full':
                self.cv_full(j)
             
            elif self.cv_approach=='last_period':
                self.cv_last_period(j)
            else:
                raise ValueError("cv_approach must be either 'full' or 'last_period'")
            
        self.best_init_idx = int(np.argmin(self.cv_errors_inits))
        self.best_cv_error = self.cv_errors_inits[self.best_init_idx]
        
        #retrain the model on the full dataset using the best initialization
      
        train_on_full_sample(self)
        
        return self.best_cv_error, self.node, self.model_full.model_visual, self.model_full.beta
            
    def cv_full(self, j):
        
        """
        cross validation that does the following:
           # 1. fit on the full dataset and save the fixed effects
           # 2. for each test split, predict using the fixed effects estimate from step 1 and the model trained on the training set
        """
        
        #1 - training on the full dataset and saving the fixed effects
        train_idx = np.array([True] * len(self.growth['global']))
        
        model_full = train_itteration(self, train_idx)
        self.country_FE = model_full.alpha
        self.time_FE = model_full.beta
        
        # 2. - for each split, train the model on the training set. This gives you an estimated neural network. 
        # Take this and the fixed effects from the full training set and use them to predict on the test set.
        errors_j = []  
        for train_idx, test_idx in self.panel_split.split():
                
                self.models_tmp[j]= train_itteration(self, train_idx)
                
                self.model_instance = self.models_tmp[j]
                
                country_FE = self.country_FE
                test_int=np.where(test_idx==True)[0][0]
                time_FE = np.array(self.time_FE)[test_int-1]
                
                mse=test_itteration(self, test_idx, country_FE, time_FE)
                
                errors_j.append(mse)
                
        
        self.cv_errors_inits[j] = np.mean(errors_j)
            
        return None
    

    def cv_last_period(self, j):
         
        """
        cross validation that does the following:
         #train on the first T-1 periods and save the fixed effects from period T-1
         #for each test split (period T), predict using the fixed effects estimate from period T-1 and the model trained on the training set
        """
      
        errors_j = []
        for train_idx, test_idx in self.panel_split.split():
                
                self.models_tmp[j]=train_itteration(self, train_idx)
                
                country_FE = self.models_tmp[j].alpha
                time_FE = np.array(self.models_tmp[j].beta)[-1]
                self.model_instance = self.models_tmp[j]
                
                mse=test_itteration(self, test_idx, country_FE, time_FE)
                
                errors_j.append(mse)
                
        self.cv_errors_inits[j] = np.mean(errors_j)
        
        return None


 