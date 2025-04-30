import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import warnings
from Models.model_functions import Prepare, load_data
from Models import MultivariateModelGlobal as Model


warnings.filterwarnings("ignore")
os.environ['PYTHONHASHSEED'] = str(0)


def main_loop(node_index, Model_selection, nodes_list, no_inits, seed_value, lr, min_delta, patience, verbose, dropout, n_splits, formulation):

    
    growth, precip, temp= load_data('IC', n_splits, formulation)
   
    # List to save model instances and performance metrics from each initialization.
    models_tmp =  np.zeros(no_inits, dtype=object)
    
    #fit a model for each key in the growth dictionary
    x_train = {'temp':temp, 'precip':precip}
    BIC_list = np.zeros(no_inits)
    AIC_list = np.zeros(no_inits)
    # Loop over each initialization
    for j in range(no_inits):
        current_seed = seed_value + j  # update seed
        tf.keras.backend.clear_session()
        tf.random.set_seed(current_seed)
        np.random.default_rng(current_seed)
        random.seed(current_seed)

        model_instance = Model(nodes=nodes_list[0], x_train=x_train, y_train=growth, dropout=dropout, formulation=formulation)
        model_instance.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)
        model_instance.in_sample_predictions()
        models_tmp[j] = model_instance

        #saves the average AIC and BIC values for each model configuration
        BIC_list[j] = model_instance.BIC
        AIC_list[j] = model_instance.AIC

        print(f"Process {os.getpid()} completed initialization {j+1}/{no_inits} (IC mode) for node {nodes_list[node_index]}", flush=True)

    # Select the best initialization based on BIC (or AIC)
    best_idx_BIC = int(np.argmin(BIC_list))
    best_idx_AIC = int(np.argmin(AIC_list))

    # Save the best model parameters for each model configuration
    models_tmp[best_idx_BIC].save_params('Model Parameters/global/BIC/' +  str(nodes_list[node_index]) + '.weights.h5')
    models_tmp[best_idx_AIC].save_params('Model Parameters/global/AIC/'  +  str(nodes_list[node_index]) + '.weights.h5')
    
    return BIC_list[best_idx_BIC], AIC_list[best_idx_AIC], nodes_list[node_index]





