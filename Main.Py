
from multiprocessing import Pool
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import warnings
from tqdm import tqdm
from Static.ModelFunctions_global import prepare

warnings.filterwarnings("ignore")

specification='Multivariate'

#choose whether it is univariate or multivariate

if specification=='univariate':
    from Static import univariate_model as Model
    
else: 
    from Static import multivariate_model as Model
    
     
nodes_list = [(2,), (4,), (8,), (16,), (32,), (2,2,), (4,2,), (4,4,), (8,2,), (8,4,), (8,8,), (16,2,), (16,4,), (16,8,),
              (16,16,), (32,2,), (32,4,), (32,8,), (32,16,), (32,32,), (2,2,2,), (4,2,2,), (4,4,2,), (4,4,4,), (8,2,2,),
                (8,4,4,), (8,8,2), (8,8,4), (8,8,8,), (16,2,2,), (16,4,2,), (16,4,4,), (16,8,2,), (16,8,4,),
              (16,8,8,), (16,16,2,), (16,16,4,), (16,16,8,), (16,16,16,), (32,2,2,), (32,4,2,), (32,4,4,), (32,8,2,),
              (32,8,4,), (32,8,8,), (32,16,2,), (32,16,4,), (32,16,8,), (32,16,16,), (32,32,2,), (32,32,4,), (32,32,8,),
              (32,32,16,), (32,32,32,)]


no_inits = 20                  # number of different initializations
seed_value = 0                                                
lr = 0.001                        # initial learning rate for the Adam optimizer
min_delta = 1e-6                  # tolerance to be used for optimization
patience = 50                    # patience to be used for optimization
verbose = 0                   # verbosity mode for optimization


def setup():
    # Importing libraries

    
    data=pd.read_excel('data/MainData.xlsx')

    growth, precip, temp = prepare.Prepare(data)
        
    return growth, precip, temp


def model(node):
    
    growth, precip, temp = setup()
    
    if specification=='Univariate':
         x_train=temp
    else:
         x_train=[temp, precip]
        
    
    models_tmp = [None] * no_inits
    BIC_tmp = [None] * no_inits
    AIC_tmp = [None] * no_inits
    seed_value_tmp = seed_value
    
    
    for j in range(no_inits):
        seed_value_tmp =  seed_value_tmp + 1  # Ensure each process uses a different seed
        
        tf.keras.backend.clear_session()  # Clear previous models and sessions
        tf.random.set_seed(seed_value_tmp)
        np.random.default_rng(seed_value_tmp)
        random.seed(seed_value_tmp)

        models_tmp[j]= Model(nodes=nodes_list[node], x_train=x_train, y_train=growth)
        
        models_tmp[j].fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)
  
        models_tmp[j].in_sample_predictions()
        BIC_tmp[j] = models_tmp[j].BIC
        AIC_tmp[j] = models_tmp[j].AIC
      
        print(f"Process {os.getpid()} completed initialization {j+1}/{no_inits} for node {nodes_list[node]}", flush=True)

          
    #saving BIC for all initializations
    np.save('Model Selection/BIC/' + str(nodes_list[node]) + '.npy' , BIC_tmp)
    np.save('Model Selection/AIC/' + str(nodes_list[node]) + '.npy' , AIC_tmp)
    
    #Finding the best initialisation meassured by BIC
    best_idx_BIC = np.argmin(BIC_tmp)
    best_idx_AIC= np.argmin(AIC_tmp)
    
    #saving the best models parameters
    models_tmp[best_idx_BIC].save_params('Model Parameters/BIC/' + str(nodes_list[node])+ '.weights.h5')
    models_tmp[best_idx_AIC].save_params('Model Parameters/AIC/' + str(nodes_list[node])+ '.weights.h5')
    
    return BIC_tmp[best_idx_BIC], AIC_tmp[best_idx_AIC], nodes_list[node]





# Multiprocesssing function
def multiprocessing_model():

    storage=dict()
    # Create a pool of worker processes
    with Pool(55) as pool:
        # Distribute the work across processes
        results = pool.imap_unordered(model, range(len(nodes_list)))  # Non-blocking iterator
        
        # Iterate over the results to ensure execution
        for bic, aic, node in tqdm(results, total=len(nodes_list), desc="Processing nodes", unit="node"):
            storage[node] = [bic, aic]
     
    return storage




def main(): 
    
    results = multiprocessing_model()
    
    np.save('results/results.npy', results) 
    
    return results
    

main()


