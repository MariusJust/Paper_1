from Static.Model import static_model as Model 
from multiprocessing import Pool
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import warnings
# Importing libraries

from Helper import prepare




#import the python packages from utils


warnings.filterwarnings("ignore")
data=pd.read_excel('data/MainData.xlsx')

growth, precip, temp = prepare(data)



nodes_list = [(2,), (4,), (8,), (16,), (32,), (2,2,), (4,2,), (4,4,), (8,2,), (8,4,), (8,8,), (16,2,), (16,4,), (16,8,),
              (16,16,), (32,2,), (32,4,), (32,8,), (32,16,), (32,32,), (2,2,2,), (4,2,2,), (4,4,2,), (4,4,4,), (8,2,2,),
              (8,4,2,), (8,4,4,), (8,8,2), (8,8,4), (8,8,8,), (16,2,2,), (16,4,2,), (16,4,4,), (16,8,2,), (16,8,4,),
              (16,8,8,), (16,16,2,), (16,16,4,), (16,16,8,), (16,16,16,), (32,2,2,), (32,4,2,), (32,4,4,), (32,8,2,),
              (32,8,4,), (32,8,8,), (32,16,2,), (32,16,4,), (32,16,8,), (32,16,16,), (32,32,2,), (32,32,4,), (32,32,8,),
              (32,32,16,), (32,32,32,)]


# Setting optional choice parameters
no_inits = 10                    # number of different initializations
seed_value = 0                                                
lr = 0.001                        # initial learning rate for the Adam optimizer
min_delta = 1e-4                  # tolerance to be used for optimization
patience = 20                    # patience to be used for optimization
verbose = False                   # verbosity mode for optimization



            
def model(node):
    print(f"Process {os.getpid()} started working on node {nodes_list[node]}. \n")
    
    models_tmp = [None] * no_inits
    BIC_tmp = [None] * no_inits
    seed_value_tmp = seed_value
    
    for j in range(no_inits):
        seed_value_tmp = seed_value + j  # Ensure each process uses a different seed

        tf.keras.backend.clear_session()  # Clear previous models and sessions
        tf.random.set_seed(seed_value_tmp)
        np.random.default_rng(seed_value_tmp)
        random.seed(seed_value_tmp)

        models_tmp[j] = Model(nodes=nodes_list[node], x_train=temp, y_train=growth, formulation='Global')

        models_tmp[j].fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)

        models_tmp[j].in_sample_predictions()
        BIC_tmp[j] = models_tmp[j].BIC
        print(f"Process {os.getpid()} finished calculating the BIC in itteration {j} \n")
    #return the best BIC and the model architecture
    
    best_idx = np.argmin(BIC_tmp)
    return BIC_tmp[best_idx], nodes_list[node]




# Multiprocessing function
def multiprocessing_model():
    start_t = time.perf_counter()
    
    storage=dict()
    # Create a pool of worker processes
    with Pool() as pool:
        # Distribute the work across processes
        results = pool.imap_unordered(model, range(len(nodes_list)))  # Non-blocking iterator
        
        # Iterate over the results to ensure execution
        for bic, node in results:
            storage[node]=bic  
    
    end_t = time.perf_counter()
    print(f"Finished in {end_t - start_t:.2f} seconds")
    
    return storage


# Call the multiprocessing function
results=multiprocessing_model()

#save the results to a file for later use
np.save('results.npy', results)  # Save the results to a file








