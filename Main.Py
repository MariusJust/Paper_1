
from multiprocessing import Pool
import time
import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import warnings
from tqdm import tqdm
from Static.ModelFunctions_global import prepare

specification='Multivariate'

#choose whether it is univariate or multivariate

if specification=='univariate':
    from Static import univariate_model as Model
    
else: 
    from Static import multivariate_model as Model
    
     
# nodes_list = [(2,), (4,), (8,), (16,), (32,), (2,2,), (4,2,), (4,4,), (8,2,), (8,4,), (8,8,), (16,2,), (16,4,), (16,8,),
#               (16,16,), (32,2,), (32,4,), (32,8,), (32,16,), (32,32,), (2,2,2,), (4,2,2,), (4,4,2,), (4,4,4,), (8,2,2,),
#               (8,4,2,), (8,4,4,), (8,8,2), (8,8,4), (8,8,8,), (16,2,2,), (16,4,2,), (16,4,4,), (16,8,2,), (16,8,4,),
#               (16,8,8,), (16,16,2,), (16,16,4,), (16,16,8,), (16,16,16,), (32,2,2,), (32,4,2,), (32,4,4,), (32,8,2,),
#               (32,8,4,), (32,8,8,), (32,16,2,), (32,16,4,), (32,16,8,), (32,16,16,), (32,32,2,), (32,32,4,), (32,32,8,),
#               (32,32,16,), (32,32,32,)]

nodes_list =[(2,)]

no_inits = 1                    # number of different initializations
seed_value = 0                                                
lr = 0.001                        # initial learning rate for the Adam optimizer
min_delta = 1e-2                  # tolerance to be used for optimization
patience = 10                    # patience to be used for optimization
verbose = 2                   # verbosity mode for optimization


def setup():
    # Importing libraries

    warnings.filterwarnings("ignore")
    data=pd.read_excel('data/MainData.xlsx')

    growth, precip, temp = prepare.Prepare(data)

    return growth, precip, temp


def model(node):

    
    growth, precip, temp = setup()
    
    if specification=='Univariate':
         x_train=temp
    else:
         x_train=[temp, precip]
        
    
    print(f"Process {os.getpid()} started working on node {nodes_list[node]}. \n")
    
    models_tmp = [None] * no_inits
    BIC_tmp = [None] * no_inits
    R2_tmp = [None] * no_inits
    seed_value_tmp = seed_value
    
    for j in range(no_inits):
        seed_value_tmp =  seed_value_tmp + 1  # Ensure each process uses a different seed
        
        tf.keras.backend.clear_session()  # Clear previous models and sessions
        tf.random.set_seed(seed_value_tmp)
        np.random.default_rng(seed_value_tmp)
        random.seed(seed_value_tmp)

        models_tmp[j] = Model(nodes=nodes_list[0], x_train=x_train, y_train=growth, formulation='Global')

        models_tmp[j].fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)

        models_tmp[j].in_sample_predictions()
        BIC_tmp[j] = models_tmp[j].BIC
       
        print(f"Process {os.getpid()} finished working on node {nodes_list[node]}. \n")
        
        
    #saving BIC for all initializations
    np.save('BIC/' + str(nodes_list[node]) + '.npy' , BIC_tmp)
    
    #Finding the best initialisation meassured by BIC
    best_idx = np.argmin(BIC_tmp)
    
    #saving the best models parameters
    models_tmp[best_idx].save_params('Model Parameters/' + str(nodes_list[node])+ '.weights.h5')
    
    return BIC_tmp[best_idx], nodes_list[node]

for node in enumerate(nodes_list):
    model(node[0])



# # Multiprocesssing function
# def multiprocessing_model():
#     start_t = time.perf_counter()
    
#     storage=dict()
#     # Create a pool of worker processes
#     with Pool(58) as pool:
#         # Distribute the work across processes
#         results = pool.imap_unordered(model, range(len(nodes_list)))  # Non-blocking iterator
        
#         # Iterate over the results to ensure execution
#         for bic, node in tqdm(results, total=len(nodes_list), desc="Processing nodes", unit="node"):
#             storage[node] = bic
    
#     end_t = time.perf_counter()
#     print(f"Finished in {end_t - start_t:.2f} seconds")
    
#     return storage




# def main(): 
    
#     results = multiprocessing_model()
    
#     np.save('results.npy', results) 
    
#     return results
    

# main()






