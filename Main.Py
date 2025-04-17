from setup import multiprocessing_model
import numpy as np
   
def main():
    
    # List of node configurations to try:
    nodes_list = [
        (2,), (4,), (8,), (16,), (32,), (2,2,), (4,2,), (4,4,), (8,2,), (8,4,),
        (8,8,), (16,2,), (16,4,), (16,8,), (16,16,), (32,2,), (32,4,), (32,8,),
        (32,16,), (32,32,), (2,2,2,), (4,2,2,), (4,4,2,), (4,4,4,), (8,2,2,),
        (8,4,4,), (8,8,2), (8,8,4), (8,8,8,), (16,2,2,), (16,4,2,), (16,4,4,),
        (16,8,2,), (16,8,4,), (16,8,8,), (16,16,2,), (16,16,4,), (16,16,8,),
        (16,16,16,), (32,2,2,), (32,4,2,), (32,4,4,), (32,8,2,), (32,8,4,),
        (32,8,8,), (32,16,2,), (32,16,4,), (32,16,8,), (32,16,16,), (32,32,2,),
        (32,32,4,), (32,32,8,), (32,32,16,), (32,32,32,)
    ]


    
    no_inits = 12                   # Number of different initializations per node
    seed_value = 0                  # Seed value for random number generation                                      
    lr = 0.001                      # Learning rate
    min_delta = 1e-6                # Tolerance for optimization
    patience = 50                   # Patience for early stopping
    verbose = 0                     # Verbosity mode for optimization
    dropout = 0.2                   # Dropout rate (0 means no dropout)
    n_process = 55                  # Number of cores to run on
    Model_selection= 'CV'           # Model selection, "IC" for information criteria, "CV" for cross-validation
    specification = 'global'        # Model specification, "global" or "regional"
    n_splits = 5                    # Number of splits for cross-validation, if Model_selection = IC, argument is ignored

    results = multiprocessing_model(Model_selection, nodes_list, no_inits, seed_value, lr, min_delta, patience, verbose, dropout, n_splits, n_process)
    np.save('results/results.npy', results)

    return results

if __name__ == '__main__':
    main()
