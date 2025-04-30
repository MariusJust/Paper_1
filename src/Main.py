# from utils import multiprocessing_model
from ..config.logging_config import setup_logging
import numpy as np
import hydra
from omegaconf import DictConfig, OmegaConf
import ast

@hydra.main(version_base=None, config_path="../config", config_name="config")
def main(cfg: DictConfig):
    
    print(OmegaConf.to_yaml(cfg))                      # :contentReference[oaicite:4]{index=4}

    # Access nested values
    print("Learning rate:", cfg.instance.lr)           # :contentReference[oaicite:5]{index=5}

    # Convert node_list strings back to actual tuples
    raw = cfg.instance.node_list                       # list of strings
    node_list = [ast.literal_eval(s) for s in raw]
    print("Parsed node list:", node_list)
    
    # List of node configurations to try:
    # nodes_list = [
    #     (2,), (4,), (8,), (16,), (32,), (2,2,), (4,2,), (4,4,), (8,2,), (8,4,),
    #     (8,8,), (16,2,), (16,4,), (16,8,), (16,16,), (32,2,), (32,4,), (32,8,),
    #     (32,16,), (32,32,), (2,2,2,), (4,2,2,), (4,4,2,), (4,4,4,), (8,2,2,),
    #     (8,4,4,), (8,8,2), (8,8,4), (8,8,8,), (16,2,2,), (16,4,2,), (16,4,4,),
    #     (16,8,2,), (16,8,4,), (16,8,8,), (16,16,2,), (16,16,4,), (16,16,8,),
    #     (16,16,16,), (32,2,2,), (32,4,2,), (32,4,4,), (32,8,2,), (32,8,4,),
    #     (32,8,8,), (32,16,2,), (32,16,4,), (32,16,8,), (32,16,16,), (32,32,2,),
    #     (32,32,4,), (32,32,8,), (32,32,16,), (32,32,32,)
    # ]


    # no_inits = 15                   # Number of different initializations per node
    # seed_value = 0                  # Seed value for random number generation                                      
    # lr = 0.001                      # Learning rate
    # min_delta = 1e-6                # Tolerance for optimization
    # patience = 50                   # Patience for early stopping
    # verbose = 0                     # Verbosity mode for optimization
    # dropout = 0                   # Dropout rate (0 means no dropout)
    # n_process = 55                  # Number of cores to run on
    # Model_selection= 'IC'           # Model selection, "IC" for information criteria, "CV" for cross-validation
    # formulation = 'regional'          # Model formulation, "global" or "regional"
    # n_splits = 5                    # Number of splits for cross-validation, if Model_selection = IC, argument is ignored
    
    #Set up logging
    setup_logging()
    
    # results = multiprocessing_model(Model_selection, nodes_list, no_inits, seed_value, lr, min_delta, patience, verbose, dropout, n_splits, n_process, formulation)
    # np.save('results/results.npy', results)

    # return results

if __name__ == '__main__':
    main()
