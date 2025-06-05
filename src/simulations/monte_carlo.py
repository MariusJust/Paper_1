import hydra
from omegaconf import OmegaConf, DictConfig
from utils import multiprocessing_model,turn_off_warnings
import pandas as pd
from simulations.simulation_functions import Simulate_data
from simulations import multiprocessing_mc, mc_worker
from utils import multiprocessing_model, save_model_weights, save_yaml
import ast
import numpy as np
from models import MultivariateModelGlobal as Model            
from datetime import datetime
import multiprocessing as mp
turn_off_warnings()
import tensorflow as tf
import os


#we employ the following simulation procedure: 

# 1. Generate one sample of synthetic data using a fixed seed.
# 2. train all models (using multiproccesing) on the synthetic data and save the best model configuration (best_node) using information criteria.
# 3. Simulate a number of Monte Carlo replications (e.g. 100) using different seeds. For each replication, use the best_node from step 2 and train the model on the synthetic data.
# 4. Save the results as the average of the model weights across the Monte Carlo replications.

###############################################################################


@hydra.main(config_path="../../config", config_name="config_mc")
def run_mc(cfg: DictConfig):
    
    # #make sure the weights directory exists
    if not os.path.exists(f"../../../results/Model Parameters/MonteCarlo"):
        raise FileNotFoundError("The directory for saving model weights does not exist.")
        
    specs=cfg.mc.specifications
    breakpoints = cfg.mc.breakpoints
    

    for spec in specs:
        
            
            print(f"\n=== Running Monte Carlo for specification: {spec} ===")
            
            print(f"\n=== Running initial training loop ===")
            
            tf.keras.backend.clear_session()
        ## step 1
            nodes = [ast.literal_eval(s) for s in cfg.instance.nodes_list]
            train_kwargs = OmegaConf.to_container(cfg.instance, resolve=True)
            train_kwargs["data"] = Simulate_data.run(seed=cfg.mc.base_seed, n_countries=196, n_years=63, specification=spec, add_noise=True)
            train_kwargs["nodes_list"] = nodes
            
        ##step 2
            results = multiprocessing_model(**train_kwargs)
            best_node_BIC=min(results, key=lambda k: results[k][0])
            
            
            best_node=best_node_BIC
            best_node_idx=nodes.index(best_node)
            
            print(f"\n=== Running {cfg.mc.reps} Monte Carlo itterations for node {best_node} ===")
            
            
            
        ## step 3
            weights = multiprocessing_mc(
                node_index=best_node_idx,
                nodes_list=nodes,
                no_inits=cfg.instance.no_inits,
                seed_value=cfg.mc.base_seed,
                lr=cfg.instance.lr,
                min_delta=cfg.instance.min_delta,
                patience=cfg.instance.patience,
                verbose=cfg.instance.verbose,
                dropout=cfg.instance.dropout,
                n_splits=cfg.instance.n_splits,
                n_processes=cfg.instance.n_process,
                formulation=cfg.instance.formulation,
                reps=cfg.mc.reps,
                specification=spec,
                penalty=cfg.instance.penalty,
                breakpoints=breakpoints,
                cfg=cfg.instance
            )
            
            weights_lists = list(zip(*weights))

            # Now average across the rep dimension for each layer
            avg_weights = [
                np.mean(np.stack(list, axis=0), axis=0)
                for list in weights_lists
            ]

        ## step 4
            #setup the model using the average weights

            from simulations.simulation_functions.Simulate_data import Pivot
            growth, precip, temp = Pivot(train_kwargs["data"])
            x_train = {0:temp, 1:precip}
            
            factory = Model(nodes=best_node, x_train=x_train, y_train=growth, dropout=cfg.instance.dropout, formulation=cfg.instance.formulation, penalty=cfg.instance.penalty)
            
            ensemble_model = factory.get_model()
            ensemble_model.model.set_weights(avg_weights)
            
            # Save the model weights
            path=f"/home/mjust/Paper_1/results/Model Parameters/MonteCarlo/{spec}/{datetime.today().strftime('%Y-%m-%d')}/{best_node}.weights.h5"
            save_model_weights(path, ensemble_model)
        
        # Save the configuration
            path= f"/Paper_1/results/config/MonteCarlo/{spec}/{datetime.today().strftime('%Y-%m-%d')}/config.yaml"
            save_yaml(path, OmegaConf.to_yaml(cfg))
        
    print("\nAll specifications processed.")
    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    run_mc()
   
        
