import hydra
import tensorflow as tf
import os
import ast
import multiprocessing as mp
from datetime import datetime
from omegaconf import OmegaConf, DictConfig
from simulations.simulation_functions import Pivot, simulate, illustrate_synthetic_data
from utils.parallel import MultiprocessingMC, Multiprocess
from utils.miscelaneous import save_yaml, save_numpy

def mc_loop(cfg, spec):
    
###########################################################################################################################################################
#we employ the following simulation procedure: 

# 1. Generate one sample of synthetic data using a fixed seed.
# 2. train all models (using multiproccesing) on the synthetic data and save the best model configuration (best_node) using information criteria.
# 3. Simulate a number of Monte Carlo replications (e.g. 100) using different seeds. For each replication, use the best_node from step 2 and train the model on the synthetic data.
# 4. Save the results as the average of the model weights across the Monte Carlo replications.

###############################################################################################################################################################

   
## step 1
    nodes = [ast.literal_eval(s) for s in cfg.instance.nodes_list]
    train_kwargs = {
    "cfg": cfg.instance,
    "data": simulate(
        seed=cfg.mc.base_seed,
        n_countries=196,
        n_years=63,
        specification=spec,
        add_noise=True,
    ),
    }
    
    #illustrate the data
    # import numpy as np
    # growth, precip, temp = Pivot(train_kwargs["data"])
    # illustrate_synthetic_data(np.array(temp['global']).flatten(), np.array(precip['global']).flatten(),np.array( growth['global']).flatten())

    
## step 2
    worker = Multiprocess(**train_kwargs)
    results = worker.run()
    if cfg.instance.model_selection == "IC":
        # Select the best node based on BIC
        best_node = min(results, key=lambda k: results[k][0])
    else:
        # Select the best node based on cross validation error
        best_node= min(results, key=lambda k: results[k])
       
    best_node_idx=nodes.index(best_node)
 
    
    print(f"\n=== Running {cfg.mc.reps} Monte Carlo itterations for node {best_node} ===")

    
## step 3
    worker_mc = MultiprocessingMC(
        node_index=best_node_idx,
        nodes_list=nodes,
        cfg=cfg,
        specification=spec,
    )
    all_surfaces, country_FE = worker_mc.run() 
    

## step 4    
    # growth, precip, temp = Pivot(train_kwargs["data"])
    # x_train = {0:temp, 1:precip}
    
    # factory = Model(node=best_node, x_train=x_train, y_train=growth, dropout=cfg.instance.dropout, formulation=cfg.instance.formulation, penalty=cfg.instance.penalty)
    
    # ensemble_model = factory.get_model()
    # ensemble_model.model.set_weights(avg_weights)
    
    # Save the model weights
    
    path = f"results/MonteCarlo/{spec}/{datetime.today().strftime('%Y-%m-%d')}/_avg_surface.np"
    save_numpy(path, all_surfaces)

    path=f"results/MonteCarlo/{spec}/{datetime.today().strftime('%Y-%m-%d')}/_country_FE.np"
    save_numpy(path, country_FE)

    path= f"results/config/MonteCarlo/{spec}/{datetime.today().strftime('%Y-%m-%d')}/config.yaml"
    save_yaml(path, OmegaConf.to_yaml(cfg))
        



######################################################################### Initializer ######################################################################################

    
if __name__ == "__main__":
    mp.set_start_method("spawn", force=True)
    
    @hydra.main(config_path="../../config", config_name="config_mc", version_base="1.2")
    def run_mc(cfg: DictConfig):
        # unpack configs
        specs      = cfg.mc.specifications
        breakpoints= cfg.mc.breakpoints

        for spec in specs:
            if not os.path.exists(f"results/MonteCarlo"):
                raise FileNotFoundError("The directory for saving model weights does not exist.")
        

            print(f"\n======== Running Monte Carlo for specification: {spec} ============")
            
            print(f"\n=== Running initial training loop ===")
    
            tf.keras.backend.clear_session()
        
            # Hand off to loop
            mc_loop(cfg, spec)
            
        print("\nAll specifications processed.")
    run_mc()
    

        
