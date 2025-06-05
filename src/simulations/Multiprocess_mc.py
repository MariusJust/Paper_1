from multiprocessing import Pool
import multiprocessing as mp
from tqdm import tqdm
from functools import partial
from models.global_model.information_criteria.run_experiment_ic import main_loop
from simulations.simulation_functions import Simulate_data
from simulations.simulation_functions.bias import calculate_bias

def mc_worker(args):
    (
        node_index, nodes_list, replication_seed, model_selection, no_inits, lr,
        min_delta, patience, verbose, dropout, n_splits, penalty, data
    ) = args
    formulation = model_selection
    
    *_, weights_BIC, _=main_loop(node_index, nodes_list, no_inits,
        replication_seed, lr, min_delta, patience,verbose, dropout, formulation, penalty, data
    )
    

    return weights_BIC
  


def multiprocessing_mc( node_index,
    nodes_list,
    no_inits,
    seed_value,
    lr,
    min_delta,
    patience,
    verbose,
    dropout,
    n_splits,
    n_processes,
    formulation, 
    reps, 
    penalty,
    specification,
    breakpoints,
    cfg):
    
    """
    Main function to run Monte Carlo simulations using multiprocessing.
    """
    
    model_selection = formulation
    
    manager=mp.Manager()
    bias_dict=manager.dict({k: 0 for k in breakpoints})
    lock=manager.Lock()
    
    
        
  
    rep_args = [
        (
            node_index,
            nodes_list,
            seed_value + rep,  # replication_seed
            model_selection,
            no_inits,
            lr,
            min_delta,
            patience,
            verbose,
            dropout,
            n_splits,
            penalty,
            Simulate_data.run(seed=seed_value + rep + 1, n_countries=196, n_years=63, specification=specification, add_noise=True)  # data
        )
        for rep in range(reps)
    ]
    
    counter= manager.Value('i', 0)
    
    def callback_one(result):
        with lock:
            counter.value += 1
            all_weights.append(result)
        
        if counter.value in breakpoints:
            best_node = nodes_list[node_index]
            bias_dict[counter.value] = calculate_bias(all_weights, specification, best_node, cfg)
        
        return result
    
    
    
            
    pool = mp.Pool(processes=n_processes)
    all_weights = []

    # Here we `apply_async` each job and let the callback store bias if needed.
    # We also collect the return values in `all_weights`.
    for args_dict in rep_args:
        pool.apply_async(
            mc_worker,                 # your existing mc_worker function
            args=(args_dict, ),            # pass each dict as keyword‐args to mc_worker
            callback=callback_one 
            
        )

    pool.close()
    pool.join()
    
    return all_weights, dict(bias_dict)