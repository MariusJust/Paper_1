from multiprocessing import Pool
from tqdm import tqdm
from multiprocessing import TimeoutError
from functools import partial

def multiprocessing_model(
    Model_selection, nodes_list, no_inits,
    seed_value, lr, min_delta, patience,
    verbose, dropout, n_splits,
    n_process, formulation, cv_approach, penalty, timeout_per_node,
    n_countries, time_periods,
    data=None
):
    storage = {}

    # Each worker just needs the node‐index and the shared args
    arg_list_cv = [
        (node_index,
         nodes_list, no_inits, seed_value, lr,
         min_delta, patience, verbose, dropout,
         n_splits, formulation, cv_approach, penalty, n_countries, time_periods)
        for node_index in range(len(nodes_list))
    ]
    
    arg_list_ic = [
        (node_index,
         nodes_list, no_inits, seed_value, lr,
         min_delta, patience, verbose, dropout, formulation, penalty, data)
        for node_index in range(len(nodes_list))
    ]

    arg_list=arg_list_cv if Model_selection=='CV' else arg_list_ic
        
    # Create a version of `worker` with Model_selection already bound in
    worker_fn = partial(worker, Model_selection=Model_selection)

    pool = Pool(n_process)
    async_results = [
        pool.apply_async(worker_fn, args=(args,), kwds={'Model_selection': Model_selection})
        for args in arg_list
    ]
    pool.close()
    
    for i, async_result in enumerate(tqdm(async_results, desc="Processing nodes", unit="node")):
        try:
            result = async_result.get(timeout=timeout_per_node)  # Adjust timeout as needed
        except TimeoutError:
            print(f"Timeout occurred for node {i} with args {arg_list[i]}")
            storage[i]=None
            continue 
        
        if Model_selection=='CV':
            cv_error, node = result
            storage[node] = [cv_error]
        else:
            bic,aic,node,weights_BIC,weights_AIC = result
            storage[node] = [bic,aic]

    pool.terminate()
    pool.join()

    print("finished calculating")
    
    return storage


def worker(args, Model_selection):
    # Dynamically import the right experiment‐runner
    if Model_selection == 'CV':
        from models.global_model.cross_validation.run_experiment_cv import MainLoop
        ml = MainLoop(*args)
        return ml.run_experiment()
    else:  # 'IC'
        from models.global_model.information_criteria.run_experiment_ic import main_loop
        #no need to pass the last argument (cv_approach) to main_loop when using information criteria
        return main_loop(*args)

   
    