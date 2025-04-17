def mc_worker(args):
    """
    Worker for one Monte Carlo replication.
    """
    replication_seed, Model_selection, best_node, no_inits, lr, min_delta, patience, verbose, dropout, n_splits = args
    return run_model_replication(replication_seed, Model_selection, best_node, no_inits, lr, min_delta, patience, verbose, dropout, n_splits)


def multiprocessing_mc():
    """
    Main function to run Monte Carlo simulations using multiprocessing.
    """
  
    
    # Number of Monte Carlo replications you want to run.
    num_replications = 100
    
    # Prepare arguments for each replication.
    rep_args = [
        (1000 + rep, Model_selection, best_node, no_inits, lr, min_delta, patience, verbose, dropout, n_splits)
        for rep in range(num_replications)
    ]
    
    # Use multiprocessing to run MC replications concurrently.
    with Pool(processes=4) as pool:
        results = pool.map(mc_worker, rep_args)
        # Collect results
        for res in results:
            print(f"Seed: {res['seed']} | CV Error: {res['CV_error']}")
    # Optionally, save results to a file.

