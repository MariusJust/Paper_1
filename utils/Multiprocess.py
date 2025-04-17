
def multiprocessing_model(Model_selection, nodes_list, no_inits, seed_value, lr, min_delta, patience, verbose, dropout, n_splits, n_process):
    storage = {}

    # Prepare an iterable of arguments for each node.
    arg_list = [
        (node_index, Model_selection, nodes_list, no_inits, seed_value, lr, min_delta, patience, verbose, dropout, n_splits)
        for node_index in range(len(nodes_list))
    ]

    # Create a pool of processes
    with Pool(n_process) as pool:
        results = pool.imap_unordered(worker, arg_list)
        # Collect the results using tqdm for progress tracking.
        for result in tqdm(results, total=len(nodes_list), desc="Processing nodes", unit="node"):
            if Model_selection == 'CV':
                cv_error, node = result
                storage[node] = [cv_error]
            elif Model_selection == 'IC':
                bic, aic, node = result
                storage[node] = [bic, aic]

    return storage


def worker(args):
    # Unpack the tuple of arguments and call the model function.
    return model(*args)