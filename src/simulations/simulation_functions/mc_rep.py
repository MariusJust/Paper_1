

def run_mc_rep(replication_seed, best_node, no_inits, lr, min_delta, patience, verbose, dropout):
    """
    Runs one Monte Carlo replication:
      - Generate synthetic data with replication_seed.
      - Run the estimation procedure using the fixed best_node.
    
    Returns the performance metric from cross-validation (e.g., average MSE).
    """
    synthetic_data = simulate_data(replication_seed, n_countries=196, n_years=63, specification='q_Leirvik')
    
    # For this example, we only consider the 'IC' version.
    # The structure here mirrors the existing "model" function but for one fixed node.
    
    x_train = {0:temp, 1:precip}
    BIC_list = np.zeros(no_inits)
    AIC_list = np.zeros(no_inits)
    # Loop over each initialization
    for j in range(no_inits):
        current_seed = seed_value + j  # update seed
        tf.keras.backend.clear_session()
        tf.random.set_seed(current_seed)
        np.random.default_rng(current_seed)
        random.seed(current_seed)

        model_instance = Model(nodes=nodes_list[j], x_train=x_train, y_train=growth, dropout=dropout, formulation=formulation)
        model_instance.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)
        model_instance.in_sample_predictions()
        models_tmp[j] = model_instance

        #saves the average AIC and BIC values for each model configuration
        BIC_list[j] = model_instance.BIC
        AIC_list[j] = model_instance.AIC

        print(f"Process {os.getpid()} completed initialization {j+1}/{no_inits} (IC mode) for node {best_node}", flush=True)

    # Select the best initialization based on BIC (or AIC)
    best_idx_BIC = int(np.argmin(BIC_list))
    best_idx_AIC = int(np.argmin(AIC_list))

    # Return the replication results: for example, the average CV error.
    return {'seed': replication_seed, 'node': best_node}

