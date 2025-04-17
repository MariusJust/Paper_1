import os
import pandas as pd
import numpy as np
import tensorflow as tf
import random
import warnings
from tqdm import tqdm
from multiprocessing import Pool
# Import your own modules – ensure these are available in your environment.
from Model.ModelFunctions import prepare
from Model import multivariate_model as Model
from panelsplit.cross_validation import PanelSplit

warnings.filterwarnings("ignore")
os.environ['PYTHONHASHSEED'] = str(0)



###############################################################################
# 1. Synthetic Data Generation Function
###############################################################################
def simulate_data(seed, n_countries=50, n_years=20):
    """
    Simulate a synthetic panel dataset.
    
    For each country and time period:
      - Country fixed effect drawn from N(0, 0.1)
      - A linear time trend
      - Input variable: log GDP (Uniform[8,12])
      - Dependent variables temp and precipitation generated via a quadratic function plus effects and noise.
    
    Also creates two additional variables (e.g., "precip" and "temp") as random values.
    """
    np.random.seed(seed)
    years = pd.date_range(start='2000', periods=n_years, freq='Y')
    countries = [f'Country_{i}' for i in range(n_countries)]
    
    data = []
    for country in countries:
        country_effect = np.random.normal(0, 0.1)
        for year in years:
            x = np.random.uniform(8, 12)
            time_trend = 0.05 * (year.year - 2000)
            true_y = 1.5 * x - 0.1 * (x ** 2) + country_effect + time_trend
            noise = np.random.normal(0, 0.5)
            y = true_y + noise
            data.append({'Country': country, 'Year': year, 'logGDP': x, 'Growth': y})
    df = pd.DataFrame(data)
    growth = {'global': df[['Country', 'Year', 'Growth']]}
    
    precip_df = df[['Country', 'Year']].copy()
    precip_df['Value'] = np.random.uniform(0, 1, len(df))
    precip = {'global': precip_df}
    
    temp_df = df[['Country', 'Year']].copy()
    temp_df['Value'] = np.random.uniform(15, 30, len(df))
    temp = {'global': temp_df}
    
    return growth, precip, temp

###############################################################################
# 2. Modified setup to Use Synthetic Data
###############################################################################
def setup_synthetic(model_selection, n_splits, synthetic_data):
    """
    Modified version of setup() to obtain synthetic data.
    """
    growth, precip, temp = synthetic_data
    if model_selection == 'IC':
        return growth, precip, temp
    elif model_selection == 'CV':
        growth_global = growth['global'].copy().reset_index(drop=True)
        if not np.issubdtype(growth_global['Year'].dtype, np.datetime64):
            growth_global['Year'] = pd.to_datetime(growth_global['Year'])
        panel_split = PanelSplit(periods=growth_global['Year'], n_splits=n_splits, gap=0, test_size=1)
        return growth, precip, temp, panel_split
    else:
        raise ValueError("Invalid model_selection argument. Use 'IC' or 'CV'.")


###############################################################################
# 3. Estimation Routine (Using the Best Node Value)
###############################################################################

def run_model_replication(replication_seed, Model_selection, best_node, no_inits, lr, min_delta, patience, verbose, dropout, n_splits):
    """
    Runs one Monte Carlo replication:
      - Generate synthetic data with replication_seed.
      - Run the estimation procedure using the fixed best_node.
    
    Returns the performance metric from cross-validation (e.g., average MSE).
    """
    synthetic_data = simulate_data(replication_seed, n_countries=50, n_years=20)
    
    # For this example, we only consider the 'CV' version.
    # The structure here mirrors the existing "model" function but for one fixed node.
    
    growth, precip, temp, panel_split = setup_synthetic('CV', n_splits, synthetic_data)
   
    models_tmp = [None] * no_inits
    cv_errors_inits = np.zeros(no_inits)
    for j in range(no_inits):
        errors_j = []
        current_seed = replication_seed + j  # update seed for each initialization
        for train_idx, test_idx in panel_split.split():
            growth_train = {'global': growth['global'].loc[train_idx]}
            # Reshape test target (use 'Growth' column)
            growth_test = np.array(growth['global'].loc[test_idx].reset_index(drop=True)['Growth']).reshape((1, 1, -1, 1))
            
            temp_train = {'global': temp['global'].loc[train_idx].reset_index(drop=True)}
            temp_test = np.array(temp['global'].loc[test_idx].reset_index(drop=True)['Value']).reshape((1, 1, -1, 1))
            precip_train = {'global': precip['global'].loc[train_idx].reset_index(drop=True)}
            precip_test = np.array(precip['global'].loc[test_idx].reset_index(drop=True)['Value']).reshape((1, 1, -1, 1))
            
            x_train = [temp_train, precip_train]
            x_test = tf.concat([temp_test, precip_test], axis=3)
            
            # Clear session and set seeds for reproducibility
            tf.keras.backend.clear_session()
            tf.random.set_seed(current_seed)
            np.random.default_rng(current_seed)
            random.seed(current_seed)
            
            # Initialize model with best_node; note that best_node is a fixed value.
            model_instance = Model(nodes=best_node, x_train=x_train, y_train=growth_train, dropout=dropout)
            model_instance.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)
            models_tmp[j] = model_instance
            
            preds = np.reshape(model_instance.model_pred.predict(x_test), (-1, 1), order='F')
            mse = np.nanmean((preds - growth_test) ** 2)
            errors_j.append(mse)
        cv_errors_inits[j] = np.mean(errors_j)
    best_init_idx = int(np.argmin(cv_errors_inits))
    best_cv_error = cv_errors_inits[best_init_idx]
    
    # Optionally, retrain on full sample using the best seed.
    train_on_full_sample(best_init_idx, best_node, growth, precip, temp, lr, min_delta, patience, verbose, dropout)
    
    # Return the replication results: for example, the average CV error.
    return {'seed': replication_seed, 'CV_error': best_cv_error}

def train_on_full_sample(best_init_idx, node, growth, precip, temp, lr, min_delta, patience, verbose, dropout):
    best_seed = best_init_idx
    tf.keras.backend.clear_session()
    tf.random.set_seed(best_seed)
    np.random.default_rng(best_seed)
    random.seed(best_seed)
    x_train = [temp, precip]
    model_full = Model(nodes=node, x_train=x_train, y_train=growth, dropout=dropout)
    model_full.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)
    model_full.save_params('Model Parameters/cv/' + str(node) + '.weights.h5')
    return None

###############################################################################
# 4. Multiprocessing over MC Replications
###############################################################################
def mc_worker(args):
    """
    Worker for one Monte Carlo replication.
    """
    replication_seed, Model_selection, best_node, no_inits, lr, min_delta, patience, verbose, dropout, n_splits = args
    return run_model_replication(replication_seed, Model_selection, best_node, no_inits, lr, min_delta, patience, verbose, dropout, n_splits)

if __name__ == "__main__":
    # Fixed model/hyperparameter settings.
    best_node = 32           # The best node value determined earlier.
    Model_selection = 'CV'   # We are using Cross-Validation in this example.
    no_inits = 3
    lr = 0.001
    min_delta = 1e-6
    patience = 5
    verbose = 0
    dropout = 0.2
    n_splits = 5             # Number of CV splits.
    
    # Number of Monte Carlo replications you want to run.
    num_replications = 100
    
    # Prepare arguments for each replication.
    rep_args = [
        (1000 + rep, Model_selection, best_node, no_inits, lr, min_delta, patience, verbose, dropout, n_splits)
        for rep in range(num_replications)
    ]
    
    # Use multiprocessing to run MC replications concurrently.
    with Pool(processes=4) as pool:   # Adjust the number of processes as needed.
        mc_results = list(tqdm(pool.imap_unordered(mc_replication_worker, rep_args), total=num_replications, desc="MC Replications"))
    
    # Process the results (here we simply print them and optionally save them).
    for res in mc_results:
        print(f"Seed: {res['seed']} | CV Error: {res['CV_error']}")
    
    # Optionally, save the results to a CSV file.
    df_results = pd.DataFrame(mc_results)
    df_results.to_csv("mc_replications_results.csv", index=False)
