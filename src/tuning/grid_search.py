
import numpy as np
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D  
import plotly.graph_objects as go
import plotly.io as pio
import os
import h5py
from models.model_functions.helper_functions import Prepare
from utils import create_pred_input, model_confidence_plot, model_instance_pred
from models import MultivariateModelGlobal as Model
import random
import itertools
import pickle
from datetime import datetime
from multiprocessing import Pool

os.environ['PYTHONHASHSEED'] = str(0)

#model parameters                    
lr = 0.001                      # Learning rate
min_delta = 1e-4               # Tolerance for optimization
patience = 20                   # Patience for early stopping
verbose = 2                     # Verbosity mode for optimization
formulation = 'global'          # Model formulation, "global" or "regional"

base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))  # goes from tuning/ to src/
data_path = os.path.join(base_dir, 'models', 'data', 'MainData.xlsx')
data = pd.read_excel(data_path)


growth, precip, temp = Prepare(data, formulation='global')
x_train = {0:temp, 1:precip}

class GridSearch:
    def __init__(self, grid):
        self.grid = grid
        self.keys = list(grid.keys())
        self.values = list(grid.values())
        self.combinations = list(itertools.product(*self.values))

    def __iter__(self):
        for combo in self.combinations:
            yield dict(zip(self.keys, combo))

    def __len__(self):
        return len(self.combinations)

def init_worker(seed):
 
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def train_and_save(params):
    # Unpack parameters
    model = params['Models']
    dropout= params['dropout']
    penalty = params['penalty']

    # Unique name
    combo_name = f"model_{'-'.join(map(str, model))}_dropout{dropout}_penalty{penalty}"
    print(f"[{os.getpid()}] Training combo: {combo_name} \n")

    # Initialize factory and model
    factory = Model(
        nodes=model,
        x_train=x_train,
        y_train=growth,
        dropout=dropout,
        formulation=formulation,
        penalty=penalty
    )
  
    model = factory.get_model()

    # Paths
    date_of_run = datetime.now().strftime('%Y%m%d')
    base_dir = os.path.join('..', 'results', 'Model Parameters', 'GridSearch', date_of_run)
    os.makedirs(base_dir, exist_ok=True)
    weight_path = os.path.join(base_dir, f"{combo_name}.weights.h5")


    # Fit
    history = model.fit(
        lr=lr,
        min_delta=min_delta,
        patience=patience,
        verbose=verbose
    )
    
    print(f"finished training combo: {combo_name} \n")

    # Save
    model.save_params(weight_path)
    return combo_name, history, base_dir


if __name__ == '__main__':
    
    Base_seed = 40
    # Define grid
    grid = {
        'Models': [
            (8, 2), (4,), (32,8,4), (32,32,32,)
        ],
        'dropout': [0, 0.1,0.2,0.3],
        'penalty': list(np.concatenate(([0.0], np.arange(0.001, 0.011, 0.001))))
    }

    # Training hyperparameters
    lr = 0.001
    min_delta = 1e-6
    patience = 20
    verbose = 0
    formulation = 'global'

    # Create GridSearch
    gs = GridSearch(grid)
    len(gs.combinations)


    # Parallel training using all CPUs
    num_workers = 45
    with Pool(processes=num_workers,
              initializer=init_worker,
              initargs=(Base_seed,)) as pool:
        results = pool.map(train_and_save, gs)

import numpy as np
new_list = list(np.concatenate(([0.0], np.arange(0.001, 0.011, 0.001))))