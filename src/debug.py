
import hydra
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from simulations.simulation_functions import Simulate_data
from simulations.simulation_functions.Simulate_data import Pivot
import ast
import numpy as np
from models import MultivariateModelGlobal as Model            
from datetime import datetime
import multiprocessing as mp
import tensorflow as tf
import os
import plotly.io as pio
import plotly.graph_objects as go
from utils import create_pred_input, model_confidence_plot, model_instance_pred
import scipy.ndimage as ndi


# Define the model parameters

lr = 0.001                      # Learning rate
min_delta = 1e-6               # Tolerance for optimization
patience = 20                   # Patience for early stopping
verbose = 2                     # Verbosity mode for optimization
formulation = 'global'          # Model formulation, "global" or "regional"
base_seed = 0
specification = 'q_Leirvik'  # Specification for the simulation



data= Simulate_data.simulate(seed=base_seed, n_countries=196, n_years=63, specification=specification, add_noise=True)
growth, precip, temp = Pivot(data)
x_train = {0:temp, 1:precip}



#show the full dataset
Simulate_data.illustrate_synthetic_data(temp['global'].values.flatten(), precip['global'].values.flatten(), growth['global'].values.flatten())

#show the corresponding surface
pred_input, T, P=create_pred_input(True)
z = Simulate_data.surface(T,P,specification=specification)
Simulate_data.illustate_surface(T, P, z )



######### make a single monte carlo replication #########'

node=(16,)

# Create the model instance
factory= Model(node=node, x_train=x_train, y_train=growth, dropout=0, penalty=0)
model_instance=factory.get_model()
model_instance.fit( lr=lr,
        min_delta=min_delta,
        patience=patience,
        verbose=verbose)
pred_flat=model_instance.model_visual.predict(pred_input).reshape(-1,)
growth=pred_flat.reshape(T.shape)

#alternatively load the data from a file

surfaces=np.load("../results/MonteCarlo/q_Leirvik/2025-07-03/_avg_surface.np.npy")


growth=np.mean(surfaces, axis=0)


surf = go.Surface(
        x=T, y=P, z=growth.reshape(T.shape),
        colorscale='Cividis',
        opacity=0.85,
        showscale=False,
        name=f'Model {node}'
    )

benchmark = Simulate_data.surface(T, P, specification=specification)

bench_surf= go.Surface(
        x=T, y=P, z=benchmark.reshape(T.shape),
        colorscale='Cividis',
        opacity=0.85,
        showscale=False,
        name=f'Model {node}'
    )

   
plot_data=[surf, bench_surf]

fig = go.Figure(data=plot_data)
fig.update_layout(
    scene=dict(
        xaxis_title='Temperature (°C)',
        yaxis_title='Precipitation (m)',
        zaxis_title='Δ ln(Growth)',
        
        camera=dict(eye=dict(x=2.11, y=0.12, z=0.38))
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='black',
        borderwidth=1
    )
)

fig.show()


bias_surf = growth.reshape(T.shape) - benc.reshape(T.shape)

heatmap = go.Heatmap(
    x=T[0, :],            # unique T values along x
    y=P[:, 0],            # unique P values along y
    z=bias_surf,               # difference matrix
    colorscale='RdYlBu',
    zmin=-0.8,           # match your earlier z‑range
    zmax= 0.8,
    colorbar=dict(
        title='bias'
    )
)

fig = go.Figure(data=heatmap)
fig.update_layout(
    xaxis=dict(title='Temperature (°C)'),
    yaxis=dict(title='Precipitation (m)'),
    margin=dict(l=60, r=20, t=40, b=60)
)

fig.show()



