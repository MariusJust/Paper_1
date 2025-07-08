
import hydra
from omegaconf import OmegaConf, DictConfig
import pandas as pd
from simulations.simulation_functions.Simulate_data import Pivot, simulate, illustrate_synthetic_data, surface
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
verbose = 2                     # Verbosity mode for optimization        # Model formulation, "global" or "regional"
base_seed = 0
specification = 'q_Leirvik'    # Specification of the model, e.g. 'linear', 'interaction', 'q_Leirvik'


data= simulate(seed=base_seed, n_countries=196, n_years=63, specification=specification, add_noise=True)
growth, precip, temp = Pivot(data)
x_train = {0:temp, 1:precip}



#show the full dataset
illustrate_synthetic_data(temp['global'].values.flatten(), precip['global'].values.flatten(), growth['global'].values.flatten())

#show the corresponding surface
pred_input, T, P=create_pred_input(True)
z = surface(T,P,specification=specification)
illustate_surface(T, P, z )



######### make a single monte carlo replication #########'



# # Create the model instance
# factory= Model(node=node, x_train=x_train, y_train=growth, dropout=0, penalty=0)
# model_instance=factory.get_model()
# model_instance.fit( lr=lr,
#         min_delta=min_delta,
#         patience=patience,
#         verbose=verbose)
# pred_flat=model_instance.model_visual.predict(pred_input).reshape(-1,)
# growth=pred_flat.reshape(T.shape)

#alternatively load the data from a file
spec="q_Leirvik"
date="2025-07-08"
node=(2,)

# surface_linear=np.load(f"../results/MonteCarlo/linear/{date}/_avg_surface.np.npy")
# surface_interaction=np.load(f"../results/MonteCarlo/interaction/{date}/_avg_surface.np.npy")
surfaces=np.load(f"../results/MonteCarlo/{spec}/{date}/_avg_surface.np.npy")

# growth_linear=np.mean(surface_linear, axis=0)
# growth_interaction=np.mean(surface_interaction, axis=0)
growth=np.mean(surfaces, axis=0)
growth=surfaces[0]
pred_input, T, P=create_pred_input(True)
surf = go.Surface(
        x=T, y=P, z=growth.reshape(T.shape),
        colorscale='Cividis',
        opacity=0.85,
        showscale=False,
        name=f'Model {node}'
    )

# surface_interaction=go.Surface(
#         x=T, y=P, z=growth_interaction.reshape(T.shape),
#         colorscale='Cividis',
#         opacity=0.85,
#         showscale=False,
#         name=f'Model {node}'
#     )

benchmark = surface(T.flatten(), P.flatten(), specification=spec)

bench_surf= go.Surface(
        x=T, y=P, z=benchmark.reshape(T.shape),
        colorscale='Cividis',
        opacity=0.85,
        showscale=False,
        name=f'Benchmark'
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


bias_surf = growth_q.reshape(T.shape) - benchmark.reshape(T.shape)

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



########## linear model ##########