import numpy as np
import pandas as pd

import os
from models.global_model.model_functions.helper_functions.prepare_data import Prepare
from utils import create_pred_input

from models import MultivariateModelGlobal as Model       


os.environ['PYTHONHASHSEED'] = str(0)

#model parameters                    
lr = 0.001                      # Learning rate
min_delta = 1e-2               # Tolerance for optimization
patience = 20                 # Patience for early stopping
verbose = 2                     # Verbosity mode for optimization
n_countries=196
time_periods=63                 #

#prepare the data
data=pd.read_excel('../data/MainData.xlsx')
growth, precip, temp = Prepare(data, n_countries, time_periods)
x_train = {0:temp, 1:precip}

#summary statics for standardisation
mean_temp=np.nanmean(data["TempPopWeight"])
std_temp=np.nanstd(data["TempPopWeight"])
mean_precip=np.nanmean(data["PrecipPopWeight"])
std_precip=np.nanstd(data["PrecipPopWeight"])


surface=np.load(f"/workspaces/Paper_1/results/MonteCarlo/Dynamic/Burke/NN/2025-09-02/surfaces.np.npy")

surface= surface.reshape(90 * 90, 64)


pred_input, T, P= create_pred_input(mc=False, mean_T=mean_temp, std_T=std_temp, mean_P=mean_precip, std_P=std_precip, time_periods=63)




z = surface[:, 63].reshape(90, 90)   # keep what you used
x = T[:, :, 0]
y = P[:, :, 0] / 1000.0

print("shapes -> x, y, z:", x.shape, y.shape, z.shape)
print("dtypes -> x, y, z:", x.dtype, y.dtype, z.dtype)
print("z stats -> min, max, mean, nans:", np.nanmin(z), np.nanmax(z), np.nanmean(z), np.isnan(z).sum())

# quick sanity checks
if not (x.shape == y.shape == z.shape):
    raise ValueError("x, y, z must have the same shape. Got: "
                     f"{x.shape}, {y.shape}, {z.shape}")

if np.isnan(z).all():
    raise ValueError("z is all NaNs — nothing to plot. Check your surface array / reshape.")

# ---- Plotting ----
surf = go.Surface(
    x=x,
    y=y,
    z=z,
    colorscale='Cividis',
    opacity=0.85,
    showscale=False,
    name='Model'
)

fig1 = go.Figure(data=[surf])
fig1.update_layout(
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
import plotly.io as pio

try:
    pio.renderers.default = "browser"
    fig1.show()
    print("Attempted to open plot in default browser (renderer='browser').")
except Exception as e:
    print("Renderer 'browser' failed:", e)

fig1.show()


def time_plots(T,t, temp, precip): 
    
    growth = (T-t+1)/T*0.0127 * temp +  (T-t+1)/T*0.145 * precip - (t-1)/T*0.0005 * temp**2 - (t-1)/T*0.047* precip**2

    surf=go.Surface(x=temp, y=precip, z=growth,
                     colorscale='Cividis',
                    opacity=0.85,
                    showscale=False,
                    name=f'Time period {t}')
    
    fig = go.Figure(data=surf)
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
    return fig 

fig=time_plots(63,60,T[:,:,0], P[:,:,0]/1000)

fig.show()













import numpy as np
import plotly.graph_objects as go

# --- Inputs (use your existing arrays) ---
# T_total = number of time periods used in the formula (63 in your example)
T_total = 63
temp_grid = T[:, :, 0]       # 2D array of temperatures (shape: (n_precip, n_temp))
precip_grid = P[:, :, 0] / 1000.0  # convert to meters if P was mm

# --- build frames ---
frames = []
for t in range(1, T_total + 1):
    # growth formula from your function (broadcasting over the 2D grids)
    growth = (
        (T_total - t + 1) / T_total * 0.0127 * temp_grid
        + (T_total - t + 1) / T_total * 0.145 * precip_grid
        - (T_total - t + 1)/T_total * 0.0125 * temp_grid * precip_grid
        + (t-1)/T_total * 0.00029 * precip_grid * temp_grid**2
        + (t-1)/T_total * 0.007 * temp_grid * precip_grid**2
        - (t-1)/T_total * 0.00013 * temp_grid**2 * precip_grid**2
        - (t - 1) / T_total * 0.0005 * temp_grid**2
        - (t - 1) / T_total * 0.047 * precip_grid**2
    )

    surf = go.Surface(
        x=temp_grid,
        y=precip_grid,
        z=growth,
        colorscale='Cividis',
        opacity=0.85,
        showscale=False,
        name=f'Time {t}',
        hoverinfo='skip'  # optional: skip heavy hover info
    )

    frames.append(go.Frame(data=[surf], name=str(t)))

# --- initial surface is the first frame ---
initial = frames[0].data[0]

# --- build the figure with frames ---
fig = go.Figure(
    data=[initial],
    frames=frames
)

# --- slider (one step per frame) ---
steps = []
for i, fr in enumerate(frames):
    step = dict(
        method="animate",
        args=[
            [fr.name],
            dict(mode="immediate", frame=dict(duration=200, redraw=True), transition=dict(duration=0))
        ],
        label=fr.name
    )
    steps.append(step)

sliders = [dict(
    active=0,
    pad={"t": 50},
    steps=steps,
    currentvalue={"prefix": "Time: "}
)]

# --- layout (camera, fixed z-range, buttons) ---
fig.update_layout(
    title="Animated Growth Surface over Time",
    scene=dict(
        xaxis_title='Temperature (°C)',
        yaxis_title='Precipitation (m)',
        zaxis_title='Δ ln(Growth)',
        camera=dict(eye=dict(x=2.11, y=0.12, z=0.38)),
        zaxis=dict(range=[-1.5, 2])  # keep z-range fixed to avoid jumping as surface changes
    ),
    updatemenus=[
        dict(
            type="buttons",
            showactive=False,
            y=0.05,
            x=0.1,
            xanchor="right",
            yanchor="top",
            pad={"t": 60, "r": 10},
            buttons=[
                dict(label="Play",
                     method="animate",
                     args=[None, dict(frame=dict(duration=120, redraw=True),
                                      transition=dict(duration=0),
                                      fromcurrent=True,
                                      mode='immediate')]),
                dict(label="Pause",
                     method="animate",
                     args=[[None], dict(frame=dict(duration=0, redraw=False),
                                        mode='immediate',
                                        transition=dict(duration=0))])
            ],
        )
    ],
    sliders=sliders,
    showlegend=False,
    height=600,
    width=800
)

fig.show()

# Optional: save as interactive HTML
fig.write_html("growth_animation_Interactive.html")
