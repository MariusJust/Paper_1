
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os
from models.global_model.model_functions.helper_functions.prepare_data import Prepare
from utils import create_pred_input
from scipy.interpolate import griddata
from models import MultivariateModelGlobal as Model       
import tensorflow as tf
import pandas as pd 
import os 
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.keras.optimizers import Adam




from models.global_model.model_functions.helper_functions import initialize_parameters, Preprocess, individual_loss, WithinHelper



#change working directory
os.chdir(os.path.dirname(os.path.abspath(__file__)))

os.environ['PYTHONHASHSEED'] = str(0)

#model parameters                    
lr = 0.001                      # Learning rate
min_delta = 1e-4               # Tolerance for optimization
patience = 20                   # Patience for early stopping
verbose = 2                     # Verbosity mode for optimization
n_countries=196
time_periods=63                 #
holdout=5                      #number of holdout periods for validation

#prepare the data
data=pd.read_excel('../data/MainData.xlsx')
growth, precip, temp = Prepare(data, n_countries, time_periods)
x_train = {0:temp, 1:precip}

#summary statics for standardisation
mean_temp=np.nanmean(data["TempPopWeight"])
std_temp=np.nanstd(data["TempPopWeight"])
mean_precip=np.nanmean(data["PrecipPopWeight"])
std_precip=np.nanstd(data["PrecipPopWeight"])

pred_input, T, P= create_pred_input(mc=False, mean_T=mean_temp, std_T=std_temp, mean_P=mean_precip, std_P=std_precip)



######### train one model with holdout and within transformation ###########
node=(32,2)
factory = Model(
            node=None, 
            x_train=None,     
            y_train=None,
            x_train_val=None,
            y_train_val=None,
            x_val=None,
            y_val=None,
            dropout=0,
            country_trends=False,
            dynamic_model=False,
            within_transform=True,
            holdout=5, 
            add_fe=False
        )
    
    
factory.x_train = {0: temp, 1: precip}

factory.y_train = growth
            

temp_train_val = {key: df.iloc[:-holdout, :] for key, df in temp.items()}
temp_val = {key: df.iloc[-holdout:, :] for key, df in temp.items()}
precip_train_val = {key: df.iloc[:-holdout, :] for key, df in precip.items()}
precip_val = {key: df.iloc[-holdout:, :] for key, df in precip.items()}
growth_train_val = {key: df.iloc[:-holdout, :] for key, df in growth.items()}
growth_val = {key: df.iloc[-holdout:, :] for key, df in growth.items()}


factory.x_train_val = {0: temp_train_val, 1: precip_train_val}
factory.y_train_val = growth_train_val
factory.x_val = {0: temp_val, 1: precip_val}
factory.y_val = growth_val
factory.node = node

factory = factory.get_model()



#compute p matrix on whole data
P_helper_full=WithinHelper(factory.input_data_temp)
P_matrix_full=P_helper_full.calculate_P_matrix()
p_tensor=tf.convert_to_tensor(P_matrix_full, dtype=tf.float32)


y_true_target_train=tf.matmul(p_tensor, tf.cast(tf.reshape(factory.targets[~factory.Mask], (1, -1, 1)), dtype=tf.float32))[:,  :factory.noObs['train'], :]

y_true_target_val=tf.matmul(p_tensor, tf.cast(tf.reshape(factory.targets[~factory.Mask], (1, -1, 1)), dtype=tf.float32))[:, factory.noObs['train']:, :]

n_obs_holdout= factory.noObs['global'] - factory.noObs['train']

factory.model.compile(optimizer=Adam(lr), loss=individual_loss(mask=factory.Mask, p_matrix=p_tensor, n_holdout=n_obs_holdout))

print(f"PID {os.getpid()} - compiled model with within_transform and holdout of {n_obs_holdout} observations", flush=True)
if n_obs_holdout>0:
    es = EarlyStopping(monitor='val_loss', mode='min', min_delta=min_delta, patience=patience,
                        restore_best_weights=True, verbose=verbose)
    callbacks = [es]
    #validation data preprocessing
    x_train_val =  [factory.input_data_temp_train_val, factory.input_data_precip_train_val]
    x_val =  [factory.input_data_temp_val, factory.input_data_precip_val]
    
    factory.model.fit(x_train_val, y_true_target_train, callbacks=callbacks, batch_size=1, epochs=int(1e6), verbose=verbose, shuffle=False, validation_data=(x_val, y_true_target_val))


factory.holdout_loss = np.min(es.best)
        

factory.model_visual.predict([pred_input])


pred_flat = factory.model_visual.predict([pred_input]).reshape(-1,)
Growth = pred_flat.reshape(T.shape)

opacity = 0.3
surf = go.Surface(
    x=T, y=P/1000, z=Growth, #ensure that the surfaces are meassured in meters instead of milimeters
    colorscale='Cividis',
    opacity=0.85,
    showscale=False,
    name=f'Model {node}'
)

#add the true data as grid points / remember to undo the standardisation
true_data=go.Scatter3d(
x=np.array(data["TempPopWeight"]).flatten(),
y=np.array(data["PrecipPopWeight"]).flatten()/1000,
z=np.array(data["GrowthWDI"]).flatten(),
mode='markers',
marker=dict(size=1.5, opacity=0.2, color='red')
)


n_obs_holdout=holdout*n_countries

holdout_data=go.Scatter3d(
  x=true_data.x[-n_obs_holdout:],
  y=true_data.y[-n_obs_holdout:],
  z=true_data.z[-n_obs_holdout:],
  mode='markers',
  marker=dict(size=1.5, opacity=0.5, color='blue'),
  name='Holdout Data'
  )

plot_data=[surf,  holdout_data]
fig= go.Figure(data=plot_data)
fig.update_layout(
    scene=dict(
        xaxis_title='Temperature (°C)',
        yaxis_title='Precipitation (m)',
        zaxis_title='Δ ln(Growth)',
        camera=dict(eye=dict(x=2.11, y=0.12, z=0.38))
        
        ,zaxis=dict(range=[-0.5, 0.5])
    ),
    legend=dict(
        bgcolor='rgba(255,255,255,0.7)',
        bordercolor='black',
        borderwidth=1
    )
    
)

fig.show()
    
    



# ######### retrain the full model and save weights ###########

# if hasattr(factory, '_cache'):
#                 try:
#                     factory._cache.clear()
#                 except Exception:
#                     factory._cache = {}
                    
# factory.x_train = {0: temp, 1: precip}
# factory.y_train = growth
# factory.x_val = None
# factory.y_val = None
# factory.node = node
# factory.holdout=0
            
# model_full=factory.get_model()
# model_full.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)

# weights=model_full.model.get_weights()

# #save weights

# output_dir = f'../results/Model Parameters/IC/2025-04-11/'
# if not os.path.exists(output_dir):
#     os.makedirs(output_dir)
# weight_file = f'../results/Model Parameters/IC/2025-04-11/{node}.weights.h5'
# model_full.model.save_weights(weight_file)


# ###################################### Now use the weights to retrain the model without within transfomr#####################################################



# factory = Model(node, x_train, growth, dropout=0, country_trends=False, dynamic_model=False, within_transform=False, holdout=0)


# factory.Depth=len(node)
# model=factory.get_model()
# weight_file = f'../results/Model Parameters/IC/2025-04-11/{node}.weights.h5'
# model.load_params(weight_file)


# # fit & predict
# model.fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)

# pred_flat = model.model_visual.predict([pred_input]).reshape(-1,)
# Growth = pred_flat.reshape(T.shape)

