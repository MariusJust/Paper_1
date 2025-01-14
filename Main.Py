from Prepare import prepare 
from Static.Model import static_model as Model

import pandas as pd
import numpy as np
import tensorflow as tf
import random
import os




data=pd.read_excel('Paper_1/data/MainData.xlsx')

growth, precip, temp = prepare(data)

# Setting optional choice parameters
no_inits = 10                     # number of different initializations
seed_value = -1                   # initial seed value minus one

# Setting optional choice parameters
lr = 0.001                        # initial learning rate for the Adam optimizer
min_delta = 1e-6                  # tolerance to be used for optimization
patience = 100                    # patience to be used for optimization
verbose = False                   # verbosity mode for optimization


# nodes_list = [(2,), (4,), (8,), (16,), (32,), (2,2,), (4,2,), (4,4,), (8,2,), (8,4,), (8,8,), (16,2,), (16,4,), (16,8,),
#               (16,16,), (32,2,), (32,4,), (32,8,), (32,16,), (32,32,), (2,2,2,), (4,2,2,), (4,4,2,), (4,4,4,), (8,2,2,),
#               (8,4,2,), (8,4,4,), (8,8,2), (8,8,4), (8,8,8,), (16,2,2,), (16,4,2,), (16,4,4,), (16,8,2,), (16,8,4,),
#               (16,8,8,), (16,16,2,), (16,16,4,), (16,16,8,), (16,16,16,), (32,2,2,), (32,4,2,), (32,4,4,), (32,8,2,),
#               (32,8,4,), (32,8,8,), (32,16,2,), (32,16,4,), (32,16,8,), (32,16,16,), (32,32,2,), (32,32,4,), (32,32,8,),
#               (32,32,16,), (32,32,32,)]

nodes_list = [(16, 16, 16)]



# %% Computing BIC


models_tmp = [None] 
BIC_tmp = [None] 

seed_value_tmp = seed_value

for j in range(no_inits):
    seed_value_tmp = seed_value_tmp + 1

    tf.keras.backend.clear_session()
    tf.random.set_seed(seed_value_tmp)
    np.random.seed(seed_value_tmp)
    random.seed(seed_value_tmp)

    models_tmp[j] = Model(nodes=nodes_list[0], x_train=temp, y_train=growth, formulation='Global')

    models_tmp[j].fit(lr=lr, min_delta=min_delta, patience=patience, verbose=verbose)

    models_tmp[j].in_sample_predictions()
    BIC_tmp[j] = models_tmp[j].BIC








# where = np.where(BIC_tmp == np.min(BIC_tmp))[0][0]

# # Storing BIC and model from best initialization
# BIC[i] = BIC_tmp[where]
# models[i] = models_tmp[where]

# # Saving BIC from best initialization for each architecture
# np.save('BIC/' + specification.capitalize() + ' model/' + formulation.capitalize() + '/' + ghg_name + '/BIC_' + str(nodes_list[i]), np.array(models_tmp[where].BIC))

# # Saving parameters from best initialization for each architecture
# models_tmp[where].save_params('Model Parameters/' + specification.capitalize() + ' model/' + formulation.capitalize() + '/' + ghg_name + '/parameters_' + str(nodes_list[i]))







