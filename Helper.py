# Importing libraries
import numpy as np
import pandas as pd
import tensorflow as tf

from tensorflow.keras.layers import Input, Dense, Layer, Add
from tensorflow.keras import Model
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.initializers import he_normal, Zeros
from tensorflow.keras.backend import sigmoid
from tensorflow.keras.layers import Activation
from tensorflow.keras.callbacks import EarlyStopping
from tensorflow.python.keras.utils.generic_utils import get_custom_objects



class Dummies(Layer):
    """
    Layer that creates country and time dummies.
    """

    def __init__(self, N, T, time_periods_na, **kwargs):
        super(Dummies, self).__init__(**kwargs)
        self.N = N
        self.T = T
        self.time_periods_na = time_periods_na
        self.noObs = None
    
    def call(self, x):
        where_mat = tf.transpose(tf.math.is_nan(x))

        for t in range(self.T):
            # print(where_mat[:, t, 0].np())
            idx = tf.where(~where_mat[:, t, 0])
            idx = tf.reshape(idx, (-1,))

            #Constructs and identity matrix, so that all count
            D_t = tf.eye(self.N)
            #gathers from D_t according to the indices in idx
            D_t = tf.gather(D_t, idx, axis=0)

            if t == 0:
                Delta_1 = D_t

                Delta_2 = tf.matmul(D_t, tf.ones((self.N, 1)))

            else:
                Delta_1 = tf.concat([Delta_1, D_t], axis=0)

                Delta_2 = tf.concat([Delta_2, tf.zeros((tf.shape(Delta_2)[0], 1))], axis=1)

                Delta_2_tmp = tf.matmul(D_t, tf.ones((self.N, 1)))
                Delta_2_tmp = tf.concat([tf.zeros((tf.shape(Delta_2_tmp)[0], t)), Delta_2_tmp], axis=1)

                Delta_2 = tf.concat([Delta_2, Delta_2_tmp], axis=0)

        Delta_1 = Delta_1[:, 1:]
        Delta_2 = Delta_2[:, self.time_periods_na + 1:]

        self.noObs = tf.shape(Delta_1)[0]
        
        Delta_1 = tf.reshape(Delta_1, (1, self.noObs, self.N - 1))
        Delta_2 = tf.reshape(Delta_2, (1, self.noObs, self.T - (self.time_periods_na + 1)))

        return [Delta_1, Delta_2]


class Extend(Layer):
    """
    Layer that extends the second dimension of inputs.
    """

    def __init__(self, mask, **kwargs):
        super(Extend, self).__init__(**kwargs)
        self.mask = mask

    def call(self, x):
        where = ~self.mask
        indices = tf.cast(tf.where(where), tf.int32)
        scatter = tf.scatter_nd(indices, tf.reshape(x, (-1,)), shape=tf.shape(self.mask))
        scatter = tf.cast(scatter, dtype=np.float64)

        indices = tf.cast(tf.where(~where), tf.int32)
        mask_tmp = tf.cast(self.mask, tf.int32)
        x_nan = tf.ones(tf.reduce_sum(mask_tmp)) * np.nan
        scatter_nan = tf.scatter_nd(indices, x_nan, shape=tf.shape(self.mask))
        scatter_nan = tf.cast(scatter_nan, dtype=np.float64)

        return scatter + scatter_nan


# %% Creating custom loss function
def individual_loss(mask):
    """
    Loss function (in two layers so that it can be interpreted by tensorflow).

    ARGUMENTS
        * mask: mask used to identify missing observations.

    Returns
        * loss: loss function evaluated in y_true and y_pred.
    """

    def loss(y_true, y_pred):
        """
        ARGUMENTS
            * y_true: observed targets.
            * y_pred: predicted targets.

        RETURNS
            * loss function evaluated in y_true and y_pred.
        """

        y_true_transf = tf.reshape(y_true[~mask], (1, -1, 1))
        y_pred_transf = tf.reshape(y_pred[~mask], (1, -1, 1))
        loss=tf.reduce_mean(tf.math.squared_difference(y_true_transf, y_pred_transf), axis=1)
        return loss

    return loss


class Matrixize(Layer):
    """
    Layer that matrixizes the second dimension of inputs.
    """

    def __init__(self, N, T, noObs, mask, **kwargs):
        super(Matrixize, self).__init__(**kwargs)
        self.N = N
        self.T = T
        self.noObs = noObs
        self.mask = mask

    def call(self, x):
        where = ~self.mask
        indices = tf.cast(tf.where(where), tf.int32)
        scatter = tf.scatter_nd(indices, tf.reshape(x, (-1,)), shape=tf.shape(self.mask))
        scatter = tf.cast(scatter, dtype=np.float64)

        indices = tf.cast(tf.where(~where), tf.int32)
        x_nan = tf.ones(self.N * self.T - self.noObs) * np.nan
        scatter_nan = tf.scatter_nd(indices, x_nan, shape=tf.shape(self.mask))
        scatter_nan = tf.cast(scatter_nan, dtype=np.float64)

        return scatter + scatter_nan


def prepare(data):

    #the growth data should contain the following columns: year, county, and GrowthWDI
    growth=data[['CountryCode', 'RegionCode', 'Year', 'GrowthWDI']]

    #precipitation data
    precip=data[['CountryCode', 'RegionCode', 'Year', 'PrecipPopWeight']]

    #temperature data
    temp=data[['CountryCode', 'RegionCode', 'Year', 'TempPopWeight']]

    #Now I make dictionaries, to capture the region dependent variables

    growth_dict=dict()
    precip_dict=dict()
    temp_dict=dict()

    # dictionary that holds the region code and the reference country in each region
    regions = {'Asia': [142, "CHN"], 'Europe': [150, 'DEU'], 'Africa': [2, 'ZAF'], 'Americas': [19, 'USA'], 'Oceania': [9, 'AUS']}

    dict_and_dfs = [(growth_dict, growth),
        (precip_dict, precip),
        (temp_dict, temp)]

    #Now I will loop through the regions and create a dataframe for each region
    for region, value in regions.items():
        #assign the region code and the reference country to variables
        regionCode, referenceCountry = value
        for region_dict, df in dict_and_dfs:
            #get the region specific data
            region_data = df[df['RegionCode'] == regionCode]
            
            # Pivot the data so that the years are the index and the countries are the columns
            pivot_data = region_data.pivot(index='Year', columns='CountryCode', values=region_data.columns[-1])
            
            
            #Reorder the columns so that the reference country is the first column
            cols = pivot_data.columns.tolist()  # Get current list of columns
            cols.insert(0, cols.pop(cols.index(referenceCountry)))  # Move reference country to the first column
            pivot_data = pivot_data[cols]  # Reorder columns in the dataframe

            # Concatenate the reference country data on top
            region_dict[region] = pivot_data

    return growth_dict, precip_dict, temp_dict
   
     
def swish(x, beta=1):
    """
    Swish activation function.

    ARGUMENTS
        * x:    input variable.
        * beta: hyperparameter of the Swish activation function.

    Returns
        * Swish activation function applied to x.
    """

    return x * sigmoid(beta * x)   
        

# %% Creating vectorization layer
class Vectorize(Layer):
    """
    Layer that vectorizes the second dimension of inputs.
    """

    def __init__(self, **kwargs):
        super(Vectorize, self).__init__(**kwargs)
        self.dim1 = None

    def call(self, x):
        where_mat = tf.math.is_nan(x)

        y = tf.reshape(x[~where_mat], (1, -1, 1))

        self.dim1 = tf.shape(y)[1]

        return y


    
   
    
