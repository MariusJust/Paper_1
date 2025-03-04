
import tensorflow as tf 
from tensorflow.keras.layers import Layer
import numpy as np

# %% Creating vectorization layer
class Vectorize(Layer):
    """
    Layer that vectorizes the second dimension of inputs.
    """

    def __init__(self, N, **kwargs):
        super(Vectorize, self).__init__(**kwargs)
        self.N=N
        self.dim1 = None

    def call(self, x):
     

        mask = tf.math.is_nan(x[..., :self.N])
        
        # Slice the input into temperature and precipitation blocks.
        temp = x[..., :self.N]    # First N values: temperatures
        precip = x[..., self.N:]  # Next N values: precipitation

        # Apply the mask to both slices.
        temp_clean = tf.reshape(temp[~mask], (1, -1, 1))
        precip_clean = tf.reshape(precip[~mask], (1, -1, 1))

        # Store the shape of the first variable for further processing
        self.dim1 = tf.shape(temp_clean)[1]

        return [temp_clean, precip_clean]
    
    def compute_output_shape(self, input_shape):
        return [(1, self.dim1, 1), (1, self.dim1, 1)]
