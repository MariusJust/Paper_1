import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

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
        # Find the positions where the mask is False
        where = ~self.mask

        # Find indices for scatter
        indices = tf.cast(tf.where(where), tf.int32)

     
        # Reshape the input x
        reshaped_x = tf.reshape(x, (-1,))

        # Scatter the reshaped x
        scatter = tf.scatter_nd(indices, reshaped_x, shape=tf.shape(self.mask))
        scatter = tf.cast(scatter, dtype=tf.float64)

        # Now let's handle the NaN values
        indices = tf.cast(tf.where(~where), tf.int32)

        # Create x_nan and check its shape
        x_nan = tf.ones((self.N * self.T - self.noObs)/2) * np.nan
       
        # Scatter the NaN values
        scatter_nan = tf.scatter_nd(indices, x_nan, shape=tf.shape(self.mask))
        scatter_nan = tf.cast(scatter_nan, dtype=tf.float64)
      
        return scatter + scatter_nan

