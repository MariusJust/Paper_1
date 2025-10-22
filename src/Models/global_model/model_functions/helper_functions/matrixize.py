import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class Matrixize(Layer):
    def __init__(self, N, T, noObs, mask, **kwargs):
        super(Matrixize, self).__init__(**kwargs)
        self.N = N
        self.T = T
        self.noObs = noObs
        self.mask = mask

    def call(self, x):
        
        where = ~self.mask
        
        indices = tf.cast(tf.where(where), tf.int32)

        updates_obs = tf.reshape(x, (-1,))
        
        scatter = tf.scatter_nd(indices, updates_obs, shape=tf.shape(self.mask))
        scatter = tf.cast(scatter, dtype=np.float64)
        
        indices = tf.cast(tf.where(~where), tf.int32)
        x_nan = tf.ones(self.N * self.T - self.noObs) * np.nan
        scatter_nan = tf.scatter_nd(indices, x_nan, shape=tf.shape(self.mask))
        scatter_nan = tf.cast(scatter_nan, dtype=np.float64)

        return scatter + scatter_nan