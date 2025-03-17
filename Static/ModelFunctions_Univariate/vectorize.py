
import tensorflow as tf 
from tensorflow.keras.layers import Layer
import numpy as np

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

            #this could be an issue
        y = tf.reshape(x[~where_mat], (1, -1, 1))

        self.dim1 = tf.shape(y)[1]

        return y

