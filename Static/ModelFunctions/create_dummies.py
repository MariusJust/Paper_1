import tensorflow as tf
from tensorflow.keras.layers import Layer

def Create_dummies(self, input_x):
        """ Create and apply the dummies layer to input_x. """
        my_layer = Dummies(N=int(self.N['global']), T=self.T, time_periods_na=self.time_periods_na['global'])
        return my_layer(input_x)
    


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
        # Remove the batch dimension (assuming batch=1).
        x0 = x[0]  # shape: (T, N, P)
        # Create a mask that marks an observation as valid only if all predictors are non-missing.
        valid = tf.reduce_all(~tf.math.is_nan(x0), axis=-1)  # shape: (T, N)
        # Transpose so that each row corresponds to a country.
        valid = tf.transpose(valid)  # now shape: (N, T)
  

        for t in range(self.T):
            # For each time period, find indices (countries) with valid data.
            idx = tf.where(valid[:, t])
            idx = tf.reshape(idx, (-1,))

            # Create an identity matrix for the countries. 196x196
            D_t = tf.eye(self.N)
            # Select rows corresponding to the valid indices.
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

        # Remove the first column from Delta_1 and the initial time dummies from Delta_2.
        Delta_1 = Delta_1[:, 1:]
        Delta_2 = Delta_2[:, self.time_periods_na + 1:]

        self.noObs = tf.shape(Delta_1)[0]

        # Reshape to add the batch dimension back.
        Delta_1 = tf.reshape(Delta_1, (1, self.noObs, self.N - 1))
        Delta_2 = tf.reshape(Delta_2, (1, self.noObs, self.T - (self.time_periods_na + 1)))
        return [Delta_1, Delta_2]