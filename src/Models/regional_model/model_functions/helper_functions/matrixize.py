import tensorflow as tf
from tensorflow.keras.layers import Layer
import numpy as np

class Matrixize(Layer):
    def __init__(self, N, T, mask, holdout, n_obs_train, **kwargs):
        super(Matrixize, self).__init__(**kwargs)
        self.N = N
        self.T = T
        self.mask = mask
        self.holdout = holdout
        self.noObs_train = n_obs_train
    def call(self, x):

        
        n_obs=tf.shape(x)[1]
        
        # tf.print("\n================ Matrixize DEBUG ================")
        # tf.print("Input x shape:", tf.shape(x))
        # tf.print("Expected training obs:", self.noObs_train)
        # tf.print("n_obs:", n_obs)
        # tf.print("Holdout:", self.holdout)
        # tf.print("Full T:", self.T)
        # tf.print("--------------------------------------------------")

        
        def val_branch():
            mask = self.mask[:, -self.holdout:, :]
            where = tf.math.logical_not(mask)
            noObs = tf.reduce_sum(tf.cast(where, tf.int32))
            T = self.holdout

            # tf.print(">>> BRANCH: VALIDATION")
            # tf.print("mask shape:", tf.shape(mask))
            # tf.print("T used:", T)
            # tf.print("Observed entries:", noObs)

            return mask, where, noObs, T


        def train_branch():
            if self.holdout > 0:
                mask = self.mask[:, :-self.holdout, :]
            else:
                mask = self.mask

            where = tf.math.logical_not(mask)
            noObs = tf.reduce_sum(tf.cast(where, tf.int32))
            T = self.T

            # tf.print(">>> BRANCH: TRAIN")
            # tf.print("mask shape:", tf.shape(mask))
            # tf.print("T used:", T)
            # tf.print("Observed entries:", noObs)

            return mask, where, noObs, T


        chosen_mask, where, noObs, T = tf.cond(
            tf.equal(n_obs, self.noObs_train),
            train_branch,
            val_branch
        )


        out_shape = tf.shape(chosen_mask)
        
        # tf.print("--------------------------------------------------")
        # tf.print("Final mask shape:", out_shape)
        # tf.print("Number of true 'where' indices:", noObs)

        indices = tf.cast(tf.where(where), tf.int32)
        # tf.print("Indices (observed) shape:", tf.shape(indices))
        
        updates_obs = tf.reshape(x, (-1,))
        
        # tf.print("Updates_obs shape:", tf.shape(updates_obs))
        # tf.print("N * T:", self.N * T)
        # tf.print("Expected missing count:", self.N * T - noObs)
                
        scatter = tf.scatter_nd(indices, updates_obs, shape=out_shape)
        scatter = tf.cast(scatter, dtype=np.float64)
        
        indices = tf.cast(tf.where(~where), tf.int32)

        x_nan = tf.ones(self.N * T - noObs) * np.nan
        scatter_nan = tf.scatter_nd(indices, x_nan, shape=out_shape)
        scatter_nan = tf.cast(scatter_nan, dtype=np.float64)

        return scatter + scatter_nan