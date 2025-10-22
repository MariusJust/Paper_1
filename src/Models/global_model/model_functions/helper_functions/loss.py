import tensorflow as tf

def individual_loss(mask, P_matrix=None):
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

        # if P_matrix is not None:
        #     y_pred = tf.matmul(P_matrix, y_pred)
        #     y_pred_transf = tf.reshape(y_pred[~mask], (1, -1, 1))
        #     y_true_transf = tf.reshape(y_true[~mask], (1, -1, 1))
        #     loss=tf.reduce_mean(tf.math.squared_difference(y_true_transf, y_pred_transf), axis=1)
        #     return loss
        
        # else: 
        #reshape mask to a (1,N,T) tensor
        # dynamic time dimension of the batch
        time_len = tf.shape(y_true)[1]                    # e.g. 58 during training

        # slice mask to match y_true time-length
        mask_dyn = mask[:, :time_len, :]                  
        y_true_dyn = y_true[:, :time_len, :]      
        y_pred_dyn = y_pred[:, :time_len, :]

        y_true_transf = tf.reshape(y_true_dyn[~mask_dyn], (1, -1, 1))
        y_pred_transf = tf.reshape(y_pred_dyn[~mask_dyn], (1, -1, 1))
        loss=tf.reduce_mean(tf.math.squared_difference(y_true_transf, y_pred_transf), axis=1)
        return loss

    return loss
