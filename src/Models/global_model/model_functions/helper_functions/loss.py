import tensorflow as tf

def individual_loss(mask, p_matrix=None, n_holdout=0):
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

        if p_matrix is not None:
            time_len = tf.shape(y_pred)[1]  # e.g. 58 during training
            n_obs = y_true.shape[1]

        if p_matrix is not None:
        
            if n_obs == n_holdout:
                
                # slice mask to match y_true time-length
                mask_dyn = mask[:, -time_len:, :]                  
            
                # y_true_transf = tf.reshape(y_true[~mask_dyn], (1, -1, 1))
                y_pred_transf = tf.reshape(y_pred[~mask_dyn], (1, -1, 1))
                
                #count number of non-missing observations from y_true_transf

                P_dyn = p_matrix[-n_holdout:, -n_holdout:]
            else:
                mask_dyn = mask[:, :time_len, :]    
                                
                y_pred_dyn = y_pred[:, :time_len, :]

                y_pred_transf = tf.reshape(y_pred_dyn[~mask_dyn], (1, -1, 1))
                
                P_dyn = p_matrix[:n_obs, :n_obs]

            # tf.print(">>> individual_loss debug: P_dyn.shape:", tf.shape(P_dyn))

            y_pred_transf = tf.matmul(P_dyn, tf.cast(y_pred_transf, dtype=tf.float32))
            loss=tf.reduce_mean(tf.math.squared_difference(y_true, y_pred_transf), axis=1)
            return loss 
        else:
            time_len = tf.shape(y_true)[1]                    # e.g. 58 during training
            # slice mask to match y_true time-length
            mask_dyn = mask[:, :time_len, :]                  
            y_true_dyn = y_true[:, :time_len, :]      
            y_pred_dyn = y_pred[:, :time_len, :]

            y_pred_transf = tf.reshape(y_pred_dyn[~mask_dyn], (1, -1, 1))
            loss=tf.reduce_mean(tf.math.squared_difference(y_true, y_pred_transf), axis=1)
            return loss
                

    return loss



# class HoldoutLoss(tf.keras.losses.Loss):
#     """
#     Minimal holdout loss:
#       - Expects y_true, y_pred shapes: (batch, time, channels)
#       - Expects P_matrix_full shape: (proj_time, time)
#       - Expects mask shape: (batch, proj_time, channels) or broadcastable to that
#       - Computes projection, slices from holdout_period_start onward,
#         selects masked entries (~mask) and computes MSE on those entries.
#     """
#     def __init__(self, mask, P_matrix_full, holdout_period_start=0, name="holdout_loss"):
#         super().__init__(name=name)
#         self.mask = mask
#         self.P = tf.convert_to_tensor(P_matrix_full, dtype=tf.float32)
#         self.holdout_period_start = int(holdout_period_start)

#     def call(self, y_true, y_pred):
#         # slice to training-time length (like your original function)

#         #use tf.matmul to do projection
#         proj_true = tf.matmul(self.P, y_true)
#         proj_pred = tf.matmul(self.P, y_pred)

#         # slice holdout period
#         h = self.holdout_period_start
#         true_hold = proj_true[:, h:, :]
#         pred_hold = proj_pred[:, h:, :]

#         # slice mask to match holdout length (assumes mask is for projected time axis)
#         mask_hold = self.mask[:, h:, :]

#         # select masked entries (preserving your original use of ~mask)
#         true_sel = tf.reshape(true_hold[~mask_hold], (1, -1, 1))
#         pred_sel = tf.reshape(pred_hold[~mask_hold], (1, -1, 1))

#         # mean squared error on selected entries (return scalar)
#         loss = tf.reduce_mean(tf.math.squared_difference(true_sel, pred_sel))
#         return loss








# # def holdout_loss(mask, P_matrix_full=None, holdout_period_start=None):
# #     """
# #     Loss function (in two layers so that it can be interpreted by tensorflow).

# #     ARGUMENTS
# #         * mask: mask used to identify missing observations.
# #         * P_matrix_full: P matrix computed on the full sample.
# #         * holdout: size of the holdout period.

# #     Returns
# #         * loss: loss function evaluated in y_true and y_pred.
        
# #     """

# #     y_true=tf.matmul(P_matrix_full, y_true)
# #     y_pred=tf.matmul(P_matrix_full, y_pred)
    
# #     y_true_masked = tf.reshape(y_true[~mask], (1, -1, 1))
# #     y_pred_masked = tf.reshape(y_pred[~mask], (1, -1, 1))

    

# #     def loss_val(y_true_masked, y_pred_masked):
        
# #         #in this helper we take the full p matrix and multiply it by the climate part predictions. We then pick the elements that correspond to the holdout period

# #         y_true=tf.matmul(P_matrix_full, y_true_masked)
# #         y_pred=tf.matmul(P_matrix_full, y_pred_masked)
        
# #         #apply the mask, 
        

# #         #choose the holdout period values
# #         y_true_holdout=y_true[:, holdout_period_start:, :]
# #         y_pred_holdout=y_pred[:, holdout_period_start:, :]

# #         mse=tf.reduce_mean(tf.math.squared_difference(y_true_holdout, y_pred_holdout), axis=1)
        
# #         return mse

# #     def loss_train(p_train):
# #         """
# #         ARGUMENTS
# #             * p_train: training data.

# #         RETURNS
# #             * loss function evaluated in p_train.
# #         """

# #         return mse
