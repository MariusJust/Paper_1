import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Add

class HoldoutMonitor(tf.keras.callbacks.Callback):
    def __init__(self, mod, patience, min_delta, verbose, P_matrix_train=None, P_matrix_full=None, y_true=None):
        super().__init__()
        self.mod= mod
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.P_matrix_train = P_matrix_train
        self.P_matrix_full = P_matrix_full
        self.y_true = y_true

        # Early-stopping bookkeeping
        self.best_loss = np.inf
        self.wait = 0
        self.best_weights = None  # will store keras model weights (list)
        self.model_ref = None     # will hold the keras Model once set_model is called

    def set_model(self, model):
            """
            Keras training loop calls this (preferred) instead of assigning to .model.
            Save the model reference under a different name to avoid property collisions.
            """
            # store the Keras model object under model_ref
            self.model_ref = model

    def on_train_begin(self, logs=None):
            # Reset wait counter and best loss at the start of training
            self.wait = 0
            self.best_loss = np.inf

    def on_epoch_end(self, epoch, logs=None):

        #climate part prediction for the holdout period
        precip_val=tf.reshape(np.array(self.mod.input_data_precip), (1, 1, -1, 1))
        temp_val=tf.reshape(np.array(self.mod.input_data_temp), (1, 1, -1, 1))
        X_in=tf.concat([temp_val, precip_val], axis=3)
        climate_part= tf.reshape(self.mod.model_visual([X_in]), (1, self.mod.T+self.mod.holdout, self.mod.N['global']))
        
        y_pred=tf.reshape(climate_part[~self.mod.Mask_full], (1, -1, 1))
        
        final_prediction_proj=tf.matmul(self.P_matrix_full, tf.cast(y_pred, dtype=tf.float64))[:, -self.y_true.shape[1]:, :]

      

        mse = float(np.nanmean((final_prediction_proj - self.y_true) ** 2))
        
        if logs is not None:
            logs['holdout_mse'] = mse
        
        #early stopping check
        if mse + self.min_delta < self.best_loss:
            self.best_loss = mse
            self.wait = 0
            self.best_weights = self.model_ref.get_weights()  # save the best weights
            # if self.verbose > 0:
            print(f"\nEpoch {epoch+1}:{mse:.5f}")
        else:
            self.wait += 1
            # if self.verbose > 0:
            print(f"\nEpoch {epoch+1}:{mse:.5f}")
            if self.wait >= self.patience:
                # if self.verbose > 0:
                print(f"\nEpoch {epoch+1}: Early stopping triggered. Restoring best weights from epoch with HoldoutMonitor {self.best_loss:.5f}.")
                self.model_ref.stop_training = True
                if self.best_weights is not None:
                    self.model_ref.set_weights(self.best_weights)
                self.model_ref.stop_training = True
       
            


   
  