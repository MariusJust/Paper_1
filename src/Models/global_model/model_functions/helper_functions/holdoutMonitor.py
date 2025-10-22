import tensorflow as tf
import numpy as np 
from tensorflow.keras.layers import Add
from sklearn.linear_model import LinearRegression

class HoldoutMonitor(tf.keras.callbacks.Callback):
    def __init__(self, mod, patience, min_delta, verbose, P_matrix_train=None, P_matrix_val=None):
        super().__init__()
        self.model = mod
        self.patience = patience
        self.min_delta = min_delta
        self.verbose = verbose
        self.P_matrix_train = P_matrix_train
        self.P_matrix_val = P_matrix_val

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
        precip_val=tf.reshape(np.array(self.mod.input_data_precip_val), (1, 1, -1, 1))
        temp_val=tf.reshape(np.array(self.mod.input_data_temp_val), (1, 1, -1, 1))
        X_in=tf.concat([temp_val, precip_val], axis=3)
        climate_part= self.mod.model_visual([X_in]) 
        
        final_prediction=tf.matmul(self.P_matrix_val, climate_part)

        true_values=self.model_ref.y_val_transf

        if final_prediction.shape != np.array(true_values).shape:
            raise ValueError(f"Shape mismatch: final_pred.shape={final_prediction.shape}, targets_val.shape={np.array(true_values).shape}."
                             " Adjust how climate_pred / country_FE / time_FE are constructed.")

        mse = float(np.nanmean((final_prediction - true_values) ** 2))
        
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
       
            
    # def _predict_time_FE(self, time_FE_vector):
        
    #     #definning regressors
    #     X=np.arange(len(time_FE_vector)).reshape(-1, 1)+1
    #     Y=time_FE_vector
        
    #     #making the linear model
    #     model = LinearRegression() 
    #     model.fit(X, Y)
        
    #     #make prediction vector for the holdout set 
    #     pred_vector=np.arange(len(time_FE_vector), len(time_FE_vector)+self.mod.holdout).reshape(-1,1) + 1
        
    #     return model.predict(pred_vector)
        
        
        

   
  