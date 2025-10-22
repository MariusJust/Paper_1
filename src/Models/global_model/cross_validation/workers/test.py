
import numpy as np
import tensorflow as tf



def test_itteration(self, test_idx, country_FE, time_FE):
        """
        takes the test index, country fixed effect and time fixed effect as inputs
        makes a prediction on the test set by appending the fixed effects to the base neural network model
        returns the mean squared error of the prediction.
        """
       # Prepare test sets
        growth_test = np.array(self.growth['global'].loc[test_idx].reset_index(drop=True)).reshape(1, 1, -1, 1)
        temp_test = np.array(self.temp['global'].loc[test_idx].reset_index(drop=True)).reshape((1, 1, -1, 1))

        precip_test = np.array(self.precip['global'].loc[test_idx].reset_index(drop=True)).reshape((1, 1, -1, 1))
        x_test = tf.concat([temp_test, precip_test], axis=3)
        
        # Reshape the fixed effects to match the input shape of the model
        full_country_FE = np.concatenate([[0.], np.array(country_FE)[0]], axis=0)  
        full_country_FE=tf.reshape(full_country_FE, (1, 1, -1, 1))    
        time_FE=tf.reshape(time_FE, (1, 1, 1,1))
        
        # predicting x_test by using the neural network (as defined in model_pred) and appending the fixed effects
        preds = np.reshape(self.model_instance.model_pred.predict([x_test, full_country_FE, time_FE], verbose=self.verbose),  (-1, 1), order='F')
        mse = np.nanmean((preds - growth_test) ** 2)
        return mse
    
def test_dynamic(self, test_idx, country_FE):
        """
        takes the test index, country fixed effect as inputs
        makes a prediction on the test set by appending the fixed effects to the base neural network model
        returns the mean squared error of the prediction.
        """
       # Prepare test sets
        growth_test = np.array(self.growth['global'].loc[test_idx].reset_index(drop=True)).reshape(1, 1, -1, 1)
        temp_test = np.array(self.temp['global'].loc[test_idx].reset_index(drop=True)).reshape((1, 1, -1, 1))
        precip_test = np.array(self.precip['global'].loc[test_idx].reset_index(drop=True)).reshape((1, 1, -1, 1))
        test_int=np.where(test_idx==True)[0][0]
        test_array= np.repeat(np.array(test_int).reshape(1, 1, 1, 1), repeats=temp_test.shape[2], axis=2)
        
        
        x_test = tf.concat([temp_test, precip_test, test_array], axis=3)

        
        # Reshape the fixed effects to match the input shape of the model
        full_country_FE = np.concatenate([[0.], np.array(country_FE)[0]], axis=0)
        full_country_FE = tf.reshape(full_country_FE, (1, 1, -1, 1))
        
        # predicting x_test by using the neural network (as defined in model_pred) and appending the fixed effects
        preds = np.reshape(self.model_instance.model_pred.predict([x_test, full_country_FE], verbose=self.verbose),  (-1, 1), order='F')
        mse = np.nanmean((preds - growth_test) ** 2)
        return mse
