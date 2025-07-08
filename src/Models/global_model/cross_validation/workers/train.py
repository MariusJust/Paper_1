
import numpy as np
import tensorflow as tf
import random

def train_itteration(self, train_idx):

        """
        Fits the model on the training set as defined by train idx and returns the model instance.
        """
        growth_train = {'global': self.growth['global'].loc[train_idx]}
        temp_train = {'global': self.temp['global'].loc[train_idx].reset_index(drop=True)} 
        precip_train = {'global': self.precip['global'].loc[train_idx].reset_index(drop=True)}
        
        x_train = [temp_train, precip_train]
        
        self.factory.node   = self.node
        self.factory.x_train = x_train
        self.factory.y_train = growth_train
        
        #  set seeds for reproducibility
        tf.random.set_seed(self.current_seed)
        np.random.default_rng(self.current_seed)
        random.seed(self.current_seed)

        model_instance=self.factory.get_model()
        model_instance.fit(lr=self.lr, min_delta=self.min_delta, patience=self.patience, verbose=self.verbose)
        
        return model_instance
    
    
    
def train_on_full_sample(self):
    """
    Trains the model on the full dataset using the best initialization seed.
    """
        
    best_seed = self.best_init_idx

    # Clear session and reinitialize seeds 
    tf.random.set_seed(best_seed)
    np.random.default_rng(best_seed)
    random.seed(best_seed)

    x_train = [self.temp, self.precip]
    growth_train = {'global': self.growth['global']}
    
    self.factory.node   = self.node
    self.factory.x_train = x_train
    self.factory.y_train = growth_train
    self.factory.Depth=len(self.factory.node)
        
    model_full = self.factory.get_model()
    # Train the model on the full dataset.
    model_full.fit(lr=self.lr, min_delta=self.min_delta, patience=self.patience, verbose=self.verbose)

    # Save the final model weights.
    if self.data is None:
        model_full.save_params('results/Model Parameters/CV/' + str(self.node) + '.weights.h5')
    else:
        self.model_full = model_full  # Store the model instance for later use in the Monte Carlo simulation

    return None
