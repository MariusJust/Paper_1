from tensorflow.python.keras.backend import count_params

def Count_params(self):
    
    ### needs to be updated count params is deprecated
    """
    Counting the number of parameters in the model.

    RETURNS
        number of parameters in the model.
    """



    return sum(count_params(w) for w in self.model.trainable_weights)
