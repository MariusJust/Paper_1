import os
import numpy as np
import os
# import yaml


def save_numpy(filepath, array):
    """
    Save a NumPy array to the given filepath, creating directories if needed.

    Parameters:
    - filepath (str): Full path (including filename) where the array should be saved.
    - array (np.ndarray): The NumPy array to save.
    """
    # Extract directory from filepath
    dir_path = os.path.dirname(filepath)
    
    # Create directory if it doesn't exist
    os.makedirs(dir_path, exist_ok=True)
    
    # Save the NumPy array
    np.save(filepath, array)


def save_yaml(filepath, data):
    # """
    # Save data to a YAML file at the given filepath, creating directories if needed.

    # Parameters:
    # - filepath (str): Full path (including filename) where the YAML file should be saved.
    # - data (dict): The data to save in YAML format.
    # """
    # # Extract directory from filepath
    # dir_path = os.path.dirname(filepath)
    
    # # Create directory if it doesn't exist
    # os.makedirs(dir_path)
    
    # # Save the data to a YAML file with pretty formatting
    # with open(filepath, 'w') as file:
    #     yaml.dump(data, file, default_flow_style=False, sort_keys=False)
    return None

def save_model_weights(filepath, model):
    """
    Save only the weights of a Keras model to the given filepath, creating directories if needed.

    Parameters:
    - filepath (str): Full path (including filename) where the weights should be saved.
    - model (tf.keras.Model): The Keras model whose weights should be saved.
    """
    dir_path = os.path.dirname(filepath)
    os.makedirs(dir_path, exist_ok=True)
    model.save_params(filepath)
