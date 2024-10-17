
import os
import yaml
import pickle
import numpy as np
from typing import Any

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def save_pkl(obj:Any, file_path:str) -> None:
    """
    Save an object to a pickle file.

    Args:
        obj(Any): The object to be serialized and saved.
        file_path (str): The path to the file where the object will be saved.

    Returns:
        None
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_pkl(file_path:str) -> Any:
    """
    Load an object from a pickle file.

    Args:
        file_path (str): The path to the file from which the object will be loaded.

    Returns:
        Any: The object loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_config(file_path:str) -> dict:
    """
    Load configuration from a YAML file.

    Args:
        file_path (str): The path to the YAML file to be loaded.

    Returns:
        dict: A dictionary containing the loaded configuration.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def save_dict_arrays(folder_path:str, X:dict, y:dict=None) -> None:
    """
    Save two lists of arrays to specified folders as .npy files.

    Parameters:
        folder_path (str): Directory path to save the arrays.
        X (list): List of arrays to be saved.
        y (list): List of arrays to be saved (can be None).

    Returns:
        None

    """
    X_folder = os.path.join(folder_path, "X")
    y_folder = os.path.join(folder_path, "y")

    # Save X arrays
    os.makedirs(X_folder, exist_ok=True)
    for id, array in X.items():
        if array is not None:
            np.save(os.path.join(X_folder, f"X_{id}.npy"), array)

    # Save y arrays
    if y is not None and not all(i is None for i in y.values()):
        os.makedirs(y_folder, exist_ok=True)
        for id, array in y.items():
            if array is not None:
                np.save(os.path.join(y_folder, f"y_{id}.npy"), array)

#-----------------------------------------------------------------------------------------------------------------------------------------------------