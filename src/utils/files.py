
import os
import yaml
import pickle
import numpy as np
from typing import Any, Tuple, Optional, Dict

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def save_pkl(obj: Any,
             file_path: str) -> None:
    """
    Save an object to a pickle file.

    Parameters:
        obj(Any): The object to be serialized and saved.
        file_path (str): The path to the file where the object will be saved.

    Returns:
        None
    """
    with open(file_path, 'wb') as file:
        pickle.dump(obj, file)

def load_pkl(file_path: str) -> Any:
    """
    Load an object from a pickle file.

    Parameters:
        file_path (str): The path to the file from which the object will be loaded.

    Returns:
        (Any): The object loaded from the pickle file.
    """
    with open(file_path, 'rb') as file:
        return pickle.load(file)

def load_config(file_path: str) -> dict:
    """
    Load configuration from a YAML file.

    Parameters:
        file_path (str): The path to the YAML file to be loaded.

    Returns:
        (dict): A dictionary containing the loaded configuration.

    Raises:
        FileNotFoundError: If the specified file does not exist.
        yaml.YAMLError: If there is an error parsing the YAML file.
    """
    with open(file_path, "r") as file:
        config = yaml.safe_load(file)
    return config

def npy_file_to_dict(file_path: str) -> Dict[str, np.ndarray]:
    """
    Loads a single .npy file from the specified path and returns its contents
    in a dictionary with the file name (without extension) as the key.

    Parameters:
        file_path (str): The full path of the file to be loaded.

    Returns:
        dict: A dictionary with the file name (without extension) as the key
              and the contents of the file (np.ndarray) as the value.
    """
    data_dict = {}

    if os.path.exists(file_path) and file_path.endswith(".npy"):
        file_name = os.path.basename(file_path)
        file_id = file_name.split(".")[0]
        data_dict[file_id] = np.load(file_path)

    return data_dict

def load_arrays_from_folders(folder_path: str) -> Tuple[Dict[str, np.ndarray], Dict[str, np.ndarray]]:
    """
    Loads arrays from 'X' and 'y' subdirectories inside the specified folder path.
    Uses the load_single_file_to_dict function to load each file into a dictionary.

    Parameters:
        folder_path (str): The directory path where the 'X' and 'y' subfolders are located.

    Returns:
        (tuple): A tuple containing.
            X (dict): A dictionary with the arrays in the 'X' subfolder.
            y (dict): A dictionary with the arrays in the 'y' subfolder.
    """
    X_folder = os.path.join(folder_path, "X")
    y_folder = os.path.join(folder_path, "y")

    X = {}
    if os.path.exists(X_folder):
        for file_name in os.listdir(X_folder):
            if file_name.endswith(".npy"):
                file_path = os.path.join(X_folder, file_name)
                X.update(npy_file_to_dict(file_path))

    y = {}
    if os.path.exists(y_folder):
        for file_name in os.listdir(y_folder):
            if file_name.endswith(".npy"):
                file_path = os.path.join(y_folder, file_name)
                y.update(npy_file_to_dict(file_path))

    return X, y

#-----------------------------------------------------------------------------------------------------------------------------------------------------