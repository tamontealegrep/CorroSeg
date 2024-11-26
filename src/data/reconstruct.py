
import os
import numpy as np
import pandas as pd
from typing import Any, List, Dict, Tuple, Optional, Union
from src.data.expand import expand_multiple
from src.utils.files import save_dict_arrays

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def find_missing_keys(dictionary:dict) -> set:
    """
    Check for missing keys in a sequence of numeric keys in a dictionary.

    Parameters:
        dictionary (dict): A dictionary with numeric keys.

    Returns:
        set: A set of missing keys in the sequence from the minimum to the maximum key. If there are no missing keys, an empty set is returned.
    
    """
    keys = sorted(dictionary.keys())
    
    min_key = min(keys)
    max_key = max(keys)

    expected_keys = set(range(min_key, max_key + 1))
    actual_keys = set(keys)

    missing_keys = expected_keys - actual_keys
    
    return missing_keys

def load_npy_files(folder_path:str, prefix:str=None) -> List[np.ndarray]:
    """
    Load .npy files from a specified folder, with an optional prefix filter.

    Args:
        folder_path (str): The path to the folder containing .npy files.
        prefix (str, optional): A prefix to filter files. Only files starting with this prefix will be loaded. If None, all .npy files will be loaded.
                                
                                
    Returns:
        list: A sorted list of loaded data arrays. If a prefix is provided and some IDs are missing, a warning will be printed with the missing IDs.
    
    """
    files = {}
    for filename in os.listdir(folder_path):
        if filename.endswith('.npy') and (prefix is None or filename.startswith(prefix)):
            file_path = os.path.join(folder_path, filename)
            if prefix is not None:
                id = int(filename.split(prefix)[1].split(".")[0])
            else:
                id = filename.split(".")[0]
            files[id] = np.load(file_path)

    if prefix is not None:
        missing_keys = find_missing_keys(files)
        if missing_keys:
            print(f"Warning: Sequence for {prefix} is incomplete. Missing IDs: {missing_keys}")
    
    files = [files[key] for key in sorted(files.keys())]

    return files

def load_csv_and_reshape(file_path:str, prefix:str=None, shape:Tuple[int,int]=(36, 36)) -> List[np.ndarray]:
    """
    Load a CSV file and reshape each row into a specified 2D shape, optionally filtering by index prefix.

    Args:
        file_path (str): The path to the CSV file.
        prefix (str, optional): The prefix to filter rows by index. If None, all rows will be loaded.
        shape (tuple, optional): The desired shape for reshaping each row .Default is (36, 36).
        
    Returns:
        list: A sorted list of numpy.ndarray reshaped to the specified dimensions. 

    """
    file = pd.read_csv(file_path, index_col=0)
    data = {}

    for index, row in file.iterrows():
        if prefix is None or index.startswith(prefix):
            sample = np.array(row).reshape(shape)
            if prefix is not None:
                id = int(index.split(prefix)[1])
            else:
                id = index
                
            data[id] = sample

    if prefix is not None:
        missing_keys = find_missing_keys(data)
        if missing_keys:
            print(f"Warning: Sequence for {prefix} is incomplete. Missing indices: {missing_keys}")

    data = [data[key] for key in sorted(data.keys())]

    return data

def load_raw_data(X_folder: str, y_file: Optional[str] = None) -> Tuple[dict,dict]:
    """
    Load training and testing data from specified directories.

    Parameters:
        X_folder (str): Path to the folder containing X.
        y_file (str): Path to the CSV file containing y.

    Returns:
        tuple: A tuple containing:
          mapping well IDs to tuples of (X_raw, y_raw).
            X_raw (dict): Maps IDs to the data on X_folder.
            y_raw (dict): Maps IDs to the data on y_file.

    """
    # Get IDs
    ids = sorted({int(filename.split("_")[1]) for filename in os.listdir(X_folder)})

    X_raw_data = {id: load_npy_files(X_folder, f"well_{id}_patch_") for id in ids}

    y_raw_data = {}
    if y_file is not None:
        y_raw_data = {id: load_csv_and_reshape(y_file, f"well_{id}_patch_") for id in ids}

    X_raw = {id:X_raw_data[id] for id in ids}
    y_raw = {id:y_raw_data.get(id) for id in ids}

    return X_raw, y_raw

def array_reconstruct(patches:List[np.ndarray], num_columns:int) -> np.ndarray:
    """
    Reconstruct an array from a list of numpy arrays (patches) into a grid format.

    Parameters:
        patches (list): A list of 2D numpy arrays representing array patches.
        num_columns (int): The number of columns in the reconstructed image.

    Returns:
        numpy.ndarray: A 2D reconstructed array.
    
    """
    num_patches = len(patches)
    num_rows = (num_patches + num_columns - 1) // num_columns

    patch_h, patch_w = patches[0].shape

    # Create an output array with the appropriate dimensions
    output_array = np.zeros((num_rows * patch_h, num_columns * patch_w))

    for i, patch in enumerate(patches):
        row = i // num_columns
        column = i % num_columns
        output_array[row * patch_h: (row + 1) * patch_h, column * patch_w: (column + 1) * patch_w] = patch

    return output_array

def reconstruct_multiple(data_dict: Dict[Any, Optional[List[np.ndarray]]], column_config: Dict[Any, int]) -> Dict[Any, Optional[np.ndarray]]:
    """
    Reconstruct multiple arrays based on specified column configurations.

    Parameters:
        data_dict (dict): Dictionary (ID: list[np.ndarray]) of list of 2D numpy arrays.
        column_config (dict): Dictionary (ID: num_columns) to reconstruct the arrays.

    Returns:
        reconstructed_data (dict): Dictionary (ID: reconstructed_array) of reconstructed arrays.
        
    """
    reconstructed_data = {}

    for id in data_dict.keys():
        if data_dict[id] is None:
            reconstructed_data[id] = None
        else:
            reconstructed_data[id] = array_reconstruct(data_dict[id], column_config[id])

    return reconstructed_data

def data_reconstruct(X_folder_path:str,y_file_path:Union[str,None],column_config:dict, save_path:str=None) -> Tuple[Dict[Any, np.ndarray], Dict[Any,np.ndarray]]:
    """
    Load and reconstruct data arrays from patch files.

    This function takes the path to a folder containing patch files for features (X) 
    and a file for labels (y). The data is reconstructed using a provided column 
    configuration. If specified, the resulting arrays are saved to the given path.

    Parameters:
        X_folder_path (str): Path to the folder containing the X patch files.
        y_file_path (str): Path to the file containing the y patches. Can be None if (y) does not exist.
        column_config (dict): Configuration that defines how the data should be reconstructed.
        save_path (str, optional): Path where the unexpanded arrays will be saved. If None, the arrays are not saved.

    Returns:
        tuple: A tuple containing:
            X (dict): Dictionary (ID: numpy.ndarray) of reconstructed array of features.
            y (dict): Dictionary (ID: numpy.ndarray) of reconstructed array of labels.

    Raises:
        FileNotFoundError: If the provided paths do not exist.

    """
    X_patches, y_patches = load_raw_data(X_folder_path,y_file_path)

    X = reconstruct_multiple(X_patches,column_config)
    y = reconstruct_multiple(y_patches,column_config)

    if save_path is not None:
        save_dict_arrays(save_path, X, y)

    return X, y

#-----------------------------------------------------------------------------------------------------------------------------------------------------