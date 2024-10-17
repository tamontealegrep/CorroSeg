
import numpy as np
from typing import Dict, List, Optional, Any

#-----------------------------------------------------------------------------------------------------------------------------------------------------

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

#-----------------------------------------------------------------------------------------------------------------------------------------------------