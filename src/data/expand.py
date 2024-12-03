
import numpy as np
from typing import Dict, Optional, Any, Tuple

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def array_expansion(array: np.ndarray) -> np.ndarray:
    """
    Expand an array of shape (h, w) by rearranging its columns for data continuity.

    Parameters:
        array (np.ndarray): A numpy array to be expanded.

    Returns:
        output_array (np.ndarray): An array representing the expanded array.
    """
    mid = array.shape[1] // 2

    l_array = array[:, :mid]
    r_array = array[:, mid:]

    output_array = np.hstack((r_array, array, l_array))
    
    return output_array

def expand_multiple(data_dict: Dict[Any, np.ndarray]) -> Dict[Any, np.ndarray]:
    """
    Expand multiple arrays by rearranging its columns for data continuity.

    Parameters:
        data_dict (dict): Dictionary (ID: np.ndarray) of numpy arrays with shape (h, w).

    Returns:
        expanded_data (dict): Dictionary (ID: np.ndarray) of expanded arrays. 
    """
    expanded_data = {}

    for id in data_dict.keys():
        if data_dict[id] is None:
            expanded_data[id] = None
        else:
            expanded_data[id] = array_expansion(data_dict[id])

    return expanded_data

def data_expand(
        X: Dict[Any,np.ndarray],
        y: Dict[Any,np.ndarray] = None,
        ) -> Tuple[Dict[Any, np.ndarray], Dict[Any,np.ndarray]]:
    """
    Expand data arrays for cilindrical continuity.

    Parameters:
        X (dict): Dictionary (ID: np.ndarray) representing a feature array that will be expanded.
        y (dict): Dictionary (ID: np.ndarray) representing a targets array that will be expanded.
            Can be None if no targets exist. Default None.

    Returns:
        tuple: A tuple containing:
            X (dict): Dictionary (ID: np.ndarray) of expanded array of features.
            y (dict): Dictionary (ID: np.ndarray) of expanded array of targets.
    """
    X = expand_multiple(X)
    y = expand_multiple(y)

    return X, y

#-----------------------------------------------------------------------------------------------------------------------------------------------------
