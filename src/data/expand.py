
import numpy as np
from typing import Dict, List, Optional, Any, Tuple

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def array_expansion(array:np.ndarray) -> np.ndarray:
    """
    Expand an array by rearranging its columns.

    Parameters:
        array (numpy.ndarray): A 2D numpy array to be expanded.

    Returns:
        numpy.ndarray: A 2D numpy array representing the expanded array.
    
    """
    mid = array.shape[1] // 2

    l_array = array[:, :mid]
    r_array = array[:, mid:]

    output_array = np.hstack((r_array, array, l_array))
    
    return output_array

def expand_multiple(data_dict: Dict[Any, Optional[List[np.ndarray]]]) -> Dict[Any, Optional[np.ndarray]]:
    """
    Expand multiple arrays by rearranging its columns for data continuity.

    Parameters:
        data_dict (dict): Dictionary (ID: list[np.ndarray]) of list of 2D numpy arrays.

    Returns:
        expanded_data (dict): Dictionary (ID: expanded_array) of expanded arrays.
        
    """
    expanded_data = {}

    for id in data_dict.keys():
        if data_dict[id] is None:
            expanded_data[id] = None
        else:
            expanded_data[id] = array_expansion(data_dict[id])

    return expanded_data

def data_expand(X:Dict[Any,np.ndarray], y:Dict[Any,np.ndarray]=None) -> Tuple[Dict[Any, np.ndarray], Dict[Any,np.ndarray]]:
    """
    Expand data arrays.

    This function expands the data for cilindrical continuity.

    Parameters:
        X (dict): Dictionary where each key corresponds to an ID and each value is a numpy.ndarray 
                  representing a feature array that will be expanded.
        y (dict): Dictionary where each key corresponds to an ID and each value is a numpy.ndarray 
                  representing a label array that will be expanded. Can be None if no labels exist. Default None.

    Returns:
        tuple: A tuple containing:
            X (dict): Dictionary (ID: numpy.ndarray) of expanded array of features.
            y (dict): Dictionary (ID: numpy.ndarray) of expanded array of labels.

    """
    X = expand_multiple(X)
    y = expand_multiple(y)

    return X, y
#-----------------------------------------------------------------------------------------------------------------------------------------------------
