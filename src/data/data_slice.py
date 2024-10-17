
import numpy as np
from typing import Tuple, Union, Any, Dict
from src.data.utils.slice import horizontal_slicing, vertical_slicing
#-----------------------------------------------------------------------------------------------------------------------------------------------------

def data_slice(X: Dict[Any,np.ndarray],
               y: Dict[Any,np.ndarray],
               h_section_size:int=360,
               h_stride:int=360,
               v_section_size:int=36,
               v_stride:int=18,
               padding_value:int=0) -> Tuple[Dict[Any, Dict[int, np.ndarray]], Dict[Any, Dict[int, np.ndarray]]]:
    """
    Slices input and target data arrays into smaller patches by performing horizontal 
    and vertical slicing.

    The function first divides each input array in `X` into horizontal sections 
    of a specified height, then further slices those sections vertically into smaller 
    patches of specified width. Each slice can include padding if the original arrays 
    are smaller than the specified sizes.

    Parameters:
        X (Dict[Any, np.ndarray]): A dictionary where keys are identifiers and values are numpy arrays representing input data.
        y (Dict[Any, np.ndarray]): A dictionary where keys correspond to X and values are numpy arrays representing target data. It can be None.
        h_section_size (int, optional): The height of each section to be sliced. Default is 360.
        h_stride (int, optional): The number of rows to move forward for the next slice. Default is 360.
        v_section_size (int, optional): The width of each section to be sliced. Default is 36.
        v_stride (int, optional): The number of columns to move forward for the next slice. Default is 36.
        padding_value (int, optional): The value used for padding the input arrays. Default is 0.

    Returns:
        Tuple[Dict[Any, Dict[int, np.ndarray]], Dict[Any, Dict[int, np.ndarray]]]: 
        A tuple containing two dictionaries:
            X_output (dict): Contains the sliced input patches, where each key 
              corresponds to an identifier from `X`, and each value is another dictionary 
              mapping patch indices to the sliced numpy arrays.
            y_output (dict): Contains the sliced target patches, structured similarly 
              to the first.
              
    Raises:
        ValueError: If the number of input slices does not match the number of target slices.

    """
    return_none = False
    if y is None:
        return_none = True
        y = {}
        for id in X.keys():
            y[id] = None

    X_data = {}
    y_data = {}

    for id in X.keys():
        X_slices, y_slices = horizontal_slicing(X[id], y[id], h_section_size, h_stride, padding_value)
        X_data[id] = X_slices
        y_data[id] = y_slices

    if len(X_data) != len(y_data):
        raise ValueError("")
    
    
    X_output = {}
    y_output = {}

    for id in X_data.keys():
        X_v_slices = {}
        y_v_slices = {}
        patch_id = 0
        for i in range(len(X_data[id])):
            X_patches, y_patches = vertical_slicing(X_data[id][i], y_data[id][i], v_section_size, v_stride, padding_value)
            for j in range(len(X_patches)):
                X_v_slices[patch_id] = X_patches[j]
                y_v_slices[patch_id] = y_patches[j]
                patch_id += 1

        X_output[id] = X_v_slices
        y_output[id] = y_v_slices

    if return_none:
        y_output = None

    return X_output, y_output

#-----------------------------------------------------------------------------------------------------------------------------------------------------