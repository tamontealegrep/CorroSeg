
import numpy as np
from src.data.utils.slice import horizontal_slicing, vertical_slicing
from typing import Any, List, Tuple, Dict

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def split_list(data: List, fraction: float = 0.2, seed: int = None) -> Tuple[List, List]:
    """
    Split a list into two subsets: a larger subset and a smaller subset.

    This function randomly shuffles the input data and splits it into two subsets
    based on a specified fraction. The first subset contains the specified fraction 
    of the data, while the second subset contains the remaining data.

    Parameters:
        data (list): A list of data samples to be split.
        fraction (float, optional): The fraction of the dataset to include in the smaller subset.
        seed (int, optional): A seed for the random number generator to ensure reproducibility.

    Returns:
        tuple: A tuple containing:
            larger_subset (list): The larger subset of the data.
            smaller_subset (list): The smaller subset of the data.
    """
    if seed is not None:
        np.random.seed(seed)

    num_samples = len(data)
    split_index = int(num_samples * (1 - fraction))
    shuffled_indices = list(np.random.permutation(num_samples))

    larger_indices = shuffled_indices[:split_index]
    smaller_indices = shuffled_indices[split_index:]

    larger_subset = [data[i] for i in larger_indices]
    smaller_subset = [data[i] for i in smaller_indices]

    return larger_subset, smaller_subset

def data_split(X:Dict[Any,np.ndarray], y:Dict[Any,np.ndarray], h_section_size:int=360, h_stride:int=360, v_section_size:int=36, v_stride:int=18, padding_value:int=0, fraction:float=0.2, seed:int = None) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
    """
    Split data into larger and smaller subsets from dictionaries of arrays.

    This function takes two dictionaries of numpy arrays, slices the arrays into smaller sections,
    and splits the resulting data into larger and smaller subsets based on a specified fraction.
    The function also supports padding of the input arrays to ensure consistent section sizes.

    Parameters:
        X (Dict[Any, np.ndarray]): A dictionary where keys are identifiers and values are numpy arrays representing input data.
        y (Dict[Any, np.ndarray]): A dictionary where keys correspond to X and values are numpy arrays representing target data.
        h_section_size (int, optional): The height of each section to be sliced. Default is 360.
        h_stride (int, optional): The number of rows to move forward for the next slice. Default is 360.
        v_section_size (int, optional): The width of each section to be sliced. Default is 36.
        v_stride (int, optional): The number of columns to move forward for the next slice. Default is 18.
        padding_value (int, optional): The value used for padding the input arrays. Default is 0.
        fraction (float, optional): The fraction of the dataset to include in the smaller subset. Default is 0.2.
        seed (int, optional): A seed for the random number generator to ensure reproducibility. If None, a random seed will be generated.

    Returns:
        tuple: A tuple containing:
            X_larger_output (np.ndarray): Array of larger subsets of input data.
            y_larger_output (np.ndarray): Array of larger subsets of target data.
            X_smaller_output (np.ndarray): Array of smaller subsets of input data.
            y_smaller_output (np.ndarray): Array of smaller subsets of target data.

    Raises:
        ValueError: If the lengths of X and y do not match or if an error occurs during slicing.
    """
    X_data = []
    y_data = []

    for id in X.keys():
        X_slices, y_slices = horizontal_slicing(X[id], y[id], h_section_size, h_stride, padding_value)
        X_data += X_slices
        y_data += y_slices

    if len(X_data) != len(y_data):
        raise ValueError("")

    if seed is None:
        seed = np.random.randint(0,1000000)

    data = list(zip(X_data,y_data))

    larger_subset, smaller_subset = split_list(data,fraction,seed)

    X_larger_output = []
    y_larger_output = []

    for i in range(len(larger_subset)):
        X_larger_subset, y_larger_subset = vertical_slicing(larger_subset[i][0], larger_subset[i][1], v_section_size, v_stride ,padding_value)
        X_larger_output += X_larger_subset
        y_larger_output += y_larger_subset

    X_smaller_output = []
    y_smaller_output = []

    for i in range(len(smaller_subset)):   
        X_smaller_subset, y_smaller_subset = vertical_slicing(smaller_subset[i][0], smaller_subset[i][1], v_section_size, v_stride ,padding_value)
        X_smaller_output += X_smaller_subset
        y_smaller_output += y_smaller_subset

    X_larger_output = np.stack(X_larger_output, axis=0)
    y_larger_output = np.stack(y_larger_output, axis=0)
    X_smaller_output = np.stack(X_smaller_output, axis=0)
    y_smaller_output = np.stack(y_smaller_output, axis=0)

    return X_larger_output, y_larger_output, X_smaller_output, y_smaller_output

#-----------------------------------------------------------------------------------------------------------------------------------------------------