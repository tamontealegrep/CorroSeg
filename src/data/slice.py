
import math
import numpy as np
from typing import Tuple, Union, Any, Dict

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def horizontal_slicing(X: np.ndarray, y: np.ndarray, section_size: int = 36, stride: int = 36, padding_value: Union[int,float] = 0) -> Tuple[list, list]:
    """
    Slice the input arrays horizontally into smaller sections with padding.

    This function takes two 2D arrays (X and y) and slices them into 
    smaller sections of a specified height (section_size) with a given stride.
    If there is insufficient data to create a full section, it pads the input arrays
    with a specified padding value.

    Parameters:
        X (numpy.ndarray): A 2D array representing the input data.
        y (numpy.ndarray): A 2D array representing the corresponding labels.
        section_size (int, optional): The height of each section to be sliced. Default is 36.
        stride (int, optional): The number of rows to move forward for the next slice. Default is 36.
        padding_value (float, optional): The value to use for padding. Default is 0. To use NaN for padding, set this argument to `np.nan`.

    Returns:
        tuple: A tuple containing:
            X_sections (list): List containing sliced sections of X.
            y_sections (list): List containing sliced sections of y.
    
    Raises:
        ValueError: If the heights of X and y do not match.
    """
    if y is None:
        y = np.zeros_like(X)

    if X.shape[0] != y.shape[0]:
        raise ValueError("Arrays must have the same height")
    
    height = X.shape[0]

    # Calculate the total height needed with padding
    num_sections = math.ceil((height - section_size) / stride + 1 )
    total_height = stride * (num_sections - 1) + section_size
    pad_height = max(0, total_height - height)

    # Pad the arrays with the specified padding value
    X_padded = np.pad(X, ((0, pad_height), (0, 0)), mode='constant', constant_values=padding_value)
    y_padded = np.pad(y, ((0, pad_height), (0, 0)), mode='constant', constant_values=padding_value)

    X_sections = []
    y_sections = []

    for i in range(num_sections):
        start = i * stride
        X_section = X_padded[start:start + section_size, :]
        y_section = y_padded[start:start + section_size, :]

        X_sections.append(X_section)
        y_sections.append(y_section)

    return X_sections, y_sections

def vertical_slicing(X: np.ndarray, y: np.ndarray, section_size: int = 36, stride: int = 36, padding_value: float = 0) -> Tuple[list, list]:
    """
    Slice the input arrays vertically into smaller sections with padding.

    This function takes two 2D arrays (X and y) and slices them into 
    smaller sections of a specified width (section_size) with a given stride.
    If there is insufficient data to create a full section, it pads the input arrays
    with a specified padding value.

    Parameters:
        X (numpy.ndarray): A 2D array representing the input data.
        y (numpy.ndarray): A 2D array representing the corresponding labels.
        section_size (int, optional): The width of each section to be sliced. Default is 36.
        stride (int, optional): The number of columns to move forward for the next slice. Default is 36.
        padding_value (float, optional): The value to use for padding. Default is 0. To use NaN for padding, set this argument to `np.nan`.

    Returns:
        tuple: A tuple containing:
            X_sections (list): List containing sliced sections of X.
            y_sections (list): List containing sliced sections of y.

    Raises:
        ValueError: If the widths of X and y do not match.
    """
    if y is None:
        y = np.zeros_like(X)
        
    if X.shape[1] != y.shape[1]:
        raise ValueError("Arrays must have the same width")
    
    width = X.shape[1]
    
    # Calculate the total width needed with padding
    num_sections = math.ceil((width - section_size) / stride + 1 )
    total_height = stride * (num_sections - 1) + section_size
    pad_width = max(0, total_height - width)

    # Pad the arrays with the specified padding value
    X_padded = np.pad(X, ((0, 0), (0, pad_width)), mode='constant', constant_values=padding_value)
    y_padded = np.pad(y, ((0, 0), (0, pad_width)), mode='constant', constant_values=padding_value)

    X_sections = []
    y_sections = []

    for i in range(num_sections):
        start = i * stride
        X_section = X_padded[:, start:start + section_size]
        y_section = y_padded[:, start:start + section_size]

        if X_sections and np.array_equal(X_sections[0], X_section):
            break
        
        X_sections.append(X_section)
        y_sections.append(y_section)

    return X_sections, y_sections

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def data_slice(X: np.ndarray,
               y: np.ndarray = None,
               height_size: int = 360,
               height_stride: int = 360,
               width_size: int = 36,
               width_stride: int = 18,
               padding_value: int = 0) -> Tuple[np.ndarray, np.ndarray]:
    """
    Slices input and target data arrays into smaller patches by performing horizontal 
    and vertical slicing.

    The function first divides the input array `X` into horizontal sections 
    of a specified height, then further slices those sections vertically into smaller 
    patches of specified width. Each slice can include padding if the original arrays 
    are smaller than the specified sizes.

    Parameters:
        X (np.ndarray): A numpy array representing the input data.
        y (np.ndarray, optional): A numpy array representing the target data. Default is None.
        height_size(int, optional): The height of each section to be sliced. Default is 360.
        height_stride (int, optional): The number of rows to move forward for the next slice. Default is 360.
        width_size (int, optional): The width of each section to be sliced. Default is 36.
        width_stride (int, optional): The number of columns to move forward for the next slice. Default is 18.
        padding_value (int, optional): The value used for padding the input arrays. Default is 0.

    Returns:
        Tuple[np.ndarray, np.ndarray]: 
        A tuple containing two numpy arrays:
            X_output (np.ndarray): Contains the sliced input patches.
            y_output (np.ndarray): Contains the sliced target patches, or None if `y` is not provided.
              
    Raises:
        ValueError: If the number of input slices does not match the number of target slices.

    """
    X_slices, y_slices = horizontal_slicing(X, y, height_size, height_stride, padding_value)

    if len(X_slices) != len(y_slices):
        raise ValueError("The number of input slices does not match the number of target slices.")
    
    X_output = []
    y_output = []

    for i in range(len(X_slices)):
        X_patches, y_patches = vertical_slicing(X_slices[i], y_slices[i], width_size, width_stride, padding_value)
        X_output.extend(X_patches)
        y_output.extend(y_patches)
    
    X_output = np.stack(X_output, axis=0)
    y_output = np.stack(y_output, axis=0)

    if y is None:
        return X_output, None
    else:
        return X_output, y_output
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------