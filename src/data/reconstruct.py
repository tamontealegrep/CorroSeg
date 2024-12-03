
import math
import numpy as np
from typing import Optional, Tuple 

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def data_reconstruct(
        X: np.ndarray,
        height: int,
        width: int,
        height_stride: int,
        width_stride: int, 
        average: Optional[bool] = False,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Reconstructs a larger 2D array from smaller patches.

    This function reconstructs a 2D array from patches by placing the patches 
    into their corresponding locations according to the given strides. 
    If overlapping patches are placed in the same region, they can be averaged 
    to avoid duplication.

    Parameters:
        X (np.ndarray): A 2D array containing patches of data, with shape (num_patches, patch_height, patch_width).
        height (int): The total height of the reconstructed array.
        width (int): The total width of the reconstructed array.
        height_stride (int): The stride (step size) in the height direction during the patch placement.
        width_stride (int): The stride (step size) in the width direction during the patch placement.
        average (bool, optional): A flag to indicate whether to average overlapping patches. 
            If True, the sum of overlapping patches will be divided by the number of overlapping patches. Default is False.

    Returns:
        X_output (np.ndarray): The reconstructed data, with shape (height, width).

    """
    patch_height = X.shape[-2]
    patch_width = X.shape[-1]

    # Calculate the num rows and columns
    rows = math.ceil((height - patch_height) / height_stride + 1 )
    columns = math.ceil((width - patch_width) / width_stride + 1 )
    
    total_height = height_stride * (rows - 1) + patch_height
    total_width = width_stride * (columns - 1) + patch_width

    X_reconstructed = np.zeros((total_height, total_width))
    if average:
        X_counter = np.zeros((total_height, total_width))

    for idx in range(X.shape[0]):
        i = idx // columns  # row
        j = idx % columns   # column

        row_start = i * height_stride
        col_start = j * width_stride

        X_reconstructed[row_start:row_start+patch_height, col_start:col_start+patch_width] += X[idx]
        if average:
            X_counter[row_start:row_start+patch_height, col_start:col_start+patch_width] += 1

    if average:
        X_reconstructed = np.divide(X_reconstructed, X_counter, where=X_counter != 0)

    X_output = X_reconstructed[:height, :width]

    return X_output

#-----------------------------------------------------------------------------------------------------------------------------------------------------