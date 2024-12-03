
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def array_reduction(array: np.ndarray) -> np.ndarray:
    """
    Reverse the expansion operation on an array of shape (h, w), restoring the original array.

    Parameters:
        array (numpy.ndarray): Array that was expanded.

    Returns:
        output_array (numpy.ndarray): A numpy array of shape (h, w) representing the original array before expansion.
    
    """
    width = array.shape[1] // 4

    output_array = array[:, width:3*width]

    return output_array

#-----------------------------------------------------------------------------------------------------------------------------------------------------