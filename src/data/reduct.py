
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def array_reduction(array: np.ndarray) -> np.ndarray:
    """
    Reverse the expansion operation on a 2D array, restoring the original array.

    Parameters:
        array (numpy.ndarray): A 2D numpy array that was expanded.

    Returns:
        numpy.ndarray: A 2D numpy array representing the original array before expansion.
    
    """
    width = array.shape[1] // 4

    output_array = array[:, width:3*width]

    return output_array

#-----------------------------------------------------------------------------------------------------------------------------------------------------