
import numpy as np

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def v_cutmix(array_1:np.ndarray, array_2:np.ndarray, alpha:float) -> np.ndarray:
    """
    Perform Vertical CutMix data augmentation by mixing two arrays based on a given alpha value.
    
    Args:
        array_1 (np.ndarray): The first array of shape (H, W) or (H, W, C).
        array_2 (np.ndarray): The second array of the same shape as array_1.
        alpha (float): A value between 0 and 1 that determines the proportion of the first image to use.
    
    Returns:
        output_array (np.ndarray): A new array that is a combination of array_1 and array_2, 
                    with the upper portion taken from array_1 and the rest from array_2.
    
    Raises:
        ValueError: If array_1 and array_2 do not have the same shape or if alpha is not between 0 and 1.
    """
    if array_1.shape != array_2.shape:
        raise ValueError("Input arrays must have the same shape.")
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be a value between 0 and 1.")

    h, w = array_1.shape[:2]
    cut_height = int(alpha * h)

    u_array = array_1[:cut_height, :]
    l_array = array_2[cut_height:, :]
    output_array = np.vstack((u_array, l_array))
    
    return output_array

def h_cutmix(array_1:np.ndarray, array_2:np.ndarray, alpha:float) -> np.ndarray:
    """
    Perform Horizontal CutMix data augmentation by mixing two arrays based on a given alpha value.
    
    Args:
        array_1 (np.ndarray): The first array of shape (H, W).
        array_2 (np.ndarray): The second array of the same shape as array_1.
        alpha (float): A value between 0 and 1 that determines the proportion of the first image to use.
    
    Returns:
        output_array (np.ndarray): A new array that is a combination of array_1 and array_2, 
                    with the left portion taken from array_1 and the rest from array_2.
    
    Raises:
        ValueError: If array_1 and array_2 do not have the same shape or if alpha is not between 0 and 1.
    """
    if array_1.shape != array_2.shape:
        raise ValueError("Input arrays must have the same shape.")
    if not (0 <= alpha <= 1):
        raise ValueError("Alpha must be a value between 0 and 1.")

    h, w = array_1.shape[-2:]
    cut_width = int(alpha * w)

    l_array = array_1[:, :cut_width]
    r_array = array_2[:, cut_width:]
    output_array = np.hstack((l_array,r_array))
    
    return output_array

#-----------------------------------------------------------------------------------------------------------------------------------------------------