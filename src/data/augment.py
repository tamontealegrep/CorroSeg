
import numpy as np
from typing import Tuple

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def v_cutmix(
        array_1:np.ndarray,
        array_2:np.ndarray,
        alpha:float,
        ) -> np.ndarray:
    """
    Perform Vertical CutMix data augmentation by mixing two arrays.
    
    Parameters:
        array_1 (np.ndarray): The first array of shape (h, w).
        array_2 (np.ndarray): The second array of the same shape as array_1.
        alpha (float): A value between 0 and 1 that determines the proportion of the first image to use.
    
    Returns:
        output_array (np.ndarray): A new array that is a combination of array_1 and array_2, 
            with the upper portion taken from array_1 and the rest from array_2.
    
    Raises:
        ValueError: If array_1 and array_2 do not have the same shape.
        ValueError: If alpha is not between 0 and 1.
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

def h_cutmix(
        array_1:np.ndarray,
        array_2:np.ndarray,
        alpha:float,
        ) -> np.ndarray:
    """
    Perform Horizontal CutMix data augmentation by mixing two arrays.
    
    Parameters:
        array_1 (np.ndarray): The first array of shape (h, w).
        array_2 (np.ndarray): The second array of the same shape as array_1.
        alpha (float): A value between 0 and 1 that determines the proportion of the first image to use.
    
    Returns:
        output_array (np.ndarray): A new array that is a combination of array_1 and array_2, 
            with the left portion taken from array_1 and the rest from array_2.
    
    Raises:
        ValueError: If array_1 and array_2 do not have the same shape.
        ValueError: If alpha is not between 0 and 1.
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

def cutmix_augment_data(
        X:np.ndarray,
        y:np.ndarray,
        num_samples:int,
        ) -> Tuple[np.ndarray, np.ndarray]:
    """
    Augment data using random horizontal and vertical CutMix techniques.
    
    Parameters:
        X (np.ndarray): Input array of shape (n_samples, h, w).
        y (np.ndarray): Target array of shape (n_samples, h, w).
        num_samples (int): The number of augmented samples to generate.
    
    Returns:
        X_augmented (np.ndarray): Array containing the augmented input samples.
        y_augmented (np.ndarray): Array containing the augmented target samples.
    
    Raises:
        ValueError: If X and y do not have the same first dimension.
        ValueError: If num_samples is non-positive.
    """
    if X.shape[0] != y.shape[0]:
        raise ValueError("Input arrays X and y must have the same number of samples.")
    if num_samples < 0:
        raise ValueError("num_samples must be a positive integer.")

    n_samples, h, w = X.shape
    X_augmented = np.empty((num_samples, h, w))
    y_augmented = np.empty((num_samples, h, w))

    for i in range(num_samples):
        idx1 = np.random.randint(n_samples)
        idx2 = np.random.randint(n_samples)
        alpha = np.random.uniform(0.1, 0.9)

        if np.random.rand() < 0.1:
            X_augmented[i] = h_cutmix(X[idx1], X[idx2], alpha)
            y_augmented[i] = h_cutmix(y[idx1], y[idx2], alpha)
        else:
            X_augmented[i] = v_cutmix(X[idx1], X[idx2], alpha)
            y_augmented[i] = v_cutmix(y[idx1], y[idx2], alpha)

    return X_augmented, y_augmented

#-----------------------------------------------------------------------------------------------------------------------------------------------------
