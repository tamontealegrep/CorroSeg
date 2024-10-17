
import numpy as np
from typing import Tuple
from src.data.utils.augment import v_cutmix, h_cutmix

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def cutmix_augment_data(X:np.ndarray, y:np.ndarray, num_samples:int) -> Tuple[np.ndarray,np.ndarray]:
    """
    Augment data using random horizontal and vertical CutMix techniques.
    
    Args:
        X (np.ndarray): Input array of shape (n_samples, h, w).
        y (np.ndarray): Target array of shape (n_samples, h, w).
        num_samples (int): The number of augmented samples to generate.
    
    Returns:
        X_augmented (np.ndarray): Array containing the augmented input samples.
        y_augmented (np.ndarray): Array containing the augmented target samples.
    
    Raises:
        ValueError: If X and y do not have the same first dimension or if num_samples is non-positive.
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
