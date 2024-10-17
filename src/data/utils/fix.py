
import numpy as np
from scipy.ndimage import generic_filter
from typing import Tuple, Optional, Union, List, Any

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def filter_samples_nan(X: np.ndarray, y: np.ndarray, return_unfiltered: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filter out samples from features and targets that contain NaN values.

    This function removes samples from the feature and target arrays that contain 
    any NaN values. It can optionally return the unfiltered samples as well.

    Parameters:
        X (numpy.ndarray): Array of features with shape (n_samples, h, w).
        y (numpy.ndarray): Array of targets with shape (n_samples, h, w).
        return_unfiltered (bool, optional): If True, return the samples that were filtered out. Default is False.

    Returns:
        tuple: A tuple containing:
            X_filtered (numpy.ndarray): Array of features without NaN values.
            y_filtered (numpy.ndarray): Array of targets corresponding to the filtered features.
            X_unfiltered (Optional[numpy.ndarray]): Array of features that were filtered out (only if return_unfiltered is True).
            y_unfiltered (Optional[numpy.ndarray]): Array of targets corresponding to the unfiltered features (only if return_unfiltered is True).
    """
    # Create a boolean mask for valid samples (not containing NaN)
    valid_mask = ~np.isnan(X).any(axis=(1, 2))
    
    # Filter images and masks based on valid samples
    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]

    if return_unfiltered:
        invalid_mask = ~valid_mask
        # Filter images and masks based on invalid samples
        X_unfiltered = X[invalid_mask]
        y_unfiltered = y[invalid_mask]

        return X_filtered, y_filtered, X_unfiltered, y_unfiltered
    
    else:
        return X_filtered, y_filtered
    
def replace_nan_values(array: np.ndarray, new_value: Union[int, float] = 0) -> np.ndarray:
    """
    Replace NaN values in a 3D NumPy array with a specified value.

    This function scans through the input array and replaces any NaN values 
    with the provided new value.

    Parameters:
        array (numpy.ndarray): Input array of shape (n_samples, h, w).
        new_value (Union[int, float], optional): The value to replace NaN values with. Default is 0.

    Returns:
        numpy.ndarray: Array with NaN values replaced by the specified new value.
    """
    # Replace NaN values with the specified new value
    array[np.isnan(array)] = new_value

    return array

def filter_samples_placeholder(X: np.ndarray, y: np.ndarray, placeholder_values: List[Union[int, float]], return_unfiltered: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filter out samples from features and targets that contain specified placeholder values.

    This function removes samples from the feature and target arrays that contain 
    any of the provided placeholder values. It can optionally return the unfiltered samples.

    Parameters:
        X (numpy.ndarray): Array of features with shape (n_samples, h, w).
        y (numpy.ndarray): Array of targets with shape (n_samples, h, w).
        placeholder_values (List[Union[int, float]]): List of values to filter out.
        return_unfiltered (bool, optional): If True, return the samples that were filtered out. Default is False.

    Returns:
        tuple: A tuple containing:
            X_filtered (numpy.ndarray): Array of features without the specified placeholder values.
            y_filtered (numpy.ndarray): Array of targets corresponding to the filtered features.
            X_unfiltered (Optional[numpy.ndarray]): Array of features that were filtered out (only if return_unfiltered is True).
            y_unfiltered (Optional[numpy.ndarray]): Array of targets corresponding to the unfiltered features (only if return_unfiltered is True).
    """
    # Create a boolean mask for valid samples
    # Check if any value in X matches the placeholder values
    invalid_mask = np.isin(X, placeholder_values).any(axis=(1, 2))
    
    # Create a mask for samples that are valid (not containing placeholders)
    valid_mask = ~invalid_mask
    
    # Filter images and masks based on valid samples
    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]

    if return_unfiltered:
        # Filter images and masks based on invalid samples
        X_unfiltered = X[invalid_mask]
        y_unfiltered = y[invalid_mask]

        return X_filtered, y_filtered, X_unfiltered, y_unfiltered
    
    else:
        return X_filtered, y_filtered

def replace_placeholder_values(array: np.ndarray, placeholder_values: List[Union[int, float]], new_value: Union[int, float] = 0) -> np.ndarray:
    """
    Replace specified placeholder values in a 3D NumPy array with a new value.

    This function scans through the input array and replaces any values that match 
    the specified placeholder values with the provided new value.

    Parameters:
        array (numpy.ndarray): Input array of shape (n_samples, h, w).
        placeholder_values (List[Union[int, float]]): List of values to replace.
        new_value (Union[int, float], optional): The value to replace the placeholder values with. Default is 0.

    Returns:
        numpy.ndarray: Array with specified placeholder values replaced by the new value.
    """
    for i in placeholder_values:
        array[array == i] = new_value

    return array

def filter_samples_range(X: np.ndarray, y: np.ndarray, value_range: Tuple[float, float], return_unfiltered: bool = False) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], Optional[np.ndarray]]:
    """
    Filter samples from features and targets based on a specified value range.

    This function filters out samples from the feature and target arrays that contain 
    values outside the specified minimum and maximum value range. It can optionally 
    return the unfiltered samples.

    Parameters:
        X (numpy.ndarray): Array of features with shape (n_samples, h, w).
        y (numpy.ndarray): Array of targets with shape (n_samples, h, w).
        value_range (Tuple[float, float]): A tuple (min_value, max_value) specifying the valid range of values.
        return_unfiltered (bool, optional): If True, return the samples that were filtered out. Default is False.

    Returns:
        tuple: A tuple containing:
            X_filtered (numpy.ndarray): Array of features that fall within the specified value range.
            y_filtered (numpy.ndarray): Array of targets corresponding to the filtered features.
            X_unfiltered (Optional[numpy.ndarray]): Array of features that were filtered out (only if return_unfiltered is True).
            y_unfiltered (Optional[numpy.ndarray]): Array of targets corresponding to the unfiltered features (only if return_unfiltered is True).

    Raises:
        ValueError: If max_value is not greater than min_value.
    """
    min_value, max_value = value_range

    if min_value >= max_value:
        raise ValueError("max_value must be greater than min_value")
    
    # Create a boolean mask for valid samples
    valid_mask = np.all((X >= min_value) & (X <= max_value), axis=(1, 2))
    
    # Filter images and masks based on valid samples
    X_filtered = X[valid_mask]
    y_filtered = y[valid_mask]

    if return_unfiltered:
        invalid_mask = ~valid_mask
        # Filter images and masks based on invalid samples
        X_unfiltered = X[invalid_mask]
        y_unfiltered = y[invalid_mask]

        return X_filtered, y_filtered, X_unfiltered, y_unfiltered
    
    else:
        return X_filtered, y_filtered

def replace_oor_mean(X:np.ndarray, min_val:Union[float,int]=0, max_val:Union[float,int]=1):
    """
    Replaces out-of-range values in an array with the mean value of the entire array.

    This function clips the pixel values to the specified range, then calculates the mean 
    of the clipped X and replaces any pixels that are below `min_val` or above 
    `max_val` with this mean value.

    Parameters:
        X (np.ndarray): The input array.
        min_val (int, optional): The minimum allowable value. Default is 0.
        max_val (int, optional): The maximum allowable value. Default is 1.

    Returns:
        np.ndarray: The modified X with out-of-range values replaced by the mean of the X.
    """
    X_clipped = np.clip(X, min_val, max_val)

    mean_value = np.nanmean(X_clipped)

    X[np.logical_or(X < min_val, X > max_val)] = mean_value

    return X

def _mean_filter(values: Any, min_val:Union[float,int]=0, max_val:Union[float,int]=1) -> float:
    """
    Computes the mean of valid values, ignoring any that are out of the specified range.

    Parameters:
        values (Any): The array of pixel values from the local neighborhood.
        min_val (int, optional): The minimum allowable value. Default is 0.
        max_val (int, optional): The maximum allowable value. Default is 1.

    Returns:
        float: The mean of valid pixel values, or 0 if there are no valid values.
    """
    valid_values = [v for v in values if min_val <= v <= max_val]
    return np.mean(valid_values) if valid_values else 0

def replace_oor_neighbor_mean(X: np.ndarray, min_val:Union[float,int]=0, max_val:Union[float,int]=1) -> np.ndarray:
    """
    Replaces out-of-range values in an array with the mean of their valid neighboring.

    This function applies a local mean filter to replace pixels that are below `min_val` 
    or above `max_val`. The mean is computed using the neighboring pixels, ignoring 
    any out-of-range values.

    Parameters:
        X (np.ndarray): The input array.
        min_val (int, optional): The minimum allowable value. Default is 0.
        max_val (int, optional): The maximum allowable value. Default is 1.

    Returns:
        np.ndarray: The modified X with out-of-range values replaced by the mean of valid neighbors.
    """
    neighbor_means = generic_filter(X, lambda values: _mean_filter(values, min_val, max_val), size=3, mode='constant', cval=0)

    X[np.logical_or(X < min_val, X > max_val)] = neighbor_means[np.logical_or(X < min_val, X > max_val)]

    return X

#-----------------------------------------------------------------------------------------------------------------------------------------------------