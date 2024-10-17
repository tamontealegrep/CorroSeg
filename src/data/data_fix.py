
import numpy as np
from typing import Union, List
from src.data.utils.fix import replace_nan_values,replace_placeholder_values,replace_oor_neighbor_mean,replace_oor_mean
from src.data.utils.scale import RobustScaler, scaler_robust

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def fix_train_val_data(X_train:np.ndarray,
                       y_train:np.ndarray,
                       X_val:np.ndarray,
                       y_val:np.ndarray,
                       placeholders:List[Union[int,float]],
                       default_value:Union[int,float],
                       value_range_min:Union[int,float],
                       value_range_max:Union[int,float],):
    """
    Cleans and prepares training and validation datasets by addressing NaN values, 
    placeholder values, and out-of-range values, followed by scaling the data.

    This function performs the following steps:
    1. Replaces NaN values in the training and validation datasets with a specified default value.
    2. Replaces placeholder values in both datasets.
    3. Replaces out-of-range values in both datasets using a neighbor mean approach.
    4. Scales the training data using a robust scaler and applies the same transformation to the validation data.

    Parameters:
        X_train (np.ndarray): The training input data.
        y_train (np.ndarray): The training target data.
        X_val (np.ndarray): The validation input data.
        y_val (np.ndarray): The validation target data.
        placeholders (List[Union[int, float]]): A list of placeholder values to be replaced.
        default_value (Union[int, float]): The value to replace NaNs with.
        value_range_min (Union[int, float]): The minimum allowable value for the input data.
        value_range_max (Union[int, float]): The maximum allowable value for the input data.

    Returns:
        tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]: A tuple containing:
            X_train(np.ndarray): The cleaned training inputs.
            y_train(np.ndarray): The training targets.
            X_val(np.ndarray): The cleaned validation inputs.
            y_val(np.ndarray): The validation targets.
            scaler(RobustScaler: The scaler used for transformation.
    """
    X_train= X_train.copy()
    X_val = X_val.copy()
    # Nan Values
    X_train = replace_nan_values(X_train,default_value)
    X_val = replace_nan_values(X_val,default_value)
    # Placeholder Values
    X_train = replace_placeholder_values(X_train, placeholders)
    X_val = replace_placeholder_values(X_val, placeholders)
    # Out of Range
    X_train = replace_oor_mean(X_train, value_range_min, value_range_max)
    X_val = replace_oor_mean(X_val, value_range_min, value_range_max)
    # Scale
    X_train, scaler = scaler_robust(X_train)
    X_val = scaler.transform(X_val)

    return X_train, y_train, X_val, y_val, scaler

def fix_inference_data(X:np.ndarray,
                       placeholders:List[Union[int,float]],
                       default_value:Union[int,float],
                       value_range_min:Union[int,float],
                       value_range_max:Union[int,float],
                       scaler:RobustScaler):
    """
    Cleans and prepares input data for inference by addressing NaN values, 
    placeholder values, and out-of-range values, followed by scaling the data.

    This function performs the following steps:
    1. Replaces NaN values in the input data with a specified default value.
    2. Replaces placeholder values in the input data.
    3. Replaces out-of-range values using a neighbor mean approach.
    4. Scales the input data using a provided scaler.

    Parameters:
        X (np.ndarray): The input data to be cleaned and prepared for inference.
        placeholders (List[Union[int, float]]): A list of placeholder values to be replaced.
        default_value (Union[int, float]): The value to replace NaNs with.
        value_range_min (Union[int, float]): The minimum allowable value for the input data.
        value_range_max (Union[int, float]): The maximum allowable value for the input data.
        scaler (RobustScaler): The scaler used to transform the input data.

    Returns:
        np.ndarray: The cleaned and scaled input data prepared for inference.
    """
    X = X.copy()
    # Nan Values
    X = replace_nan_values(X,default_value)
    # Placeholder Values
    X = replace_placeholder_values(X, placeholders)
    # Out of Range
    X = replace_oor_mean(X, value_range_min, value_range_max)
    # Scale
    X = scaler.transform(X)

    return X

#-----------------------------------------------------------------------------------------------------------------------------------------------------