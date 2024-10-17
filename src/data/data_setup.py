
import numpy as np
from typing import Any, Tuple, Union, Dict, List
from src.data.data_split import data_split
from src.data.data_slice import data_slice
from src.data.data_fix import fix_train_val_data, fix_inference_data
from src.data.data_augment import cutmix_augment_data
from src.data.utils.scale import RobustScaler

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def setup_train_val_data(X:Dict[Any, np.ndarray],
                         y:Dict[Any, np.ndarray],
                         h_section_size:int,
                         w_section_size:int,
                         h_stride:int,
                         w_stride:int,
                         default_value:int=0,
                         val_fraction:float=0.2,
                         seed:int=None,
                         placeholders:List[Union[float,int]]=0,
                         value_range_min:Union[int,float]=0,
                         value_range_max:Union[int,float]=1,
                         augmented_ratio:float=0.5) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray,RobustScaler]:
    """
    Prepares training and validation datasets by splitting the data into patches,
    fixing invalid values, and applying data augmentation.

    This function performs the following steps:
    1. Splits the input data into training and validation sets using the specified
       section sizes and strides.
    2. Cleans the training and validation data by addressing NaN values, placeholders,
       and out-of-range values.
    3. Applies CutMix augmentation to a specified ratio of the training data.

    Parameters:
        X (Dict[Any, np.ndarray]): A dictionary of input data arrays indexed by identifiers.
        y (Dict[Any, np.ndarray]): A dictionary of target data arrays indexed by identifiers.
        h_section_size (int): Height of each section to be sliced from the input data.
        w_section_size (int): Width of each section to be sliced from the input data.
        h_stride (int): Number of rows to move forward for the next slice.
        w_stride (int): Number of columns to move forward for the next slice.
        default_value (int, optional): Value to replace NaNs with. Default is 0.
        val_fraction (float, optional): Fraction of the data to use for validation. Default is 0.2.
        seed (int, optional): Random seed for reproducibility. Default is None.
        placeholders (List[Union[float, int]], optional): List of placeholder values to replace. Default is None.
        value_range_min (Union[int, float], optional): Minimum allowable value for input data. Default is 0.
        value_range_max (Union[int, float], optional): Maximum allowable value for input data. Default is 1.
        augmented_ratio (float, optional): Ratio of data to be augmented using CutMix augmentation. Default is 0.5.

    Returns:
        Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, RobustScaler]: 
        A tuple containing the cleaned training inputs, training targets, cleaned validation inputs,
        validation targets, and the scaler used for transformation.
    """
    # Split data and make patches
    X_train, y_train, X_val, y_val = data_split(X,y,h_section_size,h_stride,w_section_size,w_stride,default_value,val_fraction,seed)
    # Fixing values (Nan, Placeholders, Out of Range)
    X_train, y_train, X_val, y_val, scaler = fix_train_val_data(X_train, y_train, X_val, y_val,placeholders,default_value,value_range_min,value_range_max)
    # Augment data
    X_aug, y_aug = cutmix_augment_data(X_train,y_train,int(X_train.shape[0] * augmented_ratio))
    X_train = np.concatenate((X_train, X_aug), axis=0)
    y_train = np.concatenate((y_train, y_aug), axis=0)

    return X_train, y_train, X_val, y_val, scaler 

def setup_inference_data(X:Dict[Any, np.ndarray],
                         h_section_size:int,
                         w_section_size:int,
                         h_stride:int,
                         w_stride:int,
                         default_value:int=0,
                         placeholders:List[Union[float,int]]=0,
                         value_range_min:Union[int,float]=0,
                         value_range_max:Union[int,float]=1,
                         scaler:RobustScaler=None) -> Dict[Any, np.ndarray]:
    """
    Prepares input data for inference by slicing the data into patches and 
    fixing invalid values.

    This function performs the following steps:
    1. Slices the input data into patches using the specified section sizes and strides.
    2. Cleans the sliced data by addressing NaN values, placeholders, and out-of-range values.

    Parameters:
        X (Dict[Any, np.ndarray]): A dictionary of input data arrays indexed by identifiers.
        h_section_size (int): Height of each section to be sliced from the input data.
        w_section_size (int): Width of each section to be sliced from the input data.
        h_stride (int): Number of rows to move forward for the next slice.
        w_stride (int): Number of columns to move forward for the next slice.
        default_value (int, optional): Value to replace NaNs with. Default is 0.
        placeholders (List[Union[float, int]], optional): List of placeholder values to replace. Default is None.
        value_range_min (Union[int, float], optional): Minimum allowable value for input data. Default is 0.
        value_range_max (Union[int, float], optional): Maximum allowable value for input data. Default is 1.
        scaler (RobustScaler, optional): Scaler used to transform the input data. Default is None.

    Returns:
        Dict[Any, np.ndarray]: A dictionary of cleaned and prepared input data arrays indexed by identifiers.
    """
    # Make patches
    X_sliced, _ = data_slice(X,None,h_section_size,h_stride,w_section_size,w_stride,default_value)
    
    # Fixing values (Nan, Placeholders, Out of Range)
    X_fixed = {}
    for id in X_sliced.keys():
        patches = {}
        for patch_id, patch in X_sliced[id].items():
            patch_fixed = fix_inference_data(patch,placeholders,default_value,value_range_min,value_range_max,scaler)
            patches[patch_id] = patch_fixed
        X_fixed[id] = patches

    X_output = {}
    for id, array_dict in X_fixed.items():
        for patch_id in sorted(array_dict.keys()):
            if id not in X_output:
                X_output[id] = []
            X_output[id].append(array_dict[patch_id])

    for id in X_output.keys():
        X_output[id] = np.stack(X_output[id], axis=0)

    return X_output


#-----------------------------------------------------------------------------------------------------------------------------------------------------