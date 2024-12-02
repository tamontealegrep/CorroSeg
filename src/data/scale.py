
import numpy as np
from typing import Tuple, Union

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class Scaler:
    """
    Base class for all scalers. This class provides the common interface for scaling operations.

    Methods:
        fit(X): Computes the necessary parameters to be used for scaling.
        transform(X): Transforms the data using the computed parameters.
        fit_transform(X): Fits the scaler and then transforms the data.
        inverse_transform(X): Transforms the scaled data back to the original representation.
    """
    def fit(self, X):
        """Compute necessary parameters from the data."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def transform(self, X):
        """Transform the data based on computed parameters."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def fit_transform(self, X):
        """Fit the scaler and then transform the data."""
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        """Transform the scaled data back to the original representation."""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def state_dict(self):
        """Return an object state"""
        raise NotImplementedError("This method should be overridden by subclasses.")

    def load_state_dict(self, state_dict):
        """Restore the state of the object using a dictionary"""
        raise NotImplementedError("This method should be overridden by subclasses.")
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------

class MinMaxScaler(Scaler):
    """
    A Min-Max scaler that scales features to a specified range.

    Parameters:
        feature_range (tuple, optional): Desired range of transformed data. Default is (0, 1).
        
    Methods:
        fit(X): Computes the minimum and maximum to be used for scaling.
        transform(X): Transforms the data using the minimum and maximum.
        fit_transform(X): Fits the scaler and then transforms the data.
        inverse_transform(X): Transforms the scaled data back to the original representation.
    """
    def __init__(self, feature_range=(0, 1)):
        super().__init__()
        self.min = None
        self.max = None
        self.feature_range = feature_range

    def fit(self, X):
        self.min = np.min(X)
        self.max = np.max(X)

    def transform(self, X):
        scaled = (X - self.min) / (self.max - self.min)
        return scaled * (self.feature_range[1] - self.feature_range[0]) + self.feature_range[0]

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return (X - self.feature_range[0]) / (self.feature_range[1] - self.feature_range[0]) * (self.max - self.min) + self.min
    
    def state_dict(self):
        return {
            "min":self.min,
            "max":self.max,
            "feature_range":self.feature_range
            }
    
    def load_state_dict(self, state_dict):
        self.min = state_dict["min"]
        self.max = state_dict["max"]
        self.feature_range = state_dict["feature_range"]

class StandardScaler(Scaler):
    """
    A Standard scaler that standardizes features by removing the mean  and scaling to unit variance.

    Methods:
        fit(X): Computes the mean and standard deviation to be used for scaling.
        transform(X): Transforms the data using the mean and standard deviation.
        fit_transform(X): Fits the scaler and then transforms the data.
        inverse_transform(X): Transforms the standardized data back to the original representation.
    """
    def __init__(self):
        super().__init__()
        self.mean = None
        self.std = None

    def fit(self, X):
        self.mean = np.mean(X)
        self.std = np.std(X)

    def transform(self, X):
        return (X - self.mean) / self.std

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.std + self.mean
    
    def state_dict(self):
        return {
            "mean":self.mean,
            "std":self.std,
            }
    
    def load_state_dict(self, state_dict):
        self.mean = state_dict["mean"]
        self.std = state_dict["std"]
    
class RobustScaler(Scaler):
    """
    A Robust scaler that scales features using statistics that are robust to outliers.

    Methods:
        fit(X): Computes the median and interquartile range to be used for scaling.
        transform(X): Transforms the data using the median and interquartile range.
        fit_transform(X): Fits the scaler and then transforms the data.
        inverse_transform(X): Transforms the scaled data back to the original representation.
    """
    def __init__(self):
        super().__init__()
        self.median = None
        self.iqr = None  # Interquartile Range

    def fit(self, X):
        Q1 = np.percentile(X, 25)
        Q3 = np.percentile(X, 75)
        self.median = np.median(X)
        self.iqr = Q3 - Q1

    def transform(self, X):
        return (X - self.median) / self.iqr

    def fit_transform(self, X):
        self.fit(X)
        return self.transform(X)

    def inverse_transform(self, X):
        return X * self.iqr + self.median
    
    def state_dict(self):
        return {
            "median":self.median,
            "iqr":self.iqr,
            }
    
    def load_state_dict(self, state_dict):
        self.median = state_dict["median"]
        self.iqr = state_dict["iqr"]

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def scaler_minmax(X:np.ndarray, feature_range:Tuple[Union[int,float],Union[int,float]]=(0,1), transform:bool=False) -> Tuple[np.ndarray,MinMaxScaler]:
    """
    Create and fit a MinMaxScaler to the provided data, returning the scaler and optionally the scaled data.

    Parameters:
        X (numpy.ndarray): The input data to be scaled, expected to have shape (n_samples, h, w).
        feature_range (tuple, optional): Desired range of transformed data. Default is (0, 1).
        transform (bool, optional): Transform and return the provided data. Default False.
    
    Returns:
        scaler (MinMaxScaler): The Min-Max scaler.
        X_scaled (numpy.ndarray): The scaled data.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    if transform:
        X_scaled = scaler.fit_transform(X)
        return scaler, X_scaled
    else:
        scaler.fit(X)
        return scaler

def scaler_standar(X:np.ndarray, transform:bool=False) -> Tuple[np.ndarray,StandardScaler]:
    """
    Create and fit a StandardScaler to the provided data, returning the scaler and optionally the scaled data.

    Parameters:
        X (numpy.ndarray): The input data to be scaled, expected to have shape (n_samples, h, w).
        transform (bool, optional): Transform and return the provided data. Default False.

    Returns:
        scaler (StandardScaler): The Standard scaler.
        X_scaled (numpy.ndarray): The scaled data.  
    """
    scaler = StandardScaler()
    if transform:
        X_scaled = scaler.fit_transform(X)
        return scaler, X_scaled
    else:
        scaler.fit(X)
        return scaler

def scaler_robust(X:np.ndarray, transform:bool=False) -> Tuple[np.ndarray,RobustScaler]:
    """
    Create and fit a RobustScaler to the provided data, returning the scaler and optionally the scaled data.

    Parameters:
        X (numpy.ndarray): The input data to be scaled, expected to have shape (n_samples, h, w).
        transform (bool, optional): Transform and return the provided data. Default False.
        
    Returns:
        scaler (RobustScaler): The Robust scaler.
        X_scaled (numpy.ndarray): The scaled data. 
    """
    scaler = RobustScaler()
    if transform:
        X_scaled = scaler.fit_transform(X)
        return scaler, X_scaled
    else:
        scaler.fit(X)
        return scaler

#-----------------------------------------------------------------------------------------------------------------------------------------------------
