
import numpy as np
from typing import Tuple, Union

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class MinMaxScaler:
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
    
class StandardScaler:
    """
    A Standard scaler that standardizes features by removing the mean  and scaling to unit variance.

    Methods:
        fit(X): Computes the mean and standard deviation to be used for scaling.
        transform(X): Transforms the data using the mean and standard deviation.
        fit_transform(X): Fits the scaler and then transforms the data.
        inverse_transform(X): Transforms the standardized data back to the original representation.
    """
    def __init__(self):
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
    
class RobustScaler:
    """
    A Robust scaler that scales features using statistics that are robust to outliers.

    Methods:
        fit(X): Computes the median and interquartile range to be used for scaling.
        transform(X): Transforms the data using the median and interquartile range.
        fit_transform(X): Fits the scaler and then transforms the data.
        inverse_transform(X): Transforms the scaled data back to the original representation.
    """
    def __init__(self):
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

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def scaler_minmax(X:np.ndarray,feature_range:Tuple[Union[int,float],Union[int,float]]=(0,1)) -> Tuple[np.ndarray,MinMaxScaler]:
    """
    Create and fit a MinMaxScaler to the provided data, returning the scaled data and the scaler.

    Parameters:
        - X (numpy.ndarray): The input data to be scaled, expected to have shape (n_samples, h, w).
        - feature_range (tuple, optional): Desired range of transformed data. Default is (0, 1).

    Returns:
        - X_scaled (numpy.ndarray): The scaled data.
        - scaler (MinMaxScaler): The Min-Max scaler.
    """
    scaler = MinMaxScaler(feature_range=feature_range)
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def scaler_standar(X:np.ndarray) -> Tuple[np.ndarray,StandardScaler]:
    """
    Create and fit a StandardScaler to the provided data, returning the scaled data and the scaler.

    Parameters:
        - X (numpy.ndarray): The input data to be scaled, expected to have shape (n_samples, h, w).

    Returns:
        - X_scaled (numpy.ndarray): The scaled data.
        - scaler (StandardScaler): The Standard scaler.
    """
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

def scaler_robust(X:np.ndarray) -> Tuple[np.ndarray,RobustScaler]:
    """
    Create and fit a RobustScaler to the provided data, returning the scaled data and the scaler.

    Parameters:
        - X (numpy.ndarray): The input data to be scaled, expected to have shape (n_samples, h, w).

    Returns:
        - X_scaled (numpy.ndarray): The scaled data.
        - scaler (RobustScaler): The Robust scaler.
    """
    scaler = RobustScaler()
    X_scaled = scaler.fit_transform(X)
    return X_scaled, scaler

#-----------------------------------------------------------------------------------------------------------------------------------------------------
