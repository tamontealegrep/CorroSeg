
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import List, Tuple, Union, Dict, Any
from torchinfo import summary

from src.models.architectures.networks import Network
from src.data.expand import data_expand
from src.data.split import data_split 
from src.data.fix import data_fix
from src.data.augment import cutmix_augment_data
from src.data.slice import data_slice
from src.data.scale import scaler_minmax, scaler_standar, scaler_robust

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ModelManager:
    """
    A general-purpose model manager for training and validating a PyTorch model. This class allows you to:
    - Define and manage a PyTorch model.
    - Create and manage datasets using a custom dataset class.
    - Configure and manage optimizers and loss functions.
    - Train and validate the model over multiple epochs.

    Args:
        network (Network): An instance of a PyTorch Network.
        dataset_class (torch.utils.data.Dataset): The custom dataset class.
        loss_fn (nn.Module): The loss function to be used during training.
        optimizer_class (torch.optim.Optimizer): A PyTorch optimizer class.
        scaler_type (str): The name of the scaler to scale the data. Options: minmax, standar or robust.
        input_shape (tuple): A tuple representing the input shape of the data (channels, height, width).
        placeholders (List[Union[int, float]]): List of placeholder values to be replaced.
        value_range (tuple): A tuple (min_value, max_value) with int or float values.
        default_value (Union[int, float]): The value to replace NaNs with.
        loss_params (dict): A dictionary with parameters for the loss function.
        optimizer_params (dict): A dictionary with parameters for the optimizer.
        scaler_params (dict): A dictionary with parameters for the scaler.
   
    """

    def __init__(self,
                 network:Network,
                 dataset_class:torch.utils.data.Dataset,
                 loss_fn:nn.Module,
                 optimizer_class:torch.optim.Optimizer,
                 scaler_type: str,
                 input_shape: Tuple[int,int,int],
                 placeholders: List[Union[int, float]],
                 value_range:Tuple[Union[int, float]],
                 default_value: Union[int, float],
                 loss_params: dict = {},
                 optimizer_params: dict = {},
                 scaler_params: dict = {},
                 ):

        self.network = network
        self.loss_fn = loss_fn(**loss_params)
        self.optimizer = optimizer_class(self.network.parameters(), **optimizer_params)
        self.dataset_class = dataset_class

        self.scaler = None
        if scaler_type == "minmax":
            self.scaler_fn = scaler_minmax
        elif scaler_type == "standar":
            self.scaler_fn = scaler_standar
        elif scaler_type == "robust":
            self.scaler_fn = scaler_robust
        else:
            raise ValueError("scaler_type must be minmax, standar or robust.")
        
        self.scaler_params = scaler_params

        if len(input_shape) != 3:
            raise ValueError("input_shape must be in the form (channels, height, width).")
        else:
            self.channels, self.height, self.width = input_shape

        self.placeholders = placeholders
        self.default_value = default_value

        if len(value_range) != 2:
            raise ValueError("value_range must be in the form (min_value, max_value).")
        elif value_range[0] >= value_range[1]:
            raise ValueError("value_range max must be greater than value_range min")
        else:
            self.value_range = value_range

        self.device = None
        self.set_device()

    def _augment_data(self, X:np.ndarray, y:np.ndarray, augmented_ratio:float) -> Tuple[np.ndarray,np.ndarray]:
        X_aug, y_aug = cutmix_augment_data(X, y,int(X.shape[0] * augmented_ratio))
        X = np.concatenate((X, X_aug), axis=0)
        y = np.concatenate((y, y_aug), axis=0)
        return X, y
    
    def _train_val_split(self, X:Dict[Any,np.ndarray], y:Dict[Any,np.ndarray], height_stride:int=0, width_stride:int=18, fraction:float=0.2, seed:int=None, expand:bool=True) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        if expand:
            X, y = data_expand(X, y)
        height_stride = height_stride if height_stride > 0 else self.height
        width_stride = width_stride if width_stride > 0 else self.width
        X_train, y_train, X_val, y_val = data_split(X,y,self.height,height_stride,self.width,width_stride,self.default_value,fraction,seed)
        return X_train, y_train, X_val, y_val
    
    def _preprocess_data(self, X:np.ndarray) -> np.ndarray:
        if self.scaler is None:
            raise RuntimeError("The scaler has not been fitted. Please fit the scaler first using 'fit_scaler'.")
        X = data_fix(X,self.placeholders,self.default_value,self.value_range[0],self.value_range[1],self.scaler)
        return X
    
    def set_device(self) -> None:
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.device = device
        self.network.device = device

    def summary(self, depth:int=5) -> None:
        print(summary(self.network, input_size=(1, self.channels, self.height, self.width),depth=depth,device=self.device))
    
    def fit_scaler(self, X:np.ndarray) -> None:
        X = data_fix(X,self.placeholders,self.default_value,self.value_range[0],self.value_range[1],None)
        scaler = self.scaler_fn(X, **self.scaler_params)
        self.scaler = scaler

    def setup_train_val_data(self, X:Dict[Any,np.ndarray], y:Dict[Any,np.ndarray], height_stride:int=0, width_stride:int=18, fraction:float=0.2, seed:int=None, expand=True, augmented_ratio:float=0.5) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        X_train, y_train, X_val, y_val = self._train_val_split(X, y, height_stride=height_stride, width_stride=width_stride, fraction=fraction,seed=seed,expand=expand)
        if self.scaler is None:
            self.fit_scaler(X_train)
        X_train = np.array([self._preprocess_data(X_train[i]) for i in range(X_train.shape[0])])
        X_val = np.array([self._preprocess_data(X_val[i]) for i in range(X_val.shape[0])])
        X_train, y_train = self._augment_data(X_train, y_train, augmented_ratio)
        return X_train, y_train, X_val, y_val
    
    def setup_data(self, X:np.ndarray, y:Union[np.ndarray,None]=None, height_stride:int=0, width_stride:int=0) -> Tuple[np.ndarray,np.ndarray]:
        height_stride = height_stride if height_stride > 0 else self.height
        width_stride = width_stride if width_stride > 0 else self.width
        X, y = data_slice(X, y, self.height, height_stride, self.width, width_stride, self.default_value)
        X = np.array([self._preprocess_data(X[i]) for i in range(X.shape[0])])
        return X, y
    
    def build_dataset(self, X:np.ndarray, y:Union[np.ndarray,None]=None, transform=None, transform_config={}) -> torch.utils.data.Dataset:
        dataset = self.dataset_class(X, y, transform, **transform_config)
        return dataset
    
    def build_dataloader(self, dataset:torch.utils.data.Dataset, batch_size:int, shuffle:bool=False) -> torch.utils.data.DataLoader:
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
        return dataloader
    
    def train(self, train_loader:torch.utils.data.DataLoader, val_loader:Union[torch.utils.data.DataLoader,None]=None, num_epochs=50) -> None:
        self.network.train_model(self.loss_fn, self.optimizer, train_loader, val_loader, num_epochs)

    def predict(self, data:Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]) -> np.ndarray:
        return self.network.predict_model(data)
    
    def save_model(self, path:str) -> None:
        """
        Saves the trained model to a file.

        Args:
            path (str): The file path where the model will be saved.
        """
        torch.save(self.network.state_dict(), path)

    def load_model(self, path:str) -> None:
        """
        Loads a trained model from a file.

        Args:
            path (str): The file path from where the model will be loaded.
        """
        self.network.load_state_dict(torch.load(path))
        self.network.to(self.device)

#-----------------------------------------------------------------------------------------------------------------------------------------------------