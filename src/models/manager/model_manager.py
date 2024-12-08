
import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import Dataset, DataLoader
from typing import Self, Optional, List, Tuple, Union, Dict, Any, Callable
from torchinfo import summary

from src.models.architectures.networks import Network
from src.models.dataset import *
from src.models.metrics import *

from src.data.expand import data_expand, array_expansion
from src.data.reduct import array_reduction
from src.data.split import data_split 
from src.data.fix import data_fix
from src.data.augment import cutmix_augment_data
from src.data.slice import data_slice
from src.data.reconstruct import data_reconstruct
from src.data.scale import Scaler, scaler_minmax, scaler_standar, scaler_robust

#-----------------------------------------------------------------------------------------------------------------------------------------------------

class ModelManager():
    """
    A class to manage the model lifecycle, including training, evaluation, and saving/loading the model.

    This class is designed to handle the training and evaluation of a PyTorch model, manage the dataset,
    and perform various utilities such as data preprocessing, augmentation, and model saving/loading.

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

    Attributes:
        config (dict): A dictionary containing the configuration parameters for 
            the model, dataset, optimizer, loss function, and scaler.
        network (torch.nn.Module): The PyTorch model to be trained or evaluated.
        dataset_class (torch.utils.data.Dataset): The PyTorch dataset used in the data.
        loss_class (torch.nn.Module): The loss function used during training.
        loss_params (dict): A dictionary containing parameters for the loss function.
        optimizer (torch.optim.Optimizer): The optimizer used for training the model.
        optimizer_params (dict): A dictionary containing parameters for the optimizer.
        scaler_fn (function): A function that implements the scaling method.
        scaler_params (dict): A dictionarArgsy containing parameters for the scaler.
        scaler (Scaler): The scaler used to preprocess input data.
        self.channels (int): The number of channels in the input data.
        self.height (int): The height (number of rows) of the input data.
        self.width (int): The width (number of columns) of the input data.
        placeholders (list): A list of placeholder values to be replaced in the input data.
        value_range (tuple): A tuple defining the valid range of values for the input data, 
            in the form (min, max).
        default_value (float or int): The value used to replace missing or invalid data in the dataset.
        device (torch.device): The device to perform calculations on.
        
    """

    def __init__(self,
                 network: Network,
                 dataset_class: torch.utils.data.Dataset,
                 loss_class: nn.Module,
                 optimizer_class: torch.optim.Optimizer,
                 scaler_type: str,
                 input_shape: Tuple[int,int,int],
                 placeholders: List[Union[int, float]],
                 value_range:Tuple[Union[int, float]],
                 default_value: Union[int, float],
                 loss_params: dict = None,
                 optimizer_params: dict = None,
                 scaler_params: dict = None,
                 ):
        # Save config
        self.config = {
            "network_class":network.__class__.__name__,
            "dataset_class":dataset_class.__name__,
            "loss_class": loss_class.__name__,
            "optimizer_class": optimizer_class.__name__,
            "scaler_type": scaler_type,
            "input_shape": input_shape,
            "placeholders": placeholders,
            "value_range": value_range,
            "default_value": default_value,
            "loss_params": loss_params or {},
            "optimizer_params": optimizer_params or {},
            "scaler_params": scaler_params or {},
        }
        # Initialize kwargs if not provided
        loss_params = loss_params or {}
        optimizer_params = optimizer_params or {}
        scaler_params = scaler_params or {}

        self.network = network
        self.dataset_class = dataset_class
        self.loss_class = loss_class(**loss_params)
        self.optimizer:torch.optim.Optimizer = optimizer_class(self.network.parameters(), **optimizer_params)
        

        self.scaler: Scaler = None
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

        self.set_device()

    def _augment_data(self,
            X: np.ndarray,
            y: np.ndarray,
            augmented_ratio: float,
            ) -> Tuple[np.ndarray,np.ndarray]:
        """
        Augments the data by applying the CutMix technique.

        Parameters:
            X (np.ndarray): The input feature array.
            y (np.ndarray): The target array.
            augmented_ratio (float): The ratio of the data to augment.

        Returns:
            (tuple): A tuple containing.
                X (np.ndarray): The augmented feature array.
                y (np.ndarray): The augmented label array.
        """
        X_aug, y_aug = cutmix_augment_data(X, y,int(X.shape[0] * augmented_ratio))
        X, y = np.concatenate((X, X_aug), axis=0), np.concatenate((y, y_aug), axis=0)
        return X, y
    
    def _train_val_split(self,
            X: Dict[Any,np.ndarray],
            y: Dict[Any,np.ndarray],
            height_stride: Optional[int] = 0,
            width_stride: Optional[int] = 18,
            fraction: Optional[float] = 0.2,
            seed: Optional[int] = None,
            expand: Optional[bool] = True,
            ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Splits the data into training and validation sets.

        Parameters:
            X (Dict[Any, np.ndarray]): Input data (features).
            y (Dict[Any, np.ndarray]): Target data (labels).
            height_stride (int, optional): Stride in the height direction. Default is 0.
            width_stride (int, optional): Stride in the width direction. Default is 18.
            fraction (float, optional): The fraction of the data to use for validation. Default is 0.2.
            seed (int, optional): Random seed for reproducibility. Default is None.
            expand (bool, optional): Whether to expand the data for lateral continuity. Default is True.

        Returns:
            (tuple): A tuple containing.
                X_train (np.ndarray): The training split of the feature array.
                y_train (np.ndarray): The training split of the target array.
                X_val (np.ndarray): The validation split of the feature array.
                y_val (np.ndarray): The validation split of the target array.
        """
        if expand:
            X, y = data_expand(X, y)
        height_stride = height_stride if height_stride > 0 else self.height
        width_stride = width_stride if width_stride > 0 else self.width
        X_train, y_train, X_val, y_val = data_split(X,y,self.height,height_stride,self.width,width_stride,self.default_value,fraction,seed)
        return X_train, y_train, X_val, y_val
    
    def _preprocess_data(self, X: np.ndarray) -> np.ndarray:
        """
        Preprocesses the input data using the fitted scaler.

        Parameters:
            X (np.ndarray): The input data to preprocess.

        Returns:
            X (np.ndarray): The preprocessed input data.

        Raises:
            RuntimeError: If the scaler has not been fitted yet.
        """
        if self.scaler is None:
            raise RuntimeError("The scaler has not been fitted. Please fit the scaler first using 'fit_scaler'.")
        X = data_fix(X,self.placeholders,self.default_value,self.value_range[0],self.value_range[1],self.scaler)
        return X
    
    def set_device(self, device_name: Optional[str] = None) -> None:
        """
        Sets the device (CPU or GPU) for the network.

        Parameters:
            device_name (str, optional): The name of the device to use. Default is None, 
                which uses the device specified in the network. Default None. Options: cpu and cuda.
        """
        self.network.set_device(device_name)
        self.device = self.network.device
        
    def summary(self, depth: Optional[int] = 5) -> None:
        """
        Prints a summary of the model architecture.

        Parameters:
            depth (int, optional): The depth of the summary to print. Default is 5.
        """
        print(summary(self.network, input_size=(1, self.channels, self.height, self.width),depth=depth,device=self.device))
    
    def fit_scaler(self, X: np.ndarray) -> None:
        """
        Fits the scaler to the provided data.

        Parameters:
            X (np.ndarray): The input data to fit the scaler.

        Raises:
            ValueError: If the input data is not in the correct format.
        """
        X = data_fix(X,self.placeholders,self.default_value,self.value_range[0],self.value_range[1],None)
        scaler = self.scaler_fn(X, **self.scaler_params)
        self.scaler = scaler

    def setup_train_data(self,
            X: Dict[Any,np.ndarray],
            y: Dict[Any,np.ndarray],
            height_stride: Optional[int] = 0,
            width_stride: Optional[int] = 18,
            expand: Optional[bool] = True,
            augmented_ratio: Optional[float] = 0.5,
            ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Prepares the data for training, applying preprocessing and augmentation.

        Parameters:
            X (Dict[Any, np.ndarray]): Input feature array.
            y (Dict[Any, np.ndarray]): Target array.
            height_stride (int, optional): Stride in the height direction. Default is 0.
            width_stride (int, optional): Stride in the width direction. Default is 18.
            expand (bool, optional): Whether to expand the data for lateral continuity. Default is True.
            augmented_ratio (float, optional): The ratio of data to augment. Default is 0.5.

        Returns:
            (tuple): A tuple containing.
                X_train (np.ndarray): The processed training split of the feature array.
                y_train (np.ndarray): The processed training split of the target array.
        """
        for key in X.keys():
            if key not in y.keys():
                raise FileNotFoundError(f"The file {key}.npy is not found in data/train/y.")
            
        if expand:
            X, y = data_expand(X, y)
        height_stride = height_stride if height_stride > 0 else self.height
        width_stride = width_stride if width_stride > 0 else self.width
        
        X_train = []
        y_train = []

        for id in X.keys():
            X_slices, y_slices = data_slice(X[id], y[id], self.height, height_stride, self.width, width_stride, self.default_value, True)
            X_train.append(X_slices)
            y_train.append(y_slices)

        X_train = np.concatenate(X_train, axis=0)
        y_train = np.concatenate(y_train, axis=0)

        if self.scaler is None:
            self.fit_scaler(X_train)

        X_train = np.array([self._preprocess_data(X_train[i]) for i in range(X_train.shape[0])])
        X_train, y_train = self._augment_data(X_train, y_train, augmented_ratio)
        return X_train, y_train

    def setup_train_val_data(self,
            X: Dict[Any,np.ndarray],
            y: Dict[Any,np.ndarray],
            height_stride: Optional[int] = 0,
            width_stride: Optional[int] = 18,
            fraction: Optional[float] = 0.2,
            seed: Optional[int] = None,
            expand: Optional[bool] = True,
            augmented_ratio: Optional[float] = 0.5,
            ) -> Tuple[np.ndarray,np.ndarray,np.ndarray,np.ndarray]:
        """
        Prepares and splits the data for training and validation, applying preprocessing and augmentation.

        Parameters:
            X (Dict[Any, np.ndarray]): Input feature array.
            y (Dict[Any, np.ndarray]): Target array.
            height_stride (int, optional): Stride in the height direction. Default is 0.
            width_stride (int, optional): Stride in the width direction. Default is 18.
            fraction (float, optional): The fraction of data to use for validation. Default is 0.2.
            seed (int, optional): Random seed for reproducibility. Default is None.
            expand (bool, optional): Whether to expand the data for lateral continuity. Default is True.
            augmented_ratio (float, optional): The ratio of data to augment. Default is 0.5.

        Returns:
            (tuple): A tuple containing.
                X_train (np.ndarray): The processed training split of the feature array.
                y_train (np.ndarray): The processed training split of the target array.
                X_val (np.ndarray): The processed validation split of the feature array.
                y_val (np.ndarray): The processed validation split of the target array.
        """
        for key in X.keys():
            if key not in y.keys():
                raise FileNotFoundError(f"The file {key}.npy is not found in data/train/y.")
            
        X_train, y_train, X_val, y_val = self._train_val_split(X,
                                                               y,
                                                               height_stride=height_stride,
                                                               width_stride=width_stride,
                                                               fraction=fraction,
                                                               seed=seed,
                                                               expand=expand)
        if self.scaler is None:
            self.fit_scaler(X_train)
        X_train = np.array([self._preprocess_data(X_train[i]) for i in range(X_train.shape[0])])
        X_val = np.array([self._preprocess_data(X_val[i]) for i in range(X_val.shape[0])])
        X_train, y_train = self._augment_data(X_train, y_train, augmented_ratio)
        return X_train, y_train, X_val, y_val
    
    def setup_data(self,
                   X: np.ndarray,
                   y: Optional[Union[np.ndarray,None]] = None,
                   height_stride: Optional[int] = 0,
                   width_stride: Optional[int] = 0,
                   ) -> Tuple[np.ndarray,np.ndarray]:
        """
        Prepares the data for training or prediction by applying preprocessing.

        Parameters:
            X (np.ndarray): Input data (features).
            y (Union[np.ndarray, None], optional): Target data (labels). Default is None.
            height_stride (int, optional): Stride in the height direction. Default is 0.
            width_stride (int, optional): Stride in the width direction. Default is 0.

        Returns:
            (tuple): A tuple containing.
                X (np.ndarray): The preprocessed input feature array.
                y (np.ndarray): The preprocessed target array.
        """
        height_stride = height_stride if height_stride > 0 else self.height
        width_stride = width_stride if width_stride > 0 else self.width
        X, y = data_slice(X, y, self.height, height_stride, self.width, width_stride, self.default_value, True)
        X = np.array([self._preprocess_data(X[i]) for i in range(X.shape[0])])
        return X, y
    
    def build_dataset(self,
                      X: np.ndarray,
                      y: Optional[Union[np.ndarray,None]] = None,
                      transform: Optional[Callable] = None,
                      transform_config: Optional[dict] = {},
                      ) -> torch.utils.data.Dataset:
        """
        Builds a custom dataset using the provided data and transformation.

        Parameters:
            X (np.ndarray): Input feature array.
            y (Union[np.ndarray, None], optional): Target array. Default is None.
            transform (Optional[callable], optional): A transformation function to apply to the data. Default is None.
            transform_config (dict, optional): Additional configuration for the transformation. Default is an empty dictionary.

        Returns:
            dataset (torch.utils.data.Dataset): A PyTorch dataset containing the input features and targets.
        """
        dataset = self.dataset_class(X, y, transform, **transform_config)
        return dataset
    
    def build_dataloader(self,
                         dataset: torch.utils.data.Dataset,
                         batch_size: int,
                         shuffle: Optional[bool] = False,
                         ) -> torch.utils.data.DataLoader:
        """
        Creates a PyTorch DataLoader for the given dataset.

        Parameters:
            dataset (torch.utils.data.Dataset): The dataset to load.
            batch_size (int): The batch size for the DataLoader.
            shuffle (bool, optional): Whether to shuffle the data. Default is False.

        Returns:
            dataloader (torch.utils.data.DataLoader): A DataLoader for the given dataset.
        """
        dataloader = DataLoader(dataset,batch_size=batch_size,shuffle=shuffle)
        return dataloader
    
    def train(self,
              train_loader: torch.utils.data.DataLoader,
              val_loader: Optional[Union[torch.utils.data.DataLoader,None]] = None,
              num_epochs: Optional[int] = 50,
              ) -> None:
        """
        Trains the model using the provided data loaders.

        Parameters:
            train_loader (torch.utils.data.DataLoader): The DataLoader for training data.
            val_loader (Union[torch.utils.data.DataLoader, None], optional): The DataLoader for validation data. Default is None.
            num_epochs (int, optional): The number of epochs to train the model. Default is 50.
        """
        self.network.train_model(self.loss_class, self.optimizer, train_loader, val_loader, num_epochs)

    def evaluate(self, data:Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]) -> Tuple[np.ndarray, List[float], float]:
        """
        Evaluates the model on the provided data.

        Parameters:
            data (Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]): The dataset or DataLoader containing the data to evaluate on.

        Returns:
            (tuple): A tuple containing.
                data_pred (np.ndarray): The predictions for input feature array.
                losses (List[float]): The individual loss for each sample.
                average_loss (float): The average loss.
        """
        return self.network.evaluate_model(data, self.loss_class)
        
    def predict(self, data:Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]) -> np.ndarray:
        """
        Makes predictions using the trained model on the provided data.

        Parameters:
            data (Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]): The dataset or DataLoader containing the data for prediction.

        Returns:
            (np.ndarray): The predicted values.
        """
        return self.network.predict_model(data)
    
    def predict_well(self,
                     X: np.ndarray,
                     height_stride: Optional[int] = 0,
                     width_stride: Optional[int] = 0,
                     batch_size: Optional[int] = 64,
                     expand: Optional[bool] = True,
                     ) -> np.ndarray:
        """
        Makes predictions on the well data with additional processing like expansion and reconstruction.

        Parameters:
            X (np.ndarray): Input well data.
            height_stride (int, optional): Stride in the height direction. Default is 0.
            width_stride (int, optional): Stride in the width direction. Default is 0.
            batch_size (int, optional): Batch size for the prediction. Default is 64.
            expand (bool, optional): Whether to expand the data before processing. Default is True.

        Returns:
            X_output (np.ndarray): The predicted data.
        """
        height_stride = height_stride if height_stride > 0 else self.height
        width_stride = width_stride if width_stride > 0 else self.width
        if expand:
            X = array_expansion(X)
        height, width = X.shape
        X, _ = self.setup_data(X, None, height_stride, width_stride)
        X_dataset = self.build_dataset(X)
        X_loader = self.build_dataloader(X_dataset, min(len(X_dataset), batch_size), False)
        X_preds = self.predict(X_loader)
        X_output = data_reconstruct(X_preds,height,width,height_stride,width_stride,True)
        if expand:
            X_output = array_reduction(X_output)
        return X_output
    
    def apply_threshold_to_data(self,
                                X: np.ndarray,
                                threshold: Optional[float] = 0.5,
                                ) -> np.ndarray:
        """
        Apply a threshold to the input data and return a binary mask.

        This function converts the input data into a binary mask by applying the specified threshold. 
        Any value greater than the threshold is set to 1, and values less than or equal to the threshold are set to 0.

        Parameters:
            data (np.ndarray): The input data array to which the threshold will be applied.
            threshold (float, optional): The threshold value. Default is 0.5.

        Returns:
            (np.ndarray): A binary mask where values greater than the threshold are 1, and others are 0.
        """
        return ((X) > threshold).astype(int)
    
    def compute_loss(self,
                     y_pred:np.ndarray,
                     y:np.ndarray,
                     ) -> float:
        """
        Computes the loss between the predicted and actual values.

        Parameters:
            y_pred (np.ndarray): The predicted output from the model.
            y (np.ndarray): The true labels.

        Returns:
            (float): The calculated loss value.
        
        Raises:
            ValueError: If the shapes of `y_pred` and `y` do not match.
        """
        if y_pred.shape != y.shape:
            raise ValueError(f"Shape mismatch: the shape of y_pred ({y_pred.shape}) does not match the shape of y ({y.shape}).")
        y_pred = torch.tensor(y_pred, dtype=torch.float32)
        y = torch.tensor(y, dtype=torch.float32)
        loss = self.loss_class(y_pred, y).item()
        return loss

    def save_model(self, path:str) -> None:
        """
        Saves the trained model to a file.

        Parameters:
            path (str): The file path where the model will be saved.
        """
        torch.save({
            "hyperparams_config":self.config,
            "network_config":self.network.config,
            "network_state_dict":self.network.state_dict(),
            "network_results": self.network.results,
            "optimizer_state_dict": self.optimizer.state_dict(),
            "scaler_state_dict": self.scaler.state_dict(),
        }, path)

    @staticmethod
    def load_model(network_class: Network,
                   path: str,
                   ) -> Self:
        """
        Loads a trained model from a file.

        Parameters:
            path (str): The file path from where the model will be loaded.

        Returns:
            (ModelManager) : The loaded model.
        """
        checkpoint = torch.load(path)
        hyperparams_config = checkpoint["hyperparams_config"]
        optimizer_state_dict = checkpoint["optimizer_state_dict"]
        scaler_state_dict = checkpoint["scaler_state_dict"]

        network = network_class.load_model(path)

        model = ModelManager.from_dict(network, hyperparams_config)

        model.optimizer.load_state_dict(optimizer_state_dict)
        model.fit_scaler(np.zeros((10, 10)))
        model.scaler.load_state_dict(scaler_state_dict)

        return model
    
    @staticmethod
    def from_dict(network: Network,
                  config_dict: dict,
                  ) -> Self:
        """
        Creates a ModelManager instance from the provided network and configuration dictionary.

        Parameters:
            network (Network): A PyTorch network.
            config_dict (dict): A dictionary containing the model's configuration.

        Returns:
            (ModelManager) : An instance of the ModelManager constructed from the dictionary.
        """
        return ModelManager(
            network = network,
            dataset_class = eval(config_dict["dataset_class"]),
            loss_class = eval(config_dict["loss_class"]),
            optimizer_class = eval(f"torch.optim.{config_dict["optimizer_class"]}"),
            scaler_type = config_dict["scaler_type"],
            input_shape = config_dict["input_shape"],
            placeholders = config_dict["placeholders"],
            value_range = config_dict["value_range"],
            default_value = config_dict["default_value"],
            loss_params = config_dict["loss_params"],
            optimizer_params = config_dict["optimizer_params"],
            scaler_params = config_dict["scaler_params"],
        )

#-----------------------------------------------------------------------------------------------------------------------------------------------------