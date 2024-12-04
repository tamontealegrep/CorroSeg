
import torch
import torch.nn as nn
import numpy as np
from typing import Optional, Union, Tuple, List

from src.models.operations.train import train
from src.models.operations.evaluate import evaluate
from src.models.operations.predict import predict

#---------------------------------------------------------------------------------------------------

class Network(nn.Module):
    def __init__(self):
        super(Network, self).__init__()
        """
        Base class for models.

        Parameters:
            config (dict): A dictionary that holds the model's configuration, including
                   hyperparameters, and other model-specific parameters. This attribute is 
                   initialized automatically and can be used to save or load the model configuration.
            device (torch.device): The device to perform calculations on.
            results (dict): Log of the results of the training and validation.

        """
        self.config = {}
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.results = {"train_loss": [], "val_loss": []}

    def set_device(self, device_name: str = None) -> None:
        """
        Sets the device for the computations.
        If no device name is provided, it will select the default device (cuda if available).

        Parameters:
            device_name (str, optional): The name of the device (e.g., "cuda" or "cpu").
                                          If not provided, the default device will be used.
        """
        if device_name is None:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        else:
            if device_name.lower() == "cpu":
                self.device = torch.device("cpu")
            elif device_name.lower() == "cuda" and torch.cuda.is_available():
                self.device = torch.device("cuda")
            else:
                raise ValueError(f"Invalid device '{device_name}'. Use 'cpu' or 'cuda'.")

    def train_model(self,
                    criterion: nn.Module,
                    optimizer: torch.optim.Optimizer,
                    train_loader: torch.utils.data.DataLoader,
                    val_loader: Optional[Union[torch.utils.data.DataLoader, None]] = None,
                    num_epochs: Optional[int] = 10) -> None:
        """
        Train and optionally validate the given model for a specified number of epochs.

        Parameters:
            criterion (torch.nn.Module): The loss function used to compute the losses.
            optimizer (torch.optim.Optimizer): The optimizer used for updating the model's weights.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation data. Default is None.
            num_epochs (int, optional): Number of epochs to train the model. Default is 10.

        Returns:
            None
        """
        train(self, criterion, optimizer, train_loader, val_loader, num_epochs)

    def evaluate_model(self,
                       data:Union[torch.utils.data.Dataset,torch.utils.data.DataLoader],
                       criterion:nn.Module,
                       ) -> Tuple[np.ndarray, List[float], float]:
        """
        Evaluate the model on the provided dataset or dataloader using the specified loss function.

        This function evaluates the model in evaluation mode, making predictions, computing the loss for each sample, 
        and returning:
        - the predictions for each sample,
        - the individual losses for each sample,
        - and the total average loss.

        Parameters:
            data (Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]): The input data, either a Dataset or a DataLoader.
            criterion (torch.nn.Module): The loss function used to compute the losses.

        Returns:
            (tuple): A tuple containing:
                outputs (numpy.ndarray): Predictions for each sample.
                losses (list): List of individual losses for each sample.
                average_loss (float): The average loss across the dataset or dataloader.
        """
        return evaluate(self, data, criterion)

    def predict_model(self,
                      data:Union[torch.utils.data.Dataset,torch.utils.data.DataLoader],
                      ) -> np.ndarray:
        """
        Predict outputs for a given dataset or dataloader using the specified model.

        Parameters:
            data (Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]): The input data.

        Returns:
            np.ndarray: An array of model predictions corresponding to the inputs in the data.
        """
        return predict(self, data)
    
    def save_model(self, path:str) -> None:
        """
        Save the model's state dictionary and configuration to a file.

        This method saves the model's learned parameters (weights) and its configuration 
        (hyperparameters and architecture details) to a specified file. This allows for 
        reloading the model later with the same architecture and weights.

        Parameters:
            path (str): The path to the file where the model's state and configuration
                            will be saved. The file will be saved in PyTorch's .pth format.
        """
        network_config = self.config
        network_state_dict = self.state_dict()
        network_results = self.results

        torch.save({
            "network_config": network_config,
            "network_state_dict": network_state_dict,
            "network_results": network_results,
        }, path)

    @classmethod
    def load_model(cls, path: str):
        """
        Static method to load a model from a file and create an instance from the saved configuration.
        This method uses the configuration dictionary and the from_dict method to create the model.

        Parameters:
            path (str): The path to the model file to load.
        
        Returns:
            network (Network): The model instance with loaded weights and configuration.
        """
        checkpoint = torch.load(path)
        network_config = checkpoint["network_config"]
        network_state_dict = checkpoint["network_state_dict"]
        network_results = checkpoint["network_results"]

        network = cls.from_dict(network_config)

        network.load_state_dict(network_state_dict)
        network.results = network_results

        return network
    
    @staticmethod
    def from_dict(config_dict):
        """
        Creates a model instance from a configuration dictionary.
        
        Args:
            config_dict (dict): A dictionary containing the model's configuration.
            
        Returns:
            Model: A model instance created from the configuration dictionary.
        
        Raises:
            NotImplementedError: If the method is not implemented in the subclass.
        """
        raise NotImplementedError("The 'from_dict' static method must be implemented in the subclass.")

#---------------------------------------------------------------------------------------------------