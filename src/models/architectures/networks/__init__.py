
import torch
import torch.nn as nn
import numpy as np
from typing import Union

from src.models.operations.train import train
from src.models.operations.predict import predict

#---------------------------------------------------------------------------------------------------

class Network(nn.Module):
    def train_model(self,
                    criterion:nn.Module,
                    optimizer:torch.optim.Optimizer,
                    train_loader:torch.utils.data.DataLoader,
                    val_loader:Union[torch.utils.data.DataLoader, None]=None,
                    num_epochs:int=10) -> None:
        """
        Train and optionally validate the given model for a specified number of epochs.

        Args:
            criterion (torch.nn.Module): The loss function used to compute the losses.
            optimizer (torch.optim.Optimizer): The optimizer used for updating the model's weights.
            train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
            val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation data. Default is None.
            num_epochs (int, optional): Number of epochs to train the model. Default is 10.

        """
        train(self, criterion, optimizer, train_loader, val_loader, num_epochs)

    def predict_model(self,
                      data:Union[torch.utils.data.Dataset,torch.utils.data.DataLoader]) -> np.ndarray:
        """
        Predict outputs for a given dataset or dataloader using the specified model.

        Args:
            data (Union[torch.utils.data.Dataset, torch.utils.data.DataLoader]): The input data.

        Returns:
            np.ndarray: An array of model predictions corresponding to the inputs in the data.
        """
        return predict(self, data)
    
#---------------------------------------------------------------------------------------------------