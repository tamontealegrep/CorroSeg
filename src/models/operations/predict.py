
import torch
import torch.nn as nn
import numpy as np
from typing import Union
from torch.utils.data import Dataset, DataLoader

#---------------------------------------------------------------------------------------------------

def predict_dataset(model:nn.Module, dataset:torch.utils.data.Dataset):
    """
    Predict outputs for a given dataset using the specified model.

    Parameters:
        model (nn.Module): The PyTorch model to use for predictions.
        dataset (torch.utils.data.Dataset): The dataset containing input data. Each item is expected to be a tuple (input, label), but the label will be ignored during prediction.

    Returns:
        np.ndarray: An array of model predictions corresponding to the inputs in the dataset.
    """
    model.to(model.device)
    model.eval()
    outputs = []

    with torch.no_grad():
        for i in range(len(dataset)):
            input, _ = dataset[i]
            input = input.to(model.device)
            input = input.unsqueeze(0) 
            output = model(input)
            output = output.squeeze().cpu().numpy()
            outputs.append(output)

    return np.array(outputs)

def predict_dataloader(model:nn.Module, dataloader:torch.utils.data.DataLoader):
    """
    Predict outputs for a given DataLoader using the specified model.

    Parameters:
        model (nn.Module): The PyTorch model to use for predictions.
        dataloader (torch.utils.data.DataLoader): The DataLoader providing input data. Each batch consists of a tuple (input_data, label), but the label will be ignored.

    Returns:
        np.ndarray: An array of model predictions corresponding to the inputs in the dataloader.
    """
    model.to(model.device)
    model.eval()
    outputs = []

    with torch.no_grad():
        for inputs, _ in dataloader:
            inputs = inputs.to(model.device)
            output = model(inputs)
            output = output.squeeze().cpu().numpy()
            outputs.append(output)

    return np.concatenate(outputs, axis=0)


def predict(model: nn.Module, data:Union[Dataset, DataLoader]):
    """
    Predict outputs for a given dataset or dataloader using the specified model.

    Parameters:
        model (nn.Module): The PyTorch model to use for predictions.
        data (Union[Dataset, DataLoader]): The input data.

    Returns:
        np.ndarray: An array of model predictions corresponding to the inputs in the data.
    """
    if isinstance(data, Dataset):
        return predict_dataset(model, data)
    elif isinstance(data, DataLoader):
        return predict_dataloader(model, data)
    else:
        raise ValueError("Input data must be either a Dataset or a DataLoader.")
    
#---------------------------------------------------------------------------------------------------