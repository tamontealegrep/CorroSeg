
import torch
from torch import nn
from timeit import default_timer as timer
from IPython.display import clear_output
import matplotlib.pyplot as plt

from src.models.operations.evaluate import _evaluation_step

#---------------------------------------------------------------------------------------------------

def _train_step(model:nn.Module, train_loader:torch.utils.data.DataLoader, criterion:nn.Module, optimizer:torch.optim.Optimizer) -> float:
    """
    Perform a single training step for the given model.

    This function sets the model to training mode, iterates over the training 
    data provided by the DataLoader, computes the loss, and updates the model's 
    weights based on the gradients.

    Parameters:
        model (torch.nn.Module): The model to be trained.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        criterion (nn.Module): The loss function used to compute the loss.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's weights.

    Returns:
        float: The average loss for the epoch, calculated as the total loss divided 
        by the number of samples in the training dataset.
    """
    model.train()  # Set the model to training mode

    running_loss = 0.0

    for inputs, targets in train_loader:
        inputs, targets = inputs.to(model.device), targets.to(model.device)

        optimizer.zero_grad()
        outputs = model(inputs)
        if outputs.size() != targets.size():
            outputs = outputs.squeeze()
        loss = criterion(outputs, targets)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()

    epoch_loss = running_loss / len(train_loader.dataset)

    return epoch_loss

def train(model:nn.Module, criterion: nn.Module, optimizer:torch.optim.Optimizer, train_loader:torch.utils.data.DataLoader, val_loader:torch.utils.data.DataLoader=None, num_epochs=10) -> None:
    """
    Train and optionally validate the given model for a specified number of epochs.

    This function moves the model to the specified device, initializes the results 
    dictionary to store training and validation losses, and iteratively performs 
    training (and validation) steps for the specified number of epochs. It records 
    the training loss and, if validation data is provided, the validation loss.

    Parameters:
        model (torch.nn.Module): The model to be trained (and validated).
        criterion (torch.nn.Module): The loss function used to compute the losses.
        optimizer (torch.optim.Optimizer): The optimizer used for updating the model's weights.
        train_loader (torch.utils.data.DataLoader): DataLoader for the training data.
        val_loader (torch.utils.data.DataLoader, optional): DataLoader for the validation data. Default is None.
        num_epochs (int, optional): Number of epochs to train the model. Default is 10.

    Returns:
        None: The function modifies the model in place and records the training 
        (and validation) results in the model's `results` attribute.
    """
    model.to(model.device)

    results = {"train_loss": [], "val_loss": []}

    if not hasattr(model, "results"):
        setattr(model, "results", results)

    for epoch in range(num_epochs):
        start_time = timer()
        train_loss = _train_step(model, train_loader, criterion, optimizer)
        val_loss = None
        if val_loader is not None:
            val_loss = _evaluation_step(model, val_loader, criterion)
        end_time = timer()

        model.results["train_loss"].append(train_loss)
        model.results["val_loss"].append(val_loss)

        clear_output(wait=True)
        if val_loss is not None:
            print(f'| Epoch [{epoch+1}/{num_epochs}] | Time: {end_time-start_time:.1f} |\n'
                  f'| Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} |\n')
        else:
            print(f'| Epoch [{epoch+1}/{num_epochs}] | Time: {end_time-start_time:.1f} |\n'
                  f'| Train Loss: {train_loss:.4f} |\n')

        plot_training(model)

def plot_training(model, save_path: str = ""):
    """
    Plots training curves of a results dictionary.

    Parameters:
        model (torch.nn.Module): The model to be ploted. Needs to have the "results" attribute {"train_loss": [...],"val_loss": [...]}
        save_path (str): path to save the plot as PNG
    """
    results = model.results

    loss = results['train_loss']
    val_loss = results['val_loss']

    epochs = range(len(results['train_loss']))

    plt.figure(figsize=(5, 5))
    plt.plot(epochs, loss, label='train_loss')
    plt.plot(epochs, val_loss, label='val_loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    if save_path:
        plt.savefig(save_path)
    else:
        plt.show()

#---------------------------------------------------------------------------------------------------
