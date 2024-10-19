
import torch
import torch.nn as nn
import numpy as np
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def plot_X_and_y(X:np.ndarray, y:np.ndarray=None, X_range:tuple=(-0.2, 0.2), y_range:tuple=(0, 1), figsize:tuple=(5, 5)):
    """
    Visualize X (features) and optionally y (mask) side by side.

    Args:
        X_data (numpy.ndarray): Features 2D array.
        y_data (numpy.ndarray, optional): Target 2D array. If None, only X_data will be visualized.
        X_range (tuple): The (min, max) range for the X image color scale.
        y_range (tuple): The (min, max) range for the Y image color scale.
        figsize (tuple): Size of the subplots.
        
    Returns:
        None: Displays the plot of the images side by side.
    """
    # Create a subplot to display the images
    num_images = 2 if y is not None else 1
    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    # Ensure axes is always a list
    if num_images == 1:
        axes = [axes]

    # Plot the primary image
    image_x = np.array(X)
    axes[0].imshow(image_x, vmin=X_range[0], vmax=X_range[1], cmap="gray")
    axes[0].set_title("Input Image")
    axes[0].axis("off")

    # Plot the secondary image if available
    if y is not None:
        image_y = np.array(y)
        axes[1].imshow(image_y, vmin=y_range[0], vmax=y_range[1], cmap="viridis")
        axes[1].set_title("Mask Image")
        axes[1].axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_sample(X, y, X_range=(-1, 1), y_range=(0, 1), sample_id=None, figsize=(10, 10)):
    """
    Plot a sample image and its corresponding mask.

    Parameters:
        X (numpy.ndarray): Array of images with shape (n_samples, h, w).
        y (numpy.ndarray): Array of masks with shape (n_samples, h, w).
        X_range (tuple, optional): Tuple with the min and max values for the Img display.
        y_range (tuple, optional): Tuple with the min and max values for the Mask display.
        sample_id (int, optional): Index of the sample to plot. If None, a random sample will be selected.
        figsize (tuple, optional): Size of the figure for the plot.
    """
    # Select a random sample if no sample_id is provided
    if sample_id is None:
        sample_id = np.random.randint(0, X.shape[0])

    plot_img = X[sample_id]
    plot_msk = y[sample_id]

    # Create subplots
    fig, axes = plt.subplots(1, 2, figsize=figsize)
    
    # Plot image
    axes[0].imshow(plot_img, vmin=X_range[0], vmax=X_range[1], cmap="gray")
    axes[0].set_title(f"Image (ID: {sample_id})")
    axes[0].axis("off")
    
    # Plot mask
    axes[1].imshow(plot_msk, vmin=y_range[0], vmax=y_range[1])
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

def plot_predictions(model:nn.Module, dataset:torch.utils.data.Dataset, num_samples:int=5, id:Optional[Union[int,list]]=None, X_range:Tuple=(-1, 1), y_range:Tuple=(0, 1), figsize:Tuple=(5, 5)):
    """
    Plot original images, true labels, and model predictions.

    Args:
        model (torch.nn.Module): The model used for predictions.
        dataset (torch.utils.data.Dataset): The dataset containing images and labels.
        num_samples (int, optional): Number of samples to plot. Default is 5.
        id (None, int, or list, optional): Specific indices to plot. Default is None, which generates random indices.
        X_range (tuple, optional): Tuple with the min and max values for the Img display.
        y_range (tuple, optional): Tuple with the min and max values for the Mask display.
        
    """
    model.eval()  # Set the model to evaluation mode

    if id is None:
        indices = np.random.choice(len(dataset), num_samples, replace=False)
    elif isinstance(id, int):
        indices = [id] if id < len(dataset) else []
    else:
        indices = [i for i in id if i < len(dataset)]

    with torch.no_grad():
        for i in indices:
            # Get the image and label
            image, label = dataset[i]
            image = image.unsqueeze(0)  # Add batch dimension if necessary

            # Move image to the same device as the model
            image = image.to(next(model.parameters()).device)

            # Get model prediction
            prediction = model(image)

            # Convert tensors to numpy for plotting
            image_np = image.squeeze().cpu().numpy()
            prediction_np = prediction.squeeze().cpu().numpy()

            # Handle label if it exists
            if label is not None:
                label_np = label.squeeze().cpu().numpy()
            else:
                label_np = None 

            # Plot
            plt.figure(figsize=figsize)
            plt.subplot(1, 3, 1)
            plt.title(f'Image {i}')
            plt.imshow(image_np, vmin=X_range[0], vmax=X_range[1], cmap='gray')
            plt.axis('off')

            if label_np is not None:
                plt.subplot(1, 3, 2)
                plt.title('True Label')
                plt.imshow(label_np, vmin=y_range[0], vmax=y_range[1])
                plt.axis('off')

            plt.subplot(1, 3, 3) if label_np is not None else plt.subplot(1, 3, 2)
            plt.title('Prediction')
            plt.imshow(prediction_np, vmin=y_range[0], vmax=y_range[1])
            plt.axis('off')

            plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------