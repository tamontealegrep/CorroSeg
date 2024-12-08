
import torch
import torch.nn as nn
import numpy as np
import PIL
from PIL import Image
import matplotlib.cm as cm
import matplotlib.pyplot as plt
from typing import Tuple, Union, Optional

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def _clip_and_scale_array(
        data: np.ndarray,
        data_range: Tuple[Union[int, float], Union[int, float]],
        scale_range: Optional[Tuple[Union[int, float], Union[int, float]]] = (0,1),
        ) -> np.ndarray:
    """
    Clips the values outside the range [data_range[0], data_range[1]], 
    then scales the array to the range [scale_range[0], scale_range[1]].

    Parameters:
        data (numpy.ndarray): The input array to be processed.
        data_range (tuple): A tuple (min, max) defining the range of values to clip in the input data.
        scale_range (tuple): A tuple (min, max) defining the range to scale the clipped data to. Default [0, 1]
    
    Returns:
        numpy.ndarray: The clipped and scaled array within the range [scale_range[0], scale_range[1]].
    
    Raises:
        ValueError: If the ranges are not tuples of length 2 or if the min value is greater than the max value.
    """
    if len(data_range) != 2:
        raise ValueError("data_range must be a tuple (min, max)")
    
    if data_range[0] > data_range[1]:
        raise ValueError(f"data_range max {data_range[1]} must be greater than data_range min {data_range[0]}")

    if len(scale_range) != 2:
        raise ValueError("scale_range must be a tuple (min, max)")
    
    if scale_range[0] > scale_range[1]:
        raise ValueError(f"scale_range max {scale_range[1]} must be greater than scale_range min {scale_range[0]}")

    data_clipped = np.clip(data, data_range[0], data_range[1])
    data_scaled = scale_range[0] + (data_clipped - data_range[0]) * (scale_range[1] - scale_range[0]) / (data_range[1] - data_range[0])
    
    return data_scaled

def np_array_to_pil(
        data: np.ndarray,
        data_range: Tuple[Union[int, float], Union[int, float]],
        scale_range: Optional[Tuple[Union[int, float], Union[int, float]]] = (0,1),
        cmap_name: Optional[str] = "gray") -> PIL.Image.Image:
    """
    Converts a numpy array to a PIL image using a colormap, with optional clipping and scaling.

    The function first clips and scales the input data to the specified `scale_range` based on the provided 
    `data_range`, then applies a colormap (such as 'gray', 'viridis', etc.) to the normalized data. The resulting 
    color-mapped data is then converted into a PIL Image.

    Parameters:
        data (numpy.ndarray): The input array to be converted to a PIL image.
        data_range (tuple): A tuple (min, max) defining the range of values in the input `data` to be clipped.
        scale_range (tuple, optional): A tuple (min, max) defining the range to scale the clipped data to. Defaults to (0, 1).
        cmap_name (str, optional): The name of the matplotlib colormap to apply. Defaults to 'gray'.

    Returns:
        PIL.Image.Image: A PIL Image object with the color-mapped data.

    Notes:
        - The input data will be normalized to the range [0, 1] before applying the colormap.
        - The colormap is applied using matplotlib's `get_cmap` function.
        - The default colormap is 'gray', but you can specify other colormaps like 'viridis', 'plasma', etc.
    """
    norm_data = _clip_and_scale_array(data,data_range,scale_range)

    cmap = cm.get_cmap(cmap_name)

    colored_data = cmap(norm_data)
    colored_data = (colored_data[:, :, :] * 255).astype(np.uint8)

    pil_image = Image.fromarray(colored_data)
    
    return pil_image

def plot_data(X: np.ndarray,
              y: Optional[np.ndarray] = None,
              z: Optional[np.ndarray] = None,
              X_range: Optional[tuple] = (-0.2, 0.2),
              y_range: Optional[tuple] = (0, 1),
              X_cmap: Optional[str] = "gray",
              y_cmap: Optional[str] = "viridis",
              vertical_range: Optional[tuple] = None,
              figsize: Optional[tuple] = (5, 5),
              ) -> None:
    """
    Visualize X (features), y (mask), and optionally z (predictions) side by side.

    Parameters:
        X (numpy.ndarray): Features 2D array.
        y (numpy.ndarray, optional): Target 2D array. If None, only X will be visualized.
        z (numpy.ndarray, optional): Predictions 2D array. If None, only X and y (if provided) will be visualized.
        X_range (tuple, optional): The (min, max) range for the X color scale. Default: (-0.2, 0.2)
        y_range (tuple, optional): The (min, max) range for the Y color scale. Default: (0, 1)
        X_cmap (str, optional): Colormap name for X. Default "gray".
        y_cmap (str, optional): Colormap name for y, and z. Default "viridis".
        vertical_range (tuple, optional): A tuple (start, end) to slice the image vertically. Deafult: None.
        figsize (tuple, optional): Size of the subplots.

    Returns:
        None: Displays the plot of the images side by side.
    """
    # Determine the number of images to display based on the presence of y and z
    num_images = 1  # Start with X
    if y is not None:
        num_images += 1  # Add y if it's provided
    if z is not None:
        num_images += 1  # Add z if it's provided

    # Create subplots to display the images
    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    # Ensure axes is always a list, even if it's just one axis
    if num_images == 1:
        axes = [axes]

    # If vertical_range is specified, slice the images vertically
    if vertical_range is not None:
        X = X[vertical_range[0]:vertical_range[1]]
        if y is not None:
            y = y[vertical_range[0]:vertical_range[1]]
        if z is not None:
            z = z[vertical_range[0]:vertical_range[1]]

    # Plot the primary image (X)
    axes[0].imshow(X, vmin=X_range[0], vmax=X_range[1], cmap=X_cmap)
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Plot the secondary image (y) if available
    if y is not None:
        axes[1].imshow(y, vmin=y_range[0], vmax=y_range[1], cmap=y_cmap)
        axes[1].set_title("Mask")
        axes[1].axis("off")

    # Plot the prediction image (z) if available
    if z is not None:
        axes[-1].imshow(z, vmin=y_range[0], vmax=y_range[1], cmap=y_cmap)
        axes[-1].set_title("Prediction")
        axes[-1].axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_overlay(X: np.ndarray,
              y: Optional[np.ndarray] = None,
              z: Optional[np.ndarray] = None,
              range: Optional[tuple] = (-0.2, 0.2),
              X_cmap: Optional[str] = "gray",
              y_cmap: Optional[str] = "gray",
              vertical_range: Optional[tuple] = None,
              figsize: Optional[tuple] = (5, 5),
              darken_factor: Optional[float] = 0,
              background_color: Optional[str] = "white"
              ) -> None:
    """
    Visualize X (features), y (mask), and optionally z (predictions) superimposed on X, where the mask/prediction
    areas with a value of 0 are darkened and areas with a value of 1 remain unchanged.

    Parameters:
        X (numpy.ndarray): Features 2D array (image).
        y (numpy.ndarray, optional): Target 2D array (mask). If None, only X will be visualized.
        z (numpy.ndarray, optional): Predictions 2D array. If None, only X and y (if provided) will be visualized.
        range (tuple, optional): The (min, max) range for the X color scale. Default: (-0.2, 0.2)
        X_cmap (str, optional): Colormap name for X. Default "gray".
        y_cmap (str, optional): Colormap name for y, and z. Default "viridis".
        vertical_range (tuple, optional): A tuple (start, end) to slice the image vertically. Default: None.
        figsize (tuple, optional): Size of the subplots.
        darken_factor (float, optional): Factor by which to darken pixels in X where y or z are 0. Default: 0.
        background_color (str, optional): Color of the background. Default white.

    Returns:
        None: Displays the plot of the images side by side with the mask/prediction overlaid.
    """
    # Determine the number of images to display based on the presence of y and z
    num_images = 1  # Start with X
    if y is not None:
        num_images += 1  # Add y if it's provided
    if z is not None:
        num_images += 1  # Add z if it's provided

    # Create subplots to display the images
    fig, axes = plt.subplots(1, num_images, figsize=figsize)

    # Ensure axes is always a list, even if it's just one axis
    if num_images == 1:
        axes = [axes]

    fig.patch.set_facecolor(background_color)

    # If vertical_range is specified, slice the images vertically
    if vertical_range is not None:
        X = X[vertical_range[0]:vertical_range[1]]
        if y is not None:
            y = y[vertical_range[0]:vertical_range[1]]
        if z is not None:
            z = z[vertical_range[0]:vertical_range[1]]

    # Plot the primary image (X)
    axes[0].imshow(X, vmin=range[0], vmax=range[1], cmap=X_cmap)
    axes[0].set_title("Input")
    axes[0].axis("off")

    # Overlay the mask (y) if available
    if y is not None:
        X_with_y = X.copy()
        X_with_y[y == 0] *= darken_factor  # Darken the pixels where y is 0
        axes[1].imshow(X_with_y, vmin=range[0], vmax=range[1], cmap=y_cmap)
        axes[1].set_title("Mask Overlay")
        axes[1].axis("off")

    # Overlay the prediction (z) if available
    if z is not None:
        X_with_z = X.copy()
        X_with_z[z == 0] *= darken_factor  # Darken the pixels where z is 0
        axes[-1].imshow(X_with_z, vmin=range[0], vmax=range[1], cmap=y_cmap)
        axes[-1].set_title("Prediction Overlay")
        axes[-1].axis("off")

    # Show the plot
    plt.tight_layout()
    plt.show()

def plot_sample(X: np.ndarray,
                y: np.ndarray,
                X_range: Optional[tuple] = (-1, 1),
                y_range: Optional[tuple] = (0, 1),
                X_cmap: Optional[str] = "gray",
                y_cmap: Optional[str] = "viridis",
                sample_id: Optional[int] = None,
                figsize: Optional[tuple] = (10, 10),
                ) -> None:
    """
    Plot a sample image and its corresponding mask.

    Parameters:
        X (numpy.ndarray): Array of images with shape (n_samples, h, w).
        y (numpy.ndarray): Array of masks with shape (n_samples, h, w).
        X_range (tuple, optional): Tuple with the min and max values for the Img display.
        y_range (tuple, optional): Tuple with the min and max values for the Mask display.
        X_cmap (str, optional): Colormap name for X. Default "gray".
        y_cmap (str, optional): Colormap name for y, and z. Default "viridis".
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
    axes[0].imshow(plot_img, vmin=X_range[0], vmax=X_range[1], cmap=X_cmap)
    axes[0].set_title(f"Image (ID: {sample_id})")
    axes[0].axis("off")
    
    # Plot mask
    axes[1].imshow(plot_msk, vmin=y_range[0], vmax=y_range[1], cmap=y_cmap)
    axes[1].set_title("Mask")
    axes[1].axis("off")

    plt.tight_layout()
    plt.show()

#-----------------------------------------------------------------------------------------------------------------------------------------------------