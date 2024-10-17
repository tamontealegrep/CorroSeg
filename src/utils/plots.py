
import numpy as np
import matplotlib.pyplot as plt

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
    
#-----------------------------------------------------------------------------------------------------------------------------------------------------