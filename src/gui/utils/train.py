
import time
import tkinter as tk
from tkinter import messagebox
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

from src.utils.files import load_config, load_arrays_from_folders
from src.models.manager import ModelManager
from src.models.dataset.transformations import random_transformation
from src.gui.utils.plot import plot_training

#-----------------------------------------------------------------------------------------------------------------------------------------------------

config = load_config("./src/config/config.yaml")
train_default = config["train_validation"]

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def train_model(
        model: ModelManager,
        fraction: tk.StringVar,
        seed: tk.IntVar,
        expand: tk.BooleanVar,
        augmented_ratio: tk.StringVar,
        batch_size: tk.StringVar,
        num_epochs: tk.StringVar,
        window: tk.Widget = None,
        canvas: FigureCanvasTkAgg = None,
        epoch_label: tk.Label = None,
        time_label: tk.Label = None,
        train_loss_label: tk.Label = None,
        val_loss_label: tk.Label = None
) -> None:
    """
    Trains the model using the configuration specified by the input parameters.

    This function sets up the training configuration, loads the data, prepares the datasets, and trains
    the model based on the parameters provided through Tkinter variables. It supports training with or
    without validation data, and handles data augmentation, batching, and randomization.

    Parameters:
        model (ModelManager): The ModelManager object, which provides access to the model's methods for loading data,
                      setting up the training process, and running training.
        fraction (tk.StringVar): Tkinter variable representing the fraction of the dataset to use for training.
        seed (tk.IntVar): Tkinter variable representing the random seed to ensure reproducibility of results.
        expand (tk.BooleanVar): Tkinter variable indicating whether data augmentation should be applied (True or False).
        augmented_ratio (tk.StringVar): Tkinter variable representing the ratio of augmented data to apply during training.
        batch_size (tk.StringVar): Tkinter variable representing the batch size to use during training.
        num_epochs (tk.StringVar): Tkinter variable representing the number of epochs for training.
                window (tk.Widget): The Tkinter window (root, Toplevel, Frame, or LabelFrame) to update during training.
        canvas (FigureCanvasTkAgg): Tkinter canvas to update the plot in real-time.

    Returns:
        None: This function does not return any value. It updates the model's state through training.

    Raises:
        ValueError: If the "fraction" value is non-positive, the training will only use the training data without validation.
        
    Side Effects:
        - Loads training and validation data from directories specified in the configuration.
        - Trains the model using the training data and validation data (if applicable).
        - Displays a success message once training completes.
    """
    
    dictionary = {
        "height_stride": train_default["height_stride"],
        "width_stride": train_default["width_stride"],
        "fraction": float(fraction.get()),
        "seed": int(seed.get()),
        "expand": expand.get(),
        "augmented_ratio": float(augmented_ratio.get()),
    }

    X, y = load_arrays_from_folders(config["folders"]["train"])
    
    if dictionary["fraction"] > 0:
        X_train, y_train, X_val, y_val = model.setup_train_val_data(X, y, **dictionary)

        train_dataset = model.build_dataset(X_train, y_train, random_transformation, config["random_transformation"])
        val_dataset = model.build_dataset(X_val,y_val)

        train_loader = model.build_dataloader(train_dataset, int(batch_size.get()), True)
        val_loader = model.build_dataloader(val_dataset, int(batch_size.get()), True)
                
        for epoch in range(int(num_epochs.get())):
            start_time = time.time() 
            model.train(train_loader, val_loader, num_epochs=1)
            epoch_time = time.time() - start_time

            epoch_label.config(text=f"Epoch: {epoch+1}/{num_epochs.get()}")
            time_label.config(text=f"Epoch Time: {epoch_time:.2f}s")
            train_loss_label.config(text=f"Train Loss: {model.network.results['train_loss'][-1]:.5f}")
            val_loss_label.config(text=f"Val Loss: {model.network.results['val_loss'][-1]:.5f}")
            
            plot_training(model.network, canvas) 
            window.update_idletasks()

    else:
        for i in ["fraction", "seed"]:
            if i in dictionary:
                del dictionary[i]

        X_train, y_train = model.setup_train_data(X,y,**dictionary)

        train_dataset = model.build_dataset(X_train, y_train, random_transformation, config["random_transformation"])
        train_loader = model.build_dataloader(train_dataset, int(batch_size.get()), True)

        for epoch in range(int(num_epochs.get())):
            start_time = time.time() 
            model.train(train_loader, num_epochs=1)
            epoch_time = time.time() - start_time

            epoch_label.config(text=f"Epoch: {epoch+1}/{num_epochs.get()}")
            time_label.config(text=f"Epoch Time: {epoch_time:.2f}s")
            train_loss_label.config(text=f"Train Loss: {model.network.results['train_loss'][-1]:.5f}")
            val_loss_label.config(text=f"Val Loss: NaN")

            plot_training(model.network, canvas)
            window.update_idletasks()

    messagebox.showinfo("Success", "Model trained successfully.")

#-----------------------------------------------------------------------------------------------------------------------------------------------------