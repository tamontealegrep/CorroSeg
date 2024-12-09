
import tkinter as tk
from tkinter import messagebox

from src.utils.files import load_config, load_arrays_from_folders
from src.models.dataset.transformations import random_transformation

#-----------------------------------------------------------------------------------------------------------------------------------------------------

config = load_config("./src/config/config.yaml")
train_default = config["train_validation"]

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def train_model(
        gui,
        fraction: tk.StringVar,
        seed: tk.IntVar,
        expand: tk.BooleanVar,
        augmented_ratio: tk.StringVar,
        batch_size: tk.StringVar,
        num_epochs: tk.StringVar,
) -> None:
    """
    Trains the model using the configuration specified by the input parameters.

    This function sets up the training configuration, loads the data, prepares the datasets, and trains
    the model based on the parameters provided through Tkinter variables. It supports training with or
    without validation data, and handles data augmentation, batching, and randomization.

    Parameters:
        gui (object): The GUI object, which provides access to the model's methods for loading data,
                      setting up the training process, and running training.
        fraction (tk.StringVar): Tkinter variable representing the fraction of the dataset to use for training.
        seed (tk.IntVar): Tkinter variable representing the random seed to ensure reproducibility of results.
        expand (tk.BooleanVar): Tkinter variable indicating whether data augmentation should be applied (True or False).
        augmented_ratio (tk.StringVar): Tkinter variable representing the ratio of augmented data to apply during training.
        batch_size (tk.StringVar): Tkinter variable representing the batch size to use during training.
        num_epochs (tk.StringVar): Tkinter variable representing the number of epochs for training.

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
        X_train, y_train, X_val, y_val = gui.model.setup_train_val_data(X, y, **dictionary)

        train_dataset = gui.model.build_dataset(X_train, y_train, random_transformation, config["random_transformation"])
        val_dataset = gui.model.build_dataset(X_val,y_val)

        train_loader = gui.model.build_dataloader(train_dataset, int(batch_size.get()), True)
        val_loader = gui.model.build_dataloader(val_dataset, int(batch_size.get()), True)
                
        gui.model.train(train_loader, val_loader, num_epochs=int(num_epochs.get()))
    
    else:
        for i in ["fraction", "seed"]:
            if i in dictionary:
                del dictionary[i]

        X_train, y_train = gui.model.setup_train_data(X,y,**dictionary)

        train_dataset = gui.model.build_dataset(X_train, y_train, random_transformation, config["random_transformation"])
        train_loader = gui.model.build_dataloader(train_dataset, int(batch_size.get()), True)

        gui.model.train(train_loader, num_epochs=int(num_epochs.get()))

    messagebox.showinfo("Success", "Model trained successfully.")

#-----------------------------------------------------------------------------------------------------------------------------------------------------