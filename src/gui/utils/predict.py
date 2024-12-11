
import tkinter as tk
from tkinter import filedialog, messagebox

from src.utils.files import npy_file_to_dict
from src.utils.files import load_config

#-----------------------------------------------------------------------------------------------------------------------------------------------------

config = load_config("./src/config/config.yaml")
predict_default = config["predict"]
threshold_default = config["threshold"]

#-----------------------------------------------------------------------------------------------------------------------------------------------------
        
def predict_model(gui):
    """
    Processes the loaded file and generates a prediction using the model.

    This function uses the model's `predict_well` method to generate a prediction based on the 
    file loaded in the `gui` object (`gui.file`). The prediction is then thresholded to create 
    a binary mask, which is stored in the `gui.prediction` attribute. A success message is shown 
    if the prediction is successful, and an error message is displayed if an error occurs during 
    processing.

    Parameters:
        gui (tk.Tk): The Tkinter GUI object that contains the model (`gui.model`), the loaded file 
        (`gui.file`), and where the prediction will be stored (`gui.prediction`).

    Returns:
        None: This function does not return any value. It updates the `gui.prediction` attribute 
        with the binary mask generated from the model's prediction.

    Raises:
        Exception: If an error occurs during the prediction process, an exception is caught and 
        an error message is displayed to the user.
    """
    try:
        prediction = gui.model.predict_well(gui.file, **predict_default)
        binary_mask = ((prediction) > threshold_default).astype(int)
        gui.y_pred = binary_mask
        messagebox.showinfo("Success", "File processed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"The file could not be processed. Error: {e}")
        return

#-----------------------------------------------------------------------------------------------------------------------------------------------------