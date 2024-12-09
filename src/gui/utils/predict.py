
import tkinter as tk
from tkinter import filedialog, messagebox

from src.utils.files import npy_file_to_dict
from src.utils.files import load_config

#-----------------------------------------------------------------------------------------------------------------------------------------------------

config = load_config("./src/config/config.yaml")
predict_default = config["predict"]
threshold_default = config["threshold"]

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def load_file(gui):
    """
    Prompts the user to select and load a .npy file, then loads its contents into the GUI.

    This function opens a file dialog to allow the user to select a `.npy` file. Upon successful 
    loading, it updates the `gui` object with the file's contents and filename. If an error occurs 
    during loading, an error message is displayed.

    Parameters:
        gui (tk.Tk): The Tkinter GUI object that will hold the loaded file and filename.

    Returns:
        None: This function modifies the state of the `gui` object directly. If the file is loaded
        successfully, it assigns the file contents and name to `gui.file` and `gui.file_name`.
        If an error occurs, an error message is shown.

    Raises:
        Exception: If an error occurs while loading the file, an exception is caught and an error 
        message is displayed.
    """
    file_path = filedialog.askopenfilename(
        title="Load file",
        filetypes=[("Data file", "*.npy")])
    
    if file_path:
        try:
            file = npy_file_to_dict(file_path)
            gui.file = list(file.values())[0]
            gui.file_name = list(file.keys())[0]
            messagebox.showinfo("Exit", "File loaded correctly.")
        except Exception as e:
            messagebox.showerror("Error", f"The File could not be loaded. Error: {e}")
            return
        
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
        gui.prediction = binary_mask
        messagebox.showinfo("Success", "File processed successfully.")
    except Exception as e:
        messagebox.showerror("Error", f"The file could not be processed. Error: {e}")
        return

#-----------------------------------------------------------------------------------------------------------------------------------------------------