
import numpy as np
from tkinter import filedialog, messagebox
from typing import Tuple

from src.utils.files import npy_file_to_dict

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def load_file() -> Tuple[np.ndarray, str]:
    """
    Prompts the user to select and load a .npy file, then loads its contents.

    This function opens a file dialog to allow the user to select a `.npy` file. Upon successful 
    loading, returns the file's contents and filename. If an error occurs 
    during loading, an error message is displayed.

    Returns:
        tuple: A tuple containing.
            file (np.ndarray): The contents of the .npy file.
            file_name (str): The name of the .npy file (without the .npy extension)

    Raises:
        Exception: If an error occurs while loading the file, an exception is caught and an error 
        message is displayed.
    """
    file_path = filedialog.askopenfilename(
        title="Load file",
        filetypes=[("Data file", "*.npy")])
    
    if file_path:
        try:
            file_dict = npy_file_to_dict(file_path)
            file = list(file_dict.values())[0]
            file_name = list(file_dict.keys())[0]
            messagebox.showinfo("Exit", f"File {file_name}.npy loaded correctly.")
        except Exception as e:
            messagebox.showerror("Error", f"The File could not be loaded. Error: {e}")
            return
        
        return file, file_name

#-----------------------------------------------------------------------------------------------------------------------------------------------------