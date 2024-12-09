
from tkinter import filedialog, messagebox

from src.models.manager import ModelManager
from src.models.architectures.networks.unet import Unet

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def load_model(gui):
        model_path = filedialog.askopenfilename(
            title="Load model",
            filetypes=[("Model file", "*.pt;*.pth")])
        
        if model_path:
            try:
                model = ModelManager.load_model(Unet, model_path)
                gui.model = model
                #gui.root.config(menu=self.enable_save_model_option())
                messagebox.showinfo("Success", "Model loaded.")
            except Exception as e:
                messagebox.showerror("Error", f"The model could not be loaded. Error: {e}")

#-----------------------------------------------------------------------------------------------------------------------------------------------------
