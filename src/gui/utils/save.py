
from tkinter import filedialog, messagebox

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def save_model(gui):
        if gui.model is None:
            messagebox.showwarning("Error", "There is no model to save.")
            return

        model_path = filedialog.asksaveasfilename(
            title="Save model",
            defaultextension=".pt",
            filetypes=[("Model file", "*.pt;*.pth")]
        )

        if model_path:
            try:
                gui.model.save_model(model_path)
                messagebox.showinfo("Success", "Model saved.")
            except Exception as e:
                messagebox.showerror("Error", f"The model could not be saved. Error: {e}")

#-----------------------------------------------------------------------------------------------------------------------------------------------------
