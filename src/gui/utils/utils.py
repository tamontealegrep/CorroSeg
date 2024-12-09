
import tkinter as tk
from typing import List

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def toggle_widgets_by_bool(
        bool_var: tk.BooleanVar,
        widgets: List[tk.Widget]) -> None:
    """
    Toggles the state of widgets based on the value of a Boolean control variable.
    
    Parameters:
        bool_var (tk.BooleanVar): The Tkinter Boolean variable that controls the state of the widgets.
        widgets (List[tk.Widget]): List of widgets to enable or disable based on the value of `bool_var`.
    
    Notes:
        - If `bool_var.get()` is `True`, the widgets will be enabled (`tk.NORMAL`).
        - If `bool_var.get()` is `False`, the widgets will be disabled (`tk.DISABLED`).
    """
    state = tk.NORMAL if bool_var.get() else tk.DISABLED
    for widget in widgets:
        widget.config(state=state)

def toggle_multiple_widgets(
            widgets_enabled: List[tk.Widget],
            widgets_disabled: List[tk.Widget]) -> None:
    """
    Enables and disables multiple widgets based on the input lists.
    
    Parameters:
        widgets_enabled (List[tk.Widget]): List of widgets to enable.
        widgets_disabled (List[tk.Widget]): List of widgets to disable.
    """
    for widget in widgets_enabled:
        widget.config(state=tk.NORMAL)

    for widget in widgets_disabled:
        widget.config(state=tk.DISABLED)

def scrollbar_command(canvases, *args):
    """
    Synchronizes the vertical scrolling of multiple canvases based on the provided scroll arguments.
    
    Parameters:
        canvases (list): A list of tk.Canvas widgets that will be synchronized.
        *args: Variable arguments to pass to the `yview` method of each canvas (usually 'moveto' or 'scroll').
    """
    for canvas in canvases:
        canvas.yview(*args)
        
#-----------------------------------------------------------------------------------------------------------------------------------------------------