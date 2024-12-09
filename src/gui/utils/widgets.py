
import tkinter as tk
from typing import Callable, Optional, Tuple

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def create_entry(
        parent: tk.Widget,
        row: int,
        column: int,
        label_text: str, 
        default_value: str,
        validate_input: Optional[Callable[[str], bool]] = None, 
        width: int = 5,
        padx: int = 0,
        pady: int = 0,
        ) -> tk.Entry:
    """
    Creates an entry field with optional validation.
    
    Parameters:
        parent (tk.Widget): The parent widget that contains this entry.
        row (int): The row position in the grid layout.
        column (int): The column position in the grid layout.
        label_text (str): The text for the label next to the entry.
        default_value (str): The initial value of the entry field.
        validate_input (Optional[Callable[[str], bool]], optional): A function to validate the input (default is None).
        width (int, optional): The width of the entry field (default is 5).
        padx (int, optional): Horizontal padding around the widget (default is 0).
        pady (int, optional): Vertical padding around the widget (default is 0).
    
    Returns:
        (tk.Entry): The created entry widget.
    """
    frame = tk.Frame(parent)
    tk.Label(frame, text=label_text).pack(side="left")

    if validate_input is None:
        entry = tk.Entry(frame, width=width)
    else:
        entry = tk.Entry(frame, validate="key", validatecommand=(parent.register(validate_input), '%P'), width=width)
    
    entry.insert(0, default_value)
    entry.pack(side="left")
    frame.grid(row=row, column=column, padx=padx, pady=pady)
    return entry

def create_slider(
        parent: tk.Widget,
        row: int,
        column: int,
        label_text: str, 
        from_: int,
        to_: int,
        default_value: int,
        resolution: int = 1, 
        padx: int = 0,
        pady: int = 0,
        ) -> tk.Scale:
    """
    Creates a slider control.
    
    Parameters:
        parent (tk.Widget): The parent widget that contains this slider.
        row (int): The row position in the grid layout.
        column (int): The column position in the grid layout.
        label_text (str): The text for the label next to the slider.
        from_ (int): The minimum value of the slider.
        to_ (int): The maximum value of the slider.
        default_value (int): The initial value of the slider.
        resolution (int, optional): The resolution of the slider (default is 1).
        padx (int, optional): Horizontal padding around the widget (default is 0).
        pady (int, optional): Vertical padding around the widget (default is 0).
    
    Returns:
        (tk.Scale): The created slider widget.
    """
    if not (from_ <= default_value <= to_):
        raise ValueError(f"default_value '{default_value}' must be in range [{from_}, {to_}].")
    
    frame = tk.Frame(parent)
    tk.Label(frame, text=label_text).pack(side="top")
    slider = tk.Scale(frame, from_=from_, to=to_, orient="horizontal", resolution=resolution)
    slider.set(default_value)
    slider.pack(side="top")
    frame.grid(row=row, column=column, padx=padx, pady=pady)
    return slider

def create_checkbox(
        parent: tk.Widget,
        row: int,
        column: int,
        label_text: str, 
        default_value: bool,
        padx: int = 0,
        pady: int = 0,
        ) -> Tuple[tk.BooleanVar, tk.Checkbutton]:
    """
    Creates a checkbox control.
    
    Parameters:
        parent (tk.Widget): The parent widget that contains this checkbox.
        row (int): The row position in the grid layout.
        column (int): The column position in the grid layout.
        label_text (str): The text for the label next to the checkbox.
        default_value (bool): The initial value of the checkbox (True or False).
        padx (int, optional): Horizontal padding around the widget (default is 0).
        pady (int, optional): Vertical padding around the widget (default is 0).
    
    Returns:
        Tuple[tk.BooleanVar, tk.Checkbutton]: The BooleanVar and Checkbutton objects.
    """
    if not isinstance(default_value, bool):
        raise ValueError(f"default_value '{default_value}' must be True or False.")
    
    frame = tk.Frame(parent)
    var = tk.BooleanVar(value=default_value)
    checkbox = tk.Checkbutton(frame, text=label_text, variable=var)
    checkbox.pack(side="top")
    frame.grid(row=row, column=column, padx=padx, pady=pady)
    return var, checkbox

def create_option_menu(
        parent: tk.Widget,
        row: int,
        column: int,
        label_text: str, 
        options: list[str],
        default_value: str,
        width: int = 10, 
        padx: int = 0,
        pady: int = 0,
        ) -> Tuple[tk.StringVar, tk.OptionMenu]:
    """
    Creates a dropdown menu with a valid default value and customizable size.
    
    Parameters:
        parent (tk.Widget): The parent widget that contains this option menu.
        row (int): The row position in the grid layout.
        column (int): The column position in the grid layout.
        label_text (str): The text for the label next to the option menu.
        options (list[str]): A list of options for the menu.
        default_value (str): The default selected value.
        width (int, optional): The width of the option menu (default is 10).
        padx (int, optional): Horizontal padding around the widget (default is 0).
        pady (int, optional): Vertical padding around the widget (default is 0).
    
    Returns:
        Tuple[tk.StringVar, tk.OptionMenu]: The StringVar and OptionMenu objects.
    """
    if default_value not in options:
        raise ValueError(f"default_value '{default_value}' is not in options: {options}.")
    
    var = tk.StringVar(value=default_value)
    frame = tk.Frame(parent)
    tk.Label(frame, text=label_text).pack(side="left")
    menu = tk.OptionMenu(frame, var, *options)
    menu.config(width=width) 
    menu.pack(side="left")
    frame.grid(row=row, column=column, padx=padx, pady=pady)
    return var, menu

#-----------------------------------------------------------------------------------------------------------------------------------------------------