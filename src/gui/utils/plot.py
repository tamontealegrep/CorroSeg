
import torch
from PIL import Image, ImageTk
import matplotlib.pyplot as plt
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from src.models.architectures.networks import Network

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def update_canvas(images, canvases, section_height, section_index):
    """
    Updates the visible section of the images displayed on the canvases based on the section index.
    
    This function crops all images in the `images` dictionary according to the specified `section_index`
    and updates the corresponding canvas images. It ensures that only the part of the images within the defined scrollable 
    region is shown, allowing the user to scroll through large images.

    Parameters:
        images (dict): Dictionary where keys are image identifiers (e.g., 'data', 'mask', 'pred') 
                       and values are the images (PIL.Image objects).
        canvases (dict): Dictionary where keys match the `images` dictionary, and values are the corresponding 
                         Tkinter canvas widgets to display those images.
        section_height (int): The height of each section to display. Determines how much of the image is shown at a time.
        section_index (int): The index of the section to be displayed. Determines the visible portion of the images.
    """
    img_width, img_height = images['data'].size 

    start_y = section_index * section_height
    end_y = min(start_y + section_height, img_height)
    
    sections = {}
    for key, img in images.items():
        sections[key] = img.crop((0, start_y, img_width, end_y))

    crop_width, crop_height = sections["data"].size
    
    tk_images = {}
    for key, img in sections.items():
        tk_images[key] = ImageTk.PhotoImage(img)

    for key, canvas in canvases.items():
        canvas.delete("all")
    
    for key, canvas in canvases.items():
        if key in tk_images:
            canvas.create_image(0, 0, image=tk_images[key], anchor="nw")
            canvas.image = tk_images[key]

    for canvas in canvases.values():
        canvas.config(scrollregion=(0, 0, crop_width, crop_height))

def plot_training(model: Network,
                  canvas: FigureCanvasTkAgg) -> None:
    """
    Plots training curves of a results dictionary. Can update the plot in real-time using a Tkinter canvas.

    Parameters:
        model (Network): The model to be plotted. Needs to have the "results"
            attribute {"train_loss": [...],"val_loss": [...]}
        canvas (FigureCanvasTkAgg): Tkinter canvas to update the plot in real-time.
    """

    results = model.results
    loss = results['train_loss']
    val_loss = results['val_loss']
    epochs = range(len(results['train_loss']))
    
    fig = canvas.figure
    ax = fig.axes[0]  
    
    ax.clear()
    
    ax.plot(epochs, loss, label='Train Loss')
    ax.plot(epochs, val_loss, label='Val Loss')
    ax.set_title('Training Loss')
    ax.set_xlabel('Epochs')
    ax.set_ylabel('Loss')
    ax.legend()

    ax.yaxis.set_ticks_position('left')
    ax.yaxis.set_label_position('right') 

    canvas.figure = fig
    canvas.draw() 

#-----------------------------------------------------------------------------------------------------------------------------------------------------