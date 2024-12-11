
from PIL import Image, ImageTk

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
    # Get the size of the data image (assuming all images are the same size)
    img_width, img_height = images['data'].size 

    start_y = section_index * section_height
    end_y = min(start_y + section_height, img_height)
    
    # Loop through the images and crop them
    sections = {}
    for key, img in images.items():
        # Crop the image
        sections[key] = img.crop((0, start_y, img_width, end_y))

    # Initialize the scroll region and references to the cropped images
    crop_width, crop_height = sections["data"].size
    
    # Convert the cropped sections to Tkinter images
    tk_images = {}
    for key, img in sections.items():
        # Convert to Tkinter image
        tk_images[key] = ImageTk.PhotoImage(img)

    # Clear previous images from the canvases
    for key, canvas in canvases.items():
        canvas.delete("all")
    
    # Add the new cropped images to the canvases
    for key, canvas in canvases.items():
        if key in tk_images:
            canvas.create_image(0, 0, image=tk_images[key], anchor="nw")
            canvas.image = tk_images[key]  # Maintain a reference to avoid garbage collection

    # Update the scroll region of all canvases
    for canvas in canvases.values():
        canvas.config(scrollregion=(0, 0, crop_width, crop_height))
        
#-----------------------------------------------------------------------------------------------------------------------------------------------------