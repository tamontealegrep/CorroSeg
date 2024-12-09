
from PIL import Image, ImageTk

#-----------------------------------------------------------------------------------------------------------------------------------------------------

def update_section(data_img, mask_img, data_canvas, mask_canvas, section_height, section_index):
    """
    Updates the visible section of the images displayed on the canvases based on the section index.
    
    This function crops the `data_img` and `mask_img` according to the specified `section_index` and 
    updates the corresponding canvas images. It ensures that only the part of the images within the 
    defined scrollable region is shown, allowing the user to scroll through large images.

    Parameters:
        data_img (PIL.Image): The image to be displayed on the data canvas.
        mask_img (PIL.Image): The image to be displayed on the mask canvas.
        data_canvas (tk.Canvas): The Tkinter canvas widget where the data image will be displayed.
        mask_canvas (tk.Canvas): The Tkinter canvas widget where the mask image will be displayed.
        section_height (int): The height of each section to display. Determines how much of the image is shown at a time.
        section_index (int): The index of the section to be displayed. Determines the visible portion of the images.
        
    """
    # Get the size of the image
    scroll_region = data_img.size 

    start_y = section_index * section_height
    end_y = min((section_index + 1) * section_height, scroll_region[1])
    
    # Crop the images to show only the visible section
    data_img_section = data_img.crop((0, start_y, scroll_region[0], end_y))
    mask_img_section = mask_img.crop((0, start_y, scroll_region[0], end_y))

    # Convert the cropped sections to Tkinter images
    tk_data_img_section = ImageTk.PhotoImage(data_img_section)
    tk_mask_img_section = ImageTk.PhotoImage(mask_img_section)

    # Clear previous images from the canvases
    data_canvas.delete("all")
    mask_canvas.delete("all")

    # Add the new cropped images to the canvases
    data_canvas.create_image(0, 0, image=tk_data_img_section, anchor="nw")
    mask_canvas.create_image(0, 0, image=tk_mask_img_section, anchor="nw")

    # Maintain references to the images to prevent them from being garbage collected
    data_canvas.image = tk_data_img_section
    mask_canvas.image = tk_mask_img_section

    # Update the scroll region of the canvases
    data_canvas.config(scrollregion=(0, start_y, scroll_region[0], end_y))
    mask_canvas.config(scrollregion=(0, start_y, scroll_region[0], end_y))

#-----------------------------------------------------------------------------------------------------------------------------------------------------