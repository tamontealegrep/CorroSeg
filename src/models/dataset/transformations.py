
import cv2
import numpy as np

#---------------------------------------------------------------------------------------------------

def vertical_reflection(X, y):
    """
    Perform vertical reflection (flipping) on input arrays X and y.

    Args:
        X (np.ndarray): Input array of shape (h, w) or (n, h, w).
        y (np.ndarray): Target array of the same shape as X.

    Returns:
        tuple: A tuple containing:
            - X_reflected (np.ndarray): Vertically reflected input samples.
            - y_reflected (np.ndarray): Vertically reflected target samples.
    """
    if X.shape != y.shape:
        raise ValueError("X and y must have the same dimentions")

    if X.ndim == 2: # (h, w)
        X_reflected = np.flipud(X)
        y_reflected = np.flipud(y)
    
    elif X.ndim == 3: # (n, h, w)
        X_reflected = np.array([np.flipud(i) for i in X])
        y_reflected = np.array([np.flipud(i) for i in y])

    else:
        raise ValueError("X and y must be a 2 or 3 dimensional array")

    return X_reflected, y_reflected

def horizontal_reflection(X, y):
    """
    Perform horizontal reflection (flipping) on input arrays X and y.

    Args:
        X (np.ndarray): Input array of shape (h, w) or (n, h, w).
        y (np.ndarray): Target array of the same shape as X.

    Returns:
        tuple: A tuple containing:
            - X_reflected (np.ndarray): Horizontally reflected input samples.
            - y_reflected (np.ndarray): Horizontally reflected target samples.
    """
    if X.shape != y.shape:
        raise ValueError("X and y must have the same dimentions")

    if X.ndim == 2: # (h, w)
        X_reflected = np.fliplr(X)
        y_reflected = np.fliplr(y)
    
    elif X.ndim == 3: # (n, h, w)
        X_reflected = np.array([np.fliplr(i) for i in X])
        y_reflected = np.array([np.fliplr(i) for i in y])

    else:
        raise ValueError("X and y must be a 2 or 3 dimensional array")

    return X_reflected, y_reflected

def gaussian_noise(X, y, alpha=0.1):
    """
    Add Gaussian noise to the input image.

    Args:
        X (np.ndarray): Input image array of shape (h, w) or (n, h, w).
        y (np.ndarray): Input label array of the same shape as X.
        alpha (float, optional): Standard deviation of the Gaussian noise to be added. Default is 0.1.

    Returns:
        tuple: A tuple containing:
            - X_noisy (np.ndarray): The input sample with Gaussian noise added.
            - y (np.ndarray): The target sample unchanged.
    """
    if X.shape != y.shape:
        raise ValueError("X and y must have the same dimentions")
    
    # Generate Gaussian noise
    noise = np.random.normal(0, alpha, X.shape)

    X_noisy = X.copy()

    # Add noise to the image
    X_noisy = X_noisy + noise

    return X_noisy, y

def salt_and_pepper_noise(X, y, salt_prob=0.025, pepper_prob=0.025, salt_value=1, pepper_value=0, noise_variation=1):
    """
    Adds salt and pepper noise to the input image and label.

    Args:
        X (np.ndarray): Input array of shape (h, w) or (n, h, w).
        y (np.ndarray): Target array of the same shape as X.
        salt_prob (float, optional): Probability of salt noise (high values). Default is 0.025.
        pepper_prob (float, optional): Probability of pepper noise (low values). Default is 0.025.
        salt_value (int, optional): Value to be used for salt noise. Default is 1.
        pepper_value (int, optional): Value to be used for pepper noise. Default is 0.
        noise_variation (int, optional): Variation range for randomizing salt and pepper values. Default is 1.

    Returns:
        tuple: A tuple containing:
            - X_noisy (np.ndarray): The input sample with salt and pepper noise added.
            - y (np.ndarray): The target sample unchanged.
    """
    if X.shape != y.shape:
        raise ValueError("X and y must have the same dimentions")
    
    X_noisy = X.copy()

    if X.ndim == 2:
        # Single image case
        h, w = X.shape
        salt_mask = np.random.rand(h, w) < salt_prob
        pepper_mask = np.random.rand(h, w) < pepper_prob
    elif X.ndim == 3:
        # Batch case
        n, h, w = X.shape
        salt_mask = np.random.rand(n, h, w) < salt_prob
        pepper_mask = np.random.rand(n, h, w) < pepper_prob
    else:
        raise ValueError("Input array X must be of shape (h, w) or (n, h, w)")


    X_noisy[salt_mask] = np.random.uniform(salt_value - noise_variation, salt_value + noise_variation, size=np.sum(salt_mask))
    X_noisy[pepper_mask] = np.random.uniform(pepper_value - noise_variation, pepper_value + noise_variation, size=np.sum(pepper_mask))


    return X_noisy, y

def cutout(X, y, min_area_fraction=0.2, max_area_fraction=0.5, value=0):
    """
    Crop a random area from the image and replace it with a specified value.

    Args:
        X (np.ndarray): Input image array of shape (h, w) or (n, h, w).
        y (np.ndarray): Input label array of the same shape as X.
        min_area_fraction (float, optional): Minimum fraction of the image area that can be hidden. Must be a value between 0 and 1. Default is 0.2.
        max_area_fraction (float, optional): Maximum fraction of the image area that can be hidden. Must be a value between 0 and 1. Default is 0.5.
        value (int, optional): Value to replace the cropped area. Default is 0.

    Returns:
        tuple: A tuple containing:
            - X_cropped (np.ndarray): The input sample with the random area set to the specified value.
            - y (np.ndarray): The target sample unchanged.
    """
    if X.shape != y.shape:
        raise ValueError("X and y must have the same dimentions")
    
    # Determine the shape
    n = X.shape[0] if X.ndim == 3 else 1
    h, w = X.shape[-2:] if X.ndim == 3 else X.shape

    X_cropped = X.copy()

    for i in range(n):
        # Random crop dimensions
        crop_x = np.random.randint(0, w)
        crop_y = np.random.randint(0, h)
        crop_x_len = np.random.randint(int(w * min_area_fraction), int(w * max_area_fraction))
        crop_y_len = np.random.randint(int(h * min_area_fraction), int(h * max_area_fraction))

        # Define cropping boundaries
        crop_x_start = max(0, crop_x)
        crop_x_end = min(w, crop_x + crop_x_len)
        crop_y_start = max(0, crop_y)
        crop_y_end = min(h, crop_y + crop_y_len)

        # Replace specified area with the specified value
        if X.ndim == 3:
            X_cropped[i, crop_y_start:crop_y_end, crop_x_start:crop_x_end] = value
        else:
            X_cropped[crop_y_start:crop_y_end, crop_x_start:crop_x_end] = value

    return X_cropped, y

def resize_and_pad(X, y, min_scale=0.8, max_scale=1.2, padding_value_x=0, padding_value_y=0):
    """
    Resize the input image and label randomly within the specified scale range,
    and pad them if they are smaller than the original size, centering the content.

    Args:
        X (np.ndarray): Input image array of shape (h, w) or (n, h, w).
        y (np.ndarray): Input label array of the same shape as X.
        min_scale (float, optional): Minimum scaling factor (for shrinking). Default is 0.8.
        max_scale (float, optional): Maximum scaling factor (for enlarging). Default is 1.2.
        padding_value_x (int or float, optional): Value to pad the image if it's smaller than original. Default is 0.
        padding_value_y (int or float, optional): Value to pad the label if it's smaller than original. Default is 0.

    Returns:
        tuple: A tuple containing:
            - X_resized (np.ndarray): The resized (and possibly padded) image.
            - y_resized (np.ndarray): The resized (and possibly padded) label.
    """
    if X.shape != y.shape:
        raise ValueError("X and y must have the same dimentions")
    
    # Determine the number of images and original dimensions
    n = X.shape[0] if X.ndim == 3 else 1
    h, w = X.shape[-2:] if X.ndim == 3 else X.shape

    X_resized = np.empty((n, h, w) if n > 1 else (h, w))
    y_resized = np.empty_like(X_resized)

    for i in range(n):
        # Generate a random scale factor
        scale_factor = np.random.uniform(min_scale, max_scale)

        # Calculate new dimensions
        new_h = int(h * scale_factor)
        new_w = int(w * scale_factor)

        # Resize the image
        if n > 1:
            X_resized_single = cv2.resize(X[i], (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            y_resized_single = cv2.resize(y[i], (new_w, new_h), interpolation=cv2.INTER_NEAREST)
        else:
            X_resized_single = cv2.resize(X, (new_w, new_h), interpolation=cv2.INTER_LINEAR)
            y_resized_single = cv2.resize(y, (new_w, new_h), interpolation=cv2.INTER_NEAREST)

        # Create padded versions to ensure centering
        if X_resized_single.shape[0] > h or X_resized_single.shape[1] > w:
            # If resized image is larger, crop it to original size
            x_start = (X_resized_single.shape[1] - w) // 2
            y_start = (X_resized_single.shape[0] - h) // 2

            X_resized_single = X_resized_single[y_start:y_start + h, x_start:x_start + w]
            y_resized_single = y_resized_single[y_start:y_start + h, x_start:y_start + w]
        else:
            # If resized image is smaller, pad it
            pad_height = h - X_resized_single.shape[0]
            pad_width = w - X_resized_single.shape[1]
            
            # Calculate padding to be applied symmetrically
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            X_resized_single = np.pad(X_resized_single, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=padding_value_x)
            y_resized_single = np.pad(y_resized_single, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=padding_value_y)

        # Store the resized results
        if n > 1:
            X_resized[i] = X_resized_single
            y_resized[i] = y_resized_single
        else:
            X_resized = X_resized_single
            y_resized = y_resized_single

    return X_resized, y_resized

def rotate_and_pad(X, y, min_angle=-30, max_angle=30, padding_value_x=0, padding_value_y=0):
    """
    Rotate the input image and label by a random angle, and pad them if they are smaller than the original size.

    Args:
        X (np.ndarray): Input image array of shape (h, w) or (n, h, w).
        y (np.ndarray): Input label array of the same shape as X.
        min_angle (float): Minimum angle in degrees to rotate the image and label.
        max_angle (float): Maximum angle in degrees to rotate the image and label.
        padding_value_x (int or float, optional): Value to pad the image if it's smaller than original. Default is 0.
        padding_value_y (int or float, optional): Value to pad the label if it's smaller than original. Default is 0.

    Returns:
        tuple: A tuple containing:
            - X_rotated (np.ndarray): The rotated and padded image.
            - y_rotated (np.ndarray): The rotated and padded label.
    """
    if X.shape != y.shape:
        raise ValueError("X and y must have the same dimentions")
    
    # Determine the number of images and original dimensions
    n = X.shape[0] if X.ndim == 3 else 1
    h, w = X.shape[-2:] if X.ndim == 3 else X.shape

    X_rotated = np.empty((n, h, w) if n > 1 else (h, w))
    y_rotated = np.empty_like(X_rotated)

    for i in range(n):
        # Calculate the center of the image for rotation
        center = (w // 2, h // 2)

        # Generate a random angle
        angle = np.random.uniform(min_angle, max_angle)

        # Get the rotation matrix
        M = cv2.getRotationMatrix2D(center, angle, 1.0)

        # Rotate the image and label with appropriate interpolation
        if n > 1:
            X_rotated_i = cv2.warpAffine(X[i], M, (w, h), flags=cv2.INTER_LINEAR)
            y_rotated_i = cv2.warpAffine(y[i], M, (w, h), flags=cv2.INTER_NEAREST)
        else:
            X_rotated_i = cv2.warpAffine(X, M, (w, h), flags=cv2.INTER_LINEAR)
            y_rotated_i = cv2.warpAffine(y, M, (w, h), flags=cv2.INTER_NEAREST)

        # Check for padding if needed
        if X_rotated_i.shape[0] < h or X_rotated_i.shape[1] < w:
            pad_height = h - X_rotated_i.shape[0]
            pad_width = w - X_rotated_i.shape[1]
            
            # Calculate padding to be applied symmetrically
            pad_top = pad_height // 2
            pad_bottom = pad_height - pad_top
            pad_left = pad_width // 2
            pad_right = pad_width - pad_left
            
            X_rotated_i = np.pad(X_rotated_i, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=padding_value_x)
            y_rotated_i = np.pad(y_rotated_i, ((pad_top, pad_bottom), (pad_left, pad_right)), mode='constant', constant_values=padding_value_y)

        # Store the rotated results
        if n > 1:
            X_rotated[i] = X_rotated_i
            y_rotated[i] = y_rotated_i
        else:
            X_rotated = X_rotated_i
            y_rotated = y_rotated_i

    return X_rotated, y_rotated

#---------------------------------------------------------------------------------------------------

def random_transformation(X,y,gaussian_alpha,salt_pepper_prob,salt_value,pepper_value,cutout_min,cutout_max,resize_min,resize_max,rotation_min,rotation_max):
    """
    Apply a series of random transformations to the input images and labels.

    The transformations include vertical and horizontal reflections, adding Gaussian noise,
    applying salt and pepper noise, cutout augmentation, resizing with padding, and rotation 
    with padding. Each transformation has a probability of being applied.

    Args:
        X (np.ndarray): Input image array of shape (h, w) or (n, h, w).
        y (np.ndarray): Input label array of the same shape as X.
        gaussian_alpha (float): Standard deviation of the Gaussian noise to be added.
        salt_pepper_prob (float): Probability of applying salt and pepper noise.
        salt_value (int or float): Value to use for salt noise.
        pepper_value (int or float): Value to use for pepper noise.
        cutout_min (float): Minimum size of the cutout to be applied.
        cutout_max (float): Maximum size of the cutout to be applied.
        resize_min (float): Minimum resizing factor.
        resize_max (float): Maximum resizing factor.
        rotation_min (int or float): Minimum rotation angle.
        rotation_max (int or float): Maximum rotation angle.

    Returns:
        tuple: A tuple containing:
            - X (np.ndarray): Transformed input images.
            - y (np.ndarray): Corresponding labels, unchanged.
    """
    if X.shape != y.shape:
        raise ValueError("X and y must have the same dimentions")

    if X.ndim == 2:  # Case for (h, w)
        X = np.expand_dims(X, axis=0)  # Add a new dimension for consistency
        y = np.expand_dims(y, axis=0)  # Add a new dimension for consistency

    n = X.shape[0] if X.ndim == 3 else 1

    # List of transformations and their corresponding probabilities
    transformations = [
        (lambda X, y: vertical_reflection(X, y), 0.25),
        (lambda X, y: horizontal_reflection(X, y), 0.25),
        (lambda X, y: gaussian_noise(X, y, gaussian_alpha), 0.25),
        (lambda X, y: salt_and_pepper_noise(X, y, salt_pepper_prob, salt_pepper_prob, salt_value, pepper_value), 0.25),
        (lambda X, y: cutout(X, y, cutout_min, cutout_max), 0.25),
        (lambda X, y: resize_and_pad(X, y, resize_min, resize_max), 0),
        (lambda X, y: rotate_and_pad(X, y, rotation_min, rotation_max), 0.25),
    ]

    X_transformed = X.copy()
    y_transformed = y.copy()

    for i in range(n):
        # Generate a random number for each transformation
        rand_vals = np.random.random(len(transformations))
        
        for (transformation, threshold), rand_val in zip(transformations, rand_vals):
            if rand_val < threshold:
                X_transformed[i], y_transformed[i] = transformation(X[i], y[i])

    return X_transformed.squeeze(), y_transformed.squeeze()
     
#---------------------------------------------------------------------------------------------------