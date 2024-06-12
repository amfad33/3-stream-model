import cv2
import numpy as np
import random


# Type 1 augments
def type1_aug(image, probability=None):
    # Generate a random float between 0 and 1 to determine which function to execute
    if probability is None:
        probability = random.random()

    # Execute functions based on the probability
    if probability < 0.15:
        image = reduce_brightness(image)
    elif probability < 0.30:
        image = rotate_clockwise(image)
    elif probability < 0.45:
        image = upscale(image)
    # else do nothing with image

    return image


# Type 2 augments
def type2_aug(image, probability=None):
    # Generate a random float between 0 and 1 to determine which function to execute
    if probability is None:
        probability = random.random()

    # Execute functions based on the probability
    if probability < 0.1:
        image = make_yellow(image)
    elif probability < 0.2:
        image = make_red(image)
    elif probability < 0.3:
        image = reduce_brightness(image)
    elif probability < 0.4:
        image = rotate_counter_clockwise(image)
    # else do nothing with image

    return image


# This function will take a loaded image and will reduce brightness to half
def reduce_brightness(image):
    # Convert the image to a floating point data type
    image = image.astype(np.float32)

    # Reduce brightness by half
    image = image * 0.5

    # Clip the values to be in the valid range [0, 255] and convert back to uint8
    image = np.clip(image, 0, 255).astype(np.uint8)

    return image


# This function will take a loaded image and will rotate it 45 degrees clockwise
def rotate_clockwise(image):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Create a rotation matrix for 45 degrees clockwise
    M = cv2.getRotationMatrix2D(center, -45, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated


# This function will take a loaded image and will rotate it 45 degrees counter-clockwise
def rotate_counter_clockwise(image):
    # Get the dimensions of the image
    (h, w) = image.shape[:2]

    # Calculate the center of the image
    center = (w // 2, h // 2)

    # Create a rotation matrix for 45 degrees clockwise
    M = cv2.getRotationMatrix2D(center, 45, 1.0)

    # Perform the rotation
    rotated = cv2.warpAffine(image, M, (w, h), borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0))

    return rotated


# This function will upscale the image to twice its size then crop and takes the center
def upscale(image):
    # Get the original dimensions of the image
    original_height, original_width = image.shape[:2]

    # Calculate the new dimensions (twice the original dimensions)
    new_width = original_width * 2
    new_height = original_height * 2

    # Resize the image to the new dimensions
    upscaled_image = cv2.resize(image, (new_width, new_height), interpolation=cv2.INTER_LINEAR)

    # Calculate the coordinates for the central crop
    x_center = new_width // 2
    y_center = new_height // 2

    x_start = x_center - (original_width // 2)
    y_start = y_center - (original_height // 2)

    x_end = x_start + original_width
    y_end = y_start + original_height

    # Crop the central area
    cropped_image = upscaled_image[y_start:y_end, x_start:x_end]

    return cropped_image


# This function will take a loaded image and will add a yellow filter over it
def make_yellow(image):
    # Create a yellow image with the same dimensions as the input image
    yellow = np.full_like(image, (0, 255, 255), dtype=np.uint8)

    # Blend the original image with the yellow image
    # You can adjust the alpha value to change the intensity of the yellow filter
    alpha = 0.5  # Transparency factor for the yellow filter
    filtered_image = cv2.addWeighted(image, 1 - alpha, yellow, alpha, 0)

    return filtered_image


# This function will take a loaded image and will add a red filter over it
def make_red(image):
    # Create a red image with the same dimensions as the input image
    red = np.full_like(image, (0, 0, 255), dtype=np.uint8)

    # Blend the original image with the red image
    # You can adjust the alpha value to change the intensity of the red filter
    alpha = 0.5  # Transparency factor for the red filter
    filtered_image = cv2.addWeighted(image, 1 - alpha, red, alpha, 0)

    return filtered_image
