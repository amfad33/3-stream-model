import cv2
import augments as aug


# This function will take a loaded image and will resize and crop it
def resize_crop(image, target_width=224, target_height=224):
    # Get the size of the image
    h, w, _ = image.shape

    # Calculate scaling factor
    scale_factor = max(target_width / w, target_height / h)

    # Resize the frame while maintaining aspect ratio
    new_width = int(w * scale_factor)
    new_height = int(h * scale_factor)
    image = cv2.resize(image, (new_width, new_height))

    # Get the new size of the central frame
    h, w, _ = image.shape

    # Resize the frame by cropping or padding
    if h > target_height or w > target_width:
        # Crop the central region
        crop_y = max((h - target_height) // 2, 0)
        crop_x = max((w - target_width) // 2, 0)
        image = image[crop_y:crop_y + target_height, crop_x:crop_x + target_width]
    else:
        # Pad the image to the target size
        pad_y = max((target_height - h) // 2, 0)
        pad_x = max((target_width - w) // 2, 0)
        image = cv2.copyMakeBorder(image, pad_y, pad_y, pad_x, pad_x, cv2.BORDER_CONSTANT, value=[0, 0, 0])

    # Resize to target dimensions
    image = cv2.resize(image, (target_width, target_height))
    return image


# This function will create a video of 30 frames from image
def video_from_image(image, output_path, target_width=224, target_height=224, with_aug=False, aug_type=1):
    # fourcc code for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 format

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, 3, (target_width, target_height))

    for i in range(30):
        if with_aug:
            if aug_type == 1:
                out.write(aug.type1_aug(image))
            elif aug_type == 2:
                out.write(aug.type2_aug(image))
        else:
            out.write(image)

    out.release()
    print("Processed video saved as:", output_path)
