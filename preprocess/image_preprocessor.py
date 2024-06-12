import pandas as pd
import cv2
import image_tools as it
import augments as aug
from pathlib import Path


# This function will load image from a csv file and will prepare it
# parameters:
# csv_file_path: address to the initial csv containing information about dataset, if none exists create one.
# id_column_name: the id value (must be unique for dataset files) used for generated file names
# image_column_name: the file name of image without jpg
# base_image_path: base os path to the image file
# save_image_path: where are we saving the image data?
# save_video_path: where are we saving the generated videos?
# target_width: the target image/video width
# target_height: the target image/video height
# with_aug: do you want this generated dataset to contain augmentations?
# aug_type: type of augmentation. check the augments.py for both defined types
def load_videos(csv_file_path, id_column_name, image_column_name, base_image_path, save_image_path, save_video_path,
                target_width=224, target_height=224, with_aug=False, aug_type=1):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)

    # create save directories
    Path(save_image_path).mkdir(parents=True, exist_ok=True)
    Path(save_video_path).mkdir(parents=True, exist_ok=True)

    # Check if the columns exists
    if id_column_name not in df.columns:
        raise ValueError(f"Column '{id_column_name}' does not exist in the CSV file.")
    if image_column_name not in df.columns:
        raise ValueError(f"Column '{image_column_name}' does not exist in the CSV file.")

    # Iterate through each image file in the specified column
    for index, row in df.iterrows():
        image_filename = df.at[index, image_column_name]
        image_path = base_image_path + str(image_filename) + '.jpg'

        if image_filename == "":   # file doesn't exist check in General
            continue

        # Load the image file
        image = cv2.imread(image_path)
        if image is None:
            continue

        # Process the image
        image = it.resize_crop(image, target_width, target_height)

        # save processed image
        save_path = save_image_path + str(df.at[index, id_column_name]) + ".jpg"
        if with_aug:
            if aug_type == 1:
                cv2.imwrite(save_path, aug.type1_aug(image))
            elif aug_type == 2:
                cv2.imwrite(save_path, aug.type2_aug(image))
        else:
            cv2.imwrite(save_path, image)
        print(f'Image saved as: {save_path}')

        # create video from image and save it
        save_path_video = save_video_path + str(df.at[index, id_column_name]) + ".mp4"
        it.video_from_image(image, save_path_video, target_width, target_height, with_aug, aug_type)

        # Close any OpenCV windows
        cv2.destroyAllWindows()


# To run this function fix the following paths before executing this file
load_videos('H:\\HVU CSV\\HVU_Train_part2.csv', "id","id",
            'G:\\input\\images\\part 2\\', 'G:\\output\\images\\part 2\\',
            'G:\\output\\videos\\part 2\\',224, 224, True, 2)
