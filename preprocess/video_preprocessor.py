import numpy as np
import pandas as pd
import cv2
import video_tools as vt
import audio_sampler as asa
import random
from pathlib import Path


# This function will load videos from a csv file and prepare their image and 30 frame video snippet
# parameters:
# csv_file_path: address to the initial csv containing information about dataset, if none exists create one.
# id_column_name: the id value (must be unique for dataset files) used for generated file names
# video_column_name: the file name of video without mp4
# start_time_column_name: The column with info of start time of section of the video
# base_video_path: base os path to the video file
# save_image_path: where are we saving the image data?
# save_processed_video_path: where are we saving the processed video?
# save_audio_path: where are we saving the processed and sampled audio
# check_column:  is there a column to check that says if data exists? (youtube videos might not exist)
# target_width: the target image/video width
# target_height: the target image/video height
# duration: the duration of the segmented video in seconds
# audio_sr: the sample rate of audios
# with_aug: do you want this generated dataset to contain augmentations?
# aug_type: type of augmentation. check the augments.py for both defined types
def load_videos(csv_file_path, id_column_name, video_column_name, start_time_column_name, base_video_path,
                save_image_path, save_processed_video_path, save_audio_path, check_column=None, target_width=224,
                target_height=224, duration=10, audio_sr=2048, with_aug=False, aug_type=1):
    # Load the CSV file
    df = pd.read_csv(csv_file_path)
    audio_df = df

    # Create directories if they don't exist
    Path(save_image_path).mkdir(parents=True, exist_ok=True)
    Path(save_processed_video_path).mkdir(parents=True, exist_ok=True)
    Path(save_audio_path).mkdir(parents=True, exist_ok=True)

    # Check if the columns exists
    if id_column_name not in df.columns:
        raise ValueError(f"Column '{id_column_name}' does not exist in the CSV file.")
    if video_column_name not in df.columns:
        raise ValueError(f"Column '{video_column_name}' does not exist in the CSV file.")
    if start_time_column_name not in df.columns:
        raise ValueError(f"Column '{start_time_column_name}' does not exist in the CSV file.")
    if check_column is not None:
        if check_column not in df.columns:
            raise ValueError(f"Column '{check_column}' does not exist in the CSV file.")

    # Iterate through each video file in the specified column
    for index, row in df.iterrows():
        video_filename = df.at[index, video_column_name]
        video_path = base_video_path + str(video_filename) + '.mp4'

        if video_filename == "":   # file doesn't exist check in General
            continue

        if check_column is not None:
            if not df.at[index, check_column]:   # file doesnt exist check in HVU CSV
                continue

        # Load the video file
        cap = cv2.VideoCapture(video_path)

        if not cap.isOpened():
            print(f"Error: Could not open video file {video_path}")
            continue

        # When preparing video dataset we use same augment for entire video data and the image:
        aug_probability = random.random()

        # Process the video
        vt.save_central_frame(cap, save_image_path + str(df.at[index, id_column_name]),
                              df.at[index, start_time_column_name], duration, target_width, target_height, with_aug,
                              aug_type, aug_probability)
        vt.process_video(cap, save_processed_video_path + str(df.at[index, id_column_name]) + ".mp4",
                         df.at[index, start_time_column_name], duration, 3 * duration, target_width, target_height,
                         with_aug, aug_type, aug_probability)

        # Process the audio
        audios = asa.process_video_audio(video_path, int(df.at[index, start_time_column_name]),
                                         int(df.at[index, start_time_column_name]) + duration, audio_sr)
        np.save(save_audio_path + str(df.at[index, id_column_name]) + '.npy', audios)

        # Release the video capture object
        cap.release()

        # Close any OpenCV windows
        cv2.destroyAllWindows()


# To run this function fix the following paths before executing this file
load_videos('H:\\HVU CSV\\HVU_Train_part2.csv', "id","title",
            "time_start", 'H:\\HVU\\part 2\\', 'G:\\outputs\\images\\part 2\\',
            'G:\\outputs\\videos\\part 2\\','G:\\outputs\\audios\\part 2\\', check_column="file",
            with_aug=False, aug_type=1)
