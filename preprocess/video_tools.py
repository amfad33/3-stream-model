import cv2
import numpy as np
import image_tools as it
import augments as aug


# This function will take a loaded video and will extract its central frame (and resizes it)
def save_central_frame(video, save_addr, start_time, duration=10, target_width=224, target_height=224, with_aug=False,
                       aug_type=1, aug_p=99):
    # Get the frames per second (fps) of the video
    fps = int(video.get(cv2.CAP_PROP_FPS))

    # Calculate the start and end frame indices
    start_frame = start_time * fps
    end_frame = start_frame + duration * fps

    # Set the video to the start frame
    video.set(cv2.CAP_PROP_POS_FRAMES, start_frame)

    # Read frames and count them
    frame_count = 0
    frames = []

    while video.isOpened():
        ret, frame = video.read()
        if not ret or frame_count >= (end_frame - start_frame):
            break
        frames.append(frame)
        frame_count += 1

    # Total frames counted
    total_frames = len(frames)
    # print(f'Total frames in the 10-second section: {total_frames}')

    # Calculate the central frame index (rounded down)
    central_frame_index = total_frames // 2

    # Get the central frame
    central_frame = frames[central_frame_index]

    # Resize to target dimensions
    central_frame_resized = it.resize_crop(central_frame,target_width, target_height)

    # Save the central frame as an image
    central_frame_path = save_addr + '.jpg'
    if with_aug:
        if aug_type == 1:
            cv2.imwrite(central_frame_path, aug.type1_aug(central_frame_resized, aug_p))
        elif aug_type == 2:
            cv2.imwrite(central_frame_path, aug.type2_aug(central_frame_resized, aug_p))
    else:
        cv2.imwrite(central_frame_path, central_frame_resized)
    print(f'Central frame saved as: {central_frame_path}')


# This function will take a loaded video and will extract 30 frames from 10 seconds of the video and resizes it
def process_video(cap, output_path, start_time, duration=10, target_frames=30, target_width=224, target_height=224,
                  with_aug=False, aug_type=1, aug_p=99):
    fps = cap.get(cv2.CAP_PROP_FPS)
    vid_dur = cap.get(cv2.CAP_PROP_FRAME_COUNT)
    start_frame = int(start_time * fps)
    end_time = start_time + duration
    end_frame = min(int(end_time * fps), vid_dur)

    # print(f'Total frames in the 10-second section: {total_frames}')

    # Calculate the indices of the frames we need to keep to have `target_frames` uniformly
    indices_to_keep = np.linspace(start_frame, end_frame - 1, target_frames).astype(int)

    # print(indices_to_keep)

    # fourcc code for the output video
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')  # Use 'mp4v' for mp4 format

    # Create a VideoWriter object
    out = cv2.VideoWriter(output_path, fourcc, 3, (target_width, target_height))

    # Go back to start of video
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    # Read and process the video frames
    frame_index = 0
    selected_frames = []

    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break

        if frame_index in indices_to_keep:
            # Resize and crop frame
            frame = it.resize_crop(frame, target_width, target_height)

            if with_aug:
                if aug_type == 1:
                    frame = aug.type1_aug(frame, aug_p)
                elif aug_type == 2:
                    frame = aug.type2_aug(frame, aug_p)

            selected_frames.append(frame)

        frame_index += 1
        if frame_index >= end_frame:
            break

    # Write the selected frames to the output video
    for frame in selected_frames:
        out.write(frame)

    out.release()
    print("Processed video saved as:", output_path)
