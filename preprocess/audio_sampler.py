import numpy as np
from moviepy.editor import VideoFileClip

# This tool will sample the audio data from a section of video. It makes audio frames of 1 second and extract SR amount
# of samples from it


# Function to load audio from a video file
def load_audio_from_video(video, start_time, end_time, sr=2048):
    audio = video.audio
    audio = audio.subclip(start_time, end_time)

    # Extract the audio as a list of samples
    audio_samples = list(audio.iter_frames(fps=sr))

    # Convert the list of samples to a NumPy array
    sound_array = np.array(audio_samples)

    # Ensure the audio is in mono
    if sound_array.ndim > 1:
        sound_array = np.mean(sound_array, axis=1)

    return sound_array


# Main function
def process_video_audio(video_path, start_time, end_time, sr=2048):
    video = VideoFileClip(video_path)
    audios = []
    for i in range(start_time, end_time):
        if i > video.duration:
            audios.append(np.zeros(sr))
        else:
            # Load audio from video
            audio = load_audio_from_video(video, i, i+1, sr)
            # print(audio.shape)
            audios.append(audio)

    audios = np.array(audios)

    return audios


# Example usage
# video_path = 'H:\\Videos\\part 1\\12 year old playing silent night on alto sax.mp4'
# start_time = 10  # start time in seconds
# end_time = 20  # end time in seconds
# audios = process_video_audio(video_path, start_time, end_time)
# print(audios.shape)
