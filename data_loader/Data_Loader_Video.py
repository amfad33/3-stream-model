import os
import csv
import torch
from torch.utils.data import Dataset
from PIL import Image
import cv2
import numpy as np


def _load_image(image_path):
    image = Image.open(image_path).convert('RGB')
    image = torch.tensor(np.array(image)).permute(2, 0, 1).float() / 255.0
    return image


def _load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    frames = []
    for _ in range(30):
        ret, frame = cap.read()
        if ret:
            frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            frames.append(frame)
        else:
            frames.append(np.zeros((224, 224, 3), dtype=np.uint8))
    cap.release()

    frames = np.stack(frames, axis=0)
    video = torch.tensor(frames).permute(3, 0, 1, 2).float() / 255.0
    return video


def _load_audio(audio_path):
    return np.load(audio_path)


def _load_classes(classes_csv):
    classes_dict = {}
    with open(classes_csv, 'r', encoding='utf-8') as file:
        reader = csv.DictReader(file)
        for row in reader:
            classes_dict[row['Tag']] = int(row['ID'])
    return classes_dict


class MultiModalDataset(Dataset):
    def __init__(self, csv_file, classes_csv, images_dir, videos_dir, audios_dir, class_count, parted=False, num_parts=1):
        self.data = []
        self.images_dir = images_dir
        self.videos_dir = videos_dir
        self.audios_dir = audios_dir
        self.csv_file = csv_file
        self.class_count = class_count
        self.parted = parted
        self.num_parts = num_parts

        # Load class tags
        self.classes_dict = _load_classes(classes_csv)

        for i in range(num_parts):
            if parted:
                csv_f = self.csv_file + f'_part{i + 1}' + '.csv'
            else:
                csv_f = self.csv_file
            with open(csv_f, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['file'].lower() == 'true':
                        self.data.append([row['id'], row['Tags']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        id = int(row[0])
        part = id // 1000

        # Load image
        image_path = os.path.join(self.images_dir, f'part {part + 1}', f'{id}.jpg')
        image = _load_image(image_path)

        # Load video
        video_path = os.path.join(self.videos_dir, f'part {part + 1}', f'{id}.mp4')
        video = _load_video(video_path)

        # Load audio
        audio_path = os.path.join(self.audios_dir, f'part {part + 1}', f'{id}.npy')
        audio = _load_audio(audio_path).astype(np.float32)

        # Process tags
        tags = row[1].split('|')
        tag_vector = np.zeros(self.class_count, dtype=np.float32)
        for tag in tags:
            if tag in self.classes_dict:
                tag_vector[self.classes_dict[tag]] = 1 / len(tags)

        tag_tensor = torch.tensor(tag_vector)

        return image, video, audio, tag_tensor


# This next dataloader only loads the images
class ImageDataset(Dataset):
    def __init__(self, csv_file, classes_csv, images_dir, class_count, parted=False, num_parts=1):
        self.data = []
        self.images_dir = images_dir
        self.csv_file = csv_file
        self.class_count = class_count
        self.parted = parted
        self.num_parts = num_parts

        # Load class tags
        self.classes_dict = _load_classes(classes_csv)

        for i in range(num_parts):
            if parted:
                csv_f = self.csv_file + f'_part{i + 1}' + '.csv'
            else:
                csv_f = self.csv_file
            with open(csv_f, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['file'].lower() == 'true':
                        self.data.append([row['id'], row['Tags']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        id = int(row[0])
        part = id // 1000

        # Load image
        image_path = os.path.join(self.images_dir, f'part {part + 1}', f'{id}.jpg')
        image = _load_image(image_path)

        # Process tags
        tags = row[1].split('|')
        tag_vector = np.zeros(self.class_count, dtype=np.float32)
        for tag in tags:
            if tag in self.classes_dict:
                tag_vector[self.classes_dict[tag]] = 1 / len(tags)

        tag_tensor = torch.tensor(tag_vector)

        return image, tag_tensor


# Only load videos
class VideoDataset(Dataset):
    def __init__(self, csv_file, classes_csv, videos_dir, class_count, parted=False, num_parts=1):
        self.data = []
        self.videos_dir = videos_dir
        self.csv_file = csv_file
        self.class_count = class_count
        self.parted = parted
        self.num_parts = num_parts

        # Load class tags
        self.classes_dict = _load_classes(classes_csv)

        for i in range(num_parts):
            if parted:
                csv_f = self.csv_file + f'_part{i + 1}' + '.csv'
            else:
                csv_f = self.csv_file
            with open(csv_f, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['file'].lower() == 'true':
                        self.data.append([row['id'], row['Tags']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        id = int(row[0])
        part = id // 1000

        # Load video
        video_path = os.path.join(self.videos_dir, f'part {part + 1}', f'{id}.mp4')
        video = _load_video(video_path)

        # Process tags
        tags = row[1].split('|')
        tag_vector = np.zeros(self.class_count, dtype=np.float32)
        for tag in tags:
            if tag in self.classes_dict:
                tag_vector[self.classes_dict[tag]] = 1 / len(tags)

        tag_tensor = torch.tensor(tag_vector)

        return video, tag_tensor


# Only load audios
class AudioDataset(Dataset):
    def __init__(self, csv_file, classes_csv, audios_dir, class_count, parted=False, num_parts=1):
        self.data = []
        self.audios_dir = audios_dir
        self.csv_file = csv_file
        self.class_count = class_count
        self.parted = parted
        self.num_parts = num_parts

        # Load class tags
        self.classes_dict = _load_classes(classes_csv)

        for i in range(num_parts):
            if parted:
                csv_f = self.csv_file + f'_part{i + 1}' + '.csv'
            else:
                csv_f = self.csv_file
            with open(csv_f, 'r', encoding='utf-8') as file:
                reader = csv.DictReader(file)
                for row in reader:
                    if row['file'].lower() == 'true':
                        self.data.append([row['id'], row['Tags']])

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        row = self.data[idx]
        id = int(row[0])
        part = id // 1000

        # Load audio
        audio_path = os.path.join(self.audios_dir, f'part {part + 1}', f'{id}.npy')
        audio = _load_audio(audio_path).astype(np.float32)

        # Process tags
        tags = row[1].split('|')
        tag_vector = np.zeros(self.class_count, dtype=np.float32)
        for tag in tags:
            if tag in self.classes_dict:
                tag_vector[self.classes_dict[tag]] = 1 / len(tags)

        tag_tensor = torch.tensor(tag_vector)

        return audio, tag_tensor