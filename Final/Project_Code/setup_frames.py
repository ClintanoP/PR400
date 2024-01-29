import tqdm
import random
import pathlib
import itertools
import collections
import zipfile

import os
import cv2
import numpy as np
import remotezip as rz

import tensorflow as tf

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Some modules to display an animation using imageio.
import imageio
from IPython import display
from urllib import request

URL = '/Users/ck/Documents/Year 4/PR400/Recording_Dataset_CK.zip'

def list_files_from_zip_url(zip_url):
    files = []
    with rz.RemoteZip(zip_url) as zip:
        for zip_info in zip.infolist():
            files.append(zip_info.filename)
    return files

def list_files_from_zip_local(zip_path):
    file_names = []
    with zipfile.ZipFile(zip_path, 'r') as zip:
        for file_info in zip.infolist():
            # Extract only the file name from the full path
            file_name = os.path.basename(file_info.filename)
            file_names.append(file_name)
    return file_names


def frames_from_video_file(video_path, n_frames, output_size = (224, 224), frame_step = 15):
    """
        Creates frames from each video file present for each category.

        Args: 
            video_path: File path to the video.
            n_frames: Number of frames to be created per video file.
            output_size: Pixel size of the output frame image.
        
        Return: 
            An NumPy array of frames in the shape of (n_frames, height, width, channels).

    """
    # Read each video frame by frame
    result = []
    src = cv2.VideoCapture(str(video_path))

    video_length = src.get(cv2.CAP_PROP_FRAME_COUNT)

    need_length = 1 + (n_frames - 1) * frame_step

def get_class(fname):
    return fname.split('_')[0]

def get_files_per_class(files):
    files_for_class = collections.defaultdict(list)
    for fname in files:
        class_name = get_class(fname)
        files_for_class[class_name].append(fname)
    return files_for_class

def select_subset_of_classes(files_for_class, classes, files_per_class):
    files_subset = dict()

    for class_name in classes:
        class_files = files_for_class[class_name]
        files_subset[class_name] = class_files[:files_per_class]
    return files_subset




def main():
    # files = list_files_from_zip_url(URL)
    files = list_files_from_zip_local(URL)
    files = [f for f in files if f.endswith('.mov')]

    print(files)
    # NUM_CLASSES = 5
    # FILES_PER_CLASS = 10

    # files_for_class = get_files_per_class(files)
    # classes = list(files_for_class.keys())

    # print('Num classes:', len(classes))
    # print('Num videos for class[0]:', len(files_for_class[classes[0]]))

    # files_subset = select_subset_of_classes(files_for_class, classes[:NUM_CLASSES], FILES_PER_CLASS)
    # list(files_subset.keys())



main()