'''
This file has various functions which are used to normalize the raw data gathered from the LeapMotion camera.

As a general flow of how these functions could be used, I have an example below. 

1. Call aggregate_gesture_data(some_file_path) - This can be one file path, or multiple depending on if what you input.
2. For each file path, the data will go into normalize_file(file_path)
3. normalize_file() will then take each hand and will process seperately with process_frames()
4. process_frames() will extract the co-ordinates from the file path, and will get the 'scale_factor'
5. calculate_scale_factor() will calculate the euclidean distance between the palm and a reference point.
   The chosen reference point is the middle finger tip. 
6. The calculated scale factor is then passed with palm frame data and the palm coordinates into normalize_coordinates()
7. These normalized frames are then returned, appended to all_gesture_data, which once finished will return the aggregated normalized frames.
'''


import json
import math
import os
import numpy as np
import glob

# Normalising the data
# Will get co-ordinate location of palm, which will be the reference point for the rest of the hand.
def normalize_coordinates(hand_joints, palm_coord, scale_factor):
    normalized_joints = {}
    for joint_name, joint_coords in hand_joints.items():
        if joint_name not in ['palm', 'wrist']:  # Exclude 'palm' and 'wrist'
            normalized_joints[joint_name] = {}
            for bone_name, coords in joint_coords.items():
                normalized_coords = [
                    (coords[0] - palm_coord[0]) / scale_factor,
                    (coords[1] - palm_coord[1]) / scale_factor,
                    (coords[2] - palm_coord[2]) / scale_factor
                ]
                normalized_joints[joint_name][bone_name] = normalized_coords
    return normalized_joints

# function to read in json file and return as obj
def read_json_file(file_path):
    try:
        with open(file_path, 'r') as json_file:
            data = json.load(json_file)
        return data
    except FileNotFoundError:
        print(f"File not found: {file_path}")
        return None
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON: {e}")
        return None

# scale factor, to account for varying hand sizes
def calculate_scale_factor(palm_coord, reference_coord):
    # Calculate the Euclidean distance between the palm and a reference point
    return math.sqrt(
        (palm_coord[0] - reference_coord[0]) ** 2 +
        (palm_coord[1] - reference_coord[1]) ** 2 +
        (palm_coord[2] - reference_coord[2]) ** 2
    )

def generate_palm_coord(mid_finger_base, wrist_coordinate):
    midpoint = tuple((mf + wrist) / 2 for mf, wrist in zip(mid_finger_base, wrist_coordinate))
    return midpoint

# function to process each frame
def process_frames(hand_frames):
    normalized_frames = []
    for frame in hand_frames:
        palm_coord = frame['palm']
        if frame['palm'] == [0, 0, 0]:  # Skip frames with zero coordinates
            palm_coord = generate_palm_coord(frame['middle']['metacarpal'], frame['wrist'])
        # Assuming the wrist coordinate is available and named 'wrist'
        wrist_coord = frame.get('wrist', palm_coord)  # Default to palm if wrist not available
        # Reference point for scale factor (e.g., tip of the middle finger distal bone)
        reference_coord = frame['middle']['distal']  
        scale_factor = calculate_scale_factor(wrist_coord, reference_coord)

        normalized_frame = normalize_coordinates(frame, palm_coord, scale_factor)
        normalized_frames.append(normalized_frame)
    return normalized_frames

# returns normalized L and R hand data from given file path
def normalize_file(file):
    json_data = read_json_file(file)
    normalized_left_hand_data = process_frames(json_data['Left'])
    normalized_right_hand_data = process_frames(json_data['Right'])
    return [normalized_left_hand_data, normalized_right_hand_data]

# normalizes data, not from a file
def normalize_data(data):
    normalized_left_hand_data = process_frames(data['Left'])
    normalized_right_hand_data = process_frames(data['Right'])
    return [normalized_left_hand_data, normalized_right_hand_data]

# aggregates the data into one big array - for files
def aggregate_gesture_data(file_paths):
    all_gesture_data = []
    for file_path in file_paths:
        gesture_data = normalize_file(file_path)
        file_has_zero = False  # Flag to track if any coordinate is 0

        # Temporarily store data for the current file
        temp_file_data = []

        for hand_data in gesture_data:  # Iterate over left and right hand data
            for frame in hand_data:  # Iterate over each frame
                frame_data = []
                for joint, bones in frame.items():  # Iterate over each joint and its bones
                    for bone, coords in bones.items():  # Get coordinates
                        if 0.0 in coords:  # Check for zeros
                            file_has_zero = True
                            break  # Stop processing this file
                        frame_data.extend(coords)  # Add coordinates to frame data
                    if file_has_zero:
                        break
                if not file_has_zero:
                    temp_file_data.append(frame_data)
            if file_has_zero:
                break

        if not file_has_zero:
            # If no zeros were found, add the file's data to the aggregated data
            all_gesture_data.extend(temp_file_data)

    return np.array(all_gesture_data)


def aggregate_gesture_data_non_file(hands_data):
    all_gesture_data = []
    gesture_data = normalize_data(hands_data)
    for hand_data in gesture_data:  # Iterate over left and right hand data
        for frame in hand_data:  # Iterate over each frame
            frame_data = []
            for joint, bones in frame.items():  # Iterate over each joint and its bones
                for bone, coords in bones.items():  # Get coordinates
                    frame_data.extend(coords)  # Add coordinates to frame data
            all_gesture_data.append(frame_data)
        print(hand_data)
    return np.array(all_gesture_data)


# FOR NEW DATA

def check_hand_coordinates(frame):
    is_zero = True
    for hand_type in ['hand_left', 'hand_right']:
        for dimension in ['x', 'y', 'z']:
            if frame['skeleton'][hand_type][dimension][0] > 0 or frame['skeleton'][hand_type][dimension][0] < 0:
                is_zero = False
            else:
                is_zero = True
                break
    return is_zero

def get_dataset_json(file_path, label, index, output_dir):
    all_json_data = {"Left": [], "Right": []}

    json_data = read_json_file(file_path)
    
    for frame in json_data['frames']:
        if check_hand_coordinates(frame):
            return
        all_json_data['Left'].append(hand_to_json(frame['skeleton']['hand_left']))
        all_json_data['Right'].append(hand_to_json(frame['skeleton']['hand_right']))

    filename = f"{label}_{index}.json"
    filepath = os.path.join(output_dir, filename)

    try:
        with open(filepath, "w") as json_file:
            json.dump(all_json_data, json_file)
            print(f'Saved as {filepath}')
    except IOError as e:
        print(f"Failed to save {filepath}: {e}")




def hand_to_json(hand):
    # need to pass it the index of each joint/bone. ie. thumb base = 1. 
    hand_data = {   "thumb": finger_to_dict(hand, 4, 3, 2, 1),
                    "index": finger_to_dict(hand, 8, 7, 6, 5),
                    "middle": finger_to_dict(hand, 12, 11, 10, 9),
                    "ring": finger_to_dict(hand, 16, 15, 14, 13),
                    "pinky": finger_to_dict(hand, 20, 19, 18, 17), 
                    "palm": [0,0,0],
                    "wrist": [hand['x'][0], hand['y'][0], hand['z'][0]]}
    return hand_data

# converting the fingers to a returned dictionary
def finger_to_dict(finger, dist, int, prox, meta):
    finger_data = { "distal": [finger['x'][dist], finger['y'][dist], finger['z'][dist]], 
                    "intermediate": [finger['x'][int], finger['y'][int], finger['z'][int]], 
                    "proximal": [finger['x'][prox], finger['y'][prox], finger['z'][prox]],
                    "metacarpal": [finger['x'][meta], finger['y'][meta], finger['z'][meta]]  }
    return finger_data

#get_dataset_json(f'C:/Users/Ck/Desktop/asl-skeleton3d/normalized/3d/accident-590901.json')

file_path = (f'/Users/ck/Desktop/asl-dataset/ck_normalised/')

def get_file_paths_and_save_filtered_counts1(base_dir):
    # Dictionary to hold the file paths for each gesture (label)
    gesture_file_paths = {}

    # List all json files in the directory
    json_files = glob.glob(os.path.join(base_dir, '*.json'))

    # Sort files into their respective lists based on the label extracted from the file name
    for file_name in json_files:
        # Extract the label (gesture name) from the file name, assuming it's before the last dash
        gesture_name = os.path.basename(file_name).rsplit('-', 1)[0]

        # Add the file path to the list associated with the label in the dictionary
        if gesture_name in gesture_file_paths:
            gesture_file_paths[gesture_name].append(file_name)
        else:
            gesture_file_paths[gesture_name] = [file_name]

    # Ensure the file paths for each label are sorted
    for gesture in gesture_file_paths:
        gesture_file_paths[gesture].sort()

    return gesture_file_paths

def get_file_paths_and_save_filtered_counts():
    # Base directory where the JSON files are located
    # base_dir = 'C:/Users/Ck/Desktop/asl-skeleton3d/ck_normalised/'
    base_dir = 'Final/JSON_Data/120fps/'
    # Dictionary to hold the file paths for each gesture
    gesture_file_paths = {}

    # List all json files in the directory
    json_files = glob.glob(os.path.join(base_dir, '*.json'))

    # Sort files into their respective lists
    for file_name in json_files:
        # Extract the gesture name from the file name by splitting on the last dash
        gesture_name = os.path.basename(file_name).rsplit('_', 1)[0]

        # Add the file path to the corresponding gesture list in the dictionary
        if gesture_name in gesture_file_paths:
            gesture_file_paths[gesture_name].append(file_name)
        else:
            gesture_file_paths[gesture_name] = [file_name]

    # Sort the lists so that the files are in the correct order
    for gesture in gesture_file_paths:
        gesture_file_paths[gesture].sort()

    # Save the counts to a text file, only for gestures with x or more occurrences
    with open('ck_mixed_dataset_gesture_filtered.txt', 'w') as f:
        for gesture, files in gesture_file_paths.items():
            # if len(files) >= 10:
                # Write the gesture name and the number of occurrences to the file
            f.write(f"{gesture}\n")
    
    return gesture_file_paths


def process_datasets(base_dir, output_dir):
    file_paths_dict = get_file_paths_and_save_filtered_counts1(base_dir)
    
    for label, files in file_paths_dict.items():
        for index, file_path in enumerate(files, start=1):
            get_dataset_json(file_path, label, index, output_dir)

# process_datasets('/Users/ck/Desktop/asl-dataset/normalized/3d/', '/Users/ck/Desktop/asl-dataset/ck_normalised/')

# get_file_paths_and_save_filtered_counts()