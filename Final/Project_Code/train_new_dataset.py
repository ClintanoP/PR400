import json
import math
import os
import numpy as np
import glob


'''
    Steps
    1. Read all json paths
    2. Read all json files 
    2.1. 
    2.2. While reading, if 0 in any place, do not add.
    3. 


'''

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

# refactor the json from the dataset to the existing, hand co-ordinate only json format, used in the local training.
def refactor_and_save_new_json(file_path, label, index, output_dir):
    # where the json will be stored before locally saving.
    all_json_data = {"Left": [], "Right": []}

    json_data = read_json_file(file_path)

    
    for frame in json_data['frames']:
        left = hand_to_json(frame['skeleton']['hand_left'])
        right = hand_to_json(frame['skeleton']['hand_right'])

        if check_for_zero_coords_in_frame(left, right):
            print(f"Skipping frame with zero coordinates in {file_path}")
            return 0
        else:
            all_json_data['Left'].append(hand_to_json(frame['skeleton']['hand_left']))
            all_json_data['Right'].append(hand_to_json(frame['skeleton']['hand_right']))
            filename = f"{label}_{index}.json"
            filepath = os.path.join(output_dir, filename)

            try:
                with open(filepath, "w") as json_file:
                    json.dump(all_json_data, json_file)
                    print(f'Saved as {filepath}')
                    return 1
            except IOError as e:
                print(f"Failed to save {filepath}: {e}")

# this will check for any zeros, ie. bad / no data, and will remove from the dataset.
def check_for_zero_coords_in_frame(left, right):
    is_zero = False
    for hand in [left, right]:
        for finger in ['thumb', 'index', 'middle', 'ring', 'pinky']:
            for bone in hand[finger]:
                if hand[finger][bone][0] == 0.0 or hand[finger][bone][1] == 0.0 or hand[finger][bone][2] == 0:
                    is_zero = True
        for extra in ['palm', 'wrist']:
            if hand[extra][0] == 0.0 or hand[extra][1] == 0.0 or hand[extra][2] == 0.0:
                is_zero = True

    return is_zero

# this will put the hand into json format, same as which is used in the local training.
def hand_to_json(hand):
    # need to pass it the index of each joint/bone. ie. thumb base = 1. 
    hand_data = {   "thumb": finger_to_dict(hand, 4, 3, 2, 1),
                    "index": finger_to_dict(hand, 8, 7, 6, 5),
                    "middle": finger_to_dict(hand, 12, 11, 10, 9),
                    "ring": finger_to_dict(hand, 16, 15, 14, 13),
                    "pinky": finger_to_dict(hand, 20, 19, 18, 17), 
                    "palm": generate_palm_coord((hand['x'][9], hand['y'][9], hand['z'][9]), (hand['x'][0], hand['y'][0], hand['z'][0])),
                    "wrist": [hand['x'][0], hand['y'][0], hand['z'][0]]}
    return hand_data

# helper function for hand_to_json, to put the finger into json format.
def finger_to_dict(finger, dist, int, prox, meta):
    finger_data = { "distal": [finger['x'][dist], finger['y'][dist], finger['z'][dist]], 
                    "intermediate": [finger['x'][int], finger['y'][int], finger['z'][int]], 
                    "proximal": [finger['x'][prox], finger['y'][prox], finger['z'][prox]],
                    "metacarpal": [finger['x'][meta], finger['y'][meta], finger['z'][meta]]  }
    return finger_data

# generates a palm coord as the dataset does not have one. Uses the base of the visible middle finger, and the wrist.
def generate_palm_coord(mid_finger_base, wrist_coordinate):
    midpoint = tuple((mf + wrist) / 2 for mf, wrist in zip(mid_finger_base, wrist_coordinate))
    return midpoint

# this will process and save the new dataset in json format
def process_and_save_json_dataset(base_dir, output_dir):
    file_paths_dict = get_json_dataset_paths(base_dir)
    total_failed = 0
    for label, files in file_paths_dict.items():
        for index, file_path in enumerate(files, start=1):
            total_failed += refactor_and_save_new_json(file_path, label, index, output_dir)

    print(f'total_failed = {total_failed}')

# this will get the json dataset paths and labels, by identifying the last '-' which signifies the end of the name/label.
def get_json_dataset_paths(base_dir):
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


def get_file_paths_and_save_filtered_counts(base_dir):

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
    with open('asl_dataset_valid_full.txt', 'w') as f:
        for gesture, files in gesture_file_paths.items():
            if len(files) >= 8:
                # Write the gesture name and the number of occurrences to the file
                f.write(f"{gesture}\n")
    
    return gesture_file_paths


def aggregate_gesture_data(file_path):
    all_gesture_data = []
    for file in file_path:
        gesture_data = normalize_file(file)
        for hand_data in gesture_data:  # Iterate over left and right hand data
            for frame in hand_data:  # Iterate over each frame
                frame_data = []
                for joint, bones in frame.items():  # Iterate over each joint and its bones
                    for bone, coords in bones.items():  # Get coordinates
                        frame_data.extend(coords)  # Add coordinates to frame data
                all_gesture_data.append(frame_data)
            print(hand_data)
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

# scale factor, to account for varying hand sizes
def calculate_scale_factor(palm_coord, reference_coord):
    # Calculate the Euclidean distance between the palm and a reference point
    return math.sqrt(
        (palm_coord[0] - reference_coord[0]) ** 2 +
        (palm_coord[1] - reference_coord[1]) ** 2 +
        (palm_coord[2] - reference_coord[2]) ** 2
    )

# function to process each frame
def process_frames(hand_frames):
    normalized_frames = []
    for frame in hand_frames:
        palm_coord = frame['palm']
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

# same as above, but instead of reading in a json file it reads in an object
def normalize_data(data):
    normalized_left_hand_data = process_frames(data['Left'])
    normalized_right_hand_data = process_frames(data['Right'])
    return [normalized_left_hand_data, normalized_right_hand_data]

# run the program
# process_and_save_json_dataset('/Users/ck/Desktop/asl-dataset/normalized/3d/', '/Users/ck/Desktop/asl-dataset/ck_normalised/')
# get_file_paths_and_save_filtered_counts('/Users/ck/Desktop/asl-dataset/ck_normalised/')
# get_file_paths_and_save_filtered_counts('Final/JSON_Data/120fps/')






