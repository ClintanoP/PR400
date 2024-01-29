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
import numpy as np

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
        for hand_data in gesture_data:  # Iterate over left and right hand data
            for frame in hand_data:  # Iterate over each frame
                frame_data = []
                for joint, bones in frame.items():  # Iterate over each joint and its bones
                    for bone, coords in bones.items():  # Get coordinates
                        frame_data.extend(coords)  # Add coordinates to frame data
                all_gesture_data.append(frame_data)
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
    return np.array(all_gesture_data)
