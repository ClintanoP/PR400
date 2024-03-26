import json
import math
import time
import leap
from leap import datatypes as ldt

# global is_recording
recorded_json = {"Left" : [], "Right" : []}
frame_count = 0
cooldown_time = 5  # Cooldown time in seconds
last_record_time = 0  # Timestamp of the last recording stop
buffer_frames = {'Left': [], 'Right': []}
is_recording = False
min_recording_duration = 2  # Minimum duration for recording in seconds
recording_start_time = 0  # Timestamp when recording starts

# converting the key data into json, to then be saved. 
def hand_to_json(hand: ldt):
    hand_data = {   "thumb": finger_to_dict(hand.digits[0]),
                    "index": finger_to_dict(hand.digits[1]),
                    "middle": finger_to_dict(hand.digits[2]),
                    "ring": finger_to_dict(hand.digits[3]),
                    "pinky": finger_to_dict(hand.digits[4]), 
                    "palm": [hand.palm.position.x, hand.palm.position.y, hand.palm.position.z],
                    "wrist": [hand.arm.next_joint.x, hand.arm.next_joint.y, hand.arm.next_joint.z]}
    return hand_data

# converting the fingers to a returned dictionary
def finger_to_dict(finger):
    finger_data = { "distal": [finger.distal.next_joint.x, finger.distal.next_joint.y, finger.distal.next_joint.z], 
                    "intermediate": [finger.intermediate.next_joint.x, finger.intermediate.next_joint.y, finger.intermediate.next_joint.z], 
                    "proximal": [finger.proximal.next_joint.x, finger.proximal.next_joint.y, finger.proximal.next_joint.z],
                    "metacarpal": [finger.metacarpal.next_joint.x, finger.metacarpal.next_joint.y, finger.metacarpal.next_joint.z]  }
    return finger_data

# when both hands make fists, the recording will start, when fists close again, recording will end
def fist_made(hand: ldt):
    return all(not finger.is_extended for finger in hand.digits)

class HandListener(leap.Listener):
    def on_tracking_event(self, event):
        if event.tracking_frame_id % 4 == 0:  # 30fps
            manage_recording_state(event)
            if is_recording:
                for hand in event.hands:
                    hand_type = 'Left' if str(hand.type) == 'HandType.Left' else 'Right'
                    recorded_json[hand_type].append(hand_to_json(hand))

def manage_recording_state(event):
    global is_recording, cooldown_time, last_record_time, recording_start_time
    fist_count = sum(fist_made(hand) for hand in event.hands)

    current_time = time.time()
    if fist_count == 2 and current_time - last_record_time > cooldown_time:
        if not is_recording:
            start_recording()
            recording_start_time = current_time  # Set the recording start time
        elif is_recording and current_time - recording_start_time > min_recording_duration:
            stop_recording()
            last_record_time = current_time

def start_recording():
    global is_recording, recorded_json, buffer_frames
    print('Prepare to start recording...')
    time.sleep(1)  # Wait for 1 second to get into position
    print('Starting recording...')
    is_recording = True
    recorded_json['Left'].extend(buffer_frames['Left'])
    recorded_json['Right'].extend(buffer_frames['Right'])
    buffer_frames = {'Left': [], 'Right': []}  # Reset buffer

def stop_recording():
    global is_recording, recorded_json
    frames_to_remove = 30  # Number of frames to remove at the end

    if len(recorded_json['Left']) > frames_to_remove:
        recorded_json['Left'] = recorded_json['Left'][:-frames_to_remove]
    if len(recorded_json['Right']) > frames_to_remove:
        recorded_json['Right'] = recorded_json['Right'][:-frames_to_remove]

    is_recording = False
    print('Stopped recording, saving file...')
    save_recording()
    recorded_json = {'Left': [], 'Right': []}

def buffer_frame(event):
    global buffer_frames
    buffer_size = 30
    for hand in event.hands:
        hand_type = 'Left' if str(hand.type) == 'HandType.Left' else 'Right'
        buffer_frames[hand_type].append(hand_to_json(hand))
        if len(buffer_frames[hand_type]) > buffer_size:
            buffer_frames[hand_type].pop(0)

def save_recording():
    global recorded_json
    filename = input("Enter a filename for recording (without extension): ")
    if (filename != 'delete'):
        json_string = json.dumps(recorded_json)

        filepath = f"Final/JSON_Data/30fps/{filename}.json"

        with open(filepath, "w") as json_file:
            json_file.write(json_string)
            print(f'Recording saved as {filepath}')
    else:
        print('Recording not saved.')
    
def main():
    listener = HandListener()

    connection = leap.Connection()
    connection.add_listener(listener)

    with connection.open():
        while True:
            time.sleep(1)

if __name__ == "__main__":
    main()
