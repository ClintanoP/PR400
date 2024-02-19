# Functionally the same as visual_translator, but without the visual output, so less taxing on the system.
import time
import leap
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import normalize_data as normalize
import recording as record
import tf_train as tf_custom

# Create a dictionary to store the hand data
hands_data = {"Left" : [], "Right" : [], "Count": 0}

# Set the batch size
batch_size = 15

# Set the default colour of the hands
hand_colour = (255, 255, 255)

# Load the model
asl_model = tf.keras.models.load_model("Final/asl_v2_4_30fps.keras")

# Strings to store the predicted gesture and confidence
predicted_gesture = ""
predicted_confidence = ""

# Tracking listener class
class TrackingListener(leap.Listener):
    def on_connection_event(self, event):
        print("Connected")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        self.store_hands(event)
        
def main():
    tracking_listener = TrackingListener()

    connection = leap.Connection()
    connection.add_listener(tracking_listener)

    running = True
    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)

def store_hands(event):
    global hands_data, asl_model, predicted_gesture, predicted_confidence, batch_size

    if len(event.hands) == 0:
        return
    
    for i in range(0, len(event.hands)):
        hand = event.hands[i]

        # continually push the last 6 frames. add one frame to the dict, remove the first.
        # then check against model 
        if event.tracking_frame_id % 4 == 0:
            hand_type = 'Left' if str(hand.type) == 'HandType.Left' else 'Right'
            hands_data[hand_type].append(record.hand_to_json(hand))
            hands_data['Count'] += 1

            if hands_data['Count'] == 45:
                # Process the 45 hand frames
                print('Processing 45 hand frames.')
                
                ###################################################################################################################

                # normal model
                # measuring what takes the longest amount of time
                # Start time for normalization and aggregation
                start_time_aggregate = time.time()
                input_gesture_data = normalize.aggregate_gesture_data_non_file(hands_data)
                end_time_aggregate = time.time()
                print(f"Normalization and aggregation time: {end_time_aggregate - start_time_aggregate} seconds")

                # Start time for prediction
                start_time_predict = time.time()
                predicted_gesture, predicted_confidence = tf_custom.predict_new_gesture_prenormalized(input_gesture_data, asl_model)
                end_time_predict = time.time()
                print(f"Prediction time: {end_time_predict - start_time_predict} seconds")

                print(f"The predicted gesture is: {predicted_gesture}, with a confidence of {predicted_confidence}%")
                
                ###################################################################################################################

                # Remove the first 15 frames
                hands_data['Left'] = hands_data['Left'][-30:]
                hands_data['Right'] = hands_data['Right'][-30:]
                hands_data['Count'] = 30

    # remove an entry if no hand in the detected area.
    if len(event.hands) == 1:
        missing_hand_type = 'Left' if str(hand.type) == 'HandType.Right' else 'Left'
        if len(hands_data[missing_hand_type]) > 0:
            hands_data[missing_hand_type].pop(0)
    
if __name__ == "__main__":
    main()