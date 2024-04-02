import time
import leap
import numpy as np
import cv2
import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import tensorflow as tf
import normalize_data as normalize
import train_new_dataset as preprocess
import recording as record
import tf_train as tf_custom


hands_data = {"Left" : [], "Right" : [], "Count": 0, "ContinuousTracking": False}

batch_size = 15

hand_colour = (255, 255, 255)

labels_list = tf_custom.read_labels_from_file("ck_mixed_dataset_gesture_filtered.txt")

# normal model
asl_model = tf.keras.models.load_model("Final/asl_v3_6_ck_mixed_500epoch_earlystop_true_mid_complex_model.keras")

had_hand_in_previous_frame = False

predicted_gesture = ""
predicted_confidence = ""

_TRACKING_MODES = {
    leap.TrackingMode.Desktop: "Desktop",
    leap.TrackingMode.HMD: "HMD",
    leap.TrackingMode.ScreenTop: "ScreenTop",
}


class Canvas:
    def __init__(self):
        global hand_colour
        self.name = "ASL Visualiser"
        self.screen_size = [600, 800]
        self.hands_colour = hand_colour
        self.font_colour = (0, 255, 44)
        self.hands_format = "Skeleton"
        self.output_image = np.zeros((self.screen_size[0], self.screen_size[1], 3), np.uint8)
        self.tracking_mode = None

    def set_tracking_mode(self, tracking_mode):
        self.tracking_mode = tracking_mode

    def set_hands_colour(self, new_colour):
        self.hands_colour = new_colour

    def toggle_hands_format(self):
        self.hands_format = "Dots" if self.hands_format == "Skeleton" else "Skeleton"
        print(f"Set hands format to {self.hands_format}")

    def get_joint_position(self, bone):
        if bone:
            return int(bone.x + (self.screen_size[1] / 2)), int(bone.z + (self.screen_size[0] / 2))
        else:
            return None

    def render_hands(self, event):
        global hands_data, asl_model, predicted_gesture, predicted_confidence, batch_size, hand_colour, had_hand_in_previous_frame
        # Clear the previous image
        self.output_image[:, :] = 0

        text = (f"Tracking Mode: {_TRACKING_MODES[self.tracking_mode]}, "
        f"Predicted: {predicted_gesture}, "
        f"Confidence: {predicted_confidence}, "
        f"Frame: {hands_data['Count']}")

        cv2.putText(
            self.output_image,
            text,
            (10, self.screen_size[0] - 10),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.5,
            self.font_colour,
            1
        )

        if len(event.hands) == 0:
            if had_hand_in_previous_frame:
                if hands_data['Count'] > 15:
                    # measuring what takes the longest amount of time
                    # Start time for normalization and aggregation
                    start_time_aggregate = time.time()
                    input_gesture_data = preprocess.aggregate_gesture_data_non_file(hands_data)
                    end_time_aggregate = time.time()
                    print(f"Normalization and aggregation time: {end_time_aggregate - start_time_aggregate} seconds")

                    # Start time for prediction
                    start_time_predict = time.time()
                    predicted_gesture, predicted_confidence = tf_custom.predict_new_gesture_prenormalized(input_gesture_data, asl_model, labels_list)


                    if predicted_confidence > 90.0:
                        hand_colour = (0, 255, 0)
                    elif predicted_confidence > 80.0:
                        hand_colour = (0, 211, 222)
                    elif predicted_confidence > 70.0:
                        hand_colour = (255, 0, 0)
                    else:
                        hand_colour = (255, 255, 255)
                        
                    self.set_hands_colour(hand_colour)
                    end_time_predict = time.time()
                    print(f"Prediction time: {end_time_predict - start_time_predict} seconds")

                    print(f"The predicted gesture is: {predicted_gesture}, with a confidence of {predicted_confidence}%")
                    hands_data['Left'] = []
                    hands_data['Right'] = []
                    hands_data['Count'] = 0
                    hands_data['ContinuousTracking'] = False
                had_hand_in_previous_frame = False
            return

        for i in range(0, len(event.hands)):
            had_hand_in_previous_frame = True
            hands_data['ContinuousTracking'] = True
            hand = event.hands[i]
            
            # render hands
            for index_digit in range(0, 5):
                digit = hand.digits[index_digit]
                for index_bone in range(0, 4):
                    bone = digit.bones[index_bone]
                    if self.hands_format == "Dots":
                        prev_joint = self.get_joint_position(bone.prev_joint)
                        next_joint = self.get_joint_position(bone.next_joint)
                        if prev_joint:
                            cv2.circle(self.output_image, prev_joint, 2, self.hands_colour, -1)

                        if next_joint:
                            cv2.circle(self.output_image, next_joint, 2, self.hands_colour, -1)

                    if self.hands_format == "Skeleton":
                        wrist = self.get_joint_position(hand.arm.next_joint)
                        elbow = self.get_joint_position(hand.arm.prev_joint)
                        if wrist:
                            cv2.circle(self.output_image, wrist, 3, self.hands_colour, -1)

                        if elbow:
                            cv2.circle(self.output_image, elbow, 3, self.hands_colour, -1)

                        if wrist and elbow:
                            cv2.line(self.output_image, wrist, elbow, self.hands_colour, 2)

                        bone_start = self.get_joint_position(bone.prev_joint)
                        bone_end = self.get_joint_position(bone.next_joint)

                        if bone_start:
                            cv2.circle(self.output_image, bone_start, 3, self.hands_colour, -1)

                        if bone_end:
                            cv2.circle(self.output_image, bone_end, 3, self.hands_colour, -1)

                        if bone_start and bone_end:
                            cv2.line(self.output_image, bone_start, bone_end, self.hands_colour, 2)

                        if ((index_digit == 0) and (index_bone == 0)) or (
                            (index_digit > 0) and (index_digit < 4) and (index_bone < 2)
                        ):
                            index_digit_next = index_digit + 1
                            digit_next = hand.digits[index_digit_next]
                            bone_next = digit_next.bones[index_bone]
                            bone_next_start = self.get_joint_position(bone_next.prev_joint)
                            if bone_start and bone_next_start:
                                cv2.line(
                                    self.output_image,
                                    bone_start,
                                    bone_next_start,
                                    self.hands_colour,
                                    2,
                                )

                        if index_bone == 0 and bone_start and wrist:
                            cv2.line(self.output_image, bone_start, wrist, self.hands_colour, 2)

            # then add to the hands_data
            hand_type = 'Left' if str(hand.type) == 'HandType.Left' else 'Right'
            hands_data[hand_type].append(record.hand_to_json(hand))
            hands_data['Count'] += 1

        # remove an entry if no hand in the detected area.
        if len(event.hands) == 1:
            missing_hand_type = 'Left' if str(hand.type) == 'HandType.Right' else 'Right'
            if len(hands_data[missing_hand_type]) > 0:
                hands_data[missing_hand_type].pop(0)
            


class TrackingListener(leap.Listener):
    def __init__(self, canvas):
        self.canvas = canvas

    def on_connection_event(self, event):
        pass

    def on_tracking_mode_event(self, event):
        self.canvas.set_tracking_mode(event.current_tracking_mode)
        print(f"Tracking mode changed to {_TRACKING_MODES[event.current_tracking_mode]}")

    def on_device_event(self, event):
        try:
            with event.device.open():
                info = event.device.get_info()
        except leap.LeapCannotOpenDeviceError:
            info = event.device.get_info()

        print(f"Found device {info.serial}")

    def on_tracking_event(self, event):
        self.canvas.render_hands(event)


def main():
    canvas = Canvas()

    print(canvas.name)
    print("")
    print("Press <key> in visualiser window to:")
    print("  x: Exit")
    print("  h: Select HMD tracking mode")
    print("  s: Select ScreenTop tracking mode")
    print("  d: Select Desktop tracking mode")
    print("  f: Toggle hands format between Skeleton/Dots")

    tracking_listener = TrackingListener(canvas)

    connection = leap.Connection()
    connection.add_listener(tracking_listener)

    running = True

    with connection.open():
        connection.set_tracking_mode(leap.TrackingMode.Desktop)
        canvas.set_tracking_mode(leap.TrackingMode.Desktop)

        while running:
            cv2.imshow(canvas.name, canvas.output_image)

            key = cv2.waitKey(1)

            if key == ord("x"):
                break
            elif key == ord("h"):
                connection.set_tracking_mode(leap.TrackingMode.HMD)
            elif key == ord("s"):
                connection.set_tracking_mode(leap.TrackingMode.ScreenTop)
            elif key == ord("d"):
                connection.set_tracking_mode(leap.TrackingMode.Desktop)
            elif key == ord("f"):
                canvas.toggle_hands_format()



if __name__ == "__main__":
    main()
