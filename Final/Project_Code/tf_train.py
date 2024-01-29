'''
This file is used to train the model, which is then saved to be used later.

Below is a brief explanation of the code:
Key info:
- Each 'gesture' has these characteristics:
    - represents a specific word or sentence in ASL. 
    - has an array of file paths. These are used for training. 
    - has a label with their plaintext value. ie. 'I Love You'

Flow:
1. File paths declared for training gestures.
2. Each hands gesture data is normalized and aggregated.
3. X and y are made by concatenating the data and the labelled data respectively.
4. The data is split into training, testing and validation.
5. The model is declared, adding in the input layer, hidden layer and final output layer.
6. The model is then compiled using the popular 'Adam' optimizer.
7. The model is then finally fit to the train and test data.
8. Lastly the model is saved to a file locally.
'''

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
import numpy as np
import glob
from sklearn.model_selection import train_test_split
import tensorflow as tf
import tensorflow_model_optimization as tfmot
import normalize_data as normalize


# getting the file paths
def get_file_paths():
    # Base directory where the JSON files are located
    base_dir = 'Final/JSON_Data/30fps/'

    # Dictionary to hold the file paths for each gesture
    gesture_file_paths = {}

    # List all json files in the directory
    json_files = glob.glob(os.path.join(base_dir, '*.json'))

    # Sort files into their respective lists
    for file_name in json_files:
        # Extract the gesture name and the number from the file name
        gesture_name = os.path.basename(file_name).split('_')[0]
        
        # Add the file path to the corresponding gesture list in the dictionary
        if gesture_name in gesture_file_paths:
            gesture_file_paths[gesture_name].append(file_name)
        else:
            gesture_file_paths[gesture_name] = [file_name]

    # Sort the lists so that the files are in the correct order
    for gesture in gesture_file_paths:
        gesture_file_paths[gesture].sort()
    
    return gesture_file_paths

def main():
    gesture_file_paths = get_file_paths()

    gesture_1_data = normalize.aggregate_gesture_data(gesture_file_paths['book'])
    gesture_2_data = normalize.aggregate_gesture_data(gesture_file_paths['hello'])
    gesture_3_data = normalize.aggregate_gesture_data(gesture_file_paths['iloveyou'])
    
    # new gestures
    gesture_4_data = normalize.aggregate_gesture_data(gesture_file_paths['yes'])
    gesture_5_data = normalize.aggregate_gesture_data(gesture_file_paths['no'])
    
    label_book = 0
    label_hello = 1
    label_iloveyou = 2

    # new labels
    label_yes = 3
    label_no = 4

    labels_gesture_1 = np.full(len(gesture_1_data), label_book)
    labels_gesture_2 = np.full(len(gesture_2_data), label_hello)
    labels_gesture_3 = np.full(len(gesture_3_data), label_iloveyou)

    # new labelled
    labels_gesture_4 = np.full(len(gesture_4_data), label_yes)
    labels_gesture_5 = np.full(len(gesture_5_data), label_no)
    # labels_gesture_4 = np.full(len(gesture_4_data), label_goodbye)


    X = np.concatenate([gesture_1_data, gesture_2_data, gesture_3_data, gesture_4_data, gesture_5_data])
    y = np.concatenate([labels_gesture_1, labels_gesture_2, labels_gesture_3, labels_gesture_4, labels_gesture_5])

    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    tf.config.optimizer.set_jit(True)

    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(48, activation='elu'),
        # adding more layers as needed
        tf.keras.layers.Dense(5, activation='softmax')  # Assuming 5 gestures
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, validation_data = (X_val, y_val))

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # print the accuracy.
    print(f"Test accuracy: {test_accuracy}")
    
    # save the model locally. 
    model.save("Final/asl_v2_5_30fps.keras")

    ###################################################################################################################
    # now to optimize using quantification
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # tflite_quant_model = converter.convert()

    # # Save the quantized model
    # with open('Final/asl_quantized_v1.tflite', 'wb') as f:
    #     f.write(tflite_quant_model)

    ###################################################################################################################




def predict_new_gesture(file_path, model):
    # Process the new gesture file
    new_gesture_data = normalize.aggregate_gesture_data([file_path])

    # Assuming your model expects data of shape (num_samples, num_features)
    # Reshape the data if necessary
    new_gesture_data = new_gesture_data.reshape(-1, 60)  # Reshape for a single sample

    # Make a prediction
    predicted_probabilities = model.predict(new_gesture_data)
    
    # Find the class with the highest probability
    predicted_class = np.argmax(predicted_probabilities, axis=1)

    # Get the confidence percentage of the prediction
    confidence = np.max(predicted_probabilities) * 100
    
    # Map the class to its label
    class_labels = {0: "book", 1: "hello", 2: "I love you", 3: "yes", 4: "no"}
    predicted_label = class_labels[predicted_class[0]]

    return predicted_label, confidence

def predict_new_gesture_prenormalized(gesture_data, model):
    # this inputs already processed and normalized data. 
    gesture_data = gesture_data.reshape(-1, 60) # reshape for a single sample

    # Make a prediction
    predicted_probabilities = model.predict(gesture_data)
    
    # Find the class with the highest probability
    predicted_class = np.argmax(predicted_probabilities, axis=1)

    # Get the confidence percentage of the prediction
    confidence = np.max(predicted_probabilities) * 100
    
    # Map the class to its label
    class_labels = {0: "book", 1: "hello", 2: "I love you", 3: "yes", 4: "no"}
    predicted_label = class_labels[predicted_class[0]]

    return predicted_label, confidence
    


# TF LITE
def convert_to_tflite(model_path, tflite_model_path):
    # Load the TensorFlow model
    model = tf.keras.models.load_model(model_path)

    # Convert the model to TensorFlow Lite format
    converter = tf.lite.TFLiteConverter.from_keras_model(model)

    # Apply quantization (Optional)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # Convert the model
    tflite_model = converter.convert()

    # Save the converted model
    with open(tflite_model_path, 'wb') as f:
        f.write(tflite_model)

    print(f"Model converted and saved to {tflite_model_path}")

def load_tflite_model(tflite_model_path):
    # Load the TFLite model and allocate tensors
    interpreter = tf.lite.Interpreter(model_path=tflite_model_path)
    interpreter.allocate_tensors()
    return interpreter

def predict_new_gesture_tflite(gesture_data, interpreter):
    # Assuming gesture_data is already aggregated and needs reshaping and type conversion

    # Get input details and check the expected input shape
    input_details = interpreter.get_input_details()
    expected_shape = input_details[0]['shape']
    print(f"Expected input shape: {expected_shape}")

    # If necessary, adjust the logic here to ensure gesture_data matches the expected_shape
    # For now, let's reshape it as expected
    # Ensure the data is of type np.float32
    gesture_data = gesture_data.astype(np.float32)

    # Reshape the data to match the expected input shape
    # Make sure the total number of elements matches
    if gesture_data.size == expected_shape[1] * expected_shape[2] * expected_shape[3]:
        gesture_data = gesture_data.reshape(expected_shape)
    else:
        raise ValueError(f"Data shape mismatch. Data has {gesture_data.size} elements, but model expects {expected_shape[1] * expected_shape[2] * expected_shape[3]}")

    # Set the tensor
    interpreter.set_tensor(input_details[0]['index'], gesture_data)

    # Run the model
    interpreter.invoke()

    # Extract the output
    output_details = interpreter.get_output_details()
    output_data = interpreter.get_tensor(output_details[0]['index'])

    print(f"Output data shape: {output_data.shape}")  # Debug print

    # Find the class with the highest probability
    predicted_class = np.argmax(output_data, axis=1)
    confidence = np.max(output_data) * 100

    print(f"Predicted class index: {predicted_class}")  # Debug print

    # Map the class to its label
    class_labels = {0: "book", 1: "hello", 2: "I love you", 3: "yes", 4: "no"}
    try:
        predicted_label = class_labels[predicted_class[0]]
    except IndexError as e:
        print(f"IndexError: {e}. Predicted class index was {predicted_class[0]}")
        predicted_label = "Unknown"

    return predicted_label, confidence

main()