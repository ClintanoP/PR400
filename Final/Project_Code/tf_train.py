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
from keras import layers, models, regularizers
from keras.callbacks import EarlyStopping
import normalize_data as normalize
import train_new_dataset as preprocess


# getting the file paths
def get_file_paths(base_dir):
    # Base directory where the JSON files are located

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
    gesture_4_data = normalize.aggregate_gesture_data(gesture_file_paths['yes'])
    gesture_5_data = normalize.aggregate_gesture_data(gesture_file_paths['no'])
    
    label_book = 0
    label_hello = 1
    label_iloveyou = 2
    label_yes = 3
    label_no = 4

    labels_gesture_1 = np.full(len(gesture_1_data), label_book)
    labels_gesture_2 = np.full(len(gesture_2_data), label_hello)
    labels_gesture_3 = np.full(len(gesture_3_data), label_iloveyou)
    labels_gesture_4 = np.full(len(gesture_4_data), label_yes)
    labels_gesture_5 = np.full(len(gesture_5_data), label_no)

    X = np.concatenate([gesture_1_data, gesture_2_data, gesture_3_data, gesture_4_data, gesture_5_data])
    y = np.concatenate([labels_gesture_1, labels_gesture_2, labels_gesture_3, labels_gesture_4, labels_gesture_5])

    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    print(f"Shape of X train: {X_train.shape}, Shape of y train: {y_train.shape}")

    tf.config.optimizer.set_jit(True)

    # dense, working model.
    model = tf.keras.Sequential([
        tf.keras.layers.InputLayer(input_shape=(X_train.shape[1],)),
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.Dense(48, activation='elu'),
        tf.keras.layers.Dense(5, activation='softmax')  # Assuming 5 gestures
    ])
    
    # new model, using convolutional layers.
    # Declare the model - using convolutional layers
    # model = tf.keras.models.Sequential([
    #     tf.keras.layers.InputLayer(input_shape=(20, 3)), # Reshaped input
    #     tf.keras.layers.Conv1D(filters=64, kernel_size=3, activation='relu'),
    #     tf.keras.layers.MaxPooling1D(pool_size=2),
    #     tf.keras.layers.Conv1D(filters=128, kernel_size=3, activation='relu'),
    #     tf.keras.layers.MaxPooling1D(pool_size=2),
    #     tf.keras.layers.Flatten(), # Flatten the output to feed into dense layers
    #     tf.keras.layers.Dense(128, activation='relu'),
    #     tf.keras.layers.Dense(5, activation='softmax') # Output layer
    # ])

    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(X_train, y_train, epochs=10, validation_data = (X_val, y_val))

    test_loss, test_accuracy = model.evaluate(X_test, y_test)

    # print the accuracy.
    print(f"Test accuracy: {test_accuracy}")
    
    # save the model locally. 
    model.save("Final/asl_v2_6_30fps.keras")

    ###################################################################################################################
    # now to optimize using quantification
    # converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # tflite_quant_model = converter.convert()

    # # Save the quantized model
    # with open('Final/asl_quantized_v1.tflite', 'wb') as f:
    #     f.write(tflite_quant_model)

    ###################################################################################################################

def read_labels_from_file(file_path):
    with open(file_path, 'r') as file:
        labels = file.read().splitlines()
    return labels

def test_train_main():
    gesture_file_paths = preprocess.get_file_paths_and_save_filtered_counts('Final/JSON_Data/120fps/')
    # Read labels from the text file
    labels_list = read_labels_from_file("ck_mixed_dataset_gesture_filtered.txt")

    # Initialize empty lists to hold aggregated data and labels
    all_gesture_data = []
    all_labels = []

    # Dictionary to hold label values
    label_values = {}

    # Loop through each label, load and process the gesture data, and generate labels
    for i, label in enumerate(labels_list):
        gesture_data = preprocess.aggregate_gesture_data(gesture_file_paths[label])
        all_gesture_data.append(gesture_data)
        
        # Generate and store labels for the current gesture data
        labels = np.full(len(gesture_data), i)
        all_labels.append(labels)
        
        # Store label value in the dictionary
        label_values[label] = i
        print(f"Label: {label}, Value: {i}")
        print(f"Shape of gesture data: {gesture_data.shape}, Shape of labels: {labels.shape}")


    # Concatenate all gesture data and labels to form the dataset
    X = np.concatenate(all_gesture_data, axis=0)
    y = np.concatenate(all_labels, axis=0)
    print(f"Shape of X: {X.shape}, Shape of y: {y.shape}")

    X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42, stratify=y)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42, stratify=y_temp)

    # X_train = X_train[..., np.newaxis]  # Adding a new axis to fit the model's expected input
    # X_val = X_val[..., np.newaxis]
    # X_test = X_test[..., np.newaxis]

    early_stopping = EarlyStopping(
        monitor='val_loss',  # Monitor the model's validation loss
        patience=10,          # Number of epochs with no improvement after which training will be stopped
        verbose=1,           # Log when training is stopped
        restore_best_weights=True  # Restore model weights from the epoch with the best value of the monitored quantity
    )

    print(f"Shape of X train: {X_train.shape}, Shape of y train: {y_train.shape}")
    

    input_shape = (X_train.shape[1], 1)  # This needs to be adjusted based on the actual sequence length and features

    model_with_dropout = create_1d_cnn_model_with_dropout((X_train.shape[1], 1), len(labels_list))

    model_with_dropout.compile(optimizer='adam',
                            loss='sparse_categorical_crossentropy',
                            metrics=['accuracy'])

    history_with_dropout = model_with_dropout.fit(
        X_train, y_train, 
        epochs=500, 
        validation_data=(X_val, y_val),
        callbacks=[early_stopping]  # Include the early stopping callback here
    )

    test_loss, test_acc = model_with_dropout.evaluate(X_test, y_test, verbose=2)
    print(f"Test accuracy: {test_acc}, Test loss: {test_loss}")

    model_with_dropout.save("Final/asl_v3_5_ck_mixed_500epoch_earlystop_true_mid_complex_model.keras")
    # model.summary()
   
def test_model():
    # Load the model
    # model = tf.keras.models.load_model("Final/asl_v3_3_ck_gestures_500epoch_earlystop_true_mid_complex_model.keras")

    model = tf.keras.models.load_model("Final/asl_v2_3_30fps.keras")

    # Load a sample gesture file
    test_file_path = 'Final/JSON_Data/testing/hello_5.json'

    # Make a prediction
    predicted_label, confidence = predict_new_gesture(test_file_path, model)
    print(f"Predicted Gesture: {predicted_label} with confidence {confidence:.2f}%")


def predict_new_gesture(file_path, model):
    # Process the new gesture file
    new_gesture_data = normalize.aggregate_gesture_data([file_path])

    # Flatten the (20, 3) data into (60,) to match the model's expected input
    # Assuming new_gesture_data is shaped (num_samples, 20, 3) after aggregation
    new_gesture_data = new_gesture_data.reshape(new_gesture_data.shape[0], -1)  # This changes shape to (num_samples, 60)
    
    # Ensure the data has the correct final shape expected by the model: (None, 60, 1)
    # This is necessary if your model expects a certain number of channels or dimensions
    new_gesture_data = np.expand_dims(new_gesture_data, axis=-1)  # Changes shape to (num_samples, 60, 1)

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


def predict_new_gesture_prenormalized(gesture_data, model, labels_list):
    # this inputs already processed and normalized data. 
    # gesture_data = gesture_data.reshape(-1, 20, 3) # reshape for a single sample

    # Make a prediction
    predicted_probabilities = model.predict(gesture_data)
    
    # Find the class with the highest probability
    predicted_class = np.argmax(predicted_probabilities, axis=1)

    # Get the confidence percentage of the prediction
    confidence = np.max(predicted_probabilities) * 100
    
    # Map the class to its label
    
    predicted_label = labels_list[predicted_class[0]]

    return predicted_label, confidence
    
def create_1d_cnn_model_with_dropout(input_shape, label_count):
    model = models.Sequential()
    
    # Input convolutional layer
    model.add(layers.Conv1D(16, 3, activation='relu', padding='same', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.2))  # Adjusting dropout slightly
    
    # Adding an additional convolutional layer in the first block for more feature extraction capability
    model.add(layers.Conv1D(16, 3, activation='relu', padding='same'))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.2))  # Keeping dropout consistent in early layers

    # Second convolutional block
    model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
    model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))  # New: Adding another layer here for depth
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Dropout(0.25))  # Slight increase in dropout for added complexity

    # Global pooling to efficiently reduce dimensionality
    model.add(layers.GlobalMaxPooling1D())

    # Enhanced dense layer
    model.add(layers.Dense(256, activation='relu'))  # Increased units for a more complex model
    model.add(layers.Dropout(0.35))  # Adjusting dropout to balance the increased dense layer complexity
    
    # Output layer
    model.add(layers.Dense(label_count, activation='softmax'))
    
    return model


# FIXME: Best model so far. Keeping this commented out for now.
# def create_1d_cnn_model_with_dropout(input_shape, label_count):
#     model = models.Sequential()
    
#     # Input convolutional layer
#     model.add(layers.Conv1D(16, 3, activation='relu', padding='same', input_shape=input_shape))
#     model.add(layers.MaxPooling1D(2))
#     model.add(layers.Dropout(0.2))  # Minimal dropout for simplicity
    
#     # Reduced to a single convolutional block
#     model.add(layers.Conv1D(32, 3, activation='relu', padding='same'))
#     model.add(layers.MaxPooling1D(2))
#     model.add(layers.Dropout(0.25))

#     # Skipping additional convolutional blocks and complex structures

#     # Global pooling to reduce dimensionality without dense layers
#     model.add(layers.GlobalMaxPooling1D())

#     # Simplified dense layer
#     model.add(layers.Dense(128, activation='relu'))
#     model.add(layers.Dropout(0.3))
    
#     # Output layer
#     model.add(layers.Dense(label_count, activation='softmax'))
    
#     return model

def create_1d_cnn_model(input_shape):
    model = models.Sequential()
    
    # Input shape would be (sequence_length, features)
    # In your case, it seems like 60 might be the total features (flattened coordinates for all joints/bones across frames)
    model.add(layers.Conv1D(64, 3, activation='relu', input_shape=input_shape))
    model.add(layers.MaxPooling1D(2))
    model.add(layers.Conv1D(128, 3, activation='relu'))
    model.add(layers.GlobalMaxPooling1D())
    model.add(layers.Dense(128, activation='relu'))
    model.add(layers.Dense(5, activation='softmax'))  # Assuming 5 gestures

    return model

# test_train_main()
#test_model()
#main()