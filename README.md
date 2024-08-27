# Copyright 2024 [Anjali Gupta]
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
# http://www.apache.org/licenses/LICENSE-2.0
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.

import tensorflow as tf
from tensorflow.keras.datasets import mnist
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout
import matplotlib.pyplot as plt
import numpy as np

# Load and preprocess the MNIST dataset
def load_and_preprocess_data():
    (x_train, y_train), (x_test, y_test) = mnist.load_data()
    
    # Normalize the images to [0, 1] range
    x_train, x_test = x_train / 255.0, x_test / 255.0
    
    # Reshape data to fit the model (add channel dimension)
    x_train = x_train.reshape(-1, 28, 28, 1)
    x_test = x_test.reshape(-1, 28, 28, 1)
    
    return x_train, y_train, x_test, y_test

# Build the CNN model
def build_model():
    model = Sequential([
        Conv2D(32, kernel_size=(3, 3), activation='relu', input_shape=(28, 28, 1)),
        MaxPooling2D(pool_size=(2, 2)),
        Conv2D(64, kernel_size=(3, 3), activation='relu'),
        MaxPooling2D(pool_size=(2, 2)),
        Flatten(),
        Dense(128, activation='relu'),
        Dropout(0.5),
        Dense(10, activation='softmax')
    ])
    
    # Compile the model
    model.compile(optimizer='adam', loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    return model

# Train the model
def train_model(model, x_train, y_train, x_test, y_test):
    model.fit(x_train, y_train, epochs=10, validation_data=(x_test, y_test))

# Evaluate the model
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f'Test accuracy: {test_acc}')

# Make predictions and visualize results
def make_predictions(model, x_test, y_test):
    predictions = model.predict(x_test)
    
    # Visualize the first 5 test images, their predicted labels, and the true labels
    for i in range(5):
        plt.imshow(x_test[i].reshape(28, 28), cmap='gray')
        plt.title(f'Predicted: {np.argmax(predictions[i])}, True: {y_test[i]}')
        plt.show()

# Save the model
def save_model(model, path='digit_recognition_model.h5'):
    model.save(path)

# Load the model
def load_model(path='digit_recognition_model.h5'):
    return tf.keras.models.load_model(path)

# Main function to run the project
def main():
    x_train, y_train, x_test, y_test = load_and_preprocess_data()
    model = build_model()
    train_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)
    make_predictions(model, x_test, y_test)
    save_model(model)

if __name__ == "__main__":
    main()
