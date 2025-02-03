import tensorflow as tf
from tensorflow import keras
from keras.models import Sequential
from keras.layers import Activation, Dense, Flatten, Conv2D, MaxPool2D
from keras.optimizers import Adam
from keras.callbacks import ReduceLROnPlateau, EarlyStopping
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import cv2
import numpy as np
import matplotlib.pyplot as plt
import warnings

warnings.simplefilter(action='ignore', category=FutureWarning)

# Dataset paths
train_path = r'D:\gesture\alphabet\train'  # Ensure this folder contains A-Z folders
test_path = r'D:\gesture\alphabet\test'    # Ensure this folder contains A-Z folders

# Data generators for training and testing (for A-Z)
train_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=train_path,
    target_size=(64, 64),
    class_mode='categorical',
    batch_size=10,
    shuffle=True
)

test_batches = ImageDataGenerator(preprocessing_function=tf.keras.applications.vgg16.preprocess_input).flow_from_directory(
    directory=test_path,
    target_size=(64, 64),
    class_mode='categorical',
    batch_size=10,
    shuffle=True
)

# Function to plot images
def plotImages(images_arr):
    fig, axes = plt.subplots(1, len(images_arr), figsize=(20, 5))
    axes = axes.flatten()
    for img, ax in zip(images_arr, axes):
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
        ax.imshow(img)
        ax.axis('off')
    plt.tight_layout()
    plt.show()

# Model definition
model = Sequential()

model.add(Conv2D(filters=32, kernel_size=(3, 3), activation='relu', input_shape=(64, 64, 3)))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=64, kernel_size=(3, 3), activation='relu', padding='same'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Conv2D(filters=128, kernel_size=(3, 3), activation='relu', padding='valid'))
model.add(MaxPool2D(pool_size=(2, 2), strides=2))

model.add(Flatten())

model.add(Dense(64, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(128, activation='relu'))
model.add(Dense(26, activation='softmax'))  # Changed from 24 to 26 for A-Z

# Compile model
model.compile(optimizer=Adam(learning_rate=0.001), loss='categorical_crossentropy', metrics=['accuracy'])

# Callbacks for learning rate reduction and early stopping
reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.0001)
early_stop = EarlyStopping(monitor='val_loss', min_delta=0, patience=2, verbose=0, mode='auto')

# Train the model
history2 = model.fit(train_batches, epochs=10, callbacks=[reduce_lr, early_stop], validation_data=test_batches)

# Save the model
model.save('best_model_alphabet_A_to_Z.h5')

# Evaluate the model on test data
imgs, labels = next(test_batches)
scores = model.evaluate(imgs, labels, verbose=0)
print(f'{model.metrics_names[0]} of {scores[0]}; {model.metrics_names[1]} of {scores[1]*100}%')

# Load model
model = keras.models.load_model('best_model_alphabet_A_to_Z.h5')
# Class dictionary (for A-Z)
word_dict = {
    0: 'A', 1: 'B', 2: 'C', 3: 'D', 4: 'E', 5: 'F', 6: 'G', 7: 'H', 8: 'I', 9: 'J',
    10: 'K', 11: 'L', 12: 'M', 13: 'N', 14: 'O', 15: 'P', 16: 'Q', 17: 'R', 18: 'S', 19: 'T',
    20: 'U', 21: 'V', 22: 'W', 23: 'X', 24: 'Y', 25: 'Z'  # Added 'Z' here
}

# Predictions on a small set of test data
predictions = model.predict(imgs, verbose=0)
print("Predictions on a small set of test data:")
for ind, i in enumerate(predictions):
    print(word_dict[np.argmax(i)], end='   ')

# Plot test images and display actual labels
plotImages(imgs)
print('Actual labels:')
for i in labels:
    print(word_dict[np.argmax(i)], end='   ')
