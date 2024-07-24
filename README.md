<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <h2>Handwriting Nastaliq Recognition</h2>
</head>
<body>
    <h1>Handwriting Nastaliq Recognition using CNN-BDLSTM Hybrid Model</h1>
    <p>This repository contains the code and resources for a project aimed at recognizing Nastaliq handwritten script, a form of Persian calligraphy. The project involves detecting writing items using contours and regrouping them based on geometric proximity. It uses a custom dataset of Nastaliq words and character combinations along with a CNN-BDLSTM hybrid model to recognize the text.</p>
    <h2>Overview</h2>
    <p>Nastaliq script presents significant challenges in handwriting recognition due to its artistic and complex nature. This project addresses these challenges by employing contour detection for initial preprocessing and a hybrid Convolutional Neural Network (CNN) and Bidirectional Long Short-Term Memory (BDLSTM) network for text recognition.</p>
    <h2>Features</h2>
    <ul>
        <li><strong>Contour Detection:</strong> Utilizes contours to detect and group writing items based on geometric properties.</li>
        <li><strong>Convolutional Neural Networks (CNNs):</strong> Extracts local higher-level features from spatial input.</li>
        <li><strong>Bidirectional Long Short-Term Memory (BDLSTM):</strong> Captures sequential patterns and dependencies in the data, considering both past and future context.</li>
        <li><strong>Data Augmentation:</strong> Uses <code>ImageDataGenerator</code> for real-time data augmentation during training.</li>
    </ul>
    <h2>Model Architecture</h2>
    <p>The model architecture includes multiple convolutional layers for feature extraction, followed by MaxPooling layers to reduce spatial dimensions. A TimeDistributed layer is used to pass information to the BDLSTM network, which captures sequential dependencies. Finally, Dense layers with activation functions are employed for classification.</p>
    <h2>Contour Detection Code</h2>
    <pre><code># -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:46:26 2018
@author: Aida Mohseni
"""
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, TimeDistributed
from keras import backend as K
from keras.layers import Bidirectional, LSTM

# Dimensions of our images.
img_width, img_height = 128, 128
train_data_dir = 'd:/data2/train'
validation_data_dir = 'd:/data2/test'
nb_train_samples = 70
nb_validation_samples = 30
epochs = 50
batch_size = 10

if K.image_data_format() == 'channels_first':
    input_shape = (3, img_width, img_height)
else:
    input_shape = (img_width, img_height, 3)

model = Sequential()
model.add(Conv2D(64, (3, 3), input_shape=input_shape))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(3, 3)))

model.add(Conv2D(32, (3, 3)))
model.add(Activation('relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('relu'))

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))

model.compile(loss='binary_crossentropy',
              optimizer='adam',
              metrics=['accuracy'])

model.summary()

# Data augmentation configuration for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# Data augmentation configuration for testing
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')
</code></pre>
    <h2>Getting Started</h2>
    <h3>Prerequisites</h3>
    <ul>
        <li>Python 3.x</li>
        <li>TensorFlow</li>
        <li>Keras</li>
        <li>Scikit-learn</li>
        <li>Matplotlib</li>
    </ul>
    <h3>Installation</h3>
    <ol>
        <li>Clone the repository:
            <pre><code>git clone https://github.com/yourusername/nastaliq-handwriting-recognition.git
cd nastaliq-handwriting-recognition
            </code></pre>
        </li>
        <li>Install the required packages:
            <pre><code>pip install -r requirements.txt</code></pre>
        </li>
    </ol>
    <h3>Dataset</h3>
    <p>Ensure you have the dataset structured as follows:</p>
    <pre><code>d:/data2/
├── train/
│   ├── class1/
│   ├── class2/
│   └── ...
└── test/
    ├── class1/
    ├── class2/
    └── ...
    </code></pre>
    <h3>Training the Model</h3>
    <ol>
        <li>Set the paths to your dataset in the script:
            <pre><code>train_data_dir = 'd:/data2/train'
validation_data_dir = 'd:/data2/test'
            </code></pre>
        </li>
        <li>Run the training script:
            <pre><code>python train.py</code></pre>
        </li>
    </ol>
    <h2>Model Summary and Visualization</h2>
    <p>The model summary provides an overview of the architecture, including layer types, output shapes, and parameters.</p>
    <h2>Evaluation</h2>
    <p>The model is evaluated using accuracy metrics, with validation performed on a separate test dataset.</p>
    <h2>Results</h2>
    <p>The model achieves significant recognition accuracy for Nastaliq script, demonstrating the effectiveness of using both natural and synthesized datasets in training robust handwriting recognition systems.</p>
    <h2>License</h2>
    <p>This project is licensed under the MIT License - see the <a href="LICENSE">LICENSE</a> file for details.</p>
    <h2>Acknowledgments</h2>
    <p>Special thanks to the contributors and the open-source community for their support and tools.</p>
</body>
</html>
