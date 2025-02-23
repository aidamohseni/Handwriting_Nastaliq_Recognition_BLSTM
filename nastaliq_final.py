# -*- coding: utf-8 -*-
"""
Created on Thu Mar  1 23:46:26 2018

@author: aida mohseni

"""
import keras
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D
from keras.layers import Activation, Dropout, Flatten, Dense, TimeDistributed
from keras import backend as K
from keras.layers import Bidirectional, LSTM

# dimensions of our images.
img_width, img_height = 128, 128
pool_size = 2
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

############################################
model.add(TimeDistributed(Flatten()))
model.add(Bidirectional(LSTM(32)))
model.add(Dropout(0.1))
model.add(Dense(1))
model.add(Activation('relu'))
###########################################

model.add(Flatten())
model.add(Dense(64))
model.add(Activation('relu'))
model.add(Dropout(0.5))
model.add(Dense(1))
model.add(Activation('sigmoid'))
###########################################


model.compile(loss='binary_crossentropy',
              # optimizer='rmsprop',
              optimizer='adam',
              metrics=['accuracy'])

# , keras.metrics.Precision(), keras.metrics.Recall()
model.summary()

# this is the augmentation configuration we will use for training
train_datagen = ImageDataGenerator(
    rescale=1. / 255,
    shear_range=0.2,
    zoom_range=0.2,
    horizontal_flip=True)

# this is the augmentation configuration we will use for testing:
# only rescaling
test_datagen = ImageDataGenerator(rescale=1. / 255)

train_generator = train_datagen.flow_from_directory(
    train_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size,
    class_mode='categorical')
# class_mode='binary')

validation_generator = test_datagen.flow_from_directory(
    validation_data_dir,
    target_size=(img_width, img_height),
    batch_size=batch_size, class_mode='categorical')

model.fit_generator(
    train_generator,
    steps_per_epoch=nb_train_samples // batch_size,
    epochs=epochs,
    validation_data=validation_generator,
    validation_steps=nb_validation_samples // batch_size)

model.save_weights('first_try.h5')
