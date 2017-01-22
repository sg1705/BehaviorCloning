#imports
from PIL import Image
import numpy as np
import os
import random
import csv
import matplotlib.pyplot as plt
import tqdm
from sklearn.model_selection import train_test_split
from sklearn.utils import shuffle

import tensorflow as tf
import math
import itertools
import json

# Keras imports
from tqdm import tqdm
from keras.layers import Conv2D, Flatten
from keras.layers.core import Dropout
from keras.layers.pooling import MaxPooling2D
from keras.models import Sequential
from keras.layers import Dense, Input, Activation
from keras.optimizers import SGD, Adam, RMSprop
from keras.utils import np_utils

#all common methods
import common

#constants
PATH_OF_CSV = 'driving_log.csv'


input_shape = (18, 80, 3)
# 10 degrees = 0.17 radian
steering_correction = 0.17

###################################
##
##  1. Read Data from CSV 
##
###################################
csvData = common.readCsv(PATH_OF_CSV)



###################################
##
##  2. Split the data into 
##  training/validation/testing sets here.
##
###################################
CSV_train, CSV_test  = train_test_split(
    csvData,test_size=0.10, random_state=42)

print('Shape of training csv {}', CSV_train.shape)
print('Shape of test csv {}', CSV_test.shape)


###################################
##
##  3. Shuffling training data
##
###################################
CSV_train = shuffle(CSV_train)

# Split randomized datasets for training and validation
CSV_train, CSV_validation = train_test_split(
    CSV_train,
    test_size=0.10,
    random_state=434339)

print('Shape of  train csv {}', CSV_train.shape)
print('Shape of validation csv {}', CSV_validation.shape)


###################################
##
##  4. Test Image Generator
##
###################################
common.test_get_image(csvData)
common.test_generator(csvData)


###################################
##
##  5. Create Model
##
###################################

N_CLASSES = 1 # The output is a single digit: a steering angle

BATCH_SIZE = 64 # The lower the better
EPOCHS = 5 # The higher the better
LEARNING_RATE = 0.0015

# has to be multiple of batch_size and 3
no_samples_per_epoch = 660
validation_samples = 320
test_samples = 303


# number of convolutional filters to use
nb_filters1 = 16
nb_filters2 = 8
nb_filters3 = 4
nb_filters4 = 2

# size of pooling area for max pooling
pool_size = (2, 2)

# convolution kernel size
kernel_size = (3, 3)

# Initiating the model
model = Sequential()

# Starting with the convolutional layer
# The first layer will turn 1 channel into 16 channels
model.add(Conv2D(nb_filters1, kernel_size[0], kernel_size[1],
                        border_mode='valid',
                        input_shape=input_shape))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 16 channels into 8 channels
model.add(Conv2D(nb_filters2, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 8 channels into 4 channels
model.add(Conv2D(nb_filters3, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# The second conv layer will convert 4 channels into 2 channels
model.add(Conv2D(nb_filters4, kernel_size[0], kernel_size[1]))
# Applying ReLU
model.add(Activation('relu'))
# Apply Max Pooling for each 2 x 2 pixels
model.add(MaxPooling2D(pool_size=pool_size))
# Apply dropout of 25%
model.add(Dropout(0.25))

# Flatten the matrix. The input has size of 360
model.add(Flatten())
# Input 360 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Input 16 Output 16
model.add(Dense(16))
# Applying ReLU
model.add(Activation('relu'))
# Apply dropout of 50%
model.add(Dropout(0.5))
# Input 16 Output 1
model.add(Dense(N_CLASSES))

model.summary()



model.compile(loss='mean_squared_error',
              optimizer=Adam(LEARNING_RATE),
              metrics=['accuracy'])


###################################
##
##  6. Create 3 generators
##  training, validation, testing
##
###################################

train_gen = common.get_infite_images_generator(CSV_train,batch_size=64)
validation_gen = common.get_infite_images_generator(CSV_validation, batch_size=32)
test_gen = common.get_infite_images_generator(CSV_test, batch_size=32)


###################################
##
##  7. Train Model
##
###################################
history = model.fit_generator(train_gen, 
                    nb_epoch=EPOCHS,
                    samples_per_epoch=no_samples_per_epoch,
                    nb_val_samples=validation_samples,
                    validation_data=validation_gen,
                    verbose=1)


score = model.evaluate_generator(test_gen, test_samples)
print('Test score:', score[0])
print('Test accuracy:', score[1])


###################################
##
##  8. Save Model
##
###################################
json_string = model.to_json()

with open('model.json', 'w') as outfile:
    json.dump(json_string, outfile)
    # save weights
    model.save_weights('./model.h5')
    print("Saved")




