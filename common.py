import os
import random
import csv
import numpy as np
import matplotlib.pyplot as plt
import itertools


steering_correction = 0.17

# read CSV data
def readCsv(PATH_OF_CSV):
    data = []
    with open(PATH_OF_CSV) as F:
        reader = csv.reader(F)
        for i in reader:
            data.append(i) 
    print('{0} images available in training data'.format(len(data)))
    #print("Imported {0} rows from CSV".format(len(data)))
    return np.array(data)


### This function will resize the images from front, left and
### right camera to 18 x 80 and turn them into lists.
### The length of the each list will be 18 x 80 = 1440
### j = 0,1,2 corresponds to center, left, right
def load_image(csvRow, jj=0, input_shape=(18,80,3)):
    # image = plt.imread(csvRow[jj].strip())[65:135:4,0:-1:4,:]
    image = plt.imread(csvRow[jj].strip())
    image = crop_image(image)
    image_list = image.flatten().tolist()
    image_array = np.reshape(np.array(image_list), newshape=input_shape)
    image_array = image_array / 255 - 0.5
    return image_array


## get image for generator
def get_image(csvRow):
    features = ()
    labels = ()
    #0=center, 1 = left, 2 = right
    for j in range(3):
        features += (load_image(csvRow,j),)
        if j == 1:
            #add steering_correction
            labels += (float(csvRow[3]) + steering_correction,)
        elif j == 2:
            labels += (float(csvRow[3]) - steering_correction,)
        else:
            labels += (float(csvRow[3]),)
#     return np.array(features).reshape(3, 18, 80, 3), np.array(labels)
    return np.array(features), np.array(labels)


def get_image_v2(csvRow):
    features = ()
    labels = ()
    j = random.choice(range(0,2))
    features += (load_image(csvRow,j),)
    labels += (float(csvRow[3]),)
    return np.array(features), np.array(labels)


def test_get_image(CSV_train):
    random_idx = np.random.choice(len(CSV_train), 1)
    csvRow = np.squeeze(CSV_train[random_idx])
    X, y = get_image(csvRow)
    print(y)



def crop_image(image):
	return image[65:135:4,0:-1:4,:]


def p_flip(image, steering_angle):
    return np.fliplr(image), -1 * steering_angle

def random_flip(images, steering_angles):
    coin = random.choice(range(0,1))
    if coin == 0:
        features = []
        labels = []
        for image,angle in zip(images, steering_angles):
            image, angle = p_flip(image, angle)
            features.append(image)
            labels.append(angle)
        return  np.array(features), np.array(labels)
    else:
        return images, steering_angles

# test flip
def test_flip(csvData):
    angle_array = getAngleArrayFromCsvData(csvData)
    y_train_max = np.argmax(angle_array)
    X_train_max, Y_train_max = p_flip(load_image(csvData[y_train_max]), angle_array[y_train_max])
    
    plt.figure(figsize=(16,8))
    plt.subplot(1,2, 1)
    plt.axis('off')
    plt.imshow(np.squeeze(load_image(csvData[y_train_max]), axis=2))
    plt.title('Original Angle: {}'.format(angle_array[y_train_max]))
    plt.subplot(1,2, 2)
    plt.axis('off')
    plt.title('Flipped Angle: {}'.format(Y_train_max))
    plt.imshow(np.squeeze(X_train_max, axis=2))




# generate for given set of rows
NO_IMAGES = 3
def get_infite_images_generator(csvRows, batch_size=64):
    infiniteCsvRows = itertools.cycle(csvRows)
    counter = 0
    X = []
    y = []    
    while True:
        if (counter >= batch_size):
            counter = 0
            X_x = np.array(X).reshape(NO_IMAGES*len(X), 18, 80, 3)
            y_y = np.array(y).reshape(NO_IMAGES*len(y))
            X = []
            y = []
            yield X_x, y_y
        csvRow = next(infiniteCsvRows)
        features, labels = get_image(np.squeeze(csvRow))
        features, labels = random_flip(features, labels)
        X.append(features.tolist())
        y.append(labels.tolist())
        counter = counter + NO_IMAGES
        
        
def test_generator(csvRows):
    gg = get_infite_images_generator(csvRows)
    X, y = next(gg)
    print(X.shape, y.shape)


