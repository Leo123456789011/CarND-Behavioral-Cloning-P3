import csv
import cv2
import numpy as np
import sklearn
from sklearn.model_selection import train_test_split
import keras
from keras.models import Sequential, load_model
from keras.layers import Flatten, Dense, Lambda, ELU, Dropout
from keras.layers.convolutional import Convolution2D
from keras.layers.pooling import MaxPooling2D
from keras.layers import Cropping2D
from keras import optimizers
import random


#Function Define
def flip_image(image, measurement):
    '''
    Flip image and measurement
    '''
    flipped_image = cv2.flip(image, 1)
    flipped_measurement = float(measurement) * -1.0
    return flipped_image, flipped_measurement
    
def generator(samples, batch_size=32):
    '''
    Use generator to generate data in order to save memory
    '''

    Data_path = "./data/IMG/"

    num_samples = len(samples)
    while 1: # Loop forever so the generator never terminates
        sklearn.utils.shuffle(samples)
        for offset in range(0, num_samples, batch_size):
            batch_samples = samples[offset:offset+batch_size]

            images = []
            measurements = []
            for batch_sample in batch_samples:

                i = random.randint(0, 2)
                correction = 0.2

                if i == 0:
                    # Load images from center, left and right cameras
                    source_path = batch_sample[i]
                    #tokens = source_path.split('\\') # for windows
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    #print(filename)
                    local_path = Data_path + filename
                    image = cv2.imread(local_path)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    measurement = float(batch_sample[3])
                    # Steering adjustment for center images
                    measurement = measurement
                elif i==1:
                    # Load images from center, left and right cameras
                    source_path = batch_sample[i]
                    #tokens = source_path.split('\\') # for windows
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    #print(filename)
                    local_path = Data_path + filename
                    image = cv2.imread(local_path)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    measurement = float(batch_sample[3])
                    # Add correction for steering for left images
                    measurement = measurement+correction
                else:
                    # Load images from center, left and right cameras
                    source_path = batch_sample[i]
                    #tokens = source_path.split('\\') # for windows
                    tokens = source_path.split('/')
                    filename = tokens[-1]
                    #print(filename)
                    local_path = Data_path + filename
                    image = cv2.imread(local_path)
                    image = cv2.cvtColor(image,cv2.COLOR_BGR2RGB)
                    measurement = float(batch_sample[3])
                    # Minus correction for steering for right images
                    measurement = measurement-correction

                j = random.randint(0, 1)

                if j == 0:
                    images.append(image)
                    measurements.append(measurement)
                else:
                    flip_image(image,measurement)
                    images.append(image)
                    measurements.append(measurement)

            X_train = np.array(images)
            y_train = np.array(measurements)
            yield sklearn.utils.shuffle(X_train, y_train)


#load driving_log
lines = []
with open('./data/driving_log.csv') as csvfile:
    next(csvfile, None)
    reader = csv.reader(csvfile)
    for line in reader:
        lines.append(line)
        
#Shuffle and Split data for validation
sklearn.utils.shuffle(lines)
train_samples, validation_samples = train_test_split(lines, test_size=0.2)

# compile and train the model using the generator function
train_generator = generator(train_samples, batch_size = 8)
validation_generator = generator(validation_samples, batch_size = 8)

###Build nVidia N2N Model with Keras###

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
model.add(Lambda(lambda x: x / 255.0 - 0.5, input_shape=(160,320,3)))
model.add(Cropping2D(cropping=((70,25),(0,0))))

model.add(Convolution2D(nb_filters1, kernel_size[0], kernel_size[1],border_mode='valid',activation='elu'))
# The second conv layer will convert 16 channels into 8 channels
model.add(Convolution2D(nb_filters2, kernel_size[0], kernel_size[1],activation='elu'))
# The second conv layer will convert 8 channels into 4 channels
model.add(Convolution2D(nb_filters3, kernel_size[0], kernel_size[1],activation='elu'))
# The second conv layer will convert 4 channels into 2 channels
model.add(Convolution2D(nb_filters4, kernel_size[0], kernel_size[1],activation='elu'))
# Apply Max Pooling for each 2 x 2 pixels
model.add(MaxPooling2D(pool_size=pool_size))
# Apply dropout of 25%
model.add(Dropout(0.25))

# Flatten the matrix. The input has size of 360
model.add(Flatten())
# Input 360 Output 16
model.add(Dense(16,activation='elu'))
# Input 16 Output 16
model.add(Dense(16,activation='elu'))
# Input 16 Output 16
model.add(Dense(16,activation='elu'))
# Apply dropout of 50%
model.add(Dropout(0.5))
# Input 16 Output 1
model.add(Dense(1))

#Load Model if needed
#model = load_model('./model.h5')

#Train Model
model.compile(loss = 'mse', optimizer = 'adam')
#model.compile(optimizer=optimizers.Adam(lr=1e-04), loss='mse')
model.fit_generator(train_generator, samples_per_epoch= 
            len(train_samples)*2, validation_data=validation_generator, 
            nb_val_samples=len(validation_samples), nb_epoch= 3)

#Save Model
model.save('model.h5')
print('model saved!')