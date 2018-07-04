
from keras.models import Sequential
from keras.layers import Convolution2D
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense


model = Sequential()

#1st Layer
model.add(Conv2D(64, (3, 3), input_shape=(64, 64, 3),activation = 'relu',data_format='channels_first'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#2nd Layer
model.add(Conv2D(128, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

#3rd Layer
model.add(Conv2D(256, (3, 3), activation = 'relu'))
model.add(MaxPooling2D(pool_size=(2, 2)))

##4th Layer
#model.add(Conv2D(256, (3, 3), activation = 'relu'))
#model.add(MaxPooling2D(pool_size=(2, 2)))

#Flattening
model.add(Flatten())

#Fully Connected
model.add(Dense(output_dim = 1024, activation = 'relu'))
model.add(Dense(output_dim = 7, activation = 'sigmoid'))

model.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
                                   shear_range = 0.2,
                                   zoom_range = 0.2,
                                   horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('DataSet/Train',
                                                 target_size = (64, 64),
                                                 batch_size = 2,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('DataSet/Test',
                                            target_size = (64, 64),
                                            batch_size = 2,
                                            class_mode = 'categorical')

model.fit_generator(training_set,
                         steps_per_epoch = 300,
                         epochs = 30,
                         validation_data = test_set,
                         validation_steps = 300,
                         use_multiprocessing=True
                         )
                         
                         
model.save('better.h5')            


import cv2
import numpy as np

from keras.preprocessing.image import load_img
from keras.preprocessing.image import img_to_array


img  = load_img('Data_Set/Train/Sad/images12.jpeg',target_size = (32,32))

img = img_to_array(img)
img = img.reshape((1, img.shape[0], img.shape[1], img.shape[2]))


g = model.predict(img)