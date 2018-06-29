from keras.models import Sequential
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Flatten
from keras.layers import Dense
import pickle

# Initialising the CNN
classifier = Sequential()
classifier.add(Conv2D(64, (3, 3), input_shape = (256, 256, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Adding a second convolutional layer
classifier.add(Conv2D(128, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
# Add a third layer
classifier.add(Conv2D(256, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))
#add a fourth layer
classifier.add(Conv2D(256, (3, 3), activation = 'relu'))
classifier.add(MaxPooling2D(pool_size = (2, 2)))

classifier.add(Flatten())
classifier.add(Dense(units = 1024, activation = 'relu'))
classifier.add(Dense(units = 384, activation = 'relu'))
classifier.add(Dense(units = 96, activation = 'relu'))
classifier.add(Dense(units = 30, activation = 'softmax'))

# Compiling the CNN
classifier.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])

# Part 2 - Fitting the CNN to the images

from keras.preprocessing.image import ImageDataGenerator

train_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

test_datagen = ImageDataGenerator(rescale = 1./255,
horizontal_flip = True,
fill_mode = "nearest",
zoom_range = 0.3,
width_shift_range = 0.3,
height_shift_range=0.3,
rotation_range=30)

training_set = train_datagen.flow_from_directory('./HE_Chal',
                                                 target_size = (256, 256),
                                                 batch_size = 8,
                                                 class_mode = 'categorical')

test_set = test_datagen.flow_from_directory('./Validation',
                                            target_size = (256,256),
                                            batch_size = 8,
                                            class_mode = 'categorical')
classifier.fit_generator(training_set,
                         steps_per_epoch = 2000,
                         epochs = 30,
                         validation_data = test_set,validation_steps = 2000)	
classifier.save_weights('classifier.h5')
classifier.predict(test_set,batch_size=8)
