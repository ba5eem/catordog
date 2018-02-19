from keras.models import Sequential
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator

classifier = Sequential()

# Convolution Step
classifier.add(Conv2D(32, (3,3), input_shape = (64,64,3), activation='relu'))
# 32 the number of filters
# (3,3) shape of each filter
# (64,64) => resolution size
# (#,#, 3) => stands for RGB
# relu => rectifier function 0 or 1

# Pooling Step
classifier.add(MaxPooling2D(pool_size = (2,2)))
# (2,2) matrix


# Flattening Step
classifier.add(Flatten())



# Full Connection Step
classifier.add(Dense(units = 128, activation = 'relu'))

# Output Layer

classifier.add(Dense(units = 1, activation = 'sigmoid'))


# Compile CNN Model

classifier.compile(optimizer = 'adam', loss = 'binary_crossentropy', metrics = ['accuracy'])

train_datagen = ImageDataGenerator(rescale = 1./255,
	shear_range = 0.2,
	zoom_range = 0.2,
	horizontal_flip = True)

test_datagen = ImageDataGenerator(rescale = 1./255)

training_set = train_datagen.flow_from_directory('training_set',
	target_size = (64,64),
	batch_size = 32,
	class_mode = 'binary')

test_set = test_datagen.flow_from_directory('test_set',
	target_size = (64,64),
	batch_size = 32,
	class_mode = 'binary')

classifier.fit_generator(training_set,
	steps_per_epoch = 8000, #number of training images
	epochs = 25, # 1 epoch is a single step in training a NN
	validation_data = test_set,
	validation_steps = 2000)

classifier.save('last_brain.h5')

# Making new predictions
import numpy as np 
from keras.preprocessing import image 

test_image = image.load_img('dataset/single_prediction/cat_or_dog_1.jpg',
	target_size = (64,64))
test_image = np.expand_dims(test_image, axis = 0)
result = classifier.predict(test_image)
training_set.class_indices

if result[0][0] == 1:
	prediction = 'dog'
else:
	prediction = 'cat'



