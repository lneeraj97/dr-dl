from keras.layers import Sequential, Conv2D, MaxPooling2D, Flatten, Dense


# Initialise a classifier
classifier = Sequential()

# First layer - Conv + Pool
classifier.add(Convolution2D(32, (3, 3), input_shape=()))
