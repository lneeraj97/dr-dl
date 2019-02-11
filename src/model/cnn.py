from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential, model_from_json
from keras.callbacks import ModelCheckpoint

TRAIN = 'data/processed/train'
TEST = 'data/processed/test'
JSON_FILE = './model/model.json'
HDF5_FILE = './model/model.h5'
WEIGHTS_FILE = './model/weights.h5'
EPOCHS = 1000
BATCH_SIZE = 64
STEPS_PER_EPOCH = 65
VALIDATION_STEPS = 22
POOL_SIZE = (2, 2)
KERNEL_SIZE = (3, 3)
INPUT_SHAPE = (224, 224, 3)
TARGET_SIZE = (224, 224)


def create_model():
    # Initialise a model
    model = Sequential()

    # First conv layer
    model.add(Conv2D(64, KERNEL_SIZE, input_shape=INPUT_SHAPE,
                     activation='relu', use_bias=True, strides=1, padding='same'))

    # Second conv layer
    model.add(Conv2D(64, KERNEL_SIZE, activation='relu',
                     strides=1, use_bias=True, padding='same'))
    # First pool layer
    model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=2))
    # Third conv layer
    model.add(Conv2D(128, KERNEL_SIZE, activation='relu',
                     strides=1, use_bias=True, padding='same'))
    # Fourth conv layer
    model.add(Conv2D(128, KERNEL_SIZE, activation='relu',
                     strides=1, use_bias=True, padding='same'))
    # Second pool layer
    model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=2))
    # Fifth conv layer
    model.add(Conv2D(256, KERNEL_SIZE, activation='relu',
                     strides=1, use_bias=True, padding='same'))
    # Third pool layer
    model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=2))
    # Sixth conv layer
    model.add(Conv2D(256, KERNEL_SIZE, activation='relu',
                     strides=1, use_bias=True, padding='same'))
    # Fourth pool layer
    model.add(MaxPooling2D(pool_size=POOL_SIZE, strides=2))

    # Flattening
    model.add(Flatten())

    # FC layers
    model.add(Dense(units=2048, activation='relu'))
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=4, activation='softmax'))

    return model


def init_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['categorical_accuracy'])
    return model


def train_model(model):
    filepath = WEIGHTS_FILE
    checkpoint = ModelCheckpoint(
        filepath, monitor='val_categorical_accuracy', verbose=1, save_best_only=True, mode='max')
    callbacks_list = [checkpoint]

    train_datagen = ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=True)
    train_set = train_datagen.flow_from_directory(
        TRAIN, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

    test_datagen = ImageDataGenerator(rescale=1./255)
    test_set = test_datagen.flow_from_directory(
        TEST, target_size=TARGET_SIZE, batch_size=BATCH_SIZE, class_mode='categorical')

    model.fit_generator(train_set, steps_per_epoch=STEPS_PER_EPOCH,
                        epochs=EPOCHS, validation_steps=VALIDATION_STEPS, validation_data=test_set, callbacks=callbacks_list)

    return model


def load_model():
    # Load model from JSON
    json_file = open(JSON_FILE, 'r')
    loaded_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model)
    # Load weights from h5 file
    model.load_weights(WEIGHTS_FILE)
    print("Model Loaded from Disk")
    return model


def save_model(model):
    model_json = model.to_json()
    with open(JSON_FILE, 'w') as json_file:
        json_file.write(model_json)
    # model.save_weights(HDF5_FILE)
    # print("Model saved...! Ready to go.")
