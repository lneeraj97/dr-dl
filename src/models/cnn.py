from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.models import Sequential
TRAIN = 'data/processed/train'
TEST = 'data/processed/test'


def create_model():
    # Initialise a model
    model = Sequential()

    # First conv layer
    model.add(Conv2D(32, (3, 3), input_shape=(
        512, 512, 3), activation='relu', use_bias=True))
    # First pool layer
    model.add(MaxPooling2D(pool_size=(2, 2)))
    # Second conv layer
    model.add(Conv2D(32, (3, 3), activation='relu'))
    # Second pool layer
    model.add(MaxPooling2D(pool_size=(2, 2)))

    # Flattening
    model.add(Flatten())

    # FC layers
    model.add(Dense(units=512, activation='relu'))
    model.add(Dense(units=4, activation='sigmoid'))

    return model


def init_model(model):
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    return model


def train_model(model):
    train_datagen = ImageDataGenerator(
        rescale=1./255, shear_range=0.2, zoom_range=0.2, horizontal_flip=False)
    test_datagen = ImageDataGenerator(rescale=1./255)
    train_set = train_datagen.flow_from_directory(
        TRAIN, target_size=(512, 512), batch_size=4, class_mode='categorical')
    test_set = test_datagen.flow_from_directory(
        TEST, target_size=(512, 512), batch_size=4, class_mode='categorical')
    model.fit_generator(train_set, steps_per_epoch=120,
                        epochs=5, validation_steps=2000, validation_data=test_set)

    return model


def save_model(model):
    model_json = model.to_json()
    with open('./model/model.json', 'w') as json_file:
        json_file.write(model_json)
    model.save_weights('./model/model.h5')
    print("Model saved...! Ready to go.")


my_model = create_model()
my_model = init_model(my_model)
print(my_model.summary())
trained_model = train_model(my_model)
save_model(trained_model)
