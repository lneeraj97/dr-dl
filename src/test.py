from keras.models import model_from_json, Sequential
import cv2 as cv
import numpy as np
from tqdm import tqdm
import pandas as pd
import os
from keras.layers import Conv2D, MaxPooling2D, Flatten, Dense
from keras.preprocessing.image import ImageDataGenerator
from keras.callbacks import ModelCheckpoint
import keras

TEST_DIR = './data/processed/test'
JSON_PATH = './model/model.json'
WEIGHTS_PATH = './model/weights.h5'


def preprocess_image(image_path):
    # Open the image
    original_image = cv.imread(image_path, 1)
    # Resize image to 256x256
    original_image = cv.resize(original_image, (224, 224))
    # Extract green channel from the image
    # NOTE: OPENCV USES BGR COLOR ORDER
    image = original_image[:, :, 1]
    # Apply median blur to remove salt and pepper noise
    image = cv.medianBlur(image, 3)

    # Apply CLAHE thresholding
    clahe = cv.createCLAHE(clipLimit=10.0, tileGridSize=(4, 4))
    image = clahe.apply(image)
    cv.imwrite(image_path, image)
    image = cv.imread(image_path, 1)
    print(image.shape)
    image = image.reshape(1, 224, 224, 3)
    return image


def load_model(JSON_PATH, WEIGHTS_PATH):
    # Load model from JSON
    json_file = open(JSON_PATH, 'r')
    loaded_model = json_file.read()
    json_file.close()
    model = model_from_json(loaded_model)
    # Load weights from h5 file
    model.load_weights(WEIGHTS_PATH)
    print("Model Loaded from Disk")
    return model


def compile_model(model):
    model.compile(optimizer='adam', loss='categorical_crossentropy',
                  metrics=['accuracy'])
    print("Compiled successfully")
    return model


def predict_image(JSON_PATH, WEIGHTS_PATH, TEST_DIR):
    model = load_model(JSON_PATH, WEIGHTS_PATH)
    model = compile_model(model)
    datagen = ImageDataGenerator(rescale=1./255)
    generator = datagen.flow_from_directory(TEST_DIR, target_size=(
        224, 224), batch_size=64, class_mode='categorical', shuffle=False)
    results = model.evaluate_generator(generator, max_queue_size=10, steps=65,
                                       workers=1, use_multiprocessing=False, verbose=1)
    print(results)


predict_image(JSON_PATH, WEIGHTS_PATH, TEST_DIR)
