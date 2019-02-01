from keras.models import model_from_json
import cv2 as cv
import numpy as np
from tqdm import tqdm
import pandas as pd
import os

TEST_DIR = './data/processed/test'
JSON_PATH = './model/model.json'
WEIGHTS_PATH = './model/model.h5'
IMAGE_PATH = '/home/neeraj/Documents/dr-dl/data/interim/test/3/20060523_43174_0100_PP.tif'


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
    model.compile(optimizer='adam', loss='binary_crossentropy',
                  metrics=['accuracy'])
    print("Compiled successfully")
    return model


def predict_image(JSON_PATH, WEIGHTS_PATH, image_path):
    model = load_model(JSON_PATH, WEIGHTS_PATH)
    model = compile_model(model)
    image = preprocess_image(image_path)
    result = model.predict(image)
    print(result)
    print(np.argmax(result, axis=1))


predict_image(JSON_PATH, WEIGHTS_PATH, IMAGE_PATH)
