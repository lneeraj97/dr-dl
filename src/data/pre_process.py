import cv2 as cv
import os
import sys
import numpy as np

SOURCE_PATH = './data/raw/Base11'
DEST_PATH = './data/interim'


def preprocess_image(image_path):
    image = cv.imread(image_path, 0)
    image = cv.resize(image, (256, 256))
    cv.imshow('image', image)
    cv.waitKey(100)


def process_folder(folder):
    # Get current and destination paths of every image and call the pre_processing function on it
    for image in os.listdir(folder):
        # Full source path of the image
        image_path = os.path.join(folder, image)
        preprocess_image(image_path)
        # preprocess_image(image_path)
        # exit()


# def start_preprocessing():
    # Call the process_folder function on each folder
    # for folder in os.listdir(SOURCE_PATH):
    #   process_folder(os.path.join(SOURCE_PATH, folder))
    # start_preprocessing()


process_folder(SOURCE_PATH)
