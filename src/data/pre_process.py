import cv2 as cv
import os
import sys
import numpy as np

SOURCE_PATH = './data/interim'
DEST_PATH = './data/processed'


def preprocess_image(curr_path, dest_path):
    image = cv.imread(curr_path, 0)
    """ cv.imshow('image', image)
    cv.waitKey(100) """
    image = cv.resize(image, (256, 256))


def process_folder(curr_folder):
    # Get path of the destination folder by calculating the relative path
    dest_folder = os.path.join(
        DEST_PATH, os.path.relpath(curr_folder, SOURCE_PATH))

    # Get current and destination paths of every image and call the pre_processing function on it
    for image in os.listdir(curr_folder):
        # Full source path of the image
        curr_path = os.path.join(curr_folder, image)

        # Full destination path of the image
        dest_path = os.path.join(dest_folder, image)

        # print(curr_path, dest_path)
        preprocess_image(curr_path, dest_path)
        # exit()


def start_preprocessing():
    # Call the process_folder function on each folder
    for folder in os.listdir(SOURCE_PATH):
        process_folder(os.path.join(SOURCE_PATH, folder))
        # print(os.path.join(SOURCE_PATH, folder))


start_preprocessing()
