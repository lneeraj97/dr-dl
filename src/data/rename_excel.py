import os
from tqdm import tqdm
import sys

BASE_PATH = './data/raw'

for root, dirs, files in os.walk(BASE_PATH):
    for filename in tqdm(files):
        if filename.find('.xls') > 0:
            dir_path = os.path.relpath(root, BASE_PATH)
            new_filename = os.path.join(dir_path, 'index.xls')
            os.rename(filename, new_filename)
