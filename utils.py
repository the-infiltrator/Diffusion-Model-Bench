import os
import glob
from concurrent.futures import ProcessPoolExecutor
from PIL import Image
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler


def find_latest_file(pattern):
    """Find the most recent file that matches a given pattern."""

    # Get a list of all files that match the pattern
    files = glob.glob(pattern)

    # Check that at least one file was found
    if not files:
        return None

    # Find the most recent file
    latest_file = max(files, key=os.path.getctime)

    return latest_file


def load_and_scale_image(image_path, size=512):
    """Load and scale an image from a given path."""
    img = Image.open(image_path)
    width, height = img.size
    aspect_ratio = width / height

    if width < height:
        new_width = size
        new_height = round(size / aspect_ratio)
    else:
        new_height = size
        new_width = round(size * aspect_ratio)

    img = img.resize((new_width, new_height), Image.ANTIALIAS)
    return os.path.basename(image_path), img


def load_images_from_directory_parallel(image_dir, max_workers=4):
    """Load and scale images from a given directory into a list in a parallel manner."""
    image_paths = [os.path.join(image_dir, image_name)
                   for image_name in os.listdir(image_dir)]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        images = list(executor.map(load_and_scale_image, image_paths))
    return images


def load_model_names(model_file):
    """Load model names from a file."""
    with open(model_file, 'r') as f:
        models = [line.strip() for line in f]
    return models
