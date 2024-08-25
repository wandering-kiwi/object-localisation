import os
import numpy as np
from PIL import Image
import pickle

def load_data(data_dir, train_ratio=0.8, cache_file='data_cache.pkl'):
    """Loads image data and corresponding object positions from a given directory,
    optionally caching the results for faster subsequent access.

    Args:
        data_dir: The directory containing the 'train' and 'test' subdirectories.
        train_ratio: The proportion of data to use for training (default: 0.8).
        cache_file: The filename for the cached data (default: 'data_cache.pkl').

    Returns:
        A tuple containing:
            - x_train: A numpy array containing the training images.
            - y_train: A numpy array containing the training object positions.
            - x_test: A numpy array containing the testing images.
            - y_test: A numpy array containing the testing object positions.
    """

    try:
        # Attempt to load cached data
        with open(cache_file, 'rb') as f:
            x_train, y_train, x_test, y_test = pickle.load(f)
            print("Loaded data from cache.")
            return x_train, y_train, x_test, y_test
    except FileNotFoundError:
        # Cache file not found, proceed with loading and caching
        pass


    # Load image and label paths
    train_image_dir = os.path.join(data_dir, 'train', 'images')
    train_label_dir = os.path.join(data_dir, 'train', 'labels')
    test_image_dir = os.path.join(data_dir, 'valid', 'images')
    test_label_dir = os.path.join(data_dir, 'valid', 'labels')
    print('all dirs found')

    train_image_files = [os.path.join(train_image_dir, f) for f in os.listdir(train_image_dir) if f.endswith('.jpg')]
    train_label_files = [os.path.join(train_label_dir, f) for f in os.listdir(train_label_dir) if f.endswith('.txt')]
    test_image_files = [os.path.join(test_image_dir, f) for f in os.listdir(test_image_dir) if f.endswith('.jpg')]
    test_label_files = [os.path.join(test_label_dir, f) for f in os.listdir(test_label_dir) if f.endswith('.txt')]
    print('created lists of dirs')

    # Ensure that image and label files are paired
    assert len(train_image_files) == len(train_label_files)
    assert len(test_image_files) == len(test_label_files)

    # Shuffle the data randomly (optional)
    # ...

    # Load images and labels
    x_train = [np.array(Image.open(image_path).convert('RGB')) for image_path in train_image_files]
    print(1)
    y_train = [parse_label(label_path) for label_path in test_label_files]
    print(2)
    x_test = [np.array(Image.open(image_path).convert('RGB')) for image_path in test_image_files]
    print(3)
    y_test = [parse_label(label_path) for label_path in test_label_files]
    print('4, all loaded')

    # Convert to numpy arrays
    x_train = np.array(x_train)
    y_train = np.array(y_train)
    x_test = np.array(x_test)
    y_test = np.array(y_test)
    print('arrayified')
    with open(cache_file, 'wb') as f:
        pickle.dump((x_train, y_train, x_test, y_test), f)
        print("Cached data.")
    print('file written')
    return x_train, y_train, x_test, y_test

# Function to parse label files (adjust as needed)
def parse_label(label_path):
    with open(label_path, 'r') as f:
        label_data = f.readline().split()
        label_data = [ float(x) for x in label_data ]
        if label_data:
            label_data.append(1.0)
        else:
            label_data.append(0.0)
        return np.array(label_data)


path = '/Users/avni_aaron/object_localisation_apples/balloon_data'
