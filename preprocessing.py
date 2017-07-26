import cv2
import csv
import os, re, pickle
import config
import tensorflow as tf
import numpy as np


def preprocess(img):
    """
	#TODO: Experiment with the colour channels
	Downscale the image by a factor of 3 and change the color channel from RGB to HSV and use only the S channel.
	Args
	img: a numpy array of 160X320X3
	Returns
	resized: a numpy array of 106X53
	"""
    resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV)),
                         (img.shape[1] / config.resize_factor, img.shape[0] / config.resize_factor))
    return resized


def imgread(imgpath):
    img = cv2.imread(imgpath)
    img = preprocess(img)
    # Convert to list for ease in tensor conversion (convert_to_tensor)
    return img.tolist()


def pad(data, labels, batch_size):
    """
	Pads the list of images and labels with zeros to complete the batch_size of the last batch.
	Args
	data: a list of images
	labels: a list of steering angles
	batch_size: size of the batch
	Returns: 
	data: a list of batched images
	labels: a list of steering angles
	"""
    num_unbatched = len(data) % batch_size
    row_size = len(data[-1])
    col_size = len(data[-1][0])
    if num_unbatched != 0:
        data += [[[[0] * config.input_channels] * col_size] * row_size] * (batch_size - num_unbatched)
        labels += [0] * (batch_size - num_unbatched)
    return data, labels


def buildDataset():
    """
	Build the dataset by augmenting data with images from left and right camera.
	Augmented dataset is dumped in a pickle core.
	"""
    data = []
    # Only use center images (8000) to train the network. No data augmentation yet.
    files_center = re.findall(r"center(?:_\d+)+.jpg", ' '.join(os.listdir(config.datadir)))
    files_left = re.findall(r"left(?:_\d+)+.jpg", ' '.join(os.listdir(config.datadir)))
    files_right = re.findall(r"right(?:_\d+)+.jpg", ' '.join(os.listdir(config.datadir)))
    files = files_center + files_left + files_right
    for num, file in enumerate(files):
        data.append(imgread(config.datadir + file))
    with open("data-aug.pkl", "w") as f:
        pickle.dump(data, f)


def load_data():
    with open('data-aug.pkl', 'r') as file:
        data = pickle.load(file)
    return data


def prepareDataset():
    """
    Processes raw data by doing mean subtraction and normalization. Pads and batches the data.
    """
    data = load_data()
    labels = ground_truth()
    batch_len = len(data) // config.batch_size
    if config.batch_size != 1:
        batch_len += 1
    data, labels = pad(data, labels, config.batch_size)
    # data = tf.convert_to_tensor(data)
    # labels = tf.convert_to_tensor(labels)
    # data = tf.reshape(data, [batch_len+1, config.batch_size, int(data.get_shape()[1]), int(data.get_shape()[2])])
    # labels = tf.reshape(labels, [batch_len+1, config.batch_size])
    data = np.array(data, dtype='float32')
    labels = np.array(labels, dtype='float32')
    if config.input_channels != 1:
        data = np.reshape(data, [batch_len, config.batch_size, int(data.shape[1]), int(data.shape[2]), int(data.shape[3])])
    else:
        data = np.reshape(data, [batch_len, config.batch_size, int(data.shape[1]), int(data.shape[2])])
        data = np.expand_dims(data, -1)
    data = ppn(data)
    labels = np.reshape(labels, [batch_len, config.batch_size])
    return data, labels


def ppn(data):
    """
    ppn stands for Per-pixel normalization. Calculates the per pixel mean for separate color channels
    and normalizes the pixel values from -1 to 1.
    """
    data -= np.mean(data, (0, 1))
    data /= np.std(data, (0, 1))
    return data


def ground_truth():
    """
    Left and right imae labels are corrected by a value of 0.25 due to camera orientation.
    :return: Lavels for the image dataset.
    """
    label_center = []
    label_left = []
    label_right = []
    with open(config.ground_truth, 'r') as csvfile:
        f = csv.reader(csvfile)
        for num, row in enumerate(f):
            if num == 0:
                continue
            label_center.append(float(row[3][1:]))
    for n, l in enumerate(label_center):
        label_left.append(l + 0.25)
    for n, l in enumerate(label_center):
        label_right.append(l - 0.25)
    label = label_center + label_left + label_right
    return label


"""
def ground_truth():
    label = []
    with open(config.ground_truth, 'r') as csvfile:
        f = csv.reader(csvfile)
        for num, row in enumerate(f):
            if num==0:
                continue
            label.append(float(row[3][1:]))
    return label
"""

"""
data = []
for file in os.listdir(config.datadir):
	data.append(imgread(config.datadir+file))
"""
