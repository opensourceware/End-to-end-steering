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
	resized = cv2.resize((cv2.cvtColor(img, cv2.COLOR_RGB2HSV))[:,:,1], (img.shape[1]/config.resize_factor, img.shape[0]/config.resize_factor))
	return resized

def imgread(imgpath):
	img = cv2.imread(imgpath)
	img = preprocess(img)
	#Convert to list for ease in tensor conversion (convert_to_tensor)
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
	num_unbatched = len(data)%batch_size
	row_size = len(data[-1])
	col_size = len(data[-1][0])
	if num_unbatched!=0:
		data+=[[[0]*col_size]*row_size]*(batch_size-num_unbatched)
		labels+=[0]*(batch_size-num_unbatched)
	return data, labels

def buildDataset():
	data = []
	#Only use center images (8000) to train the network. No data augmentation yet.
	files = re.findall(r"center(?:_\d+)+.jpg", ' '.join(os.listdir(config.datadir)))
	for num, file in enumerate(files):
		data.append(imgread(config.datadir+file))
	return data


def prepareDataset():
    """
    Processes raw data by doing mean subtraction and normalization. Pads and batches the data.
    """
    with open('data.pkl', 'r') as file:
        data = pickle.load(file)
    labels = ground_truth()
    batch_len = len(data) // config.batch_size
    data, labels = pad(data, labels, config.batch_size)
    # data = tf.convert_to_tensor(data)
    # labels = tf.convert_to_tensor(labels)
    # data = tf.reshape(data, [batch_len+1, config.batch_size, int(data.get_shape()[1]), int(data.get_shape()[2])])
    # labels = tf.reshape(labels, [batch_len+1, config.batch_size])
    data = np.array(data, dtype='float32')
    labels = np.array(labels, dtype='float32')
    data = np.reshape(data, [batch_len+1, config.batch_size, int(data.shape[1]), int(data.shape[2])])
    data = np.expand_dims(data, -1)
    data = ppn(data)
    labels = np.reshape(labels, [batch_len+1, config.batch_size])
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
	label = []
	with open(config.ground_truth, 'r') as csvfile:
	    f = csv.reader(csvfile)
	    for num, row in enumerate(f):
	    	if num==0:
	    		continue
	        label.append(float(row[3][1:]))
	return label

"""
data = []
for file in os.listdir(config.datadir):
	data.append(imgread(config.datadir+file))
"""
