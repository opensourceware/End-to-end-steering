import config
import tensorflow as tf
import numpy as np


class Model:
    def __init__(self, img_size):

        self.inp = tf.placeholder(tf.float32,
                                  shape=(config.batch_size, img_size[0], img_size[1], config.input_channels))
        xavier_init = 1.0 / np.sqrt(config.conv1_filter_size * config.conv1_filter_size * config.input_channels)
        self.filter1 = tf.Variable(tf.truncated_normal([config.conv1_filter_size, config.conv1_filter_size,
                                                        config.input_channels, config.conv1_channels], mean=0.0,
                                                       stddev=xavier_init), name='convlayer1')
        self.conv1 = tf.nn.conv2d(self.inp, self.filter1, padding='SAME', strides=[1, 1, 1, 1])
        if config.batch_size == 1:
            self.relu1 = tf.nn.relu(self.conv1)
        else:
            self.mean1, self.var1 = tf.nn.moments(self.conv1, axes=[0])
            self.bn1 = tf.nn.batch_normalization(self.conv1, self.mean1, self.var1, None, None, 0.001)
            self.relu1 = tf.nn.relu(self.bn1)
        self.maxpool1 = tf.nn.max_pool(self.relu1, ksize=[1, config.ksize, config.ksize, 1],
                                       strides=[1, config.stride, config.stride, 1], padding='SAME')

        self.filter2 = tf.Variable(tf.truncated_normal([config.conv2_filter_size, config.conv2_filter_size,
                                                        config.conv1_channels, config.output_channels], mean=0.0,
                                                       stddev=xavier_init), name='convlayer2')
        self.conv2 = tf.nn.conv2d(self.maxpool1, self.filter2, padding='SAME', strides=[1, 3, 3, 1])
        if config.batch_size == 1:
            self.relu2 = tf.nn.relu(self.conv2)
        else:
            self.mean2, self.var2 = tf.nn.moments(self.conv2, axes=[0])
            self.bn2 = tf.nn.batch_normalization(self.conv2, self.mean2, self.var2, None, None, 0.001)
            self.relu2 = tf.nn.relu(self.bn2)
        self.maxpool2 = tf.nn.max_pool(self.relu2, ksize=[1, config.ksize, config.ksize, 1],
                                       strides=[1, config.stride, config.stride, 1], padding='SAME')
        #shape = self.maxpool2.get_shape()
        #fan_in = shape[1]*shape[2]*shape[3]
        self.final_layer = tf.reshape(self.maxpool2, shape=[config.batch_size, -1])
        fan_in = int(self.final_layer.get_shape()[1])
        self.weights = tf.Variable(tf.random_normal((fan_in, 1),
                                                    stddev=1.0 / fan_in, dtype=tf.float32), name="weights",
                                   trainable=True)
        self.bias = tf.Variable(tf.zeros(1, dtype=tf.float32), name="biases", trainable=True)

        self.label = tf.placeholder(tf.float32, shape=[None])
        self.pred = tf.add(tf.matmul(self.final_layer, self.weights), self.bias)
        self.loss = tf.reduce_mean(tf.square(self.pred - self.label))
        self.learning_rate = tf.placeholder(tf.float64, shape=None)
        self.optimizer = tf.train.AdagradOptimizer(self.learning_rate)
        self.train_op = self.optimizer.minimize(self.loss)

    def assign_lr(self, lr):
        tf.assign(self.learning_rate, lr)
