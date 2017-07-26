import model, preprocessing
import config
import tensorflow as tf
import numpy as np


def loss():
	loss = 0
	for num, img in enumerate(dataset):
		pred = sess.run(network.pred, feed_dict={network.inp:dataset[num]})
		actual = angles[num]
		#print np.sum(np.square(pred-actual))
		loss += np.sum(np.square(pred-actual))
	return loss


dataset, angles = preprocessing.prepareDataset()
batch_len = int(dataset.shape[0])
img_size = (int(dataset.shape[2]), int(dataset.shape[3]))

network = model.Model(img_size)

sess = tf.Session()
init_op = tf.global_variables_initializer()
sess.run(init_op)

epoch = range(50)
for e in epoch:
	for i in range(batch_len-1):
		sess.run(network.train_op,
		 feed_dict={network.inp:dataset[i], 
			network.label:angles[i],
			network.learning_rate:0.005})
	print "epoch number "+str(e)
	print ("Epoch Loss: ", str(loss()))

sess.run(network.pred, feed_dict={network.inp:dataset[i]})
sess.run(network.filter1)
sess.run(network.filter2)
sess.run(network.weights)