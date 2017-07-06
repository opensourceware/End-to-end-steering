import model, preprocessing
import config
import tensorflow as tf
import numpy as np

def main():
	dataset, angles = preprocessing.prepareDataset()
	batch_len = int(dataset.shape[0])
	img_size = (int(dataset.shape[2]), int(dataset.shape[3]))

	network = model.Model(img_size)

	sess = tf.Session()
	init_op = tf.global_variables_initializer()
	sess.run(init_op)

	epoch = range(10)
	for _ in epoch:
		for i in range(batch_len-1):
			sess.run(network.train_op,
			 feed_dict={network.inp:dataset[i], 
				network.label:angles[i],
				network.learning_rate:0.005})
			print ("Loss:", sess.run(network.loss,
			 feed_dict={network.inp:dataset[i], 
				network.label:angles[i]}))


if __name__ == "__main__":
	main()


