from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os
import sys
import time

import numpy as np 
import tensorflow as tf 
import data_utils, _model


FLAGS = tf.app.flags.FLAGS

tf.app.flags.DEFINE_integer('batch_size', 256, 
			'size of mini-batch.')
tf.app.flags.DEFINE_integer('negative_num', 5, 
			'number of negative samples.')
tf.app.flags.DEFINE_integer('embed_size', 10, 
			'the size for embedding user and item.')
tf.app.flags.DEFINE_integer('topK', 25, 
			'truncated top items.')
tf.app.flags.DEFINE_integer('epochs', 10, 
			'the number of epochs.')
tf.app.flags.DEFINE_string('dataset', 'Baby', 
			'the pre-trained dataset.')
tf.app.flags.DEFINE_string('model_dir', './AttNCF/', 
			'the dir for saving model.')
tf.app.flags.DEFINE_string('optim', 'Adam', 
			'the optimization method.')
tf.app.flags.DEFINE_string('activation', 'ReLU', 
			'the activation function.')
tf.app.flags.DEFINE_string('gpu', '0', 
			'the gpu card number.')
tf.app.flags.DEFINE_float('lr', 0.001, 
			'learning rate.')
tf.app.flags.DEFINE_float('dropout', 0.5, 
			'dropout rate.')
tf.app.flags.DEFINE_float('l2_rate', 0.001, 
			'regularize rate.')

opt_gpu = FLAGS.gpu
os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu
config = tf.ConfigProto()
config.gpu_options.allow_growth = True


def train(train_data, test_data, user_size, item_size):
	iterator = tf.data.Iterator.from_structure(
		train_data.output_types, train_data.output_shapes)

	with tf.Session(config=config) as sess:
		######################## CREATE MODEL #######################

		model = _model.AttNCF(user_size, item_size, FLAGS.embed_size,
				iterator, FLAGS.activation, FLAGS.lr, FLAGS.optim,
				FLAGS.l2_rate, FLAGS.dropout, is_training=True)
		model.build()

		ckpt = tf.train.get_checkpoint_state(FLAGS.model_dir)
		if ckpt:
			print("Reading model parameters from %s".format(ckpt.model_checkpoint_path))
			model.saver.restore(sess, ckpt.model_checkpoint_path)
		else:
			print("Creating model with fresh parameters.")
			sess.run(tf.global_variables_initializer())

		########################## TRAINING #########################

		count = 0
		for epoch in range(FLAGS.epochs):
			sess.run(model.iterator.make_initializer(train_data))
			model.is_training = True
			# model.get_data()
			start_time = time.time()

			try:
				while True:
					model.step(sess, count)
					count += 1
			except tf.errors.OutOfRangeError:
				pass
				
		######################### EVALUATION ########################

			sess.run(model.iterator.make_initializer(test_data))
			model.is_training = False
			# model.get_data()
			RMSE = np.array([], dtype=np.float32)

			try:
				while True:
					se = model.step(sess, None)
					RMSE = np.append(RMSE, se)
			except tf.errors.OutOfRangeError:
				print("\nRMSE is %.3f".format((np.sqrt(RMSE.mean()))))
				print("Epoch %d ".format(epoch) + "Took: " + time.strftime("%H: %M: %S", 
							time.gmtime(time.time() - start_time)))

		######################### SAVE MODEL ########################

		# checkpoint_path = os.path.join(FLAGS.model_dir, "AttNCF.ckpt")
		# model.saver.save(sess, checkpoint_path)

def main(argv=None):
	train_data, user_size, item_size = data_utils.read_data(
		FLAGS.dataset, FLAGS.embed_size, FLAGS.batch_size, True)
	test_data, user_size, item_size = data_utils.read_data(
		FLAGS.dataset, FLAGS.embed_size, FLAGS.batch_size, False)

	train(train_data, test_data, user_size, item_size)
	

if __name__ == '__main__':
	tf.app.run()
