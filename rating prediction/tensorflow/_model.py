from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf 


class AttNCF(object):
	def __init__(self, user_size, item_size, embed_size, iterator,
		activation_func, lr, optim, regularizer, dropout, is_training):
		""" 
		Important Args:
		user_size: number of users.
		item_size: number of items.
		embed_size: embedding size for both users and items.
		regularizer: lr regularizer rate.
		"""
		self.user_size = user_size
		self.item_size = item_size
		self.embed_size = embed_size
		self.iterator = iterator
		self.activation_func = activation_func
		self.lr = lr
		self.optim = optim
		self.regularizer_rate = regularizer
		self.dropout = dropout
		self.is_training = is_training

	def get_data(self):
		""" Obtain the input data from tensorflow iterator."""
		sample = self.iterator.get_next()

		self.user = sample['user']
		self.item = sample['item']
		self.user_feature = sample['user_feature']
		self.item_feature = sample['item_feature']
		self.rating = sample['rating']
		
	def inference(self):
		self.regularizer = tf.contrib.layers.l2_regularizer(
							self.regularizer_rate)
		if self.activation_func == 'ReLU':
			self.activation_func = tf.nn.relu
		elif self.activation_func == 'Leaky_ReLU':
			self.activation_func = tf.nn.leaky_relu
		elif self.activation_func == 'ELU':
			self.activation_func = tf.nn.elu

		if self.optim == 'SGD':
			self.optimizer = tf.train.GradientDescentOptimizer(self.lr, 
									name='SGD')
		elif self.optim == 'RMSProp':
			self.optimizer = tf.train.RMSPropOptimizer(self.lr, decay=0.9, 
							momentum=0.0, name='RMSProp')
		elif self.optim == 'Adam':
			self.optimizer = tf.train.AdamOptimizer(self.lr, name='Adam')

	def create_model(self):
		""" Create model from scratch. """
		with tf.name_scope("input"):
			self.user_embedding = tf.get_variable("user_embed", 
				[self.user_size, self.embed_size], dtype=tf.float32)
			self.item_embedding = tf.get_variable("item_embed", 
				[self.item_size, self.embed_size], dtype=tf.float32)
			self.user_embed = tf.nn.embedding_lookup(
							self.user_embedding, self.user)
			self.item_embed = tf.nn.embedding_lookup(
							self.user_embedding, self.item)
		with tf.name_scope("fusion"):
			self.user_fusion_add = self.user_embed + self.user_feature
			self.item_fusion_add = self.item_embed + self.item_feature

			self.user_fusion = tf.layers.dense(inputs=self.user_fusion_add,
								units=self.embed_size,
								activation=self.activation_func,
								kernel_regularizer=self.regularizer,
								name='user_fusion')
			self.item_fusion = tf.layers.dense(inputs=self.item_fusion_add,
								units=self.embed_size,
								activation=self.activation_func,
								kernel_regularizer=self.regularizer,
								name='item_fusion')

		with tf.name_scope("attention"):
			self.feature_all = tf.concat([
						self.user_fusion, self.item_fusion], -1)
			self.att_layer1 = tf.layers.dense(inputs=self.feature_all,
								units=1,
								activation=self.activation_func,
								kernel_regularizer=self.regularizer,
								name='att_layer1')
			self.att_layer2 = tf.layers.dense(inputs=self.att_layer1,
								units=self.embed_size,
								activation=self.activation_func,
								kernel_regularizer=self.regularizer,
								name='att_layer2')
			self.att_weights = tf.nn.softmax(self.att_layer2, 
								axis=-1, name='att_softmax')

		with tf.name_scope("prediction"):
			self.interact = self.att_weights*self.user_fusion*self.item_fusion
			self.interact1 = tf.layers.dense(inputs=self.interact,
								units=self.embed_size,
								activation=self.activation_func,
								kernel_regularizer=self.regularizer,
								name='interact1')
			self.interact1 = tf.nn.dropout(self.interact1, self.dropout)
			self.prediction = tf.layers.dense(inputs=self.interact,
								units=1,
								activation=None,
								kernel_regularizer=self.regularizer,
								name='prediction')
			self.prediction = tf.reshape(self.prediction, [-1])

	def loss_func(self):
		reg = tf.get_collection(tf.GraphKeys.REGULARIZATION_LOSSES)
		reg_loss = tf.contrib.layers.apply_regularization(
								self.regularizer, reg)
		mse_loss = tf.losses.mean_squared_error(self.rating, self.prediction)

		self.loss = reg_loss + mse_loss

	def optimization(self):
		with tf.name_scope("optimization"):
			self.optim = self.optimizer.minimize(self.loss)

	def eval(self):
		""" Evaluate each sample."""
		self.se = tf.square(self.rating - self.prediction)

	def summary(self):
		""" Create summaries to write on tensorboard. """
		self.writer = tf.summary.FileWriter(
				'./graphs/AttNCF', tf.get_default_graph())
		with tf.name_scope("summaries"):
			tf.summary.scalar('loss', self.loss)
			self.summary_op = tf.summary.merge_all()

	def build(self):
		self.get_data()
		self.inference()
		self.create_model()
		self.loss_func()
		self.optimization()
		self.eval()
		self.summary()
		self.saver = tf.train.Saver(tf.global_variables())

	def step(self, sess, step):
		""" Train the model step by step. """
		if self.is_training:
			loss, optim, summaries = sess.run(
					[self.loss, self.optim, self.summary_op])
			self.writer.add_summary(summaries, global_step=step)
		else:
			se = sess.run([self.se])[0]

			return se
