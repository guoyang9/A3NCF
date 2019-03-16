from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

import numpy as np
import pandas as pd

import tensorflow as tf

DATA_DIR = "../data/dataset/"
LDA_DIR = "../data/lda_data/"


def read_feature(dataset, dimension, name):
	""" read user and item features into dictionary."""
	entity_feature = {}
	with open(os.path.join(LDA_DIR, dataset+"."+str(
			dimension)+"."+name+".theta")) as entity_file:
		for entity_info in entity_file:
			line = entity_info.strip().split(',')
			entity = []
			for i in line[1:]:
				entity.append(float(i))
			entity_feature[int(line[0])] = np.array(
						entity, dtype=np.float32)

	return entity_feature

def read_data(dataset, dimension, batch_size, is_training):
	user_feature = read_feature(dataset, dimension, 'user')
	item_feature = read_feature(dataset, dimension, 'item')
	user_size = len(user_feature)
	item_size = len(item_feature)

	if is_training:
		data = pd.read_csv(os.path.join(
					DATA_DIR, dataset+".train.dat"), 
					sep='\t', header=None,
					names=['user', 'item', 'rating'],
					dtype={'rating': np.float32})
	else:
		data = pd.read_csv(os.path.join(
					DATA_DIR, dataset+".test.dat"), 
					sep='\t', header=None,
					names=['user', 'item', 'rating'],
					dtype={'rating': np.float32})

	######################## ADD FEATURES #####################

	user_feature_col, item_feature_col = [], []
	for i in range(len(data)):
		user_feature_col.append(
				user_feature[data['user'][i]])
		item_feature_col.append(
				item_feature[data['item'][i]])
	# data['user_feature'] = user_feature_col
	# data['item_feature'] = item_feature_col

	data_dict = dict([('user', data['user'].values),
				('item', data['item'].values),
				('rating', data['rating'].values),
				('user_feature', np.array(user_feature_col)),
				('item_feature', np.array(item_feature_col))])
	dataset = tf.data.Dataset.from_tensor_slices(data_dict)
	if is_training:
		dataset = dataset.shuffle(10000)
	dataset = dataset.batch(batch_size)

	return dataset, user_size, item_size
