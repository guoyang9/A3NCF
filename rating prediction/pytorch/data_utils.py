from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, sys

import numpy as np
import pandas as pd

import torch
from torch.utils.data import Dataset

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


class AttNCFDataset(Dataset):
	def __init__(self, dataset, dimension, is_training):
		"""
		Important Args:
		dataset: the name of the chosen dataset
		dimension: number of dimension of user and item embeddings
		is_training: denotes training mode or not
		"""
		self.dataset = dataset
		self.dimension = dimension
		self.is_training = is_training

		self.user_feature = read_feature(dataset, dimension, 'user')
		self.item_feature = read_feature(dataset, dimension, 'item')

		if self.is_training:
			self.data = pd.read_csv(os.path.join(
						DATA_DIR, dataset+".train.dat"), 
						sep='\t', header=None,
						names=['user', 'item', 'rating'],
						dtype={'rating': np.float32})
		else:
			self.data = pd.read_csv(os.path.join(
						DATA_DIR, dataset+".test.dat"), 
						sep='\t', header=None,
						names=['user', 'item', 'rating'],
						dtype={'rating': np.float32})

		######################## ADD FEATURES #####################

		user_feature_col, item_feature_col = [], []
		for i in range(len(self.data)):
			user_feature_col.append(
					self.user_feature[self.data['user'][i]])
			item_feature_col.append(
					self.item_feature[self.data['item'][i]])
		self.data['user_feature'] = user_feature_col
		self.data['item_feature'] = item_feature_col


	def get_size(self):
		""" get the size of users and items."""
		return len(self.user_feature), len(self.item_feature)

	def __len__(self):
		return len(self.data)

	def __getitem__(self, idx):
		sample = {'user': self.data['user'][idx],
				  'item': self.data['item'][idx],
				  'user_feature': self.data['user_feature'][idx],
				  'item_feature': self.data['item_feature'][idx],
				  'rating': self.data['rating'][idx],}

		return sample

