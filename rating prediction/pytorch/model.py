from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import torch
import torch.nn as nn
import torch.nn.functional as F


class AttNCF(nn.Module):
	def __init__(self, user_size, item_size, 
				embed_size, dropout, is_training):
		super(AttNCF, self).__init__()
		""" 
		Important Args:
		user_size: number of users.
		item_size: number of items.
		embed_size: embedding size for both users and items.
		"""
		self.user_size = user_size
		self.item_size = item_size
		self.embed_size = embed_size
		self.dropout = dropout
		self.is_training = is_training

		self.user_embedding = nn.Embedding(
					self.user_size, self.embed_size)
		self.item_embedding = nn.Embedding(
					self.item_size, self.embed_size)

		nn.init.xavier_uniform_(self.user_embedding.weight)
		nn.init.xavier_uniform_(self.item_embedding.weight)

		self.user_fusion = nn.Sequential(
			nn.Linear(self.embed_size, self.embed_size),
			nn.ReLU())
		self.item_fusion = nn.Sequential(
			nn.Linear(self.embed_size, self.embed_size),
			nn.ReLU())

		self.att_layer1 = nn.Sequential(
			nn.Linear(2 * self.embed_size, 1),
			nn.ReLU())
		self.att_layer2 = nn.Linear(1, self.embed_size, bias=False)

		self.rating_predict = nn.Sequential(
			nn.Linear(self.embed_size, self.embed_size),
			nn.ReLU(),
			nn.Dropout(p=self.dropout),
			# nn.Linear(self.embed_size, self.embed_size),
			# nn.ReLU(),
			# nn.Dropout(p=self.dropout),
			nn.Linear(self.embed_size, 1))

		for m in self.modules():
			if isinstance(m, nn.Linear):
				nn.init.xavier_uniform_(m.weight)
				if m.bias is not None:
					m.bias.data.zero_()

	def forward(self, user_id, item_id, user_text, item_text):

		########################## INPUT #########################
		user_id_embed = self.user_embedding(user_id)
		item_id_embed = self.item_embedding(item_id)
		
		###################### FEATURE FUSION ####################
		user_embed = user_id_embed + user_text
		item_embed = item_id_embed + item_text
		user_embed = self.user_fusion(user_embed)
		item_embed = self.item_fusion(item_embed)

		################### ATTENTIVE INTERACTION ################
		feature_all = torch.cat((
					user_embed, item_embed), dim=-1)
		att_weights = self.att_layer2(self.att_layer1(feature_all))
		att_weights = F.softmax(att_weights, dim=-1)

		#################### RATING PREDICTION ###################
		interact = att_weights * user_embed * item_embed
		prediction = self.rating_predict(interact)

		return prediction.view(-1)
