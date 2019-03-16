from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import os, time
import argparse

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from tensorboardX import SummaryWriter

from data_utils import AttNCFDataset
from model import AttNCF
import evaluation


def main():
	parser = argparse.ArgumentParser()

	parser.add_argument("--dataset", default='Baby', type=str,
				help="choose dataset to process.")
	parser.add_argument("--embed_size", default=20, type=int,
				help="the final embedding size.")
	parser.add_argument("--lr", default=0.001, type=float,
				help="the learning rate for optimization method.")
	parser.add_argument("--dropout", default=0.5, type=float,
				help="the dropout rate.")
	parser.add_argument("--decay", default=0.0001, type=float,
				help="the weight decay rate.")
	parser.add_argument("--neg_number", default=5, type=int,
				help="negative numbers for training the triplet model.")
	parser.add_argument("--batch_size", default=256, type=int,
				help="batch size for training.")
	parser.add_argument("--top_k", default=25, type=int,
				help="topk rank items for evaluating.")
	parser.add_argument("--gpu", default='0', type=str,
				help="choose the gpu card number.")

	FLAGS = parser.parse_args()

	writer = SummaryWriter() # For visualization

	opt_gpu = FLAGS.gpu
	os.environ["CUDA_VISIBLE_DEVICES"] = opt_gpu

	############################# PREPARE DATASET ##########################

	data_train = AttNCFDataset(
				FLAGS.dataset, FLAGS.embed_size, is_training=True)
	data_test = AttNCFDataset(
				FLAGS.dataset, FLAGS.embed_size, is_training=False)
	user_size, item_size = data_train.get_size()

	dataloader_train = DataLoader(data_train, 
			batch_size=FLAGS.batch_size, shuffle=True, num_workers=4)
	dataloader_test = DataLoader(data_test, 
			batch_size=FLAGS.batch_size, shuffle=False)
	
	############################## CREATE MODEL ###########################

	model = AttNCF(user_size, item_size, 
					FLAGS.embed_size, FLAGS.dropout, is_training=True)
	model.cuda()
	optimizer = torch.optim.Adam(model.parameters(), 
						lr=FLAGS.lr, weight_decay=FLAGS.decay)
	criterion = nn.MSELoss()
	
	print("Start training......\n")
	for epoch in range(10):
		model.is_training = True
		model.train() 
		start_time = time.time()

		for idx, batch_data in enumerate(dataloader_train):
			user = batch_data['user'].cuda()
			item = batch_data['item'].cuda()
			user_feature = batch_data['user_feature'].cuda()
			item_feature = batch_data['item_feature'].cuda()
			rating = batch_data['rating'].cuda()

			model.zero_grad()

			prediction = model(user, item, user_feature, item_feature)
			loss = criterion(prediction, rating)

			loss.backward()
			optimizer.step()

			# writer.add_scalar('data/mse_loss', loss.data.item(),
			# 				epoch*len(dataloader_train)+idx)

		print("Epoch %d training is done!" %epoch)

		# Start testing
		model.eval() 
		model.is_training = False
		RMSE = evaluation.metrics(model, dataloader_test)
			
		elapsed_time = time.time() - start_time
		print("Epoch: %d\t" %epoch + "Epoch time: " + time.strftime(
						"%H: %M: %S", time.gmtime(elapsed_time)))
		print("RMSE is %.3f.\n" %RMSE)


if __name__ == "__main__":
	main()
