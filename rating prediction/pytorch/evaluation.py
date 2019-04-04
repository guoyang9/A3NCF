from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np 

def metrics(model, dataloader_test):
	RMSE = np.array([], dtype=np.float32)
	for idx, batch_data in enumerate(dataloader_test):
		user = batch_data['user'].cuda()
		item = batch_data['item'].cuda()
		user_feature = batch_data['user_feature'].cuda()
		item_feature = batch_data['item_feature'].cuda()
		rating = batch_data['rating'].cuda()

		prediction = model(user, item, user_feature, item_feature)
		SE = (prediction-rating).pow(2)
		RMSE = np.append(RMSE, SE.detach().cpu().numpy())

	return np.sqrt(RMSE.mean())

