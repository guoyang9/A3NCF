# A3NCF

A pytorch and tensorflow GPU implementation for the paper:

Zhiyong Cheng, Ying Ding, Lei Zhu, Mohan Kankanhalli (2018). Aspect-Aware Latent Factor Model:  Rating Prediction with Ratings and Reviews.

**Please cite the IJCAI'18 paper if you use our codes. Thanks!**


## The requirements are as follows:
* python==3.6

* pytorch==1.0.1

* tensorflow==1.7

* pandas==0.24.2

* numpy==1.16.2

## Example to run:
* Make sure you have the LDA features in the right directory.

* Train and test the code.
	```
	python main.py --dataset=somedata --embed_size=20 --gpu=0
	```
