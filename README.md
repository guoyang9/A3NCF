# A3NCF

A pytorch and tensorflow GPU implementation for the paper:

Zhiyong Cheng, Ying Ding, Lei Zhu, Mohan Kankanhalli (2018). Aspect-Aware Latent Factor Model:  Rating Prediction with Ratings and Reviews.

**Please cite the IJCAI'18 paper if you use our codes. Thanks!**


## The requirements are as follows:
* python 3.5

* pytorch 0.4.0

* tensorflow 1.7

## Example to run:
* Make sure you have the LDA features in the right directory.

* Train and test the code.
```
python main.py --dataset=somedata --embed_size=20 --gpu=0
```
