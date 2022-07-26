from flask import request as route_req
from keras.datasets import mnist
import torch
import json

# print('importing data')

raw_data = mnist.load_data()

# print('preprocessing')

shard_size = 500

x_train_raw = raw_data[0][0]
y_train_raw = raw_data[0][1]
x_test_raw = raw_data[1][0]
y_test_raw = raw_data[1][1]

# formatting into sample shape

# we'll use MNSIT in two shapes, flat samples for feed-forward networks and 2D samples for convolutional ones
img_sample_shape = [28, 28, 1]
flat_sample_shape = [784]

img_x_train = torch.tensor(x_train_raw, dtype=torch.float32).reshape([-1]+img_sample_shape)
img_x_test = torch.tensor(x_test_raw, dtype=torch.float32).reshape([-1]+img_sample_shape)

flat_x_train = torch.tensor(x_train_raw, dtype=torch.float32).reshape([-1]+flat_sample_shape)
flat_x_test = torch.tensor(x_test_raw, dtype=torch.float32).reshape([-1]+flat_sample_shape)

y_train = torch.tensor(y_train_raw, dtype=torch.long)
y_test = torch.tensor(y_test_raw, dtype=torch.long)

# splitting into shards

num_train_shards = y_train.shape[0]//shard_size
num_test_shards = y_test.shape[0]//shard_size

# cutting off the samples that won't be in any shard
img_x_train = img_x_train[0: num_train_shards*shard_size]
flat_x_train = flat_x_train[0: num_train_shards*shard_size]
img_x_test = img_x_test[0: num_test_shards*shard_size]
flat_x_test = flat_x_test[0: num_test_shards*shard_size]

y_train = y_train[0: num_train_shards*shard_size]
y_test = y_test[0: num_test_shards*shard_size]

# reshaping
img_x_train = img_x_train.unsqueeze(dim=0).reshape([num_train_shards, -1]+img_sample_shape)
flat_x_train = flat_x_train.unsqueeze(dim=0).reshape([num_train_shards, -1]+flat_sample_shape)
img_x_test = img_x_test.unsqueeze(dim=0).reshape([num_test_shards, -1]+img_sample_shape)
flat_x_test = flat_x_test.unsqueeze(dim=0).reshape([num_test_shards, -1]+flat_sample_shape)

y_train = y_train.unsqueeze(dim=0).reshape([num_train_shards, -1])
y_test = y_test.unsqueeze(dim=0).reshape([num_test_shards, -1])