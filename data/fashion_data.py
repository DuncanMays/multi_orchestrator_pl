import tensorflow as tf
import torch

fashion_mnist = tf.keras.datasets.fashion_mnist

# loading the data, downloading if needed
(train_images, train_labels), (test_images, test_labels) = fashion_mnist.load_data()
# convertion to pytorch
(train_images, train_labels, test_images, test_labels) = (torch.tensor(train_images, dtype=torch.float32), torch.tensor(train_labels, dtype=torch.long), torch.tensor(test_images, dtype=torch.float32), torch.tensor(test_labels, dtype=torch.long))
# reshaping
(train_images, train_labels, test_images, test_labels) = (train_images.reshape([-1, 1, 28, 28]), train_labels, test_images.reshape([-1, 1, 28, 28]), test_labels)
# normalizing
train_images = train_images
test_images = test_images

shard_size = 500
img_sample_shape = [1, 28, 28]

num_train_shards = train_images.shape[0]//shard_size
num_test_shards = test_images.shape[0]//shard_size

# cutting off the samples that won't be in any shard
img_x_train = train_images[0: num_train_shards*shard_size]
img_x_test = test_images[0: num_test_shards*shard_size]

y_train = train_labels[0: num_train_shards*shard_size]
y_test = test_labels[0: num_test_shards*shard_size]

# reshaping
fashion_train_images = img_x_train.unsqueeze(dim=0).reshape([num_train_shards, -1]+img_sample_shape)
fashion_test_images = img_x_test.unsqueeze(dim=0).reshape([num_test_shards, -1]+img_sample_shape)
fashion_train_labels = y_train.unsqueeze(dim=0).reshape([num_train_shards, -1])
fashion_test_labels = y_test.unsqueeze(dim=0).reshape([num_test_shards, -1])
